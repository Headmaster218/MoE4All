//! [`ParallelSeam`] — the N-slot concurrent generation engine behind `infr serve --parallel N`.
//!
//! # What this is (and what it is not)
//!
//! Every sequence owns one KV slot. Supported Qwen hybrid text requests register with one persistent compute
//! worker before tokenization, then enqueue their prepared task and remain under that scheduler
//! through completion; no request thread doubles as the cohort leader. Decode rows run
//! layer-synchronously. Up to 96 uncached prompt tokens advance as teacher-forced rows in the same
//! Decode-LRU graphs; longer prompts share the configured ubatch in one exclusive Prefill-ring
//! phase before decode resumes. Stateless work and paged expert traffic are shared while every
//! sequence retains independent positions, KV/recurrent state and sampling. Qwen3.8 dense/sparse
//! QSA rows form compatible subgroups but never fall back to independent request loops. Other architectures
//! and constrained generation retain the established interleaved path through
//! [`crate::sampling::StepGate`].
//!
//! # VRAM: how `-np` interacts with `--ctx`
//!
//! N slots means N independent KV/recurrent states. The fit solver prices all N states plus one
//! shared runtime workspace and returns the maximum per-slot context, so raising `-np` cannot OOM
//! a device that `-np 1` fit. (It is NOT the same footprint — when the trained window is below the
//! fit, `-np 4` may allocate more total state than `-np 1`; what it cannot do is exceed the budget.
//! The visible cost can be a smaller per-request window.) An
//! explicit `--ctx C` is used verbatim per slot, and the Vulkan alloc-time budget guard is left to
//! fail it cleanly if `N * C` truly doesn't fit.
//!
//! Slots are forked EAGERLY at startup (weights are shared through `Arc<SeamWeights>`; a fork costs
//! only its own KV + IO buffers). That means a VRAM refusal happens at boot with a clear message,
//! never halfway through serving.
//!
//! When `kv.session_cache_dir` is explicitly set, a background worker streams idle dynamic Q8 KV
//! state to checksummed files and releases its physical 32K segments. A later prefix match restores
//! that state into any free resident slot. This extends the number of retained conversations; it
//! does not alter the scheduler policy described above.

#[path = "parallel_report.rs"]
mod parallel_report;

use crate::paged_cache::PagedKv;
use crate::ram_cache::{self, RadixCache};
use crate::sampling::{ParallelSampler, RequestCtx, RequestSampling, StepGate};
use crate::seam::token_delivery::{self, ParallelTokenCallback, TokenDelivery};
use crate::seam::{SeamKv, SegmentedKvLayout};
use crate::session_cache::{MtpCacheSource, SessionCache};
use crate::{Config, GenStats, SeamModel};
use anyhow::{anyhow, Context, Result};
use infr_core::{backend::Backend, WeightSource};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const DECODE_BATCH_WAIT: Duration = Duration::from_secs(1);
const PREFILL_COHORT_WAIT: Duration = Duration::from_millis(50);
const MAX_DECODE_BATCH: usize = infr_core::backend::MAX_DECODE_ROWS;
const SHORT_PREFILL_TOKENS: usize = 96;
const MTP_PLAIN_SYNC_QUANTUM: usize = 1024;
const MTP_CONCURRENT_PROBE_CYCLES: usize = 8;
const MTP_CONCURRENT_MIN_ACCEPT_PERCENT: usize = 80;
const LOOKUP_CALIBRATION_TOKENS: usize = 16;
const LOOKUP_COST_COOLDOWN_TOKENS: usize = 128;

/// Disambiguates default disk-backing filenames when one process builds several paged trees
/// (the temp path is per-process + per-seq; an explicit `kv.disk_cache_path` bypasses this).
static DISK_BACKING_SEQ: AtomicUsize = AtomicUsize::new(0);

/// Aux (recurrent/PLE) snapshot cap for the corridor-only tree (`kv.paged` with no
/// `kv.paged_cache_max`), where the RAM tier has no size to take an eighth of. One Qwen3.8 boundary
/// snapshot measures ~113 MiB, so this holds a handful of resumable frontiers without pretending to
/// be a prefix cache. A `kv.paged_cache_max` pool still derives its own (larger) budget as before.
const PAGED_AUX_CORRIDOR_BUDGET: usize = 512 << 20;

/// One projector result handed to the text model. Kept backend-neutral so `infr-llama` does not
/// depend on the optional vision crate.
#[derive(Clone)]
pub struct MultimodalEmbedding {
    pub values: Arc<Vec<f32>>,
    pub grid_nx: usize,
    pub grid_ny: usize,
    pub fingerprint: [u8; 32],
}

type MultimodalKey = [u8; 32];

fn multimodal_key(images: &[MultimodalEmbedding]) -> MultimodalKey {
    let mut hash = Sha256::new();
    hash.update((images.len() as u64).to_le_bytes());
    for image in images {
        hash.update(image.fingerprint);
        hash.update((image.grid_nx as u64).to_le_bytes());
        hash.update((image.grid_ny as u64).to_le_bytes());
    }
    hash.finalize().into()
}

fn expand_multimodal_prompt(
    tokens: &[u32],
    image_pad_id: u32,
    images: Vec<MultimodalEmbedding>,
    n_embd: usize,
) -> Result<(Vec<u32>, crate::seam::MropePlan)> {
    let mut expanded = Vec::new();
    let mut positions4 = Vec::new();
    let mut spans = Vec::with_capacity(images.len());
    let mut images = images.into_iter();
    let mut used = 0usize;
    let mut cursor = 0i32;

    for &token in tokens {
        if token != image_pad_id {
            expanded.push(token);
            positions4.extend_from_slice(&[cursor, cursor, cursor, 0]);
            cursor = cursor
                .checked_add(1)
                .ok_or_else(|| anyhow!("multimodal position overflow"))?;
            continue;
        }

        let image = images.next().ok_or_else(|| {
            anyhow!("rendered prompt contains more <|image_pad|> markers than image payloads")
        })?;
        used += 1;
        let n_tokens = image
            .grid_nx
            .checked_mul(image.grid_ny)
            .ok_or_else(|| anyhow!("image #{used} token grid overflows"))?;
        let expected_values = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| anyhow!("image #{used} embedding size overflows"))?;
        if n_tokens == 0 || image.values.len() != expected_values {
            return Err(anyhow!(
                "image #{used} projector output has {} values for a {}x{} grid; expected {}",
                image.values.len(),
                image.grid_nx,
                image.grid_ny,
                expected_values
            ));
        }
        let start = expanded.len();
        for index in 0..n_tokens {
            let y = i32::try_from(index / image.grid_nx)
                .map_err(|_| anyhow!("image #{used} row exceeds i32"))?;
            let x = i32::try_from(index % image.grid_nx)
                .map_err(|_| anyhow!("image #{used} column exceeds i32"))?;
            expanded.push(image_pad_id);
            positions4.extend_from_slice(&[
                cursor,
                cursor
                    .checked_add(y)
                    .ok_or_else(|| anyhow!("image #{used} row position overflow"))?,
                cursor
                    .checked_add(x)
                    .ok_or_else(|| anyhow!("image #{used} column position overflow"))?,
                0,
            ]);
        }
        let extent = i32::try_from(image.grid_nx.max(image.grid_ny))
            .map_err(|_| anyhow!("image #{used} grid exceeds i32"))?;
        cursor = cursor
            .checked_add(extent)
            .ok_or_else(|| anyhow!("image #{used} position overflow"))?;
        spans.push(crate::seam::ImageSpanEmbeds {
            start,
            n_tokens,
            embeds: image.values,
        });
    }
    if images.next().is_some() {
        return Err(anyhow!(
            "request carries more image payloads than rendered <|image_pad|> markers"
        ));
    }
    Ok((
        expanded,
        crate::seam::MropePlan {
            prompt_pos4: positions4,
            spans,
            decode_base: cursor,
        },
    ))
}

fn multimodal_token_position(plan: Option<&crate::seam::MropePlan>, token: usize) -> Result<i32> {
    let Some(plan) = plan else {
        return i32::try_from(token).map_err(|_| anyhow!("text position exceeds i32"));
    };
    let prompt_len = plan.prompt_pos4.len() / 4;
    if token < prompt_len {
        return Ok(plan.prompt_pos4[token * 4]);
    }
    let delta = i32::try_from(token - prompt_len)
        .map_err(|_| anyhow!("multimodal decode position exceeds i32"))?;
    plan.decode_base
        .checked_add(delta)
        .ok_or_else(|| anyhow!("multimodal decode position overflow"))
}

fn should_handoff_multimodal_to_scheduler(
    scheduler_available: bool,
    mtp_heads_present: bool,
    frontier_is_image: bool,
    max_new: usize,
) -> bool {
    scheduler_available && !mtp_heads_present && !frontier_is_image && max_new > 0
}

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
    /// Set only when the opt-in cold session cache is active. Ordinary serving does not read the
    /// wall clock during slot selection or return.
    idle_since: Option<Instant>,
    /// Image identity for multimodal KV. `None` marks ordinary token-only state.
    multimodal_key: Option<MultimodalKey>,
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
enum BatchEvent {
    Progress {
        progress: infr_core::GenerationProgress,
    },
    Token {
        id: u32,
        progress: infr_core::GenerationProgress,
    },
    Complete {
        kv: SeamKv,
        stats: GenStats,
    },
    Failed {
        kv: SeamKv,
        error: String,
    },
}

impl BatchEvent {
    fn into_kv(self) -> Option<SeamKv> {
        match self {
            Self::Progress { .. } | Self::Token { .. } => None,
            Self::Complete { kv, .. } | Self::Failed { kv, .. } => Some(kv),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchPhase {
    Unprepared,
    ShortPrefill,
    LongPrefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SchedulerMode {
    DecodeRows,
    ShortPrefillRows,
    DecodeAndShortPrefillRows,
    LongPrefill,
}

fn scheduler_mode(phases: impl IntoIterator<Item = BatchPhase>) -> Option<SchedulerMode> {
    let mut decode = false;
    let mut short_prefill = false;
    let mut any = false;
    for phase in phases {
        any = true;
        match phase {
            BatchPhase::Unprepared => unreachable!("scheduler classified an unprepared task"),
            BatchPhase::LongPrefill => return Some(SchedulerMode::LongPrefill),
            BatchPhase::ShortPrefill => short_prefill = true,
            BatchPhase::Decode => decode = true,
        }
    }
    match (any, decode, short_prefill) {
        (false, _, _) => None,
        (true, true, true) => Some(SchedulerMode::DecodeAndShortPrefillRows),
        (true, true, false) => Some(SchedulerMode::DecodeRows),
        (true, false, true) => Some(SchedulerMode::ShortPrefillRows),
        (true, false, false) => unreachable!("a prepared batch has a known phase"),
    }
}

fn phase_for_remaining_prefill(tokens: usize) -> BatchPhase {
    if tokens == 0 {
        BatchPhase::Decode
    } else if tokens <= SHORT_PREFILL_TOKENS {
        BatchPhase::ShortPrefill
    } else {
        BatchPhase::LongPrefill
    }
}

fn long_prefill_phase_prompt_end(
    start: usize,
    prompt_end: usize,
    checkpoint_boundaries: &[Option<usize>; crate::seam::TURN_CHECKPOINT_COUNT],
) -> usize {
    let frontier = prompt_end.saturating_sub(1);
    checkpoint_boundaries
        .iter()
        .flatten()
        .copied()
        .filter(|&boundary| {
            boundary > start && boundary < frontier && frontier - boundary <= SHORT_PREFILL_TOKENS
        })
        .max()
        .map(|boundary| boundary + 1)
        .unwrap_or(prompt_end)
}

fn long_prefill_cohort_limit(config: &Config) -> usize {
    if config.qwen4exp {
        MAX_DECODE_BATCH
    } else {
        1
    }
}

fn mtp_accepted_rows(candidates: &[u32], verified: &[u32]) -> usize {
    crate::speculative_verify::accepted_rows(candidates, verified)
        .expect("MTP Verify returned validated 2..=4-row results")
}

fn mtp_verify_rows(k: usize) -> usize {
    k.clamp(2, crate::mtp::DRAFT_TOKENS)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LookupTokenDelivery {
    Disconnected,
    Continue,
    Stop,
}

#[derive(Debug, PartialEq, Eq)]
struct LookupDelivery {
    accepted_drafts: usize,
    keep_inputs: usize,
    new_tokens: usize,
    stopped: bool,
}

/// Pending was emitted before Verify. Publish the matching candidates and then the target
/// prediction after the last accepted input. That bonus is emitted but never a retained row.
fn deliver_lookup_prefix(
    candidates: &[u32],
    verified: &[u32],
    remaining: usize,
    mut publish: impl FnMut(u32) -> LookupTokenDelivery,
) -> Result<LookupDelivery> {
    let accepted = crate::speculative_verify::accepted_rows(candidates, verified)?;
    let mut delivery = LookupDelivery {
        accepted_drafts: accepted - 1,
        keep_inputs: 1,
        new_tokens: 0,
        stopped: remaining == 0,
    };
    for row in 0..accepted.min(remaining) {
        let token = if row < accepted - 1 {
            candidates[row]
        } else {
            verified[accepted - 1]
        };
        let result = publish(token);
        if result == LookupTokenDelivery::Disconnected {
            delivery.stopped = true;
            break;
        }
        delivery.new_tokens += 1;
        if row < accepted - 1 {
            delivery.keep_inputs += 1;
        }
        if result == LookupTokenDelivery::Stop {
            delivery.stopped = true;
            break;
        }
    }
    delivery.stopped |= delivery.new_tokens == remaining;
    Ok(delivery)
}

fn lookup_greedy_eligible(req: &RequestCtx, cfg: &infr_core::config::SamplingCfg) -> bool {
    let effective = crate::sampling::Sampler::resolve(Some(req), cfg);
    (effective.temp <= 0.0 || effective.top_k == 1) && !req.sampling().penalties_active()
}

fn lookup_ordinary_sample_eligible(phase: BatchPhase, generated: usize, lanes: usize) -> bool {
    phase == BatchPhase::Decode && generated > 0 && lanes == 1
}

fn lookup_draft_budget(
    configured: usize,
    remaining_output: usize,
    history_len: usize,
    max_ctx: usize,
    qsa_threshold: usize,
) -> usize {
    let before_boundary = if history_len <= qsa_threshold {
        qsa_threshold - history_len
    } else {
        usize::MAX
    };
    configured
        .min(crate::speculative_verify::MAX_VERIFY_ROWS - 1)
        .min(remaining_output.saturating_sub(1))
        .min(max_ctx.saturating_sub(history_len))
        .min(before_boundary)
}

fn lookup_scheduler_supported(
    cfg: &Config,
    ec: &crate::EngineConfig,
    embed_gather: bool,
    embedding_dtype: Option<infr_core::DType>,
) -> bool {
    ec.spec.lookup
        && crate::seam::target_verify_supported(cfg)
        && ec.spec.gpu_embed
        && embed_gather
        && embedding_dtype.is_some_and(infr_vulkan::linear::embed_gather_supported)
}

fn scheduler_requested(
    n_slots: usize,
    heads_loaded: bool,
    lookup_enabled: bool,
    lookup_supported: impl FnOnce() -> bool,
) -> bool {
    n_slots > 1 || heads_loaded || (lookup_enabled && lookup_supported())
}

fn lookup_fallback_reason(
    cfg: &Config,
    ec: &crate::EngineConfig,
    req: &RequestCtx,
    heads_loaded: bool,
    embed_gather: bool,
    embedding_dtype: Option<infr_core::DType>,
) -> Option<&'static str> {
    if !ec.spec.lookup {
        return Some("lookup_disabled");
    }
    if heads_loaded {
        return Some("mtp_head_loaded");
    }
    if !crate::seam::target_verify_supported(cfg) {
        return Some("unsupported_target_state");
    }
    if !ec.spec.gpu_embed {
        return Some("gpu_embedding_disabled");
    }
    if (0..cfg.n_layer).any(|layer| cfg.is_recurrent_layer(layer)) {
        if !ec.kernels.vulkan.dn_chunk {
            return Some("gdn_chunk_disabled");
        }
        if !ec.kernels.vulkan.dn_chunk_scan {
            return Some("gdn_chunk_scan_disabled");
        }
        if !ec.kernels.vulkan.dn_split {
            return Some("gdn_split_disabled");
        }
    }
    if !crate::seam::target_verify_enabled(cfg, ec) {
        return Some("target_verify_disabled");
    }
    if !embed_gather {
        return Some("backend_embed_gather_unsupported");
    }
    if ec.spec.lookup_max_draft == 0 {
        return Some("zero_draft_budget");
    }
    if req.sampling().penalties_active() {
        return Some("request_penalties_active");
    }
    if !lookup_greedy_eligible(req, &ec.sampling) {
        return Some("request_sampling_not_greedy");
    }
    match embedding_dtype {
        None => Some("token_embedding_missing"),
        Some(dtype) if !infr_vulkan::linear::embed_gather_supported(dtype) => {
            Some("token_embedding_dtype_unsupported")
        }
        Some(_) => None,
    }
}

fn repair_lookup_predictions(
    cfg: &Config,
    mut guard: crate::sampling::Qwen4ReasoningEosGuard,
    candidates: &[u32],
    ids: &mut [u32],
    logits: &mut [f32],
    ignore_eos: bool,
) -> Result<()> {
    crate::speculative_verify::accepted_rows(candidates, ids)?;
    let rows = ids.len();
    for row in 0..rows {
        // The persistent guard already observed pending during ordinary delivery.
        if row > 0 {
            guard.observe(cfg, candidates[row - 1]);
        }
        crate::sampling::ensure_sampled_token(ids[row], cfg.vocab, "Lookup Verify")?;
        if !guard.blocks(cfg, ids[row], ignore_eos) {
            continue;
        }
        let expected = rows
            .checked_mul(cfg.vocab)
            .context("Lookup logits size overflow")?;
        anyhow::ensure!(
            logits.len() == expected,
            "Lookup EOS repair requires full target logits"
        );
        let row_logits = &mut logits[row * cfg.vocab..(row + 1) * cfg.vocab];
        // Validate before the guard intentionally masks EOS to negative infinity.
        crate::sampling::ensure_finite_logits(row_logits, "Lookup EOS repair")?;
        guard.mask_eos(cfg, row_logits);
        anyhow::ensure!(
            row_logits.iter().any(|value| value.is_finite()),
            "Lookup EOS repair masked every target token"
        );
        ids[row] = crate::sampling::argmax(row_logits) as u32;
    }
    Ok(())
}

#[derive(Default)]
struct LookupBatchStats {
    searches: usize,
    cycles: usize,
    drafted: usize,
    accepted: usize,
    delivered: usize,
    ordinary_fallbacks: usize,
    cost_backoffs: usize,
    search_secs: f64,
    snapshot_secs: f64,
    verify_secs: f64,
    restore_secs: f64,
    cycle_secs: f64,
}

struct LookupBatchState {
    proposer: crate::speculative_lookup::LookupProposer,
    indexed_len: usize,
    ordinary_token_secs: Option<f64>,
    calibration_tokens: usize,
    calibration_secs: f64,
    pending_proposal: Option<(usize, crate::speculative_lookup::Candidates, f64)>,
    slow_cycles: usize,
    cost_cooldown: usize,
    missed_search_secs: f64,
    stats: LookupBatchStats,
}

impl LookupBatchState {
    fn new(config: crate::speculative_lookup::LookupConfig, prompt: &[u32]) -> Result<Self> {
        let mut proposer = crate::speculative_lookup::LookupProposer::new(config)
            .map_err(|error| anyhow!(error))?;
        proposer.sync(prompt);
        Ok(Self {
            proposer,
            indexed_len: prompt.len(),
            ordinary_token_secs: None,
            calibration_tokens: 0,
            calibration_secs: 0.0,
            pending_proposal: None,
            slow_cycles: 0,
            cost_cooldown: 0,
            missed_search_secs: 0.0,
            stats: LookupBatchStats::default(),
        })
    }

    fn append_confirmed(&mut self, tokens: &[u32]) {
        if !tokens.is_empty() {
            if let Some((_, _, seconds)) = self.pending_proposal.take() {
                self.missed_search_secs += seconds;
            }
        }
        self.proposer.append_confirmed(tokens);
        self.indexed_len += tokens.len();
    }

    fn sync_history(&mut self, history: &[u32]) {
        if let Some((_, _, seconds)) = self.pending_proposal.take() {
            self.missed_search_secs += seconds;
        }
        self.proposer.sync(history);
        self.indexed_len = history.len();
    }

    fn finish_ordinary_outputs(&mut self, prompt_len: usize, outputs: &[u32]) -> Result<()> {
        let indexed = self
            .indexed_len
            .checked_sub(prompt_len)
            .ok_or_else(|| anyhow!("Lookup ordinary callback lost its history frontier"))?;
        anyhow::ensure!(
            indexed <= outputs.len(),
            "Lookup indexed extra ordinary outputs"
        );
        // EOS bypasses on_token. Other outputs were already indexed after their ACK.
        let tail = &outputs[indexed..];
        self.append_confirmed(tail);
        self.cost_cooldown = self.cost_cooldown.saturating_sub(tail.len());
        Ok(())
    }

    fn record_ordinary(&mut self, seconds: f64, tokens: usize) {
        self.stats.ordinary_fallbacks += 1;
        if tokens > 0 && seconds.is_finite() && seconds > 0.0 {
            let measured = seconds / tokens as f64;
            if let Some(previous) = self.ordinary_token_secs {
                // Proposal-triggered re-entry setup is not continuous Decode cost.
                // Only amortized ordinary chunks may refresh the reference rate.
                if tokens >= LOOKUP_CALIBRATION_TOKENS {
                    self.ordinary_token_secs = Some(0.75 * previous + 0.25 * measured);
                }
            } else {
                self.calibration_tokens += tokens;
                self.calibration_secs += seconds;
                if self.calibration_tokens >= LOOKUP_CALIBRATION_TOKENS {
                    self.ordinary_token_secs =
                        Some(self.calibration_secs / self.calibration_tokens as f64);
                }
            }
        }
    }

    fn search(&mut self, budget: usize) -> (Option<crate::speculative_lookup::Candidates>, f64) {
        if budget == 0 || self.cost_cooldown > 0 {
            return (None, 0.0);
        }
        let t0 = Instant::now();
        let timeline = infr_core::timeline::span("lookup_search");
        let candidates = self.proposer.propose(None, budget);
        drop(timeline);
        let seconds = t0.elapsed().as_secs_f64();
        self.stats.searches += 1;
        self.stats.search_secs += seconds;
        if candidates.is_none() {
            self.missed_search_secs += seconds;
        }
        (candidates, seconds)
    }

    /// Index once after delivery/ACK. A proposal requests a scheduler yield, not request completion.
    fn observe_ordinary_token(
        &mut self,
        token: u32,
        acknowledged: bool,
        single_decode: bool,
        budget: usize,
        interrupt: &AtomicBool,
    ) -> f64 {
        self.append_confirmed(&[token]);
        if self.cost_cooldown > 0 {
            self.cost_cooldown -= 1;
            return 0.0;
        }
        if !acknowledged
            || !single_decode
            || self.ordinary_token_secs.is_none()
            || interrupt.load(Ordering::Acquire)
        {
            return 0.0;
        }
        let (candidates, seconds) = self.search(budget);
        if let Some(candidates) = candidates {
            self.pending_proposal = Some((self.indexed_len, candidates, seconds));
            interrupt.store(true, Ordering::Release);
        }
        seconds
    }

    fn take_pending_proposal(
        &mut self,
        budget: usize,
    ) -> Option<(crate::speculative_lookup::Candidates, f64)> {
        let (frontier, mut candidates, seconds) = self.pending_proposal.take()?;
        if frontier != self.indexed_len || budget == 0 || self.cost_cooldown > 0 {
            self.missed_search_secs += seconds;
            return None;
        }
        candidates.tokens.truncate(budget);
        Some((candidates, seconds))
    }

    /// Compare measured complete CPU/GPU cycle time per newly delivered token with ordinary
    /// decode, excluding channel/ACK delivery from both. Full acceptance can still lose on cost.
    fn record_cycle(&mut self, accepted: usize, produced: usize, advanced: usize, seconds: f64) {
        self.proposer.record_result(accepted, produced);
        self.stats.cycles += 1;
        self.stats.drafted += produced;
        self.stats.accepted += accepted;
        self.stats.delivered += advanced;
        let seconds = seconds + std::mem::take(&mut self.missed_search_secs);
        self.stats.cycle_secs += seconds;
        if advanced == 0 {
            return;
        }
        if let Some(ordinary) = self.ordinary_token_secs {
            if seconds / advanced as f64 >= ordinary * 1.05 {
                self.slow_cycles += 1;
                if self.slow_cycles >= 2 {
                    self.cost_cooldown = LOOKUP_COST_COOLDOWN_TOKENS;
                    self.stats.cost_backoffs += 1;
                    self.slow_cycles = 0;
                }
            } else {
                self.slow_cycles = 0;
            }
        }
    }

    fn report(&self, slot: usize) {
        let stats = &self.stats;
        let outcome = if stats.cycles > 0 {
            "verified"
        } else if stats.searches > 0 {
            "ordinary_only_no_candidates"
        } else {
            "ordinary_only_no_eligible_verify_window"
        };
        tracing::info!(
            slot, searches = stats.searches, cycles = stats.cycles,
            outcome,
            drafted = stats.drafted, accepted = stats.accepted, delivered = stats.delivered,
            ordinary_fallbacks = stats.ordinary_fallbacks, cost_backoffs = stats.cost_backoffs,
            search_ms = stats.search_secs * 1000.0,
            snapshot_ms = stats.snapshot_secs * 1000.0,
            verify_ms = stats.verify_secs * 1000.0,
            restore_ms = stats.restore_secs * 1000.0,
            cycle_ms = stats.cycle_secs * 1000.0,
            ordinary_token_ms = ?self.ordinary_token_secs.map(|seconds| seconds * 1000.0),
            "independent Lookup request summary"
        );
    }
}

fn lookup_plain_quantum(
    lanes: usize,
    generated: usize,
    remaining_prefill: usize,
    lookup: Option<&LookupBatchState>,
) -> usize {
    let Some(lookup) = lookup.filter(|_| lanes == 1) else {
        return usize::MAX;
    };
    if generated == 0 {
        remaining_prefill.saturating_add(1)
    } else if lookup.ordinary_token_secs.is_none() {
        LOOKUP_CALIBRATION_TOKENS
            .saturating_sub(lookup.calibration_tokens)
            .max(1)
    } else {
        usize::MAX
    }
}

fn concurrent_mtp_profitable(accepted: usize, drafted: usize) -> bool {
    drafted > 0
        && accepted.saturating_mul(100) >= drafted.saturating_mul(MTP_CONCURRENT_MIN_ACCEPT_PERCENT)
}

fn concurrent_mtp_ready(enabled: bool, plain_selected: bool, all_lanes_ready: bool) -> bool {
    enabled && !plain_selected && all_lanes_ready
}

fn mtp_context_profitable(max_context: usize, visible_tokens: usize) -> bool {
    max_context == 0 || visible_tokens <= max_context
}

#[derive(Default)]
struct MtpBatchCycleStats {
    accepted: usize,
    drafted: usize,
}

struct BatchWork {
    slot: usize,
    kv: Option<SeamKv>,
    prompt: Vec<u32>,
    prompt_end: usize,
    max_new: usize,
    generated: usize,
    stats: GenStats,
    phase: BatchPhase,
    prefill_start: usize,
    checkpoint_boundaries: [Option<usize>; crate::seam::TURN_CHECKPOINT_COUNT],
    /// Paged tier: the page-aligned prefill position at which this work's slot captures the
    /// recurrent/PLE snapshot its radix entry publishes (see
    /// [`crate::seam::PreparedParallelPrompt::paged_frontier`]). Cleared once crossed, like a
    /// turn checkpoint — a later phase stopping at the same position would otherwise capture the
    /// state AFTER the boundary and publish it under the boundary's label.
    paged_frontier: Option<usize>,
    turn_checkpoint: Option<crate::seam::TurnCheckpoint>,
    mrope_plan: Option<crate::seam::MropePlan>,
    multimodal_key: Option<MultimodalKey>,
    finished: bool,
    sampling: RequestSampling,
    sampler: Option<ParallelSampler>,
    reasoning_guard: crate::sampling::Qwen4ReasoningEosGuard,
    mtp_ready: bool,
    lookup: Option<LookupBatchState>,
    channels: Option<BatchChannels>,
}

struct MtpSlotState {
    head: Arc<crate::mtp::Qwen4MtpSession>,
    cached: Vec<u32>,
    last_h: Vec<f32>,
    turn_checkpoints:
        [Option<crate::mtp::Qwen4MtpCheckpointState>; crate::seam::TURN_CHECKPOINT_COUNT],
    pending_id: Option<u32>,
    /// Whether `pending_id` has already been streamed by ordinary decode. MTP still has to feed
    /// that row into the target, but must not stream or count it a second time.
    pending_emitted: bool,
    /// Ordinary decode advances the live recurrent state beyond the last speculative checkpoint.
    target_snapshot_dirty: bool,
    multimodal_key: Option<MultimodalKey>,
    live_valid: bool,
}

impl MtpSlotState {
    fn invalidate_all(&mut self) {
        self.cached.clear();
        self.last_h.fill(0.0);
        self.turn_checkpoints.fill(None);
        self.pending_id = None;
        self.pending_emitted = false;
        self.target_snapshot_dirty = false;
        self.multimodal_key = None;
        self.live_valid = false;
    }

    fn invalidate_live(&mut self) {
        self.pending_id = None;
        self.pending_emitted = false;
        self.target_snapshot_dirty = false;
        self.live_valid = false;
    }

    fn cache_state_for(&self, target: &[u32]) -> Option<crate::mtp::Qwen4MtpCacheState> {
        (self.live_valid && self.multimodal_key.is_none() && self.cached == target).then(|| {
            crate::mtp::Qwen4MtpCacheState {
                cached: self.cached.clone(),
                last_h: self.last_h.clone(),
                turn_checkpoints: self.turn_checkpoints.clone(),
                pending_id: self.pending_id,
                pending_emitted: self.pending_emitted,
            }
        })
    }

    fn install_cache_state(&mut self, state: crate::mtp::Qwen4MtpCacheState) {
        self.cached = state.cached;
        self.last_h = state.last_h;
        self.turn_checkpoints = state.turn_checkpoints;
        self.pending_id = state.pending_id;
        self.pending_emitted = state.pending_emitted;
        self.target_snapshot_dirty = false;
        self.multimodal_key = None;
        self.live_valid = true;
    }

    fn seed_from(
        &mut self,
        backend: &dyn infr_core::backend::Backend,
        source: &MtpSlotState,
        tokens: &[u32],
    ) -> Result<bool> {
        if source.multimodal_key.is_some() || tokens.is_empty() {
            self.invalidate_all();
            return Ok(false);
        }
        let live = (source.live_valid && source.cached == tokens).then(|| {
            (
                source.last_h.clone(),
                source.pending_id,
                source.pending_emitted,
            )
        });
        let checkpoint = source
            .turn_checkpoints
            .iter()
            .flatten()
            .find(|checkpoint| {
                checkpoint.tokens == tokens
                    && source
                        .head
                        .retains_rewind_prefix(source.cached.len(), checkpoint.tokens.len())
            })
            .map(|checkpoint| (checkpoint.last_h.clone(), None, false));
        let Some((last_h, pending_id, pending_emitted)) = live.or(checkpoint) else {
            self.invalidate_all();
            return Ok(false);
        };
        self.head
            .copy_prefix_from(backend, &source.head, tokens.len())?;
        self.cached.clear();
        self.cached.extend_from_slice(tokens);
        self.last_h = last_h;
        self.turn_checkpoints = source.turn_checkpoints.clone().map(|checkpoint| {
            checkpoint.filter(|checkpoint| tokens.starts_with(&checkpoint.tokens))
        });
        self.pending_id = pending_id;
        self.pending_emitted = pending_emitted;
        self.target_snapshot_dirty = false;
        self.multimodal_key = None;
        self.live_valid = true;
        Ok(true)
    }
}

fn mtp_slot_pair_mut(
    slots: &mut [MtpSlotState],
    target: usize,
    source: usize,
) -> (&mut MtpSlotState, &MtpSlotState) {
    debug_assert_ne!(target, source);
    if target < source {
        let (left, right) = slots.split_at_mut(source);
        (&mut left[target], &right[0])
    } else {
        let (left, right) = slots.split_at_mut(target);
        (&mut right[0], &left[source])
    }
}

impl BatchWork {
    fn kv(&self) -> &SeamKv {
        self.kv.as_ref().expect("batch work owns a KV slot")
    }

    fn remaining_prefill(&self) -> usize {
        self.prompt_end
            .saturating_sub(1)
            .saturating_sub(self.kv().cached_len())
    }

    fn next_qsa_position(&self) -> usize {
        self.kv().cached_len() + 1
    }
}

fn token_lane_sort_key(
    has_mrope: bool,
    remaining_prefill: usize,
    slot: usize,
) -> (bool, std::cmp::Reverse<usize>, usize) {
    (!has_mrope, std::cmp::Reverse(remaining_prefill), slot)
}

fn sort_token_lane_indices(active: &[BatchWork], indices: &mut [usize]) {
    indices.sort_by_key(|&index| {
        let work = &active[index];
        token_lane_sort_key(
            work.mrope_plan.is_some(),
            work.remaining_prefill(),
            work.slot,
        )
    });
}

// Return a cohort-relative lane index and a Prefill-only step cap, not a physical KV slot id.
fn token_lane_prefill_fallback(
    lanes: impl IntoIterator<Item = (bool, usize)>,
) -> Option<(usize, usize)> {
    let mut previous = usize::MAX;
    let mut out_of_order = false;
    let mut text_prefill = None;
    for (lane, (has_mrope, remaining)) in lanes.into_iter().enumerate() {
        out_of_order |= previous < remaining;
        previous = remaining;
        if !has_mrope
            && remaining > 0
            && text_prefill.is_none_or(|(_, longest)| remaining > longest)
        {
            text_prefill = Some((lane, remaining));
        }
    }
    out_of_order.then_some(text_prefill).flatten()
}

struct BatchChannels {
    events: SyncSender<BatchEvent>,
    acknowledgements: Receiver<bool>,
}

struct BatchTokenDelivery<'a> {
    engine: &'a ParallelSeam,
    channels: &'a [Option<BatchChannels>],
    progress_bases: &'a [(usize, usize, usize)],
    streamed: &'a mut [usize],
}

impl TokenDelivery for BatchTokenDelivery<'_> {
    fn send_token(&mut self, lane: usize, id: u32) -> bool {
        let timeline_output = infr_core::timeline::span("decode_output_progress");
        self.streamed[lane] += 1;
        let (prompt_tokens, cached_prompt_tokens, generated) = self.progress_bases[lane];
        let progress = self.engine.batch_decode_progress(
            prompt_tokens,
            cached_prompt_tokens,
            generated.saturating_add(self.streamed[lane]),
        );
        drop(timeline_output);
        let _timeline_send = infr_core::timeline::span("decode_output_send");
        self.channels
            .get(lane)
            .and_then(Option::as_ref)
            .is_some_and(|channels| {
                channels
                    .events
                    .send(BatchEvent::Token { id, progress })
                    .is_ok()
            })
    }

    fn receive_ack(&mut self, lane: usize) -> bool {
        let _timeline_ack = infr_core::timeline::span("decode_output_ack_wait");
        self.channels
            .get(lane)
            .and_then(Option::as_ref)
            .is_some_and(|channels| channels.acknowledgements.recv().unwrap_or(false))
    }
}

#[derive(Default)]
struct DecodeBatchQueue {
    /// Eligible requests registered before tokenization/checkout but not yet ready for scheduling.
    arriving: usize,
    waiting: VecDeque<BatchWork>,
}

struct BatchRegistration<'a> {
    engine: &'a ParallelSeam,
    active: bool,
}

impl<'a> BatchRegistration<'a> {
    fn new(engine: &'a ParallelSeam) -> Self {
        engine
            .decode_batch
            .lock()
            .expect("decode batch queue poisoned")
            .arriving += 1;
        Self {
            engine,
            active: true,
        }
    }

    fn arrive(&mut self, queue: &mut DecodeBatchQueue) {
        debug_assert!(self.active);
        queue.arriving = queue.arriving.saturating_sub(1);
        self.active = false;
    }
}

impl Drop for BatchRegistration<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut queue = match self.engine.decode_batch.lock() {
            Ok(queue) => queue,
            Err(error) => error.into_inner(),
        };
        queue.arriving = queue.arriving.saturating_sub(1);
        drop(queue);
        self.engine.decode_ready.notify_all();
    }
}

struct SlotGuard<'a> {
    engine: &'a ParallelSeam,
    idx: usize,
    kv: Option<SeamKv>,
    detached: bool,
    multimodal_key: Option<MultimodalKey>,
}

impl SlotGuard<'_> {
    fn detach(&mut self) -> SeamKv {
        self.detached = true;
        self.kv.take().expect("checked-out slot has KV")
    }

    fn reattach(&mut self, kv: SeamKv) {
        self.kv = Some(kv);
        self.detached = false;
    }

    fn abandon_detached(&mut self) {
        self.kv = None;
        self.detached = false;
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if self.detached {
            return;
        }
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
        s.multimodal_key = self.multimodal_key;
        if self.engine.session_cache.is_some() {
            s.idle_since = Some(Instant::now());
        }
        drop(p);
        if self.engine.session_cache.is_some() {
            // Both a queued request and the deadline worker may be asleep on this condition.
            self.engine.freed.notify_all();
        } else {
            self.engine.freed.notify_one();
        }
    }
}

struct ColdWorker {
    stop: Arc<AtomicBool>,
    wake: Arc<Condvar>,
    thread: Option<JoinHandle<()>>,
}

struct SchedulerWorker {
    stop: Arc<AtomicBool>,
    wake: Arc<Condvar>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for SchedulerWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.wake.notify_all();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::warn!("parallel scheduler thread panicked while shutting down");
            }
        }
    }
}

impl Drop for ColdWorker {
    fn drop(&mut self) {
        tracing::info!("cold KV cache: flushing resident sessions before shutdown");
        self.stop.store(true, Ordering::Release);
        self.wake.notify_all();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::warn!("cold KV maintenance thread panicked while shutting down");
            } else {
                tracing::info!("cold KV cache: shutdown flush complete");
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cold_session_worker(
    pool: Arc<Mutex<Pool>>,
    wake: Arc<Condvar>,
    cache: Arc<Mutex<SessionCache>>,
    backend: Arc<infr_vulkan::VulkanBackend>,
    gate: Option<Arc<StepGate>>,
    mtp_heads: Option<Arc<Mutex<Vec<MtpSlotState>>>>,
    model_cfg: Config,
    idle_for: Duration,
    stop: Arc<AtomicBool>,
) {
    loop {
        let selected = {
            let mut pool_guard = match pool.lock() {
                Ok(pool) => pool,
                Err(poisoned) => poisoned.into_inner(),
            };
            loop {
                let shutting_down = stop.load(Ordering::Acquire);
                let now = Instant::now();
                let mut next_wait: Option<Duration> = None;
                let mut candidate: Option<usize> = None;
                for (index, slot) in pool_guard.slots.iter().enumerate() {
                    let Some(idle_since) = slot.idle_since else {
                        continue;
                    };
                    if slot.busy
                        || slot.multimodal_key.is_some()
                        || slot.kv.as_ref().is_none_or(|kv| kv.cached_len() == 0)
                    {
                        continue;
                    }
                    let elapsed = now.saturating_duration_since(idle_since);
                    if shutting_down || elapsed >= idle_for {
                        if candidate.is_none_or(|old| slot.tick < pool_guard.slots[old].tick) {
                            candidate = Some(index);
                        }
                    } else {
                        let remaining = idle_for - elapsed;
                        next_wait = Some(next_wait.map_or(remaining, |old| old.min(remaining)));
                    }
                }
                if let Some(index) = candidate {
                    pool_guard.slots[index].busy = true;
                    pool_guard.slots[index].idle_since = None;
                    let kv = pool_guard.slots[index]
                        .kv
                        .take()
                        .expect("cold candidate has a resident KV slot");
                    break Some((index, kv, shutting_down));
                }
                if shutting_down {
                    break None;
                }
                pool_guard = match next_wait {
                    Some(timeout) => {
                        wake.wait_timeout(pool_guard, timeout)
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .0
                    }
                    None => wake
                        .wait(pool_guard)
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                };
            }
        };
        let Some((index, mut kv, shutting_down)) = selected else {
            break;
        };

        // This slot stays busy until publication. Own its immutable source independently of the
        // other MTP lanes and use the inference gate only for bounded GPU transfers/release.
        let mtp_snapshot = mtp_heads.as_ref().and_then(|heads| {
            let heads = heads
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            heads[index]
                .cache_state_for(kv.cached_tokens())
                .map(|state| (Arc::clone(&heads[index].head), state))
        });
        let mut writer = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .spill_writer();
        let mtp_source = mtp_snapshot
            .as_ref()
            .map(|(head, state)| MtpCacheSource { head, state });
        let spill_failed = match writer.spill_with_gate(
            &mut kv,
            backend.as_ref(),
            &model_cfg,
            mtp_source,
            gate.as_deref(),
        ) {
            Ok(true) => {
                let _gate = gate.as_deref().map(StepGate::enter);
                if let Some(heads) = mtp_heads.as_ref() {
                    let mut heads = heads
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if let Err(error) = heads[index].head.release_kv(backend.as_ref()) {
                        tracing::warn!(
                            slot = index,
                            "cold KV idle spill could not release stale MTP head KV: {error}"
                        );
                    }
                    heads[index].invalidate_all();
                }
                false
            }
            Ok(false) => kv.cached_len() != 0,
            Err(error) => {
                tracing::warn!(
                    slot = index,
                    "cold KV idle spill failed; keeping the resident state: {error:#}"
                );
                true
            }
        };
        {
            // Serialize GC with restores/catalog publication, but never hold the GPU gate here.
            let mut cache = cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cache.publish_spills(writer);
            if let Err(error) = cache.gc() {
                tracing::warn!("cold KV cache maintenance failed: {error}");
            }
        }

        let mut pool_guard = match pool.lock() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        let slot = &mut pool_guard.slots[index];
        slot.kv = Some(kv);
        slot.busy = false;
        // A failed final spill must not be selected forever while `ColdWorker::drop` waits to join.
        let final_shutdown = shutting_down || stop.load(Ordering::Acquire);
        slot.idle_since = (!final_shutdown || !spill_failed).then(Instant::now);
        drop(pool_guard);
        wake.notify_all();
    }
}

/// The concurrent seam engine. Request threads own tokenization, streaming and one checked-out
/// session; a persistent worker exclusively owns batched compute scheduling.
pub struct ParallelSeam {
    /// Declared first so `Drop` joins both workers before model/backend fields drop.
    scheduler_worker: Option<SchedulerWorker>,
    cold_worker: Option<ColdWorker>,
    model: Arc<SeamModel>,
    vk: Arc<infr_vulkan::VulkanBackend>,
    pool: Arc<Mutex<Pool>>,
    /// Signalled when a slot is returned — a queued request waits here.
    freed: Arc<Condvar>,
    decode_batch: Arc<Mutex<DecodeBatchQueue>>,
    /// Wakes the dedicated scheduler when a registered request becomes ready or abandons arrival.
    decode_ready: Arc<Condvar>,
    /// Set only when the scheduler has new work to admit. The runner polls it once per
    /// aggregated token; an empty steady-state decode takes no queue lock.
    batch_interrupt: Arc<AtomicBool>,
    /// The GPU baton. Ordinary one-slot serving keeps this absent; one-slot MTP uses it to hand
    /// work to the persistent speculative scheduler.
    gate: Option<Arc<StepGate>>,
    /// Opt-in disk-backed conversation catalog. `None` keeps the pre-existing checkout path
    /// byte-for-byte isolated from file I/O and cache locking.
    session_cache: Option<Arc<Mutex<SessionCache>>>,
    /// Opt-in host-RAM radix-tree prefix cache (SGLang-style tier below the VRAM slots).
    /// Exclusive with `session_cache`: `init_session_cache` opens the RAM tree first and the
    /// disk tier stays closed, so nothing is ever written to disk while this is set.
    ram_cache: Option<Arc<Mutex<RadixCache>>>,
    /// Opt-in paged radix prefix cache (`kv.paged_cache_max`): the tree holds FRAME CHAINS inside
    /// one unified-VRAM frame corridor, so a prefix hit is a page-table write (zero byte copies)
    /// and fixed-size frames make VRAM external fragmentation impossible by construction.
    /// Exclusive with `ram_cache`; the disk tier stays closed while this is set.
    paged_cache: Option<Arc<Mutex<PagedKv>>>,
    session_idle: Duration,
    max_ctx: usize,
    /// This model's OWN placement pins (pinned prefill chunk / auto-q8 KV — see
    /// [`crate::seam::PlacementPins`]). Per-engine so a multi-model host (`infr multi` runs N of
    /// these concurrently) never leaks one model's ladder decision to another. Entered as the
    /// current [`crate::seam::PlacementScope`] around placement (construction/warmup) and every
    /// request's decode; concurrent requests on THIS engine all point at this one shared cell.
    pins: Arc<crate::seam::PlacementPins>,
    mtp_heads: Option<Arc<Mutex<Vec<MtpSlotState>>>>,
}

impl ParallelSeam {
    fn batch_decode_progress(
        &self,
        prompt_tokens: usize,
        cached_prompt_tokens: usize,
        completion_tokens: usize,
    ) -> infr_core::GenerationProgress {
        let cached_prompt_tokens = cached_prompt_tokens.min(prompt_tokens);
        infr_core::GenerationProgress {
            phase: infr_core::GenerationPhase::Decode,
            prompt_tokens: prompt_tokens as u64,
            cached_prompt_tokens: cached_prompt_tokens as u64,
            prefill_tokens: prompt_tokens.saturating_sub(cached_prompt_tokens) as u64,
            completion_tokens: completion_tokens as u64,
            context_tokens: prompt_tokens.saturating_add(completion_tokens) as u64,
            context_limit: self.max_ctx as u64,
        }
    }

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
        let detached_mtp = crate::seam::detached_mtp_enabled(model.config(), model.engine_cfg());
        if detached_mtp && n_slots > 2 {
            return Err(anyhow!("Qwen3.8 concurrent MTP supports at most two slots"));
        }
        // This engine's own placement pins, entered as the current scope for the whole
        // placement phase (the clamp inside `vulkan_slot_ctx` + the `init_slots` warmup, which is
        // where the binder pins the prefill chunk / auto-q8 KV). See `PlacementPins`.
        let pins = Arc::new(crate::seam::PlacementPins::for_slots(n_slots));
        let scope = crate::seam::PlacementScope::enter(pins.clone());
        let max_ctx = model.vulkan_slot_ctx(&vk, n_slots, want_ctx)?;
        let mtp_heads = if detached_mtp {
            let sampler = crate::sampling::Sampler::from_cfg(&model.engine_cfg().sampling);
            if sampler.temp > 0.0 && sampler.top_k != 1 {
                tracing::warn!(
                    "Qwen3.8 MTP scheduler is greedy-only; requests use ordinary decode unless temperature is 0 or top_k is 1"
                );
            }
            let sidecar = model
                .engine_cfg()
                .spec
                .draft
                .as_deref()
                .ok_or_else(|| anyhow!("Qwen3.8 MTP needs a draft sidecar"))?;
            let fixed = crate::mtp::Qwen4MtpSession::load_fixed_vulkan_with_vocab(
                &vk,
                sidecar,
                model.config(),
                model.gguf(),
                model.engine_cfg().spec.mtp_vocab.as_deref(),
            )?;
            let catch = crate::mtp::Qwen4MtpSession::shared_catch_workspace(
                &vk,
                &fixed,
                max_ctx,
                crate::seam::ubatch_rows(model.engine_cfg()),
            )?;
            let mut heads = Vec::with_capacity(n_slots);
            for _ in 0..n_slots {
                heads.push(MtpSlotState {
                    head: Arc::new(crate::mtp::Qwen4MtpSession::with_fixed_and_catch(
                        &vk,
                        Arc::clone(&fixed),
                        max_ctx,
                        Arc::clone(&catch),
                        model.engine_cfg().spec.mtp_context,
                    )?),
                    cached: Vec::new(),
                    last_h: vec![0.0; model.config().hc_mult * model.config().n_embd],
                    turn_checkpoints: std::array::from_fn(|_| None),
                    pending_id: None,
                    pending_emitted: false,
                    target_snapshot_dirty: false,
                    multimodal_key: None,
                    live_valid: false,
                });
            }
            Some(Arc::new(Mutex::new(heads)))
        } else {
            None
        };
        let scheduler_enabled = scheduler_requested(
            n_slots,
            mtp_heads.is_some(),
            model.engine_cfg().spec.lookup,
            || {
                lookup_scheduler_supported(
                    model.config(),
                    model.engine_cfg(),
                    vk.capabilities().embed_gather,
                    model
                        .gguf()
                        .tensors()
                        .iter()
                        .find(|tensor| tensor.name == "token_embd.weight")
                        .map(|tensor| tensor.dtype),
                )
            },
        );
        let session_idle = Duration::from_secs(model.engine_cfg().kv.session_idle_secs);
        let mut engine = Self {
            scheduler_worker: None,
            cold_worker: None,
            model: Arc::new(model),
            vk: Arc::new(vk),
            pool: Arc::new(Mutex::new(Pool {
                slots: Vec::new(),
                tick: 0,
            })),
            freed: Arc::new(Condvar::new()),
            decode_batch: Arc::new(Mutex::new(DecodeBatchQueue::default())),
            decode_ready: Arc::new(Condvar::new()),
            batch_interrupt: Arc::new(AtomicBool::new(false)),
            // Speculative one-slot requests also hand ownership to the persistent worker.
            gate: scheduler_enabled.then(|| Arc::new(StepGate::new())),
            session_cache: None,
            ram_cache: None,
            paged_cache: None,
            session_idle,
            max_ctx,
            pins,
            mtp_heads,
        };
        engine.init_slots(n_slots)?;
        engine.init_session_cache()?;
        drop(scope);
        engine.start_cold_worker()?;
        engine.start_scheduler_worker()?;
        Ok(engine)
    }

    /// Refusal used by every persistent session-state tier: the tree, the RAM pool and the disk
    /// cache all release a conversation's rows and re-attach them elsewhere, which IS the segmented
    /// KV layout. Each used to fail with the same sentence naming `kv.paged_cache_max` even when the
    /// operator had asked for `kv.paged`, and none of them said WHICH gate of the AND the session
    /// missed — on a device with more VRAM the answer is usually "the experts never got paged, so
    /// there is no arena", which no config spelling of `kv.*` can fix.
    fn segmented_kv_refusal(&self, requested: &str) -> anyhow::Error {
        use infr_core::backend::Backend;
        let ec = self.model.engine_cfg();
        let cfg = self.model.config().clone();
        let arena = Backend::segmented_kv_available(self.vk.as_ref());
        let blockers = {
            let pool = self.pool.lock().expect("fresh pool");
            match pool.slots.first().and_then(|slot| slot.kv.as_ref()) {
                Some(kv) => {
                    let meta = kv.session_state_meta();
                    crate::seam::segmented_kv_blockers(
                        &cfg,
                        ec,
                        kv.kv_ring(),
                        meta.k_fmt,
                        meta.v_fmt,
                        arena,
                    )
                }
                None => vec!["no KV slot exists yet".to_string()],
            }
        };
        anyhow!(
            "{requested} needs dynamic segmented KV, and this session does not have it. Unmet \
             condition(s):\n  - {}",
            blockers.join("\n  - ")
        )
    }

    fn init_session_cache(&mut self) -> Result<()> {
        let ec = self.model.engine_cfg();
        // Two switches used to be one: `kv.paged_cache_max` both opened the tree and sized its
        // host-RAM frame pool. `kv.paged` splits them so the tree can run entirely inside the VRAM
        // frame corridor. A pool still implies the tree, so every pre-existing config resolves the
        // same way it did before this field existed.
        let paged_tree = ec.kv.paged || ec.kv.paged_cache_max.is_some();
        if ec.kv.disk_cache_max.is_some() && !paged_tree {
            // The disk tier only exists behind the paged tree's RAM pool; on the blob/RAM-blob
            // paths it would silently do nothing, so make the half-config loud.
            tracing::warn!(
                "kv.disk_cache_max is set without a paged KV tree: the disk frame tier only \
                 backs the paged KV tree and stays closed (enable kv.paged or kv.paged_cache_max)"
            );
        }
        if !paged_tree && ec.kv.paged_page_tokens != crate::seam::KV_GROW_ROWS {
            // Shrink-wrapped pages only pay off inside the paged tree; on the plain blob path every
            // page is another grow step with nothing shared. Usually a forgotten `paged_cache_max`
            // rather than an intentional geometry experiment, so make the half-config loud.
            tracing::warn!(
            page_tokens = ec.kv.paged_page_tokens,
            "kv.paged_page_tokens is set without kv.paged_cache_max: the blob KV grows in small segments and shares nothing (geometry-experiment mode; enable kv.paged_cache_max for real prefix reuse)"
        );
        }
        if paged_tree {
            use infr_core::backend::Backend;
            if ec.kv.ram_cache_max.is_some() {
                return Err(anyhow!(
                "kv.paged_cache_max and kv.ram_cache_max are mutually exclusive; enable one KV prefix tier"
            ));
            }
            // A zero pool disables the RAM TIER exactly like the RAM blob tier's zero (see the
            // symmetric branch below): the switch stays "set" yet nothing is claimed, and the disk
            // session cache, when configured, keeps working (PR #86 review point 3 semantics). With
            // `kv.paged` the tree itself is separately requested, so a zero pool means "corridor
            // only", not "off".
            let paged_budget = match ec.kv.paged_cache_max {
                None => 0,
                Some(spec) => match spec {
                    infr_core::SizeSpec::Bytes(bytes)
                    | infr_core::SizeSpec::FrozenAutomaticBytes(bytes) => bytes,
                    infr_core::SizeSpec::Percent(fraction) => {
                        match infr_core::hostmem::total_bytes() {
                            Some(total) => (total as f64 * fraction) as u64,
                            None => {
                                return Err(anyhow!(
                                "kv.paged_cache_max percentages need a known physical RAM total; \
                         use an absolute GiB/MiB size"
                            ))
                            }
                        }
                    }
                },
            };
            if paged_budget == 0 && !ec.kv.paged {
                tracing::info!("paged KV prefix cache disabled by a zero size limit");
                return Ok(());
            }
            let page_tokens = ec.kv.paged_page_tokens;
            if !page_tokens.is_power_of_two() {
                return Err(anyhow!("kv.paged_page_tokens must be a power of two"));
            }
            if page_tokens >= crate::seam::KV_GROW_ROWS {
                return Err(anyhow!(
                "kv.paged_page_tokens must be smaller than the legacy {}-token segment: whole-context pages share nothing",
                crate::seam::KV_GROW_ROWS
            ));
            }
            let (supports_release, meta, n_slots, cfg) = {
                let pool = self.pool.lock().expect("fresh pool");
                let kv = pool
                    .slots
                    .first()
                    .and_then(|slot| slot.kv.as_ref())
                    .ok_or_else(|| anyhow!("paged KV cache initialized before slot 0"))?;
                (
                    kv.can_release_session_state(),
                    kv.session_state_meta(),
                    pool.slots.len(),
                    self.model.config().clone(),
                )
            };
            if !supports_release {
                let requested = match (ec.kv.paged, ec.kv.paged_cache_max.is_some()) {
                    (true, true) => "kv.paged + kv.paged_cache_max",
                    (true, false) => "kv.paged (the corridor-only paged KV tree)",
                    _ => "kv.paged_cache_max",
                };
                return Err(self.segmented_kv_refusal(requested));
            }
            if ec.kv.session_cache_dir.is_some() {
                tracing::warn!(
                "paged KV prefix cache enabled; the disk session cache stays closed (no disk writes)"
            );
            }
            if self.mtp_heads.is_some() {
                // The draft head is a SECOND KV holder: it commits its own segments from the unified
                // pool outside the frame corridor (extra VRAM the corridor sizing does not count), and
                // a tree restore carries no draft state, so the head must re-warm. Never validated as
                // a combo - say so loudly instead of silently double-spending VRAM.
                tracing::warn!(
                "paged KV prefix cache is enabled together with MTP: the draft head commits KV OUTSIDE the frame corridor (extra VRAM) and loses its draft state on every tree restore; prefer spec.mtp=false"
            );
            }
            // The warmup generation on slot 0 ran BEFORE this init existed (slots are built first),
            // so its prefill committed blob segments AND froze the whole per-plane reservation grid
            // over the KV corridor the frame pool is about to claim. Hand every slot's blob state
            // back first: `release_segmented_kv` drops committed segments AND reservations, so the
            // single contiguous corridor claim can fit the exact full-KV-space footprint. In paged
            // mode no slot ever re-enters the blob path (ensure_depth frames only), so nothing needs
            // the grid again.
            {
                let cfg = self.model.config().clone();
                let mut pool = self.pool.lock().expect("fresh pool");
                for slot in &mut pool.slots {
                    if let Some(kv) = slot.kv.as_mut() {
                        kv.release_session_state(self.vk.as_ref(), &cfg)?;
                    }
                }
            }
            let layout = SegmentedKvLayout::for_qwen(
                &cfg,
                meta.max_ctx,
                meta.k_fmt,
                meta.v_fmt,
                page_tokens,
            )
            .ok_or_else(|| anyhow!("paged KV cache enabled for a non-Qwen session"))?;
            // One frame = one page x every plane, contiguous (see kv_frame_pool). The plane offsets
            // are the prefix sums of the plane segment sizes, in the SAME order the session collects
            // its plane buffers, so `plane_addr(frame, i)` and the table at index `i` agree.
            let mut plane_offsets = Vec::with_capacity(layout.planes.len());
            let mut frame_bytes = 0u64;
            for plane in &layout.planes {
                plane_offsets.push(frame_bytes);
                frame_bytes += plane.segment_bytes() as u64;
            }
            // Phase B sizes the frame space to the FULL VRAM KV space (every slot x every page),
            // per the plan: sessions and the tree share one frame pool, the tree is what's left
            // after the resident chains. The RAM tier budget (`kv.paged_cache_max`) lands with
            // Phase C's DMA pool. The arena is sharded (~2 GiB ReBAR pieces), so this request arrives
            // as one corridor at small ctx and as several when it exceeds a shard's largest free gap.
            let budget_frames = n_slots * meta.max_ctx.div_ceil(page_tokens);
            let corridors = Backend::alloc_kv_frames(self.vk.as_ref(), frame_bytes, budget_frames)?
                .ok_or_else(|| {
                    anyhow!("the unified VRAM arena cannot fit the KV frame corridor")
                })?;
            // The corridor tiling rounds every physical shard piece down to whole frames, so under
            // fragmentation the backend can place fewer frames than the budget. Trust what was
            // actually installed (FramePool::set_segments asserts capacity == corridor frames);
            // fewer frames only means cold tree entries evict earlier.
            let vram_frames: usize = corridors.iter().map(|corridor| corridor.frames).sum();
            if vram_frames < budget_frames {
                tracing::warn!(
                    budget_frames,
                    placed_frames = vram_frames,
                    corridors = corridors.len(),
                    "paged KV frame pool is smaller than the full ctx budget; cold tree entries evict earlier"
                );
            }
            // Phase C: `paged_budget` is the RAM tier budget. Frames sunk out of the VRAM corridor
            // live in a lazily grown host backing (high-water mark = ram_frames * frame_bytes), so a
            // big budget costs nothing until prefixes actually demote. Zero budget keeps the tier off
            // and the tree behaves exactly like Phase B (cold entries evict instead of sink) — which
            // is the whole point of `kv.paged`, the switch that asks for a tree without a pool.
            let ram_frames = (paged_budget / frame_bytes) as usize;
            // Phase D: `kv.disk_cache_max` sizes the sparse-file tier behind RAM. Resolution
            // mirrors the RAM budget's grammar (percent = of physical RAM, an odd denominator
            // for disk but consistent); zero/unset keeps the two-tier cascade byte-for-byte off.
            let disk_budget = match ec.kv.disk_cache_max {
                None => 0,
                Some(spec) => match spec {
                    infr_core::SizeSpec::Bytes(bytes)
                    | infr_core::SizeSpec::FrozenAutomaticBytes(bytes) => bytes,
                    infr_core::SizeSpec::Percent(fraction) => {
                        match infr_core::hostmem::total_bytes() {
                            Some(total) => (total as f64 * fraction) as u64,
                            None => {
                                return Err(anyhow!(
                                "kv.disk_cache_max percentages need a known physical RAM total; \
                                     use an absolute GiB/MiB size"
                            ))
                            }
                        }
                    }
                },
            };
            // Frames only ever reach disk through a RAM frame: demotion writes VRAM->RAM and the
            // cascade moves RAM->disk, and a hit reads back the same way. With no RAM tier there is
            // no route, so the disk budget stays closed rather than half-wired.
            let disk_frames = if disk_budget == 0 || ram_frames == 0 {
                if disk_budget > 0 {
                    tracing::warn!(
                        "kv.disk_cache_max needs the RAM frame tier that sinks through it: enable \
                         kv.paged_cache_max (the corridor-only tree has no route to the disk tier)"
                    );
                }
                0
            } else {
                (disk_budget / frame_bytes) as usize
            };
            let disk_path = (disk_frames > 0).then(|| {
                ec.kv.disk_cache_path.clone().unwrap_or_else(|| {
                    // Per-process (and per-engine within a process) default under the temp dir:
                    // the backing is process-scoped bookkeeping, never restored across restarts.
                    let seq = DISK_BACKING_SEQ.fetch_add(1, Ordering::Relaxed);
                    std::env::temp_dir().join(format!(
                        "infr-paged-kv-disk-{}-{seq}.bin",
                        std::process::id()
                    ))
                })
            });
            let mut tree = PagedKv::new(
                frame_bytes,
                page_tokens,
                plane_offsets,
                vram_frames,
                ram_frames,
                disk_frames,
                disk_path,
            );
            tree.set_vram_segments(&corridors);
            if ram_frames == 0 {
                // The `RAM tier / 8` derivation has no tier to divide. A zero aux cap is not a
                // smaller cache here, it is NO cache: `PagedKv::hit_prefix` admits a recurrent
                // resume point only at a snapshot boundary, so a hybrid model (DeltaNet/KDA/PLE)
                // would silently re-prefill every tree hit. Snapshots still cost host RAM one
                // buffer at a time — a Qwen3.8 boundary snapshot is ~113 MiB — never reserved up
                // front, which is what makes a pool-less tree cheap without making it inert.
                tree.set_aux_budget(PAGED_AUX_CORRIDOR_BUDGET);
            }
            tracing::info!(
                page_tokens,
                vram_frames,
                frame_bytes,
                corridors = corridors.len(),
                corridor_bytes = frame_bytes * vram_frames as u64,
                ram_budget = ?ec.kv.paged_cache_max,
                ram_frames,
                ram_bytes = frame_bytes * ram_frames as u64,
                pool_less = ram_frames == 0,
                disk_budget = ?ec.kv.disk_cache_max,
                disk_frames,
                disk_bytes = frame_bytes * disk_frames as u64,
                tree = %tree.stats_line(),
                "paged KV radix prefix cache ready (frames held by the tree, zero-copy prefix reuse, multimodal sessions never enter the tree; cold chains sink to the RAM tier and cascade to the disk tier before eviction when a RAM tier is configured, and evict outright when there is none)"
            );
            let tree = Arc::new(Mutex::new(tree));
            self.paged_cache = Some(Arc::clone(&tree));
            for slot in &mut self.pool.lock().expect("fresh pool").slots {
                if let Some(kv) = slot.kv.as_mut() {
                    kv.set_paged(Arc::clone(&tree));
                }
            }
            return Ok(());
        }
        if let Some(spec) = ec.kv.ram_cache_max {
            let max_bytes = match spec {
                infr_core::SizeSpec::Bytes(bytes)
                | infr_core::SizeSpec::FrozenAutomaticBytes(bytes) => bytes,
                infr_core::SizeSpec::Percent(fraction) => infr_core::hostmem::total_bytes()
                    .map(|total| (total as f64 * fraction) as u64)
                    .ok_or_else(|| {
                        anyhow!(
                            "kv.ram_cache_max percentages need a known physical RAM total; \
                         use an absolute GiB/MiB size"
                        )
                    })?,
            };
            if max_bytes == 0 {
                tracing::info!("RAM KV prefix cache disabled by a zero size limit");
                return Ok(());
            }
            let (supports_release, meta) = {
                let pool = self.pool.lock().expect("fresh pool");
                let kv = pool
                    .slots
                    .first()
                    .and_then(|slot| slot.kv.as_ref())
                    .ok_or_else(|| anyhow!("RAM KV cache initialized before slot 0"))?;
                (kv.can_release_session_state(), kv.session_state_meta())
            };
            if !supports_release {
                return Err(self.segmented_kv_refusal("kv.ram_cache_max"));
            }
            if ec.kv.session_cache_dir.is_some() {
                tracing::warn!(
                "RAM KV prefix cache enabled; the disk session cache stays closed (no disk writes)"
            );
            }
            self.ram_cache = Some(Arc::new(Mutex::new(RadixCache::new(&meta, max_bytes))));
            tracing::info!(
            max_bytes,
            gib = max_bytes as f64 / (1u64 << 30) as f64,
            "RAM KV radix prefix cache ready (checkout demotion + prefix restore, never on disk)"
        );
            return Ok(());
        }
        if self.model.engine_cfg().kv.session_cache_dir.is_none() {
            return Ok(());
        }
        let (supports_release, meta) = {
            let pool = self.pool.lock().expect("fresh pool");
            let kv = pool
                .slots
                .first()
                .and_then(|slot| slot.kv.as_ref())
                .ok_or_else(|| anyhow!("cold KV cache initialized before slot 0"))?;
            (kv.can_release_session_state(), kv.session_state_meta())
        };
        if !supports_release {
            return Err(self.segmented_kv_refusal("kv.session_cache_dir"));
        }
        self.session_cache = SessionCache::open(self.model.engine_cfg(), self.model.gguf(), &meta)?
            .map(|cache| Arc::new(Mutex::new(cache)));
        if self.session_cache.is_some() {
            let now = Instant::now();
            for slot in &mut self.pool.lock().expect("fresh pool").slots {
                slot.idle_since = Some(now);
            }
        }
        Ok(())
    }

    fn start_cold_worker(&mut self) -> Result<()> {
        // The RAM tier deliberately has NO idle worker. Demotion happens synchronously in
        // `checkout_with_ram_cache` when a different-prefix request needs a slot: capture ->
        // publish -> release under the GPU baton, so the session is never in neither place.
        // A time-driven demoter only opened a black-hole window (the slot's KV was taken before
        // the slow capture/published, so a replay of that prefix re-prefilled from scratch) and
        // a resident continuation always beats a restore anyway — 2026-10-15 A/B regression.
        let Some(cache) = self.session_cache.as_ref().map(Arc::clone) else {
            return Ok(());
        };
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let pool = Arc::clone(&self.pool);
        let wake = Arc::clone(&self.freed);
        let backend = Arc::clone(&self.vk);
        let gate = self.gate.as_ref().map(Arc::clone);
        let model_cfg = self.model.config().clone();
        let mtp_heads = self.mtp_heads.as_ref().map(Arc::clone);
        let idle = self.session_idle;
        let thread = std::thread::Builder::new()
            .name("infr-kv-cold".into())
            .spawn(move || {
                cold_session_worker(
                    pool,
                    wake,
                    cache,
                    backend,
                    gate,
                    mtp_heads,
                    model_cfg,
                    idle,
                    worker_stop,
                )
            })
            .context("start cold KV maintenance thread")?;
        self.cold_worker = Some(ColdWorker {
            stop,
            wake: Arc::clone(&self.freed),
            thread: Some(thread),
        });
        Ok(())
    }

    /// Build the shallow engine handle owned by the scheduler thread. It shares all model,
    /// allocator and queue state but owns neither background worker, avoiding a self-reference.
    fn scheduler_handle(&self) -> Self {
        Self {
            scheduler_worker: None,
            cold_worker: None,
            model: Arc::clone(&self.model),
            vk: Arc::clone(&self.vk),
            pool: Arc::clone(&self.pool),
            freed: Arc::clone(&self.freed),
            decode_batch: Arc::clone(&self.decode_batch),
            decode_ready: Arc::clone(&self.decode_ready),
            batch_interrupt: Arc::clone(&self.batch_interrupt),
            gate: self.gate.as_ref().map(Arc::clone),
            session_cache: self.session_cache.as_ref().map(Arc::clone),
            ram_cache: self.ram_cache.as_ref().map(Arc::clone),
            paged_cache: self.paged_cache.as_ref().map(Arc::clone),
            session_idle: self.session_idle,
            max_ctx: self.max_ctx,
            pins: Arc::clone(&self.pins),
            mtp_heads: self.mtp_heads.as_ref().map(Arc::clone),
        }
    }

    fn start_scheduler_worker(&mut self) -> Result<()> {
        if self.gate.is_none() || !self.model.config().supports_parallel_token_rows() {
            return Ok(());
        }
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let wake = Arc::clone(&self.decode_ready);
        let scheduler = self.scheduler_handle();
        let thread = std::thread::Builder::new()
            .name("infr-parallel".into())
            .spawn(move || scheduler.scheduler_loop(worker_stop))
            .context("start parallel scheduler thread")?;
        self.scheduler_worker = Some(SchedulerWorker {
            stop,
            wake,
            thread: Some(thread),
        });
        Ok(())
    }

    fn scheduler_loop(self, stop: Arc<AtomicBool>) {
        let gate = self
            .gate
            .as_ref()
            .expect("parallel scheduler requires a shared GPU gate")
            .clone();
        let req = RequestCtx::with_gate(crate::sampling::RequestSampling::default(), gate);
        let _scope = crate::seam::PlacementScope::enter(Arc::clone(&self.pins));
        while let Some(work) = self.wait_for_scheduler_work(stop.as_ref()) {
            if self.mtp_heads.is_some() {
                self.run_mtp_batch(work, &req);
            } else {
                self.run_unified_batch(work, &req);
            }
        }
        self.close_decode_batch(Some("parallel scheduler is shutting down"));
    }

    pub fn fork_embedding_backend(&self) -> Result<infr_vulkan::VulkanBackend> {
        self.vk
            .fork_embedding_client()
            .map_err(|error| anyhow!("derive unified Embedding backend: {error}"))
    }

    pub fn fork_vision_backend(&self) -> Result<infr_vulkan::VulkanBackend> {
        self.vk
            .fork_vision_client()
            .map_err(|error| anyhow!("derive unified Vision backend: {error}"))
    }

    pub fn unified_vram_stats(&self) -> Option<infr_vulkan::unified::UnifiedVramStats> {
        self.vk.unified_vram().map(|pool| pool.stats())
    }

    /// Bytes still allocatable under the backend's live unified VRAM guard.
    pub fn unified_alloc_room(&self) -> u64 {
        self.vk.alloc_room()
    }

    /// Fork Vision with a private unified arena when the primary session kept every MoE expert
    /// resident (the pager only builds its arena when experts are actually paged). The primary
    /// LLM keeps its normal allocation path; only the vision client allocates from this arena.
    pub fn fork_vision_backend_with_arena(
        &self,
        bytes: usize,
    ) -> Result<infr_vulkan::VulkanBackend> {
        self.vk
            .fork_vision_client_with_arena(bytes)
            .map_err(|error| anyhow!("derive unified Vision backend: {error}"))
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
                Some(crate::seam::TurnCheckpoint::ENABLED),
                None, // constraint
                None, // req: startup, not a request — env sampling, no gate
                None, // multimodal plan
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

        let preallocated = slot0.take_preallocated_siblings();
        if !preallocated.is_empty() && preallocated.len() != n_slots.saturating_sub(1) {
            return Err(anyhow!(
                "startup materialized {} sibling slots, expected {}",
                preallocated.len(),
                n_slots.saturating_sub(1),
            ));
        }
        let mut preallocated = preallocated.into_iter();
        let mut slots = Vec::with_capacity(n_slots);
        for i in 1..n_slots {
            // A fork shares the `Arc<SeamWeights>` — it costs only its own KV + IO buffers. If VRAM
            // refuses, say so HERE, at boot, with the two knobs that fix it. Never mid-request.
            let kv = match preallocated.next() {
                Some(kv) => kv,
                None => slot0
                    .fork(
                        self.vk.as_ref(),
                        self.model.config(),
                        self.model.engine_cfg(),
                    )
                    .map_err(|e| {
                        anyhow!(
                            "could not allocate KV slot {}/{n_slots} at ctx {}: {e}\n\
                         lower --parallel, or lower --ctx (each slot owns a full context of KV cache)",
                            i + 1,
                            self.max_ctx,
                        )
                    })?,
            };
            slots.push(Slot {
                kv: Some(kv),
                busy: false,
                tick: 0,
                idle_since: None,
                multimodal_key: None,
            });
        }
        slots.insert(
            0,
            Slot {
                kv: Some(slot0),
                busy: false,
                tick: 0,
                idle_since: None,
                multimodal_key: None,
            },
        );
        self.pool.lock().expect("fresh pool").slots = slots;
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

    /// Live device-memory snapshot for diagnostics and benchmark result attribution.
    pub fn vram_info(&self) -> infr_vulkan::VramInfo {
        self.vk.vram()
    }

    /// Selected backend's actual cold upload payload, including rewrites and generated constants.
    /// This is the model footprint, not the paged arena's current resident expert contents.
    pub fn weight_footprint(&self) -> Result<crate::weights::WeightFootprint> {
        crate::seam::planned_weight_footprint(
            &self.vk,
            self.model.gguf(),
            self.model.config(),
            self.vk.cfg(),
        )
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

    /// Fill one real prompt into each slot without counting setup prompts as concurrent Prefill.
    ///
    /// The automatic benchmark measures slot 0 as its single Prefill result, then needs equivalent
    /// real contexts in the remaining slots before it can measure x2..xN Decode. Driving those
    /// prompts through ordinary checkout would repeatedly select the best-prefix slot instead of
    /// deterministically filling every slot. This setup-only entry point pins prompt `i` to slot
    /// `i`, while using the same runner, KV layout and turn checkpoint machinery as serving.
    pub fn prepare_benchmark_slots(
        &self,
        prompts: &[String],
        progress: Option<&[infr_core::GenerationProgressCallback]>,
    ) -> Result<Vec<GenStats>> {
        anyhow::ensure!(
            !prompts.is_empty() && prompts.len() <= self.n_slots(),
            "benchmark needs 1..={} prompts, got {}",
            self.n_slots(),
            prompts.len(),
        );
        anyhow::ensure!(
            self.mtp_heads.is_none(),
            "automatic benchmark slot preparation does not support MTP yet"
        );
        anyhow::ensure!(
            progress.is_none_or(|callbacks| callbacks.len() == prompts.len()),
            "benchmark Prefill progress callback count must match the prompt count"
        );

        let mut results = Vec::with_capacity(prompts.len());
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        for (index, prompt) in prompts.iter().enumerate() {
            let prompt_tokens = self.model.encode(prompt)?;
            let checkpoint = self.model.turn_checkpoints(
                &prompt_tokens,
                crate::seam::TurnCheckpointPrefixes::benchmark_frontier(),
            )?;
            let req = self
                .request_ctx(crate::sampling::RequestSampling::default())
                .with_progress(progress.and_then(|callbacks| callbacks.get(index).cloned()));
            let mut kv = {
                let mut pool = self.pool.lock().expect("pool poisoned");
                let slot = &mut pool.slots[index];
                anyhow::ensure!(!slot.busy, "benchmark slot {index} is busy");
                slot.busy = true;
                slot.multimodal_key = None;
                slot.kv.take()
            };
            let run = crate::seam::generate_dense_vulkan_session(
                self.vk.as_ref(),
                self.model.gguf(),
                self.model.config(),
                self.model.engine_cfg(),
                self.model.embd(),
                self.model.per_layer_embd(),
                &prompt_tokens,
                0,
                |_| {},
                &mut kv,
                self.max_ctx,
                checkpoint,
                None,
                Some(&req),
                None,
            );
            {
                let mut pool = self.pool.lock().expect("pool poisoned");
                pool.tick += 1;
                let tick = pool.tick;
                let slot = &mut pool.slots[index];
                slot.kv = kv;
                slot.busy = false;
                slot.tick = tick;
                slot.idle_since = self.session_cache.is_some().then(Instant::now);
            }
            self.freed.notify_all();
            results.push((run?).1);
        }
        Ok(results)
    }

    /// Prefill the common background once, copy its live state, then append each lane's question.
    /// This is setup-only: ordinary request checkout and serving cache policy are unchanged.
    pub fn prepare_benchmark_shared_slots(
        &self,
        background: &str,
        prompts: &[String],
        progress: Option<&[infr_core::GenerationProgressCallback]>,
    ) -> Result<Vec<GenStats>> {
        anyhow::ensure!(
            !prompts.is_empty() && prompts.len() <= self.n_slots(),
            "invalid shared benchmark slot count"
        );
        anyhow::ensure!(
            progress.is_none_or(|callbacks| callbacks.len() == prompts.len()),
            "benchmark progress callback count must match the prompt count"
        );
        let background_tokens = self.model.encode(background)?;
        anyhow::ensure!(background_tokens.len() > 1, "benchmark background is empty");
        for prompt in prompts {
            let tokens = self.model.encode(prompt)?;
            anyhow::ensure!(
                tokens.starts_with(&background_tokens[..background_tokens.len() - 1]),
                "benchmark question changed the shared token prefix"
            );
        }
        let mut scored = self.prepare_benchmark_slots(
            &[background.to_owned()],
            progress.map(|callbacks| &callbacks[..1]),
        )?;
        if prompts.len() > 1 {
            let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
            let mut states = {
                let mut pool = self.pool.lock().expect("pool poisoned");
                anyhow::ensure!(
                    pool.slots.iter().all(|slot| !slot.busy),
                    "cannot copy benchmark state while a slot is busy"
                );
                anyhow::ensure!(
                    pool.slots[..prompts.len()]
                        .iter()
                        .all(|slot| slot.kv.is_some()),
                    "benchmark slot has no KV"
                );
                pool.slots[..prompts.len()]
                    .iter_mut()
                    .map(|slot| {
                        slot.busy = true;
                        slot.kv.take().expect("validated benchmark KV")
                    })
                    .collect::<Vec<_>>()
            };
            let copied = (|| -> Result<()> {
                let (source, targets) = states.split_at_mut(1);
                for target in targets {
                    let copied = target.seed_benchmark_from(
                        self.vk.as_ref(),
                        self.model.config(),
                        self.model.engine_cfg(),
                        &source[0],
                    )?;
                    anyhow::ensure!(
                        copied == background_tokens.len() - 1,
                        "benchmark could not copy the complete shared frontier"
                    );
                }
                Ok(())
            })();
            // Always return all reservations, including when a device copy fails.
            let mut pool = self.pool.lock().expect("pool poisoned");
            for (slot, state) in pool.slots.iter_mut().zip(states) {
                slot.kv = Some(state);
                slot.busy = false;
            }
            drop(pool);
            self.freed.notify_all();
            copied?;
        }
        // Materialize the short questions before any measured Decode lane starts.
        let questions = self.prepare_benchmark_slots(prompts, None)?;
        for (index, stats) in questions.into_iter().enumerate() {
            if index == 0 {
                continue;
            }
            scored.push(stats);
        }
        Ok(scored)
    }

    /// Reset benchmark contents and return unused dynamic segments before the next context case.
    /// Keep logical buffers/address tables and fixed recurrent/IO state alive: frozen flat
    /// allocations cannot be recreated, while segmented physical ranges support recommit.
    pub fn clear_benchmark_slots(&self) -> Result<()> {
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        let mut pool = self.pool.lock().expect("pool poisoned");
        anyhow::ensure!(
            pool.slots.iter().all(|slot| !slot.busy),
            "cannot clear benchmark slots while a request is active"
        );
        for slot in &mut pool.slots {
            if let Some(kv) = slot.kv.as_mut() {
                kv.reset_benchmark_session_state(self.vk.as_ref(), self.model.config())?;
            }
            slot.multimodal_key = None;
            slot.idle_since = None;
        }
        if let Some(heads) = &self.mtp_heads {
            for head in heads.lock().expect("MTP heads poisoned").iter_mut() {
                head.invalidate_all();
            }
        }
        Ok(())
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
        if self.paged_cache.is_some() {
            self.checkout_with_paged_cache(prompt, req)
        } else if self.ram_cache.is_some() {
            self.checkout_with_ram_cache(prompt, req)
        } else if self.session_cache.is_some() {
            self.checkout_with_cold_cache(prompt, req)
        } else {
            self.checkout_resident(prompt, req)
        }
    }

    /// Original resident-only checkout path. Keeping it separate means an unset cold-cache path
    /// does not add a file-cache mutex or wall-clock query to ordinary serving.
    fn checkout_resident(&self, prompt: &[u32], req: &RequestCtx) -> Result<SlotGuard<'_>> {
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
            let score = |s: &Slot| {
                if s.multimodal_key.is_some() {
                    0
                } else {
                    s.kv.as_ref()
                        .map_or(0, |k| k.seedable_prefix_len(cfg, prompt))
                }
            };
            // 1. This conversation continuing: the free slot with the LONGEST reusable prefix among
            //    those the prompt extends (or equals) — not merely the first such slot (which would
            //    re-prefill more suffix). Pure decision, unit-tested via `pick_continuation`.
            let cont = pick_continuation(
                free.iter().filter_map(|&i| {
                    (p.slots[i].multimodal_key.is_none())
                        .then_some(p.slots[i].kv.as_ref())
                        .flatten()
                        .and_then(|k| {
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
            if cont.is_none() && p.slots[target].multimodal_key.is_none() {
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
                        let r: Result<usize> = {
                            let _gp = req.gate_pass();
                            let seeded = match dst.as_mut() {
                                Some(dst) => dst.seed_from(self.vk.as_ref(), cfg, ec, &src, best_s),
                                None => Ok(0),
                            };
                            seeded.inspect(|&seeded| {
                                if seeded != 0 {
                                    if let Some(heads) = &self.mtp_heads {
                                        let mut heads = heads.lock().expect("MTP heads poisoned");
                                        let tokens = &src.cached_tokens()[..seeded];
                                        let (target_lane, source_lane) =
                                            mtp_slot_pair_mut(&mut heads, target, best);
                                        if let Err(error) = target_lane.seed_from(
                                            self.vk.as_ref(),
                                            source_lane,
                                            tokens,
                                        ) {
                                            target_lane.invalidate_all();
                                            tracing::warn!(
                                                slot = target,
                                                source_slot = best,
                                                "MTP prefix seed failed; preserving the seeded main KV and using ordinary decode: {error}"
                                            );
                                        }
                                    }
                                }
                            })
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
                            if let Some(kv) = p.slots[target].kv.as_mut() {
                                kv.reset();
                            }
                            if let Some(heads) = &self.mtp_heads {
                                heads.lock().expect("MTP heads poisoned")[target].invalidate_all();
                            }
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
            let incompatible = p.slots[target].multimodal_key.take().is_some();
            let mut kv = p.slots[target].kv.take();
            if incompatible {
                if let Some(kv) = kv.as_mut() {
                    kv.reset();
                }
                if let Some(heads) = &self.mtp_heads {
                    heads.lock().expect("MTP heads poisoned")[target].invalidate_all();
                }
            }
            return Ok(SlotGuard {
                engine: self,
                idx: target,
                kv,
                detached: false,
                multimodal_key: None,
            });
        }
    }

    /// Capture one resident slot into the radix tree and release its VRAM segments. `Ok(true)` means
    /// the conversation now lives in RAM and the slot is logically empty; `Ok(false)` means the
    /// slot held nothing to keep. On `Err` the snapshot was refunded (ledger-drop) and the slot
    /// keeps its resident state — the caller decides between recycling and reset. The caller must
    /// already hold the GPU baton, matching the disk path's in-checkout spill (no gate handed to
    /// `capture`).
    fn demote_to_ram(
        &self,
        slot: usize,
        tree: &mut RadixCache,
        kv: &mut SeamKv,
        cfg: &Config,
    ) -> Result<bool> {
        let Some(inner) = ram_cache::capture(kv, self.vk.as_ref(), tree.geometry(), tree.ledger())?
        else {
            return Ok(false);
        };
        let tokens = inner.meta.cached.clone();
        tree.publish(inner.clone())?;
        match kv.release_session_state(self.vk.as_ref(), cfg) {
            Ok(()) => {
                // The only demotion path left (no idle worker): make it observable so a
                // checkout swap's cost is attributable in the engine log.
                tracing::info!(
                    slot,
                    tokens = tokens.len(),
                    tree = %tree.stats_line(),
                    "demoted replaced conversation KV to the RAM radix cache"
                );
                Ok(true)
            }
            Err(error) => {
                tree.discard(&tokens, &inner);
                Err(error)
            }
        }
    }

    /// Radix-tree-extended slot checkout (`kv.ram_cache_max`). Same reservation discipline as the
    /// disk path: targets reserved under `pool`, all GPU/tree work after that lock is dropped. A
    /// resident conversation the incoming prompt cannot continue is demoted to a RAM snapshot in
    /// place — the automatic different-prefix swap — never written to disk — and the deepest tree
    /// match is then restored into the target slot.
    fn checkout_with_ram_cache(&self, prompt: &[u32], req: &RequestCtx) -> Result<SlotGuard<'_>> {
        let tree_mutex = self
            .ram_cache
            .as_ref()
            .expect("RAM checkout requires a radix cache");
        let cfg: &Config = self.model.config();
        let mut pool = self.pool.lock().expect("pool poisoned");
        loop {
            let free = (0..pool.slots.len())
                .filter(|&index| !pool.slots[index].busy)
                .collect::<Vec<_>>();
            if free.is_empty() {
                pool = self.freed.wait(pool).expect("pool poisoned");
                continue;
            }
            let continuation = pick_continuation(
                free.iter().filter_map(|&index| {
                    (pool.slots[index].multimodal_key.is_none())
                        .then_some(pool.slots[index].kv.as_ref())
                        .flatten()
                        .and_then(|kv| {
                            kv.continuation_prefix_len(prompt)
                                .map(|prefix| (index, prefix, prefix))
                        })
                }),
                prompt.len(),
            );
            let continuation_prefix = continuation
                .and_then(|index| {
                    pool.slots[index]
                        .kv
                        .as_ref()
                        .and_then(|kv| kv.continuation_prefix_len(prompt))
                })
                .unwrap_or(0);
            let target = continuation.unwrap_or_else(|| {
                *free
                    .iter()
                    .min_by_key(|&&index| {
                        let slot = &pool.slots[index];
                        let empty = slot.kv.as_ref().is_none_or(|kv| kv.cached_len() == 0);
                        (!empty, slot.tick)
                    })
                    .expect("free is non-empty")
            });
            let seed_source = continuation
                .is_none()
                .then(|| {
                    free.iter()
                        .copied()
                        .filter(|&index| {
                            index != target && pool.slots[index].multimodal_key.is_none()
                        })
                        .filter_map(|index| {
                            let prefix = pool.slots[index]
                                .kv
                                .as_ref()
                                .map_or(0, |kv| kv.seedable_prefix_len(cfg, prompt));
                            (prefix >= 16).then_some((index, prefix))
                        })
                        .max_by_key(|&(_, prefix)| prefix)
                })
                .flatten();
            let resident_prefix =
                continuation_prefix.max(seed_source.map_or(0, |(_, prefix)| prefix));

            pool.tick += 1;
            let tick = pool.tick;
            pool.slots[target].busy = true;
            pool.slots[target].tick = tick;
            pool.slots[target].idle_since = None;
            let incompatible = pool.slots[target].multimodal_key.take().is_some();
            let mut target_kv = pool.slots[target].kv.take();
            let mut seed_kv = seed_source.map(|(source, prefix)| {
                pool.slots[source].busy = true;
                pool.slots[source].idle_since = None;
                let kv = pool.slots[source]
                    .kv
                    .take()
                    .expect("scored RAM seed slot has KV");
                (source, prefix, kv)
            });
            drop(pool);

            {
                // One pass owns both the inference baton and the tree. No other request can
                // submit against a state buffer while it is captured or restored.
                let _gate = req.gate_pass();
                let mut heads = self
                    .mtp_heads
                    .as_ref()
                    .map(|heads| heads.lock().expect("MTP heads poisoned"));
                let mut tree = tree_mutex
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if incompatible {
                    if let Some(kv) = target_kv.as_mut() {
                        kv.reset();
                    }
                    if let Some(heads) = heads.as_mut() {
                        heads[target].invalidate_all();
                    }
                }

                let ram_prefix = tree.best_continuation_len(prompt);
                let taken = (ram_prefix > resident_prefix)
                    .then(|| tree.take_best(prompt))
                    .flatten();
                if let (Some(taken), Some(kv)) = (taken, target_kv.as_mut()) {
                    tracing::info!(
                        slot = target,
                        cached_prefix_tokens = ram_prefix,
                        "RAM KV radix prefix hit"
                    );
                    let mut released = false;
                    if kv.cached_len() != 0 {
                        match self.demote_to_ram(target, &mut tree, kv, cfg) {
                            Ok(true) => released = true,
                            Ok(false) => {}
                            Err(error) => {
                                tracing::warn!(
                                    slot = target,
                                    "RAM KV replacement demotion failed; recycling the slot: {error}"
                                );
                                kv.reset();
                                if let Some(heads) = heads.as_mut() {
                                    heads[target].invalidate_all();
                                }
                            }
                        }
                    }
                    if !released {
                        released = kv.release_session_state(self.vk.as_ref(), cfg).is_ok();
                    }
                    if released {
                        // The RAM tier carries no MTP draft state; free the head's stale KV and
                        // let it re-warm on ordinary decode (the disk store's sidecar fallback).
                        if let Some(heads) = heads.as_mut() {
                            if let Err(error) = heads[target].head.release_kv(self.vk.as_ref()) {
                                tracing::warn!(
                                    slot = target,
                                    "RAM KV replacement could not release stale MTP head KV: {error}"
                                );
                            }
                            heads[target].invalidate_all();
                        }
                        if let Err(error) =
                            ram_cache::restore(taken.inner(), kv, self.vk.as_ref(), cfg)
                        {
                            if let Some(heads) = heads.as_mut() {
                                heads[target].invalidate_all();
                            }
                            tracing::warn!(
                                slot = target,
                                "RAM KV restore failed; re-prefilling the request: {error}"
                            );
                        }
                    } else {
                        tracing::warn!(
                            slot = target,
                            "could not release the target KV slot for RAM restore; re-prefilling"
                        );
                        kv.reset();
                        if let Some(heads) = heads.as_mut() {
                            heads[target].invalidate_all();
                        }
                    }
                    tree.release(taken);
                } else if continuation.is_none() {
                    // Different prefix, no deeper tree match: swap the target's out-of-the-way
                    // conversation into RAM (best effort) so a later prefix hit can bring it back,
                    // then seed the new prompt from the best resident prefix as the disk path does.
                    if let Some(kv) = target_kv.as_mut().filter(|kv| kv.cached_len() != 0) {
                        if let Err(error) = self.demote_to_ram(target, &mut tree, kv, cfg) {
                            tracing::warn!(
                                slot = target,
                                "RAM KV replacement demotion failed; forgetting the old conversation: {error}"
                            );
                            kv.reset();
                            if let Some(heads) = heads.as_mut() {
                                heads[target].invalidate_all();
                            }
                        } else if let Some(heads) = heads.as_mut() {
                            if let Err(error) = heads[target].head.release_kv(self.vk.as_ref()) {
                                tracing::warn!(
                                    slot = target,
                                    "RAM KV seed target could not release stale MTP head KV: {error}"
                                );
                            }
                            heads[target].invalidate_all();
                        }
                    }
                    if let (Some(kv), Some((source, prefix, source_kv))) =
                        (target_kv.as_mut(), seed_kv.as_ref())
                    {
                        match kv.seed_from(
                            self.vk.as_ref(),
                            cfg,
                            self.model.engine_cfg(),
                            source_kv,
                            *prefix,
                        ) {
                            Ok(seeded) if seeded != 0 => {
                                if let Some(heads) = heads.as_mut() {
                                    let tokens = &source_kv.cached_tokens()[..seeded];
                                    let (target_lane, source_lane) =
                                        mtp_slot_pair_mut(heads, target, *source);
                                    if let Err(error) =
                                        target_lane.seed_from(self.vk.as_ref(), source_lane, tokens)
                                    {
                                        target_lane.invalidate_all();
                                        tracing::warn!(
                                            slot = target,
                                            source_slot = *source,
                                            "RAM-cache MTP prefix seed failed; preserving the seeded main KV and using ordinary decode: {error}"
                                        );
                                    }
                                }
                            }
                            Ok(_) => {}
                            Err(error) => {
                                kv.reset();
                                if let Some(heads) = heads.as_mut() {
                                    heads[target].invalidate_all();
                                }
                                tracing::warn!(
                                    slot = target,
                                    source_slot = *source,
                                    "RAM-cache resident prefix seed failed; re-prefilling instead: {error}"
                                );
                            }
                        }
                    }
                }
            }

            if let Some((source, _, source_kv)) = seed_kv.take() {
                let mut pool = self.pool.lock().expect("pool poisoned");
                pool.slots[source].kv = Some(source_kv);
                pool.slots[source].busy = false;
                pool.slots[source].idle_since = Some(Instant::now());
                drop(pool);
                self.freed.notify_all();
            }

            return Ok(SlotGuard {
                engine: self,
                idx: target,
                kv: target_kv,
                detached: false,
                multimodal_key: None,
            });
        }
    }

    /// Paged-frame-extended slot checkout (`kv.paged_cache_max`). Same reservation discipline as
    /// the RAM path: the target is reserved under `pool`, then all GPU/tree work runs after that
    /// lock is dropped, under the inference baton AND the tree lock. Unlike the RAM tier there is
    /// nothing to copy in either direction: a replaced conversation is published to the tree IN
    /// PLACE (its frames keep living where they are — demotion is a chain move plus table zeroes)
    /// and a tree hit is attached with table writes only. Publish strictly precedes attach so the
    /// swap never opens the PR #86 black-hole window: until `publish` returns the frames carry the
    /// session's owners; after it the tree carries its own, so no intermediate state can strand
    /// or evict a conversation mid-swap.
    fn checkout_with_paged_cache(&self, prompt: &[u32], req: &RequestCtx) -> Result<SlotGuard<'_>> {
        let tree_mutex = self
            .paged_cache
            .as_ref()
            .expect("paged checkout requires a paged cache");
        let cfg: &Config = self.model.config();
        // Hybrid recurrent models pin reuse to snapshot boundaries; a dense model resumes anywhere.
        let recurrent = cfg.qwen35 || cfg.qwen4exp || cfg.bailingmoe3;
        let mut pool = self.pool.lock().expect("pool poisoned");
        loop {
            let free = (0..pool.slots.len())
                .filter(|&index| !pool.slots[index].busy)
                .collect::<Vec<_>>();
            if free.is_empty() {
                pool = self.freed.wait(pool).expect("pool poisoned");
                continue;
            }
            let continuation = pick_continuation(
                free.iter().filter_map(|&index| {
                    (pool.slots[index].multimodal_key.is_none())
                        .then_some(pool.slots[index].kv.as_ref())
                        .flatten()
                        .and_then(|kv| {
                            kv.continuation_prefix_len(prompt)
                                .map(|prefix| (index, prefix, prefix))
                        })
                }),
                prompt.len(),
            );
            let continuation_prefix = continuation
                .and_then(|index| {
                    pool.slots[index]
                        .kv
                        .as_ref()
                        .and_then(|kv| kv.continuation_prefix_len(prompt))
                })
                .unwrap_or(0);
            let target = continuation.unwrap_or_else(|| {
                *free
                    .iter()
                    .min_by_key(|&&index| {
                        let slot = &pool.slots[index];
                        let empty = slot.kv.as_ref().is_none_or(|kv| kv.cached_len() == 0);
                        (!empty, slot.tick)
                    })
                    .expect("free is non-empty")
            });
            pool.tick += 1;
            let tick = pool.tick;
            pool.slots[target].busy = true;
            pool.slots[target].tick = tick;
            pool.slots[target].idle_since = None;
            let incompatible = pool.slots[target].multimodal_key.take().is_some();
            let mut target_kv = pool.slots[target].kv.take();
            drop(pool);

            // A hybrid recurrent slot resumes only INSIDE its own conversation: the runner needs a
            // strict extension of `cached`, since the DeltaNet/PLE summary cannot rewind to a
            // shorter history. Replaying the same prompt therefore keeps nothing resident — and
            // publishing it is what turns the replay into an attach: the entry's frontier snapshot
            // resumes the request at the last page boundary, so a 30K repeat costs the sub-page
            // tail instead of a second 30K prefill.
            let continuation_prefix = if recurrent && continuation_prefix >= prompt.len() {
                0
            } else {
                continuation_prefix
            };

            let mut tree_attached = 0usize;
            {
                // One pass owns both the inference baton and the tree: no dispatch can observe a
                // half-written page table, and no other checkout can match a chain mid-swap.
                let _gate = req.gate_pass();
                let mut heads = self
                    .mtp_heads
                    .as_ref()
                    .map(|heads| heads.lock().expect("MTP heads poisoned"));
                let mut tree = tree_mutex
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(kv) = target_kv.as_mut() {
                    // The tree is engine state, not conversation state: a forked/rebuilt slot
                    // arrives without the handle, so every checkout re-injects it.
                    kv.set_paged(Arc::clone(tree_mutex));
                }
                if incompatible {
                    // Multimodal KV is keyed by opaque content, not tokens — never publish it to
                    // the token tree. `reset` keeps the frames as this slot's warm reuse capacity;
                    // a later tree attach frees them through the empty-publish detach.
                    if let Some(kv) = target_kv.as_mut() {
                        kv.reset();
                    }
                    if let Some(heads) = heads.as_mut() {
                        heads[target].invalidate_all();
                    }
                }

                // 1. The old conversation leaves FIRST (different-prefix swap), so the tree sees
                //    it before the new prompt matches, and the frames are never in limbo. A
                //    continuation that survived the replay rule above keeps its slot warm instead.
                if continuation_prefix == 0 {
                    if let Some(kv) = target_kv.as_mut() {
                        if kv.cached_len() != 0 || kv.paged_holds_frames() {
                            match kv.paged_publish_detach(self.vk.as_ref(), cfg, &mut tree) {
                                Ok(published) => {
                                    tracing::info!(
                                        slot = target,
                                        published,
                                        tree = %tree.stats_line(),
                                        "published replaced conversation KV to the paged radix cache (frames stay in place)"
                                    );
                                    if let Some(heads) = heads.as_mut() {
                                        if let Err(error) =
                                            heads[target].head.release_kv(self.vk.as_ref())
                                        {
                                            tracing::warn!(
                                                slot = target,
                                                "paged KV replacement could not release stale MTP head KV: {error}"
                                            );
                                        }
                                        heads[target].invalidate_all();
                                    }
                                }
                                Err(error) => {
                                    tracing::warn!(
                                        slot = target,
                                        "paged KV replacement publish failed; forgetting the old conversation: {error}"
                                    );
                                    kv.reset();
                                    if let Some(heads) = heads.as_mut() {
                                        heads[target].invalidate_all();
                                    }
                                }
                            }
                        }
                    }
                }

                // 2. Deepest page-aligned chain the tree can serve (counts a hit/miss itself).
                //    A resident continuation only loses to a STRICTLY deeper tree chain.
                if let Some((node, matched)) = tree.best_match(prompt) {
                    // A dense model resumes wherever it matched; a hybrid recurrent model only at a
                    // boundary its entry carries a snapshot for (see `PagedKv::hit_prefix`).
                    let pages = tree.page_tokens();
                    let (mut reusable, mut ckpt) = tree.hit_prefix(node, recurrent, matched);
                    if reusable > continuation_prefix {
                        // A continuation slot that skipped step 1 may still hold its conversation
                        // (and the failed-publish path leaves the chain warm). Attach needs an
                        // empty slot: detach through the same publish path (full pages = 0 after
                        // a reset, so this arm can only free frames, never publish stale state).
                        let occupied = target_kv
                            .as_ref()
                            .is_some_and(|kv| kv.cached_len() != 0 || kv.paged_holds_frames());
                        let mut ready = !occupied;
                        if occupied {
                            if let Some(kv) = target_kv.as_mut() {
                                match kv.paged_publish_detach(self.vk.as_ref(), cfg, &mut tree) {
                                    Ok(_) => ready = true,
                                    Err(error) => {
                                        tracing::warn!(
                                            slot = target,
                                            "paged KV could not empty the slot for a tree attach; re-prefilling: {error}"
                                        );
                                        kv.reset();
                                        if let Some(heads) = heads.as_mut() {
                                            heads[target].invalidate_all();
                                        }
                                    }
                                }
                            }
                        }
                        if ready {
                            // Phase C: RAM frames in the matched chain DMA back into VRAM before
                            // the attach (the page table never sees a RAM address). VRAM that
                            // cannot host the whole chain attaches at its deepest promotable
                            // prefix; the un-promoted tail stays sunk and the rest re-prefills.
                            let mut chain =
                                tree.entry_chain_promoted(self.vk.as_ref(), node, reusable);
                            // Promotion may fall short of the request. Re-pick against what
                            // actually arrived, then cut the chain back to the reuse point: a
                            // snapshot must never describe MORE tokens than the attached chain
                            // covers, and frames the slot never points at stay tree-owned.
                            let promoted = chain.len() * pages;
                            if promoted < reusable {
                                (reusable, ckpt) = tree.hit_prefix(node, recurrent, promoted);
                                chain.truncate(reusable / pages);
                            }
                            if let Some(kv) = target_kv.as_mut().filter(|_| !chain.is_empty()) {
                                let aux = ckpt.is_some();
                                match kv.paged_attach(
                                    self.vk.as_ref(),
                                    cfg,
                                    &mut tree,
                                    &chain,
                                    &prompt[..reusable],
                                    ckpt,
                                ) {
                                    Ok(()) => {
                                        tree_attached = reusable;
                                        tracing::info!(
                                            slot = target,
                                            cached_prefix_tokens = reusable,
                                            aux,
                                            tree = %tree.stats_line(),
                                            "paged KV prefix hit (zero-copy frame attach)"
                                        );
                                        // The tree carries no MTP draft state; free the head's
                                        // stale KV and let it re-warm on ordinary decode.
                                        if let Some(heads) = heads.as_mut() {
                                            if let Err(error) =
                                                heads[target].head.release_kv(self.vk.as_ref())
                                            {
                                                tracing::warn!(
                                                    slot = target,
                                                    "paged KV attach could not release stale MTP head KV: {error}"
                                                );
                                            }
                                            heads[target].invalidate_all();
                                        }
                                    }
                                    Err(error) => {
                                        tracing::warn!(
                                            slot = target,
                                            "paged KV attach failed; re-prefilling the request: {error}"
                                        );
                                        if let Some(heads) = heads.as_mut() {
                                            heads[target].invalidate_all();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Console-visible per-request ledger: what THIS prompt reuses (tree attach or the
                // resident continuation left in the slot) and what the corridor currently holds —
                // allocated frames cover both tree chains and live sessions, so `free` is what a
                // growing request can still claim before cold eviction starts.
                let reused = tree_attached.max(continuation_prefix);
                tracing::info!(
                    slot = target,
                    prompt_tokens = prompt.len(),
                    reused_tokens = reused,
                    prefill_tokens = prompt.len().saturating_sub(reused),
                    cache = %tree.stats_line(),
                    "paged KV prefix cache status"
                );
            }

            return Ok(SlotGuard {
                engine: self,
                idx: target,
                kv: target_kv,
                detached: false,
                multimodal_key: None,
            });
        }
    }

    /// Disk-extended slot checkout. Cold storage is deliberately serialized independently of the
    /// slot pool: a target is reserved under `pool`, then all GPU/file work runs after that lock is
    /// dropped. The maintenance thread handles timeout eviction independently.
    fn checkout_with_cold_cache(&self, prompt: &[u32], req: &RequestCtx) -> Result<SlotGuard<'_>> {
        let cache_mutex = self
            .session_cache
            .as_ref()
            .expect("cold checkout requires a session cache");
        let cfg: &Config = self.model.config();
        let mut pool = self.pool.lock().expect("pool poisoned");
        loop {
            let free = (0..pool.slots.len())
                .filter(|&index| !pool.slots[index].busy)
                .collect::<Vec<_>>();
            if free.is_empty() {
                pool = self.freed.wait(pool).expect("pool poisoned");
                continue;
            }
            let continuation = pick_continuation(
                free.iter().filter_map(|&index| {
                    (pool.slots[index].multimodal_key.is_none())
                        .then_some(pool.slots[index].kv.as_ref())
                        .flatten()
                        .and_then(|kv| {
                            kv.continuation_prefix_len(prompt)
                                .map(|prefix| (index, prefix, prefix))
                        })
                }),
                prompt.len(),
            );
            let continuation_prefix = continuation
                .and_then(|index| {
                    pool.slots[index]
                        .kv
                        .as_ref()
                        .and_then(|kv| kv.continuation_prefix_len(prompt))
                })
                .unwrap_or(0);
            let target = continuation.unwrap_or_else(|| {
                *free
                    .iter()
                    .min_by_key(|&&index| {
                        let slot = &pool.slots[index];
                        let empty = slot.kv.as_ref().is_none_or(|kv| kv.cached_len() == 0);
                        (!empty, slot.tick)
                    })
                    .expect("free is non-empty")
            });
            let seed_source = continuation
                .is_none()
                .then(|| {
                    free.iter()
                        .copied()
                        .filter(|&index| {
                            index != target && pool.slots[index].multimodal_key.is_none()
                        })
                        .filter_map(|index| {
                            let prefix = pool.slots[index]
                                .kv
                                .as_ref()
                                .map_or(0, |kv| kv.seedable_prefix_len(cfg, prompt));
                            (prefix >= 16).then_some((index, prefix))
                        })
                        .max_by_key(|&(_, prefix)| prefix)
                })
                .flatten();
            let resident_prefix =
                continuation_prefix.max(seed_source.map_or(0, |(_, prefix)| prefix));

            pool.tick += 1;
            let tick = pool.tick;
            pool.slots[target].busy = true;
            pool.slots[target].tick = tick;
            pool.slots[target].idle_since = None;
            let incompatible = pool.slots[target].multimodal_key.take().is_some();
            let mut target_kv = pool.slots[target].kv.take();
            let mut seed_kv = seed_source.map(|(source, prefix)| {
                pool.slots[source].busy = true;
                pool.slots[source].idle_since = None;
                let kv = pool.slots[source]
                    .kv
                    .take()
                    .expect("scored cold-cache seed slot has KV");
                (source, prefix, kv)
            });
            drop(pool);

            {
                // One pass owns both the inference baton and the cache catalog. No other request
                // can submit against a state buffer while it is being downloaded or restored.
                let _gate = req.gate_pass();
                let mut heads = self
                    .mtp_heads
                    .as_ref()
                    .map(|heads| heads.lock().expect("MTP heads poisoned"));
                let mut cache = match cache_mutex.lock() {
                    Ok(cache) => cache,
                    Err(poisoned) => poisoned.into_inner(),
                };
                if incompatible {
                    if let Some(kv) = target_kv.as_mut() {
                        kv.reset();
                    }
                    if let Some(heads) = heads.as_mut() {
                        heads[target].invalidate_all();
                    }
                }

                let cold_prefix = cache.best_continuation_len(prompt);
                let cold = (cold_prefix > resident_prefix && target_kv.is_some())
                    .then(|| cache.take_best_continuation(prompt))
                    .flatten();
                if let (Some(entry), Some(kv)) = (cold, target_kv.as_mut()) {
                    tracing::info!(
                        slot = target,
                        cached_prefix_tokens = cold_prefix,
                        "cold KV reusable prefix hit"
                    );
                    let mut released = false;
                    if kv.cached_len() != 0 {
                        let mtp_state = heads
                            .as_ref()
                            .and_then(|heads| heads[target].cache_state_for(kv.cached_tokens()));
                        let mtp_source =
                            heads
                                .as_ref()
                                .zip(mtp_state.as_ref())
                                .map(|(heads, state)| MtpCacheSource {
                                    head: &heads[target].head,
                                    state,
                                });
                        match cache.spill(kv, self.vk.as_ref(), cfg, mtp_source) {
                            Ok(true) => {
                                released = true;
                                if let Some(heads) = heads.as_mut() {
                                    heads[target].invalidate_all();
                                }
                            }
                            Ok(false) => {}
                            Err(error) => {
                                tracing::warn!(
                                    slot = target,
                                    "cold KV replacement spill failed; recycling the slot: {error}"
                                );
                                kv.reset();
                                if let Some(heads) = heads.as_mut() {
                                    heads[target].invalidate_all();
                                }
                            }
                        }
                    }
                    if !released {
                        released = kv.release_session_state(self.vk.as_ref(), cfg).is_ok()
                            || kv.release_session_state(self.vk.as_ref(), cfg).is_ok();
                    }
                    if released {
                        if let Some(heads) = heads.as_mut() {
                            if let Err(error) = heads[target].head.release_kv(self.vk.as_ref()) {
                                tracing::warn!(
                                    slot = target,
                                    "cold KV replacement could not release stale MTP head KV: {error}"
                                );
                            }
                            heads[target].invalidate_all();
                        }
                        let mtp_head = heads.as_ref().map(|heads| heads[target].head.as_ref());
                        match cache.restore(entry, kv, self.vk.as_ref(), cfg, mtp_head) {
                            Ok(Some(state)) => {
                                if let Some(heads) = heads.as_mut() {
                                    heads[target].install_cache_state(state);
                                }
                            }
                            Ok(None) => {
                                if let Some(heads) = heads.as_mut() {
                                    heads[target].invalidate_all();
                                }
                            }
                            Err(error) => {
                                if let Some(heads) = heads.as_mut() {
                                    heads[target].invalidate_all();
                                }
                                tracing::warn!(
                                    slot = target,
                                    "cold KV restore failed; re-prefilling the request: {error}"
                                );
                            }
                        }
                    } else {
                        tracing::warn!(
                            slot = target,
                            "could not release the target KV slot for cold restore; re-prefilling"
                        );
                        cache.return_entry(entry);
                        kv.reset();
                        if let Some(heads) = heads.as_mut() {
                            heads[target].invalidate_all();
                        }
                    }
                } else if continuation.is_none() {
                    if let Some(kv) = target_kv.as_mut().filter(|kv| kv.cached_len() != 0) {
                        let mtp_state = heads
                            .as_ref()
                            .and_then(|heads| heads[target].cache_state_for(kv.cached_tokens()));
                        let mtp_source =
                            heads
                                .as_ref()
                                .zip(mtp_state.as_ref())
                                .map(|(heads, state)| MtpCacheSource {
                                    head: &heads[target].head,
                                    state,
                                });
                        if let Err(error) = cache.spill(kv, self.vk.as_ref(), cfg, mtp_source) {
                            tracing::warn!(
                                slot = target,
                                "cold KV replacement spill failed; forgetting the old conversation: {error}"
                            );
                            kv.reset();
                            if let Some(heads) = heads.as_mut() {
                                heads[target].invalidate_all();
                            }
                        } else if let Some(heads) = heads.as_mut() {
                            if let Err(error) = heads[target].head.release_kv(self.vk.as_ref()) {
                                tracing::warn!(
                                    slot = target,
                                    "cold KV seed target could not release stale MTP head KV: {error}"
                                );
                            }
                            heads[target].invalidate_all();
                        }
                    }
                    if let (Some(kv), Some((source, prefix, source_kv))) =
                        (target_kv.as_mut(), seed_kv.as_ref())
                    {
                        match kv.seed_from(
                            self.vk.as_ref(),
                            cfg,
                            self.model.engine_cfg(),
                            source_kv,
                            *prefix,
                        ) {
                            Ok(seeded) if seeded != 0 => {
                                if let Some(heads) = heads.as_mut() {
                                    let tokens = &source_kv.cached_tokens()[..seeded];
                                    let (target_lane, source_lane) =
                                        mtp_slot_pair_mut(heads, target, *source);
                                    if let Err(error) =
                                        target_lane.seed_from(self.vk.as_ref(), source_lane, tokens)
                                    {
                                        target_lane.invalidate_all();
                                        tracing::warn!(
                                            slot = target,
                                            source_slot = *source,
                                            "cold-cache MTP prefix seed failed; preserving the seeded main KV and using ordinary decode: {error}"
                                        );
                                    }
                                }
                            }
                            Ok(_) => {}
                            Err(error) => {
                                kv.reset();
                                if let Some(heads) = heads.as_mut() {
                                    heads[target].invalidate_all();
                                }
                                tracing::warn!(
                                    slot = target,
                                    source_slot = *source,
                                    "cold-cache resident prefix seed failed; re-prefilling instead: {error}"
                                );
                            }
                        }
                    }
                }
                if let Err(error) = cache.gc() {
                    tracing::warn!("cold KV cache maintenance failed: {error}");
                }
            }

            if let Some((source, _, source_kv)) = seed_kv.take() {
                let mut pool = self.pool.lock().expect("pool poisoned");
                pool.slots[source].kv = Some(source_kv);
                pool.slots[source].busy = false;
                pool.slots[source].idle_since = Some(Instant::now());
                drop(pool);
                self.freed.notify_all();
            }

            return Ok(SlotGuard {
                engine: self,
                idx: target,
                kv: target_kv,
                detached: false,
                multimodal_key: None,
            });
        }
    }

    /// Take a slot whose expanded token prefix and image identity can both be continued.
    fn checkout_multimodal(
        &self,
        prompt: &[u32],
        key: MultimodalKey,
        req: &RequestCtx,
    ) -> SlotGuard<'_> {
        if self.session_cache.is_some() {
            self.checkout_multimodal_with_cold_cache(prompt, key, req)
        } else {
            self.checkout_multimodal_resident(prompt, key)
        }
    }

    fn checkout_multimodal_resident(&self, prompt: &[u32], key: MultimodalKey) -> SlotGuard<'_> {
        let mut pool = self.pool.lock().expect("pool poisoned");
        loop {
            let free = (0..pool.slots.len())
                .filter(|&index| !pool.slots[index].busy)
                .collect::<Vec<_>>();
            if !free.is_empty() {
                let continuation = pick_continuation(
                    free.iter().filter_map(|&index| {
                        (pool.slots[index].multimodal_key == Some(key))
                            .then_some(pool.slots[index].kv.as_ref())
                            .flatten()
                            .and_then(|kv| {
                                kv.continuation_prefix_len(prompt)
                                    .map(|prefix| (index, prefix, prefix))
                            })
                    }),
                    prompt.len(),
                );
                let target = continuation.unwrap_or_else(|| {
                    *free
                        .iter()
                        .min_by_key(|&&index| {
                            let slot = &pool.slots[index];
                            let empty = slot.kv.as_ref().is_none_or(|kv| kv.cached_len() == 0);
                            (!empty, slot.tick)
                        })
                        .expect("free is not empty")
                });
                pool.tick += 1;
                let tick = pool.tick;
                pool.slots[target].busy = true;
                pool.slots[target].tick = tick;
                pool.slots[target].idle_since = None;
                let old_key = pool.slots[target].multimodal_key.take();
                let mut kv = pool.slots[target].kv.take();
                if continuation != Some(target) || old_key != Some(key) {
                    if let Some(kv) = kv.as_mut() {
                        kv.reset();
                    }
                    if let Some(heads) = &self.mtp_heads {
                        heads.lock().expect("MTP heads poisoned")[target].invalidate_all();
                    }
                }
                return SlotGuard {
                    engine: self,
                    idx: target,
                    kv,
                    detached: false,
                    multimodal_key: Some(key),
                };
            }
            pool = self.freed.wait(pool).expect("pool poisoned");
        }
    }

    fn checkout_multimodal_with_cold_cache(
        &self,
        prompt: &[u32],
        key: MultimodalKey,
        req: &RequestCtx,
    ) -> SlotGuard<'_> {
        let cache_mutex = self
            .session_cache
            .as_ref()
            .expect("multimodal cold checkout requires a session cache");
        let mut pool = self.pool.lock().expect("pool poisoned");
        loop {
            let free = (0..pool.slots.len())
                .filter(|&index| !pool.slots[index].busy)
                .collect::<Vec<_>>();
            if !free.is_empty() {
                let continuation = pick_continuation(
                    free.iter().filter_map(|&index| {
                        (pool.slots[index].multimodal_key == Some(key))
                            .then_some(pool.slots[index].kv.as_ref())
                            .flatten()
                            .and_then(|kv| {
                                kv.continuation_prefix_len(prompt)
                                    .map(|prefix| (index, prefix, prefix))
                            })
                    }),
                    prompt.len(),
                );
                let target = continuation.unwrap_or_else(|| {
                    *free
                        .iter()
                        .min_by_key(|&&index| {
                            let slot = &pool.slots[index];
                            let empty = slot.kv.as_ref().is_none_or(|kv| kv.cached_len() == 0);
                            (!empty, slot.tick)
                        })
                        .expect("free is not empty")
                });
                pool.tick += 1;
                let tick = pool.tick;
                pool.slots[target].busy = true;
                pool.slots[target].tick = tick;
                pool.slots[target].idle_since = None;
                let old_key = pool.slots[target].multimodal_key.take();
                let mut kv = pool.slots[target].kv.take();
                drop(pool);
                if continuation != Some(target) || old_key != Some(key) {
                    if let Some(kv) = kv.as_mut().filter(|kv| kv.cached_len() != 0) {
                        if old_key.is_some() {
                            kv.reset();
                            if let Some(heads) = &self.mtp_heads {
                                heads.lock().expect("MTP heads poisoned")[target].invalidate_all();
                            }
                        } else {
                            let _gate = req.gate_pass();
                            let mut heads = self
                                .mtp_heads
                                .as_ref()
                                .map(|heads| heads.lock().expect("MTP heads poisoned"));
                            let mut cache = match cache_mutex.lock() {
                                Ok(cache) => cache,
                                Err(poisoned) => poisoned.into_inner(),
                            };
                            let mtp_state = heads.as_ref().and_then(|heads| {
                                heads[target].cache_state_for(kv.cached_tokens())
                            });
                            let mtp_source =
                                heads
                                    .as_ref()
                                    .zip(mtp_state.as_ref())
                                    .map(|(heads, state)| MtpCacheSource {
                                        head: &heads[target].head,
                                        state,
                                    });
                            if let Err(error) =
                                cache.spill(kv, self.vk.as_ref(), self.model.config(), mtp_source)
                            {
                                tracing::warn!(
                                    slot = target,
                                    "cold KV spill before multimodal reuse failed; forgetting the old conversation: {error}"
                                );
                                kv.reset();
                            }
                            if let Some(heads) = heads.as_mut() {
                                if let Err(error) = heads[target].head.release_kv(self.vk.as_ref())
                                {
                                    tracing::warn!(
                                        slot = target,
                                        "multimodal checkout could not release stale MTP head KV: {error}"
                                    );
                                }
                                heads[target].invalidate_all();
                            }
                            if let Err(error) = cache.gc() {
                                tracing::warn!("cold KV cache maintenance failed: {error}");
                            }
                        }
                    } else if let Some(kv) = kv.as_mut() {
                        kv.reset();
                        if let Some(heads) = &self.mtp_heads {
                            heads.lock().expect("MTP heads poisoned")[target].invalidate_all();
                        }
                    }
                }
                return SlotGuard {
                    engine: self,
                    idx: target,
                    kv,
                    detached: false,
                    multimodal_key: Some(key),
                };
            }
            pool = self.freed.wait(pool).expect("pool poisoned");
        }
    }

    /// Render an OpenAI conversation through the model's own chat template.
    /// Return a KV whose request thread no longer owns it while a decode cohort is active.
    fn return_detached_slot(&self, index: usize, kv: SeamKv) {
        let mut pool = match self.pool.lock() {
            Ok(pool) => pool,
            Err(error) => error.into_inner(),
        };
        pool.tick += 1;
        let tick = pool.tick;
        let Some(slot) = pool.slots.get_mut(index) else {
            tracing::error!(slot = index, "decode batch returned an unknown KV slot");
            return;
        };
        if !slot.busy || slot.kv.is_some() {
            tracing::error!(
                slot = index,
                busy = slot.busy,
                has_kv = slot.kv.is_some(),
                "decode batch found an inconsistent detached KV slot"
            );
            return;
        }
        slot.kv = Some(kv);
        slot.busy = false;
        slot.tick = tick;
        slot.multimodal_key = None;
        if self.session_cache.is_some() {
            slot.idle_since = Some(Instant::now());
        }
        drop(pool);
        if self.session_cache.is_some() {
            self.freed.notify_all();
        } else {
            self.freed.notify_one();
        }
    }

    fn deliver_batch_state(&self, slot: usize, tx: SyncSender<BatchEvent>, event: BatchEvent) {
        if let Err(mpsc::SendError(event)) = tx.send(event) {
            if let Some(kv) = event.into_kv() {
                self.return_detached_slot(slot, kv);
            }
        }
    }

    fn fail_batch_work(&self, mut work: BatchWork, error: &str) {
        if let Some(lookup) = &work.lookup {
            lookup.report(work.slot);
        }
        let kv = work.kv.take().expect("active decode work owns a KV slot");
        let Some(channels) = work.channels.take() else {
            self.return_detached_slot(work.slot, kv);
            return;
        };
        self.deliver_batch_state(
            work.slot,
            channels.events,
            BatchEvent::Failed {
                kv,
                error: error.to_owned(),
            },
        );
    }

    fn complete_batch_work(&self, mut work: BatchWork) {
        if let Some(lookup) = &work.lookup {
            lookup.report(work.slot);
        }
        let kv = work
            .kv
            .take()
            .expect("completed decode work owns a KV slot");
        if self.model.engine_cfg().debug.state_trace {
            let checkpoints = kv
                .session_state_meta()
                .checkpoint_tokens
                .map(|tokens| tokens.map(|tokens| tokens.len()));
            tracing::warn!(
                "[parallel state trace] slot={} complete target={} recurrent_checkpoints={:?}",
                work.slot,
                kv.cached_len(),
                checkpoints,
            );
        }
        let Some(channels) = work.channels.take() else {
            self.return_detached_slot(work.slot, kv);
            return;
        };
        self.deliver_batch_state(
            work.slot,
            channels.events,
            BatchEvent::Complete {
                kv,
                stats: work.stats,
            },
        );
    }

    /// Fail and drain work that has not entered the active cohort. Used after a batch error and
    /// when the persistent worker shuts down.
    fn close_decode_batch(&self, error: Option<&str>) {
        let waiting = {
            let mut queue = self
                .decode_batch
                .lock()
                .expect("decode batch queue poisoned");
            let waiting = queue.waiting.drain(..).collect::<Vec<_>>();
            self.batch_interrupt.store(false, Ordering::Release);
            waiting
        };
        for work in waiting {
            if let Some(error) = error {
                self.fail_batch_work(work, error);
            } else {
                self.fail_batch_work(work, "parallel scheduler closed with queued work");
            }
        }
    }

    fn take_pending_batch_work(&self, capacity: usize, force: bool) -> Vec<BatchWork> {
        if capacity == 0 || (!force && !self.batch_interrupt.swap(false, Ordering::AcqRel)) {
            return Vec::new();
        }
        let accepted = {
            let mut queue = self
                .decode_batch
                .lock()
                .expect("decode batch queue poisoned");
            let take = capacity.min(queue.waiting.len());
            let accepted = queue.waiting.drain(..take).collect::<Vec<_>>();
            if !queue.waiting.is_empty() {
                self.batch_interrupt.store(true, Ordering::Release);
            }
            accepted
        };
        if !accepted.is_empty() {
            tracing::debug!(
                lanes = accepted.len(),
                "admitting work into the unified Qwen3.8 scheduler"
            );
        }
        accepted
    }

    fn wait_for_scheduler_work(&self, stop: &AtomicBool) -> Option<Vec<BatchWork>> {
        let mut queue = self
            .decode_batch
            .lock()
            .expect("decode batch queue poisoned");
        while queue.waiting.is_empty() && !stop.load(Ordering::Acquire) {
            queue = self
                .decode_ready
                .wait(queue)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        if stop.load(Ordering::Acquire) {
            return None;
        }

        if queue.arriving > 0 && queue.waiting.len() < MAX_DECODE_BATCH {
            (queue, _) = self
                .decode_ready
                .wait_timeout_while(queue, DECODE_BATCH_WAIT, |queue| {
                    !stop.load(Ordering::Acquire)
                        && queue.arriving > 0
                        && queue.waiting.len() < MAX_DECODE_BATCH
                })
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        if stop.load(Ordering::Acquire) {
            return None;
        }

        let take = MAX_DECODE_BATCH.min(queue.waiting.len());
        let work = queue.waiting.drain(..take).collect::<Vec<_>>();
        self.batch_interrupt
            .store(!queue.waiting.is_empty(), Ordering::Release);
        Some(work)
    }

    fn take_prefill_cohort_peers(&self, capacity: usize) -> Vec<BatchWork> {
        if capacity == 0 {
            return Vec::new();
        }
        let queue = self
            .decode_batch
            .lock()
            .expect("decode batch queue poisoned");
        let timeout = if queue.arriving > 0 {
            DECODE_BATCH_WAIT
        } else {
            PREFILL_COHORT_WAIT
        };
        let (mut queue, _) = self
            .decode_ready
            .wait_timeout_while(queue, timeout, |queue| queue.waiting.is_empty())
            .expect("decode batch queue poisoned");
        let take = capacity.min(queue.waiting.len());
        let peers = queue.waiting.drain(..take).collect::<Vec<_>>();
        self.batch_interrupt
            .store(!queue.waiting.is_empty(), Ordering::Release);
        peers
    }

    /// Called only after the active set becomes empty. Briefly include requests that registered
    /// before tokenization; otherwise return control to the persistent worker's idle wait.
    fn refill_batch(&self) -> Option<Vec<BatchWork>> {
        let queue = self
            .decode_batch
            .lock()
            .expect("decode batch queue poisoned");
        let (mut queue, _) = self
            .decode_ready
            .wait_timeout_while(queue, DECODE_BATCH_WAIT, |queue| {
                queue.waiting.is_empty() && queue.arriving > 0
            })
            .expect("decode batch queue poisoned");
        if queue.waiting.is_empty() {
            self.batch_interrupt.store(false, Ordering::Release);
            None
        } else {
            let take = MAX_DECODE_BATCH.min(queue.waiting.len());
            let work = queue.waiting.drain(..take).collect::<Vec<_>>();
            self.batch_interrupt
                .store(!queue.waiting.is_empty(), Ordering::Release);
            Some(work)
        }
    }

    fn qsa_sparse_at(&self, visible_tokens: usize) -> bool {
        visible_tokens > self.qsa_threshold()
    }

    fn qsa_threshold(&self) -> usize {
        let cfg = self.model.config();
        let ratio = cfg
            .compress_ratios
            .iter()
            .copied()
            .max()
            .unwrap_or(4)
            .max(1);
        cfg.indexer_top_k + ratio - 1
    }

    fn token_steps_before_qsa_boundary<'a>(
        &self,
        active: impl IntoIterator<Item = &'a BatchWork>,
    ) -> usize {
        let threshold = self.qsa_threshold();
        active
            .into_iter()
            .filter(|work| !self.qsa_sparse_at(work.next_qsa_position()))
            .map(|work| {
                threshold
                    .saturating_sub(work.next_qsa_position())
                    .saturating_add(1)
            })
            .min()
            .unwrap_or(usize::MAX)
    }

    #[allow(clippy::too_many_arguments)]
    fn wait_for_decode_batch<F: FnMut(&str)>(
        &self,
        guard: &mut SlotGuard<'_>,
        events: Receiver<BatchEvent>,
        acknowledgements: SyncSender<bool>,
        req: &RequestCtx,
        acc: &mut Vec<u32>,
        printed: &mut usize,
        on_piece: &mut F,
        stream_text: bool,
    ) -> Result<GenStats> {
        loop {
            let event = match events.recv() {
                Ok(event) => event,
                Err(_) => {
                    guard.abandon_detached();
                    return Err(anyhow!(
                        "parallel decode worker stopped before returning its KV"
                    ));
                }
            };
            match event {
                BatchEvent::Progress { progress } => req.report_progress(progress),
                BatchEvent::Token { id, progress } => {
                    req.report_progress(progress);
                    let keep_going = if crate::sampling::abort_requested(Some(req)) {
                        false
                    } else {
                        if stream_text {
                            crate::stream_token(self.model.tokenizer(), acc, printed, id, on_piece);
                        }
                        !crate::sampling::abort_requested(Some(req))
                    };
                    let _ = acknowledgements.send(keep_going);
                }
                BatchEvent::Complete { kv, stats } => {
                    guard.reattach(kv);
                    return Ok(stats);
                }
                BatchEvent::Failed { kv, error } => {
                    guard.reattach(kv);
                    return Err(anyhow!(error));
                }
            }
        }
    }

    fn report_lookup_policy(
        &self,
        req: &RequestCtx,
        slot: usize,
        path: &'static str,
        reason: Option<&str>,
    ) {
        let cfg = self.model.config();
        let ec = self.model.engine_cfg();
        if !ec.spec.lookup {
            return;
        }
        let effective = crate::sampling::Sampler::resolve(Some(req), &ec.sampling);
        let dtype = self
            .model
            .gguf()
            .tensors()
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .map(|tensor| tensor.dtype);
        tracing::info!(slot, path, eligible = reason.is_none(), reason = reason.unwrap_or("ready"),
            effective_temp = effective.temp, effective_top_k = effective.top_k,
            penalties = req.sampling().penalties_active(), heads_loaded = self.mtp_heads.is_some(),
            scheduler_available = self.gate.is_some(), parallel_rows = cfg.supports_parallel_token_rows(),
            target_supported = crate::seam::target_verify_supported(cfg),
            target_enabled = crate::seam::target_verify_enabled(cfg, ec),
            gpu_embed = ec.spec.gpu_embed, backend_embed_gather = self.vk.capabilities().embed_gather,
            embedding_dtype = ?dtype, dn_chunk = ec.kernels.vulkan.dn_chunk,
            dn_chunk_scan = ec.kernels.vulkan.dn_chunk_scan, dn_split = ec.kernels.vulkan.dn_split,
            layers = cfg.n_layer, width = cfg.n_embd, hc_mult = cfg.hc_mult,
            ssm_state = cfg.ssm_d_state, ssm_inner = cfg.ssm_d_inner, ssm_rank = cfg.ssm_dt_rank,
            "independent Lookup request policy");
    }

    fn scheduler_admission_eligible(&self) -> bool {
        self.gate.is_some()
            && (!self.model.engine_cfg().spec.lookup
                || scheduler_requested(self.n_slots(), self.mtp_heads.is_some(), true, || {
                    lookup_scheduler_supported(
                        self.model.config(),
                        self.model.engine_cfg(),
                        self.vk.capabilities().embed_gather,
                        self.model
                            .gguf()
                            .tensors()
                            .iter()
                            .find(|tensor| tensor.name == "token_embd.weight")
                            .map(|tensor| tensor.dtype),
                    )
                }))
    }

    fn new_lookup_work(
        &self,
        req: &RequestCtx,
        prompt: &[u32],
        slot: usize,
        path: &'static str,
    ) -> Option<LookupBatchState> {
        let ec = self.model.engine_cfg();
        if !ec.spec.lookup {
            return None;
        }
        let dtype = self
            .model
            .gguf()
            .tensors()
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .map(|tensor| tensor.dtype);
        // A loaded head owns the target frontier until explicitly synchronized.
        if let Some(reason) = lookup_fallback_reason(
            self.model.config(),
            ec,
            req,
            self.mtp_heads.is_some(),
            self.vk.capabilities().embed_gather,
            dtype,
        ) {
            self.report_lookup_policy(req, slot, path, Some(reason));
            return None;
        }
        let config = crate::speculative_lookup::LookupConfig {
            min_match: ec.spec.lookup_min_match,
            adaptive_width: ec.spec.lookup_adaptive_width,
            history_capacity: self.max_ctx.clamp(65_536, 262_144),
            max_drafts: ec
                .spec
                .lookup_max_draft
                .min(crate::speculative_verify::MAX_VERIFY_ROWS - 1),
            ..Default::default()
        };
        match LookupBatchState::new(config, prompt) {
            Ok(lookup) => {
                self.report_lookup_policy(req, slot, path, None);
                Some(lookup)
            }
            Err(error) => {
                self.report_lookup_policy(req, slot, path, Some("invalid_lookup_policy"));
                tracing::warn!(%error, "invalid Lookup policy; using ordinary decode");
                None
            }
        }
    }

    fn prepare_unified_work(&self, work: &mut BatchWork, req: &RequestCtx) -> Result<()> {
        if work.phase != BatchPhase::Unprepared {
            return Ok(());
        }
        if let Some(lookup) = &mut work.lookup {
            lookup.sync_history(&work.prompt);
        }
        let checkpoint = work.turn_checkpoint.take();
        let prepared = {
            let _gate = req.gate_pass();
            crate::seam::prepare_dense_vulkan_parallel_prompt_session(
                self.vk.as_ref(),
                self.model.config(),
                self.model.engine_cfg(),
                work.kv.as_mut().expect("unprepared work owns a KV slot"),
                &work.prompt[..work.prompt_end],
                checkpoint,
            )?
        };
        work.prefill_start = prepared.start;
        work.checkpoint_boundaries = prepared.checkpoint_boundaries;
        work.paged_frontier = prepared.paged_frontier;
        work.stats.n_prompt = work
            .stats
            .n_prompt
            .saturating_add(work.prompt_end.saturating_sub(prepared.start));
        work.stats.n_cached = work.stats.n_cached.saturating_add(prepared.start);
        work.phase = phase_for_remaining_prefill(work.remaining_prefill());
        tracing::debug!(
            slot = work.slot,
            start = prepared.start,
            prompt_tokens = work.prompt_end,
            remaining = work.remaining_prefill(),
            phase = ?work.phase,
            "prepared work for the unified Qwen hybrid scheduler"
        );
        Ok(())
    }

    fn run_long_prefill_phase(&self, active: &mut Vec<BatchWork>, req: &RequestCtx) -> Result<()> {
        let mut long = Vec::new();
        let mut token_work = Vec::with_capacity(active.len());
        let long_prefill_limit = long_prefill_cohort_limit(self.model.config());
        let batch_prefill_rows = long_prefill_limit > 1;
        for work in active.drain(..) {
            if work.phase == BatchPhase::LongPrefill && long.len() < long_prefill_limit {
                long.push(work);
            } else {
                token_work.push(work);
            }
        }
        *active = token_work;
        if long.is_empty() {
            return Ok(());
        }
        let phase_prompt_ends = long
            .iter()
            .map(|work| {
                long_prefill_phase_prompt_end(
                    work.kv().cached_len(),
                    work.prompt_end,
                    &work.checkpoint_boundaries,
                )
            })
            .collect::<Vec<_>>();
        let prompts = long
            .iter()
            .zip(&phase_prompt_ends)
            .map(|(work, &end)| work.prompt[..end].to_vec())
            .collect::<Vec<_>>();
        let prepared = long
            .iter()
            .map(|work| crate::seam::PreparedParallelPrompt {
                start: work.prefill_start,
                checkpoint_boundaries: work.checkpoint_boundaries,
                paged_frontier: work.paged_frontier,
            })
            .collect::<Vec<_>>();
        let progress_senders = long
            .iter()
            .map(|work| {
                work.channels
                    .as_ref()
                    .map(|channels| channels.events.clone())
            })
            .collect::<Vec<_>>();
        let report_progress = |lane: usize, progress: infr_core::GenerationProgress| {
            if let Some(Some(events)) = progress_senders.get(lane) {
                let _ = events.send(BatchEvent::Progress { progress });
            }
        };
        let mut primary = long[0].kv.take();
        let mut peers = long
            .iter_mut()
            .skip(1)
            .map(|work| work.kv.take().expect("long prefill work owns a KV slot"))
            .collect::<Vec<_>>();
        let result = {
            // Long prefill owns the pager phase as one transaction. Passing no RequestCtx below
            // avoids per-chunk baton release and prevents unrelated legacy work from rebuilding
            // Decode LRU between ring chunks.
            let _gate = req.gate_pass();
            if batch_prefill_rows {
                crate::seam::generate_dense_vulkan_parallel_prefill_session(
                    self.vk.as_ref(),
                    self.model.gguf(),
                    self.model.config(),
                    self.model.engine_cfg(),
                    self.model.embd(),
                    self.model.per_layer_embd(),
                    &prompts,
                    &mut primary,
                    &mut peers,
                    self.max_ctx,
                    &prepared,
                    Some(&report_progress),
                    None,
                )
            } else {
                debug_assert_eq!(prompts.len(), 1);
                debug_assert!(peers.is_empty());
                crate::seam::generate_dense_vulkan_prepared_prefill_session(
                    self.vk.as_ref(),
                    self.model.gguf(),
                    self.model.config(),
                    self.model.engine_cfg(),
                    self.model.embd(),
                    self.model.per_layer_embd(),
                    &prompts[0],
                    &mut primary,
                    self.max_ctx,
                    prepared[0],
                )
                .map(|stats| vec![stats])
            }
        };
        long[0].kv = primary;
        for (work, kv) in long.iter_mut().skip(1).zip(peers) {
            work.kv = Some(kv);
        }

        let stats = match result {
            Ok(stats) if stats.len() == long.len() => stats,
            Ok(stats) => {
                active.extend(long);
                return Err(anyhow!(
                    "parallel prefill returned {} stats for {} lanes",
                    stats.len(),
                    prompts.len()
                ));
            }
            Err(error) => {
                active.extend(long);
                return Err(error);
            }
        };
        if long
            .iter()
            .zip(&phase_prompt_ends)
            .any(|(work, &end)| work.kv().cached_len() + 1 != end)
        {
            active.extend(long);
            return Err(anyhow!(
                "parallel long prefill stopped before every lane reached its decode frontier"
            ));
        }
        for (work, stats) in long.iter_mut().zip(stats) {
            work.stats.prompt_secs += stats.prompt_secs;
            let cached_len = work.kv().cached_len();
            for boundary in &mut work.checkpoint_boundaries {
                if boundary.is_some_and(|boundary| cached_len >= boundary) {
                    *boundary = None;
                }
            }
            if work
                .paged_frontier
                .is_some_and(|frontier| cached_len >= frontier)
            {
                work.paged_frontier = None;
            }
            work.phase = phase_for_remaining_prefill(work.remaining_prefill());
            let progress = infr_core::GenerationProgress {
                phase: infr_core::GenerationPhase::Prefill,
                prompt_tokens: work.prompt_end as u64,
                cached_prompt_tokens: work.prefill_start as u64,
                prefill_tokens: cached_len.saturating_sub(work.prefill_start) as u64,
                completion_tokens: 0,
                context_tokens: cached_len as u64,
                context_limit: self.max_ctx as u64,
            };
            if let Some(channels) = work.channels.as_ref() {
                if channels
                    .events
                    .send(BatchEvent::Progress { progress })
                    .is_err()
                {
                    work.finished = true;
                }
            }
        }
        tracing::debug!(
            lanes = long.len(),
            "completed one exclusive long-prefill phase"
        );
        active.extend(long);
        Ok(())
    }

    fn materialize_mtp_frontiers(&self, active: &mut [BatchWork], indices: &[usize]) -> Result<()> {
        let heads = self.mtp_heads.as_ref().expect("MTP scheduler has heads");
        for &index in indices {
            let (pending, already_emitted) = {
                let heads = heads.lock().expect("MTP heads poisoned");
                let work = &active[index];
                let lane = &heads[work.slot];
                anyhow::ensure!(
                    lane.cached.as_slice() == work.kv().cached_tokens(),
                    "MTP lane {} target and head prefixes diverged before plain decode",
                    work.slot
                );
                let pending = lane.pending_id.ok_or_else(|| {
                    anyhow!("MTP lane {} has no frontier for plain decode", work.slot)
                })?;
                if lane.pending_emitted {
                    anyhow::ensure!(
                        work.prompt.len() == lane.cached.len() + 1
                            && work.prompt.last() == Some(&pending),
                        "MTP lane {} has an invalid emitted frontier",
                        work.slot
                    );
                } else {
                    anyhow::ensure!(
                        work.prompt.as_slice() == lane.cached.as_slice(),
                        "MTP lane {} has an invalid uncommitted frontier",
                        work.slot
                    );
                }
                (pending, lane.pending_emitted)
            };
            if already_emitted {
                continue;
            }

            let work = &mut active[index];
            let eos = !self.model.engine_cfg().sampling.ignore_eos
                && (self.model.config().eos_ids.contains(&pending)
                    || pending == self.model.config().eos);
            let mut committed = eos;
            let mut stopped = eos;
            if !eos {
                let progress = self.batch_decode_progress(
                    work.prompt_end,
                    work.prefill_start,
                    work.generated + 1,
                );
                let Some(channels) = work.channels.as_ref() else {
                    work.finished = true;
                    continue;
                };
                if channels
                    .events
                    .send(BatchEvent::Token {
                        id: pending,
                        progress,
                    })
                    .is_ok()
                {
                    committed = true;
                    stopped = !channels.acknowledgements.recv().unwrap_or(false);
                } else {
                    stopped = true;
                }
            }
            if committed {
                work.prompt.push(pending);
                work.reasoning_guard.observe(self.model.config(), pending);
                work.generated += 1;
                work.stats.n_gen += 1;
                heads.lock().expect("MTP heads poisoned")[work.slot].pending_emitted = true;
            }
            work.finished = stopped || work.generated >= work.max_new;
        }
        Ok(())
    }

    fn deactivate_mtp_lanes(
        &self,
        active: &mut [BatchWork],
        indices: &[usize],
        reason: &'static str,
    ) -> Result<()> {
        let ready = indices
            .iter()
            .copied()
            .filter(|&index| active[index].mtp_ready)
            .collect::<Vec<_>>();
        self.materialize_mtp_frontiers(active, &ready)?;

        let mut heads = self
            .mtp_heads
            .as_ref()
            .expect("MTP scheduler has heads")
            .lock()
            .expect("MTP heads poisoned");
        for &index in indices {
            let work = &mut active[index];
            heads[work.slot].invalidate_live();
            work.mtp_ready = false;
        }
        tracing::info!(
            lanes = indices.len(),
            reason,
            "Qwen3.8 MTP switched to ordinary decode"
        );
        Ok(())
    }

    fn sync_mtp_after_plain_decode(
        &self,
        active: &[BatchWork],
        indices: &[usize],
        outputs: &[Vec<u32>],
        hidden: &[Vec<f32>],
    ) -> Result<()> {
        let cfg = self.model.config();
        let h_width = cfg.hc_mult * cfg.n_embd;
        anyhow::ensure!(
            outputs.len() == indices.len() && hidden.len() == indices.len(),
            "plain decode returned an incomplete MTP synchronization trace"
        );
        let mut heads = self
            .mtp_heads
            .as_ref()
            .expect("MTP scheduler has heads")
            .lock()
            .expect("MTP heads poisoned");
        for (lane_index, &index) in indices.iter().enumerate() {
            let work = &active[index];
            if !work.mtp_ready {
                continue;
            }
            let lane = &mut heads[work.slot];
            anyhow::ensure!(
                lane.pending_emitted,
                "MTP lane {} entered plain decode with an unstreamed frontier",
                work.slot
            );
            let target = work.kv().cached_tokens();
            anyhow::ensure!(
                target.starts_with(&lane.cached),
                "MTP lane {} target no longer extends the head prefix",
                work.slot
            );
            let start = lane.cached.len();
            let rows = target.len() - start;
            anyhow::ensure!(
                rows > 0
                    && outputs[lane_index].len() == rows
                    && hidden[lane_index].len() == rows * h_width,
                "MTP lane {} plain synchronization has rows={rows}, outputs={}, hidden={} (width {h_width})",
                work.slot,
                outputs[lane_index].len(),
                hidden[lane_index].len()
            );
            let mut shifted = vec![0.0f32; rows * h_width];
            shifted[..h_width].copy_from_slice(&lane.last_h);
            if rows > 1 {
                shifted[h_width..].copy_from_slice(&hidden[lane_index][..(rows - 1) * h_width]);
            }
            let catch_up_t0 = Instant::now();
            lane.head.catch_up(
                self.vk.as_ref(),
                &target[start..],
                &shifted,
                start,
                work.mrope_plan.as_ref(),
                work.kv().mtp_shared_weights(),
            )?;
            tracing::debug!(
                slot = work.slot,
                rows,
                elapsed_ms = catch_up_t0.elapsed().as_secs_f64() * 1e3,
                "synchronized MTP head after ordinary concurrent decode"
            );
            lane.cached.extend_from_slice(&target[start..]);
            lane.last_h
                .copy_from_slice(&hidden[lane_index][(rows - 1) * h_width..]);
            lane.pending_id = outputs[lane_index].last().copied();
            lane.pending_emitted = true;
            lane.target_snapshot_dirty = true;
            lane.live_valid = true;
        }
        Ok(())
    }

    fn refresh_mtp_snapshots(&self, active: &mut [BatchWork], indices: &[usize]) -> Result<()> {
        let cfg = self.model.config();
        let mut heads = self
            .mtp_heads
            .as_ref()
            .expect("MTP scheduler has heads")
            .lock()
            .expect("MTP heads poisoned");
        for &index in indices {
            let work = &mut active[index];
            let lane = &mut heads[work.slot];
            if !lane.target_snapshot_dirty {
                continue;
            }
            work.kv
                .as_mut()
                .expect("MTP work has target KV")
                .mtp_snapshot_delta(self.vk.as_ref(), cfg)?;
            lane.target_snapshot_dirty = false;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_token_group(
        &self,
        active: &mut [BatchWork],
        indices: &[usize],
        quantum: usize,
        sync_mtp: bool,
    ) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        // MRoPE needs a visual primary, but sampling needs all Prefill rows before Decode rows.
        // If these orders conflict, drain one text suffix without sampling, then rejoin the cohort.
        let prefill_fallback = token_lane_prefill_fallback(indices.iter().map(|&index| {
            let work = &active[index];
            (work.mrope_plan.is_some(), work.remaining_prefill())
        }));
        let prefill_index = prefill_fallback.map(|(lane, _)| indices[lane]);
        let indices = prefill_index
            .as_ref()
            .map(std::slice::from_ref)
            .unwrap_or(indices);
        let lookup_t0 = (indices.len() == 1
            && active[indices[0]].lookup.is_some()
            && active[indices[0]].phase == BatchPhase::Decode)
            .then(Instant::now);
        let quantum = prefill_fallback.map_or(quantum, |(_, remaining)| quantum.min(remaining));
        let max_steps = indices
            .iter()
            .map(|&index| {
                let work = &active[index];
                work.remaining_prefill()
                    .saturating_add(work.max_new.saturating_sub(work.generated))
            })
            .min()
            .unwrap_or(0)
            .min(self.token_steps_before_qsa_boundary(indices.iter().map(|&index| &active[index])))
            .min(quantum);
        if max_steps == 0 {
            return Err(anyhow!("parallel token group contains exhausted work"));
        }
        let prompts = indices
            .iter()
            .map(|&index| active[index].prompt.clone())
            .collect::<Vec<_>>();
        let prompt_ends = indices
            .iter()
            .map(|&index| active[index].prompt_end)
            .collect::<Vec<_>>();
        let checkpoints = indices
            .iter()
            .map(|&index| active[index].checkpoint_boundaries)
            .collect::<Vec<_>>();
        let mut owned_mrope_plans = indices
            .iter()
            .map(|&index| active[index].mrope_plan.take())
            .collect::<Vec<_>>();
        let mrope_plans = owned_mrope_plans
            .iter()
            .map(Option::as_ref)
            .collect::<Vec<_>>();
        let mut primary = active[indices[0]].kv.take();
        let mut peers = indices[1..]
            .iter()
            .map(|&index| {
                active[index]
                    .kv
                    .take()
                    .expect("parallel token work owns a KV slot")
            })
            .collect::<Vec<_>>();
        let mut samplers = indices
            .iter()
            .map(|&index| {
                active[index]
                    .sampler
                    .take()
                    .expect("parallel token work owns a sampler")
            })
            .collect::<Vec<_>>();
        let mut reasoning_guards = indices
            .iter()
            .map(|&index| active[index].reasoning_guard)
            .collect::<Vec<_>>();
        let mut channels = indices
            .iter()
            .map(|&index| active[index].channels.take())
            .collect::<Vec<_>>();
        let progress_bases = indices
            .iter()
            .map(|&index| {
                let work = &active[index];
                (work.prompt_end, work.prefill_start, work.generated)
            })
            .collect::<Vec<_>>();
        let mut streamed = vec![0usize; indices.len()];
        let mut accepted = vec![true; indices.len()];
        let lookup_single_decode = active.len() == 1
            && !sync_mtp
            && lookup_ordinary_sample_eligible(
                active[indices[0]].phase,
                active[indices[0]].generated,
                indices.len(),
            );
        let lookup_output_limit = active[indices[0]].max_new;
        let lookup_max_draft = self.model.engine_cfg().spec.lookup_max_draft;
        let lookup_qsa_threshold = self.qsa_threshold();
        let mut lookups = if indices.iter().any(|&index| active[index].lookup.is_some()) {
            indices
                .iter()
                .map(|&index| {
                    let work = &mut active[index];
                    let mut lookup = work.lookup.take();
                    if let Some(lookup) = &mut lookup {
                        if lookup.indexed_len != work.prompt.len() {
                            lookup.sync_history(&work.prompt);
                        }
                    }
                    lookup
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut lookup_search_secs = vec![0.0; lookups.len()];
        let mut lookup_delivery_secs = 0.0;
        let group_req = match &self.gate {
            Some(gate) => {
                RequestCtx::with_gate(active[indices[0]].sampling.clone(), Arc::clone(gate))
            }
            None => RequestCtx::new(active[indices[0]].sampling.clone()),
        };
        // This shared service path reaches execute_parallel_token_forward in the runner:
        // split PLE Decode admits L0 and tail together before either plan submits work.
        let mut run = |on_token: ParallelTokenCallback<'_>| {
            crate::seam::generate_dense_vulkan_parallel_sampled_session(
                self.vk.as_ref(),
                self.model.gguf(),
                self.model.config(),
                self.model.engine_cfg(),
                self.model.embd(),
                self.model.per_layer_embd(),
                &prompts,
                &prompt_ends,
                &checkpoints,
                &mrope_plans,
                max_steps,
                &mut primary,
                &mut peers,
                self.max_ctx,
                &mut samplers,
                &mut reasoning_guards,
                on_token,
                Some(&self.batch_interrupt),
                Some(&group_req),
                sync_mtp,
            )
        };
        let batch_delivery = !sync_mtp
            && !self.model.engine_cfg().spec.mtp
            && lookups.is_empty()
            && indices
                .iter()
                .all(|&index| active[index].phase == BatchPhase::Decode);
        let result = if batch_delivery {
            let mut delivery = BatchTokenDelivery {
                engine: self,
                channels: &channels,
                progress_bases: &progress_bases,
                streamed: &mut streamed,
            };
            let mut stream = |tokens: &[(usize, u32)]| {
                let batch_accepted =
                    token_delivery::deliver_token_batch::<MAX_DECODE_BATCH>(tokens, &mut delivery);
                for &(lane, _) in tokens {
                    accepted[lane] = batch_accepted[lane];
                }
                batch_accepted
            };
            run(ParallelTokenCallback::Batch(&mut stream))
        } else {
            // Keep Lookup's post-ACK indexing/yield and the MTP/Prefill callback order intact.
            let mut stream = |lane: usize, id: u32| {
                let delivery_t0 = lookup_t0.map(|_| Instant::now());
                let timeline_output = infr_core::timeline::span("decode_output_progress");
                streamed[lane] += 1;
                let (prompt_tokens, cached_prompt_tokens, generated) = progress_bases[lane];
                let progress = self.batch_decode_progress(
                    prompt_tokens,
                    cached_prompt_tokens,
                    generated.saturating_add(streamed[lane]),
                );
                drop(timeline_output);
                let keep_going =
                    channels
                        .get_mut(lane)
                        .and_then(Option::as_mut)
                        .is_some_and(|channels| {
                            let timeline_send = infr_core::timeline::span("decode_output_send");
                            let sent = channels
                                .events
                                .send(BatchEvent::Token { id, progress })
                                .is_ok();
                            drop(timeline_send);
                            let _timeline_ack = infr_core::timeline::span("decode_output_ack_wait");
                            sent && channels.acknowledgements.recv().unwrap_or(false)
                        });
                accepted[lane] = keep_going;
                if let Some(lookup) = lookups.get_mut(lane).and_then(Option::as_mut) {
                    let budget = if lookup_single_decode {
                        lookup_draft_budget(
                            lookup_max_draft,
                            lookup_output_limit.saturating_sub(generated + streamed[lane]),
                            lookup.indexed_len + 1,
                            self.max_ctx,
                            lookup_qsa_threshold,
                        )
                    } else {
                        0
                    };
                    lookup_search_secs[lane] += lookup.observe_ordinary_token(
                        id,
                        keep_going,
                        lookup_single_decode,
                        budget,
                        self.batch_interrupt.as_ref(),
                    );
                }
                if let Some(t0) = delivery_t0 {
                    lookup_delivery_secs += t0.elapsed().as_secs_f64();
                }
                keep_going
            };
            run(ParallelTokenCallback::Legacy(&mut stream))
        };
        active[indices[0]].kv = primary;
        for (&index, kv) in indices[1..].iter().zip(peers) {
            active[index].kv = Some(kv);
        }
        for (&index, sampler) in indices.iter().zip(samplers) {
            active[index].sampler = Some(sampler);
        }
        for (&index, guard) in indices.iter().zip(reasoning_guards) {
            active[index].reasoning_guard = guard;
        }
        for (&index, channel) in indices.iter().zip(channels) {
            active[index].channels = channel;
        }
        // Restore ownership before inspecting result: error cleanup must still see Lookup slots.
        for (&index, lookup) in indices.iter().zip(lookups) {
            active[index].lookup = lookup;
        }
        drop(mrope_plans);
        for (&index, plan) in indices.iter().zip(owned_mrope_plans.drain(..)) {
            active[index].mrope_plan = plan;
        }

        let lookup_wall_secs =
            lookup_t0.map(|t0| (t0.elapsed().as_secs_f64() - lookup_delivery_secs).max(0.0));
        // L0 may already have changed recurrent/KV state when a staged PLE handoff fails.
        // All slot ownership is restored above; propagate the error without replaying the group.
        let result = result?;
        let outputs = result.tokens;
        let prompt_secs = result.prompt_secs;
        let decode_secs = result.decode_secs;
        if outputs.len() != indices.len()
            || prompt_secs.len() != indices.len()
            || decode_secs.len() != indices.len()
        {
            return Err(anyhow!(
                "parallel token runner returned inconsistent lane counts"
            ));
        }
        if sync_mtp {
            self.sync_mtp_after_plain_decode(active, indices, &outputs, &result.hidden)?;
        }
        let cfg = self.model.config();
        for (lane, &index) in indices.iter().enumerate() {
            let work = &mut active[index];
            let lookup_sample =
                lookup_ordinary_sample_eligible(work.phase, work.generated, indices.len());
            work.stats.prompt_secs += prompt_secs[lane];
            work.stats.decode_secs += decode_secs[lane];
            work.generated += outputs[lane].len();
            work.stats.n_gen += outputs[lane].len();
            work.prompt.extend_from_slice(&outputs[lane]);
            if let Some(lookup) = &mut work.lookup {
                lookup.finish_ordinary_outputs(prompts[lane].len(), &outputs[lane])?;
                work.stats.decode_secs += lookup_search_secs[lane];
                lookup.record_ordinary(
                    if lookup_sample {
                        lookup_wall_secs.unwrap_or(decode_secs[lane])
                    } else {
                        0.0
                    },
                    outputs[lane].len(),
                );
            }
            let cached_len = work.kv().cached_len();
            for boundary in &mut work.checkpoint_boundaries {
                if boundary.is_some_and(|boundary| cached_len >= boundary) {
                    *boundary = None;
                }
            }
            if work
                .paged_frontier
                .is_some_and(|frontier| cached_len >= frontier)
            {
                work.paged_frontier = None;
            }
            work.phase = phase_for_remaining_prefill(work.remaining_prefill());
            let eos = outputs[lane].last().is_some_and(|token| {
                !self.model.engine_cfg().sampling.ignore_eos
                    && (cfg.eos_ids.contains(token) || *token == cfg.eos)
            });
            work.finished = eos || !accepted[lane] || work.generated >= work.max_new;

            if work.phase == BatchPhase::ShortPrefill {
                let progress = infr_core::GenerationProgress {
                    phase: infr_core::GenerationPhase::Prefill,
                    prompt_tokens: work.prompt_end as u64,
                    cached_prompt_tokens: work.prefill_start as u64,
                    prefill_tokens: work.kv().cached_len().saturating_sub(work.prefill_start)
                        as u64,
                    completion_tokens: work.generated as u64,
                    context_tokens: work.kv().cached_len() as u64,
                    context_limit: self.max_ctx as u64,
                };
                if let Some(channels) = work.channels.as_ref() {
                    if channels
                        .events
                        .send(BatchEvent::Progress { progress })
                        .is_err()
                    {
                        work.finished = true;
                    }
                }
            }
        }
        Ok(())
    }

    fn retire_finished_work(&self, active: &mut Vec<BatchWork>) {
        for index in (0..active.len()).rev() {
            if !active[index].finished {
                continue;
            }
            self.complete_batch_work(active.remove(index));
        }
    }

    fn fail_unified_scheduler(&self, active: &mut Vec<BatchWork>, mut error: anyhow::Error) {
        let numerical = crate::sampling::is_numerical_corruption(&error);
        for work in active.iter_mut() {
            if numerical || self.mtp_heads.is_some() || work.lookup.is_some() {
                work.kv.as_mut().expect("batch work owns a KV slot").reset();
            }
        }
        if numerical {
            tracing::error!(
                active_slots = active.len(),
                %error,
                "discarded KV state after non-finite model output"
            );
        }
        if let Some(heads) = &self.mtp_heads {
            let _gate = self.gate.as_ref().map(|gate| gate.enter());
            let mut heads = heads.lock().expect("MTP heads poisoned");
            for work in active.iter() {
                let lane = &mut heads[work.slot];
                if let Err(restore) = lane.head.settle_draft(self.vk.as_ref(), 0) {
                    tracing::error!(slot = work.slot, %restore,
                        "MTP head rollback failed; discarded the affected target/head prefix");
                    error = error.context(format!(
                        "MTP head slot {} rollback also failed: {restore:#}",
                        work.slot
                    ));
                }
                lane.invalidate_all();
            }
        }
        let message = format!("{error:#}");
        for work in active.drain(..) {
            self.fail_batch_work(work, &message);
        }
        self.close_decode_batch(Some(&message));
        self.vk.release_primary_runtime_after_cohort_shrink();
        tracing::warn!(%error, "parallel scheduler batch failed");
    }

    fn prime_mtp_work(&self, work: &mut BatchWork, req: &RequestCtx) -> Result<()> {
        if work.phase != BatchPhase::Unprepared {
            return Ok(());
        }
        let cfg = self.model.config();
        let ec = self.model.engine_cfg();
        let prompt = &work.prompt[..work.prompt_end];
        let _gate = req.gate_pass();
        let prepared = crate::seam::prepare_dense_vulkan_parallel_prompt_session(
            self.vk.as_ref(),
            cfg,
            ec,
            work.kv.as_mut().expect("MTP work has target KV"),
            prompt,
            work.turn_checkpoint,
        )?;
        let heads = self.mtp_heads.as_ref().expect("MTP scheduler has heads");
        let mut heads = heads.lock().expect("MTP heads poisoned");
        let lane = &mut heads[work.slot];
        if ec.debug.state_trace {
            let recurrent = work
                .kv()
                .session_state_meta()
                .checkpoint_tokens
                .map(|tokens| tokens.map(|tokens| tokens.len()));
            let head = lane.turn_checkpoints.each_ref().map(|checkpoint| {
                checkpoint
                    .as_ref()
                    .map(|checkpoint| checkpoint.tokens.len())
            });
            tracing::warn!(
                "[MTP state trace] slot={} target={} head_live={} recurrent_checkpoints={:?} head_checkpoints={:?}",
                work.slot,
                work.kv().cached_len(),
                lane.cached.len(),
                recurrent,
                head,
            );
        }
        for checkpoint in &mut lane.turn_checkpoints {
            if checkpoint
                .as_ref()
                .is_some_and(|checkpoint| !prompt.starts_with(&checkpoint.tokens))
            {
                *checkpoint = None;
            }
        }
        let target_prefix = work.kv().cached_tokens();
        let reusable = lane.live_valid
            && lane.multimodal_key == work.multimodal_key
            && lane.cached == target_prefix
            && prompt.starts_with(&lane.cached);
        let restored_head = (!reusable && lane.multimodal_key == work.multimodal_key)
            .then(|| {
                lane.turn_checkpoints.iter().flatten().find(|checkpoint| {
                    checkpoint.tokens.as_slice() == target_prefix
                        && lane.cached.starts_with(&checkpoint.tokens)
                        && lane
                            .head
                            .retains_rewind_prefix(lane.cached.len(), checkpoint.tokens.len())
                })
            })
            .flatten()
            .map(|checkpoint| (checkpoint.tokens.clone(), checkpoint.last_h.clone()));
        if let Some((tokens, last_h)) = restored_head {
            lane.cached = tokens;
            lane.last_h = last_h;
            lane.pending_id = None;
            lane.pending_emitted = false;
            lane.target_snapshot_dirty = false;
            lane.live_valid = true;
            tracing::debug!(
                slot = work.slot,
                cached_tokens = lane.cached.len(),
                "restored MTP head turn checkpoint"
            );
        } else if !reusable && target_prefix.is_empty() {
            lane.invalidate_all();
            lane.multimodal_key = work.multimodal_key;
            lane.live_valid = true;
        } else if !reusable {
            let start = prepared.start;
            work.prefill_start = start;
            work.stats.n_cached = work.stats.n_cached.saturating_add(start);
            work.stats.n_prompt = work
                .stats
                .n_prompt
                .saturating_add(prompt.len().saturating_sub(start));
            work.checkpoint_boundaries = prepared.checkpoint_boundaries;
            work.paged_frontier = prepared.paged_frontier;
            work.phase = phase_for_remaining_prefill(work.remaining_prefill());
            work.turn_checkpoint = None;
            work.mtp_ready = false;
            lane.invalidate_live();
            tracing::info!(
                slot = work.slot,
                cached_tokens = start,
                prompt_tokens = prompt.len(),
                "MTP side state is unavailable for the reusable main KV; using ordinary decode for this turn"
            );
            return Ok(());
        }
        let start = lane.cached.len();
        anyhow::ensure!(
            start == prepared.start,
            "MTP lane {} restored {} tokens while the main KV restored {}",
            work.slot,
            start,
            prepared.start
        );
        work.prefill_start = start;
        work.stats.n_cached += start;
        work.stats.n_prompt += prompt.len() - start;
        let progress_events = work
            .channels
            .as_ref()
            .map(|channels| channels.events.clone());
        if let Some(events) = progress_events.as_ref() {
            let _ = events.send(BatchEvent::Progress {
                progress: infr_core::GenerationProgress {
                    phase: infr_core::GenerationPhase::Prefill,
                    prompt_tokens: work.prompt_end as u64,
                    cached_prompt_tokens: start as u64,
                    prefill_tokens: 0,
                    completion_tokens: 0,
                    context_tokens: start as u64,
                    context_limit: self.max_ctx as u64,
                },
            });
        }
        let t0 = Instant::now();
        let (bind, finish) = crate::seam::vulkan_moe_binder(
            self.vk.as_ref(),
            self.model.gguf(),
            cfg,
            ec,
            false,
            self.max_ctx,
        )?;
        let checkpoint_boundaries = prepared.checkpoint_boundaries;
        let initial_h = lane.last_h.clone();
        let report_prime_progress = |chunk_end: usize| {
            if let Some(events) = progress_events.as_ref() {
                let _ = events.send(BatchEvent::Progress {
                    progress: infr_core::GenerationProgress {
                        phase: infr_core::GenerationPhase::Prefill,
                        prompt_tokens: work.prompt_end as u64,
                        cached_prompt_tokens: start as u64,
                        prefill_tokens: chunk_end.saturating_sub(start) as u64,
                        completion_tokens: 0,
                        context_tokens: chunk_end as u64,
                        context_limit: self.max_ctx as u64,
                    },
                });
            }
        };
        let mut prime_sink = crate::mtp::Qwen4MtpPrimeSink::new(
            &lane.head,
            &initial_h,
            checkpoint_boundaries,
            Some(&report_prime_progress),
        )?;
        let (mut pending, mut pending_logits) = crate::mtp::run_qwen4_prime_frontier_with_finish(
            self.vk.as_ref(),
            &bind,
            self.model.gguf(),
            cfg,
            ec,
            self.model.embd(),
            prompt,
            &mut work.kv,
            self.max_ctx,
            work.mrope_plan.as_ref(),
            finish.as_deref(),
            &mut prime_sink,
            Some(prepared),
        )?;
        if work
            .reasoning_guard
            .blocks(cfg, pending, ec.sampling.ignore_eos)
        {
            anyhow::ensure!(
                pending_logits.len() == cfg.vocab,
                "Qwen3.8 MTP prime EOS repair expected {} logits, got {}",
                cfg.vocab,
                pending_logits.len()
            );
            work.reasoning_guard.mask_eos(cfg, &mut pending_logits);
            let blocked = pending;
            pending = crate::mtp::argmax_row(&pending_logits)?;
            tracing::warn!(
                slot = work.slot,
                blocked_token = blocked,
                replacement_token = pending,
                "Qwen3.8 MTP suppressed premature EOS inside an open reasoning block"
            );
        }
        let prime_capture = prime_sink.finish(self.vk.as_ref())?;
        lane.last_h = prime_capture.last_h;
        for (index, checkpoint_h) in prime_capture.checkpoint_h.into_iter().enumerate() {
            if let (Some(boundary), Some(last_h)) = (checkpoint_boundaries[index], checkpoint_h) {
                lane.turn_checkpoints[index] = Some(crate::mtp::Qwen4MtpCheckpointState {
                    tokens: prompt[..boundary].to_vec(),
                    last_h,
                });
            }
        }
        lane.pending_id = Some(pending);
        lane.pending_emitted = false;
        lane.cached.extend_from_slice(&prompt[start..]);
        lane.multimodal_key = work.multimodal_key;
        anyhow::ensure!(lane.pending_id.is_some(), "MTP prime has no frontier token");
        work.kv
            .as_mut()
            .expect("MTP work has target KV")
            .mtp_snapshot_delta(self.vk.as_ref(), cfg)?;
        if ec.debug.state_trace {
            let recurrent = work
                .kv()
                .session_state_meta()
                .checkpoint_tokens
                .map(|tokens| tokens.map(|tokens| tokens.len()));
            tracing::warn!(
                "[MTP state trace] slot={} prime_complete target={} recurrent_checkpoints={:?}",
                work.slot,
                work.kv().cached_len(),
                recurrent,
            );
        }
        lane.target_snapshot_dirty = false;
        lane.live_valid = true;
        work.stats.prompt_secs += t0.elapsed().as_secs_f64();
        work.phase = BatchPhase::Decode;
        work.turn_checkpoint = None;
        work.mtp_ready = true;
        Ok(())
    }

    fn run_mtp_batch(&self, work: Vec<BatchWork>, req: &RequestCtx) {
        let mut active = work;
        let verify_rows = mtp_verify_rows(self.model.engine_cfg().spec.k);
        let mut concurrent_plain = false;
        let mut concurrent_probe_cycles = 0usize;
        let mut concurrent_accepted = 0usize;
        let mut concurrent_drafted = 0usize;
        let mut concurrent_reported_plain = None;
        loop {
            let capacity = MAX_DECODE_BATCH.saturating_sub(active.len());
            let pending = self.take_pending_batch_work(capacity, false);
            if !pending.is_empty() && !active.is_empty() {
                concurrent_plain = false;
                concurrent_probe_cycles = 0;
                concurrent_accepted = 0;
                concurrent_drafted = 0;
                concurrent_reported_plain = None;
            }
            active.extend(pending);
            if active.is_empty() {
                match self.refill_batch() {
                    Some(work) => active = work,
                    None => return,
                }
            }
            let force_plain_concurrent =
                active.len() > 1 && !self.model.engine_cfg().spec.mtp_concurrent;
            if force_plain_concurrent {
                let ready = (0..active.len())
                    .filter(|&index| active[index].mtp_ready)
                    .collect::<Vec<_>>();
                if !ready.is_empty() {
                    if let Err(error) =
                        self.deactivate_mtp_lanes(&mut active, &ready, "multiple active lanes")
                    {
                        self.fail_unified_scheduler(&mut active, error);
                        return;
                    }
                }
            }
            for work in &mut active {
                let was_unprepared = work.phase == BatchPhase::Unprepared;
                let use_mtp = !force_plain_concurrent
                    && mtp_context_profitable(
                        self.model.engine_cfg().spec.mtp_max_context,
                        work.prompt_end,
                    );
                let prepared = if use_mtp {
                    self.prime_mtp_work(work, req)
                } else {
                    if was_unprepared {
                        let mut heads = self
                            .mtp_heads
                            .as_ref()
                            .expect("MTP scheduler has heads")
                            .lock()
                            .expect("MTP heads poisoned");
                        heads[work.slot].invalidate_live();
                        tracing::info!(
                            slot = work.slot,
                            prompt_tokens = work.prompt_end,
                            max_context = self.model.engine_cfg().spec.mtp_max_context,
                            reason = if force_plain_concurrent {
                                "multiple active lanes"
                            } else {
                                "context cutoff"
                            },
                            "Qwen3.8 MTP prime skipped"
                        );
                    }
                    self.prepare_unified_work(work, req)
                };
                if let Err(error) = prepared {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
                if was_unprepared && work.mtp_ready {
                    let progress = infr_core::GenerationProgress {
                        phase: infr_core::GenerationPhase::Prefill,
                        prompt_tokens: work.prompt_end as u64,
                        cached_prompt_tokens: work.prefill_start as u64,
                        prefill_tokens: (work.prompt_end - work.prefill_start) as u64,
                        completion_tokens: 0,
                        context_tokens: work.prompt_end as u64,
                        context_limit: self.max_ctx as u64,
                    };
                    if work.channels.as_ref().is_none_or(|channels| {
                        channels
                            .events
                            .send(BatchEvent::Progress { progress })
                            .is_err()
                    }) {
                        work.finished = true;
                    }
                }
            }
            self.retire_finished_work(&mut active);
            if active.is_empty() {
                continue;
            }
            if active
                .iter()
                .any(|work| work.phase == BatchPhase::LongPrefill)
            {
                if let Err(error) = self.run_long_prefill_phase(&mut active, req) {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
                self.retire_finished_work(&mut active);
                continue;
            }
            if active.len() > 1
                && !concurrent_mtp_ready(
                    self.model.engine_cfg().spec.mtp_concurrent,
                    concurrent_plain,
                    active.iter().all(|work| work.mtp_ready),
                )
            {
                if !concurrent_plain {
                    tracing::info!(
                        lanes = active.len(),
                        "MTP scheduler switching concurrent decode to the ordinary batched path"
                    );
                    concurrent_plain = true;
                }
                let mut mtp_indices = (0..active.len())
                    .filter(|&index| active[index].mtp_ready)
                    .collect::<Vec<_>>();
                mtp_indices
                    .sort_by_key(|&index| (active[index].mrope_plan.is_none(), active[index].slot));
                if let Err(error) = self.materialize_mtp_frontiers(&mut active, &mtp_indices) {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
                let before = active.len();
                self.retire_finished_work(&mut active);
                if active.len() < before {
                    self.vk.release_primary_runtime_after_cohort_shrink();
                }
                if active.len() < 2 {
                    continue;
                }
                let mut indices = (0..active.len()).collect::<Vec<_>>();
                sort_token_lane_indices(&active, &mut indices);
                let sync_mtp = indices.iter().any(|&index| active[index].mtp_ready);
                if let Err(error) =
                    self.run_token_group(&mut active, &indices, MTP_PLAIN_SYNC_QUANTUM, sync_mtp)
                {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
                let before = active.len();
                self.retire_finished_work(&mut active);
                if active.len() < before {
                    self.vk.release_primary_runtime_after_cohort_shrink();
                }
                continue;
            }
            if concurrent_plain {
                if active[0].mtp_ready {
                    tracing::info!(
                        "MTP scheduler returning the surviving lane to speculative decode"
                    );
                }
                concurrent_plain = false;
            }
            if active.len() == 1 {
                concurrent_probe_cycles = 0;
                concurrent_accepted = 0;
                concurrent_drafted = 0;
            }
            let context_fallback = (0..active.len())
                .filter(|&index| {
                    active[index].mtp_ready
                        && !mtp_context_profitable(
                            self.model.engine_cfg().spec.mtp_max_context,
                            active[index].kv().cached_len().saturating_add(verify_rows),
                        )
                })
                .collect::<Vec<_>>();
            if !context_fallback.is_empty() {
                if let Err(error) =
                    self.deactivate_mtp_lanes(&mut active, &context_fallback, "context cutoff")
                {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
            }
            if !active[0].mtp_ready {
                if let Err(error) =
                    self.run_token_group(&mut active, &[0], MTP_PLAIN_SYNC_QUANTUM, false)
                {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
                self.retire_finished_work(&mut active);
                continue;
            }
            let fallback = (0..active.len())
                .filter(|&index| active[index].kv().cached_len() + verify_rows > self.max_ctx)
                .collect::<Vec<_>>();
            if !fallback.is_empty() {
                if let Err(error) = self.materialize_mtp_frontiers(&mut active, &fallback) {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
                self.retire_finished_work(&mut active);
                if active.is_empty() {
                    continue;
                }
                let fallback = (0..active.len()).collect::<Vec<_>>();
                if let Err(error) = self.run_token_group(&mut active, &fallback, 1, true) {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
                self.retire_finished_work(&mut active);
                continue;
            }
            let mut indices = (0..active.len()).collect::<Vec<_>>();
            indices.sort_by_key(|&index| (active[index].mrope_plan.is_none(), active[index].slot));
            let cycle = match self.run_mtp_verify_cycle(&mut active, &indices, req) {
                Ok(cycle) => cycle,
                Err(error) => {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
            };
            if active.len() > 1 && self.model.engine_cfg().spec.mtp_concurrent {
                concurrent_probe_cycles += 1;
                concurrent_accepted += cycle.accepted;
                concurrent_drafted += cycle.drafted;
                if concurrent_probe_cycles >= MTP_CONCURRENT_PROBE_CYCLES {
                    let alpha = concurrent_accepted as f64 / concurrent_drafted.max(1) as f64;
                    let next_plain =
                        !concurrent_mtp_profitable(concurrent_accepted, concurrent_drafted);
                    if concurrent_reported_plain != Some(next_plain) {
                        tracing::info!(
                            cycles = concurrent_probe_cycles,
                            accepted = concurrent_accepted,
                            drafted = concurrent_drafted,
                            alpha,
                            mode = if next_plain { "plain" } else { "mtp" },
                            "Qwen3.8 concurrent MTP acceptance window selected decode mode"
                        );
                        concurrent_reported_plain = Some(next_plain);
                    } else {
                        tracing::debug!(
                            cycles = concurrent_probe_cycles,
                            accepted = concurrent_accepted,
                            drafted = concurrent_drafted,
                            alpha,
                            mode = if next_plain { "plain" } else { "mtp" },
                            "Qwen3.8 concurrent MTP acceptance window retained decode mode"
                        );
                    }
                    concurrent_plain = next_plain;
                    concurrent_probe_cycles = 0;
                    concurrent_accepted = 0;
                    concurrent_drafted = 0;
                }
            }
            let before = active.len();
            self.retire_finished_work(&mut active);
            if active.len() < before {
                self.vk.release_primary_runtime_after_cohort_shrink();
            }
        }
    }

    fn run_mtp_verify_cycle(
        &self,
        active: &mut [BatchWork],
        indices: &[usize],
        req: &RequestCtx,
    ) -> Result<MtpBatchCycleStats> {
        let cfg = self.model.config();
        let ec = self.model.engine_cfg();
        let h_width = cfg.hc_mult * cfg.n_embd;
        let rows = mtp_verify_rows(ec.spec.k);
        let timeline_positions = indices
            .iter()
            .map(|&index| active[index].kv().cached_len())
            .collect::<Vec<_>>();
        let _timeline_cycle = infr_core::timeline::begin_step(&timeline_positions);
        let snapshot_t0 = Instant::now();
        let timeline_snapshot = infr_core::timeline::span("mtp_snapshot_refresh");
        self.refresh_mtp_snapshots(active, indices)?;
        drop(timeline_snapshot);
        let snapshot_secs = snapshot_t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let timeline_draft = infr_core::timeline::span("mtp_draft");
        let (feeds, predictions) = {
            let timeline_gate_wait = infr_core::timeline::span("mtp_draft_gate_wait");
            let _gate = req.gate_pass();
            drop(timeline_gate_wait);
            let mut heads = self
                .mtp_heads
                .as_ref()
                .expect("MTP scheduler has heads")
                .lock()
                .expect("MTP heads poisoned");
            let mut feeds = Vec::with_capacity(indices.len());
            let mut predictions = Vec::with_capacity(indices.len());
            for &index in indices {
                let work = &mut active[index];
                let lane = &mut heads[work.slot];
                let pending = lane.pending_id.ok_or_else(|| {
                    anyhow!("MTP lane {} entered VERIFY without a frontier", work.slot)
                })?;
                let physical_position = work.kv().cached_len();
                let rope_position =
                    multimodal_token_position(work.mrope_plan.as_ref(), physical_position)?;
                let candidates = lane.head.draft_with_tail_mode(
                    self.vk.as_ref(),
                    pending,
                    &lane.last_h,
                    physical_position,
                    rope_position,
                    rows,
                    work.kv().mtp_shared_weights(),
                    ec.spec.mtp_tail_kv_only,
                )?;
                let mut feed = work.kv().cached_tokens().to_vec();
                feed.push(pending);
                feed.extend_from_slice(&candidates);
                work.kv
                    .as_mut()
                    .expect("MTP work has target KV")
                    .mtp_arm_delta_trace(rows)?;
                feeds.push(feed);
                predictions.push(candidates);
            }
            (feeds, predictions)
        };
        drop(timeline_draft);
        let draft_secs = t0.elapsed().as_secs_f64();

        let timeline_verify = infr_core::timeline::span("mtp_verify");
        let (mut verify_ids, mut verify_logits, verify_hidden) = {
            let timeline_gate_wait = infr_core::timeline::span("mtp_verify_gate_wait");
            let _gate = req.gate_pass();
            drop(timeline_gate_wait);
            let mut owned_mrope_plans = indices
                .iter()
                .map(|&index| active[index].mrope_plan.take())
                .collect::<Vec<_>>();
            let mrope_plans = owned_mrope_plans
                .iter()
                .map(Option::as_ref)
                .collect::<Vec<_>>();
            let mut primary = active[indices[0]].kv.take();
            let mut peers = indices[1..]
                .iter()
                .map(|&index| active[index].kv.take().expect("MTP peer has target KV"))
                .collect::<Vec<_>>();
            let result = crate::seam::generate_dense_vulkan_parallel_mtp_verify_session(
                self.vk.as_ref(),
                self.model.gguf(),
                cfg,
                ec,
                self.model.embd(),
                &feeds,
                &mrope_plans,
                &mut primary,
                &mut peers,
                self.max_ctx,
            );
            active[indices[0]].kv = primary;
            for (&index, kv) in indices[1..].iter().zip(peers) {
                active[index].kv = Some(kv);
            }
            drop(mrope_plans);
            for (&index, plan) in indices.iter().zip(owned_mrope_plans.drain(..)) {
                active[index].mrope_plan = plan;
            }
            result?
        };
        drop(timeline_verify);
        anyhow::ensure!(
            verify_ids.len() == indices.len()
                && verify_logits.len() == indices.len()
                && verify_hidden.len() == indices.len(),
            "parallel MTP VERIFY returned inconsistent lane counts"
        );
        let compute_secs = t0.elapsed().as_secs_f64();

        let mut cycle = MtpBatchCycleStats::default();
        for (lane_index, &index) in indices.iter().enumerate() {
            let timeline_accept = infr_core::timeline::span("mtp_accept_emit");
            let work = &mut active[index];
            let first_cycle = work.generated == 0;
            let pending_emitted = self
                .mtp_heads
                .as_ref()
                .expect("MTP scheduler has heads")
                .lock()
                .expect("MTP heads poisoned")[work.slot]
                .pending_emitted;
            let old_len = feeds[lane_index].len() - rows;
            let ids = &mut verify_ids[lane_index];
            let logits = &mut verify_logits[lane_index];
            let hidden = &verify_hidden[lane_index];
            anyhow::ensure!(
                ids.len() == rows && hidden.len() == rows * h_width,
                "parallel MTP VERIFY lane {lane_index} returned incomplete rows"
            );
            let mut verify_guard = work.reasoning_guard;
            for row in 0..rows {
                verify_guard.observe(cfg, feeds[lane_index][old_len + row]);
                if !verify_guard.blocks(cfg, ids[row], ec.sampling.ignore_eos) {
                    continue;
                }
                anyhow::ensure!(
                    logits.len() == rows * cfg.vocab,
                    "parallel MTP EOS repair lane {lane_index} expected {} logits, got {}",
                    rows * cfg.vocab,
                    logits.len()
                );
                let row_logits = &mut logits[row * cfg.vocab..(row + 1) * cfg.vocab];
                verify_guard.mask_eos(cfg, row_logits);
                let blocked = ids[row];
                ids[row] = crate::mtp::argmax_row(row_logits)?;
                tracing::warn!(
                    slot = work.slot,
                    row,
                    blocked_token = blocked,
                    replacement_token = ids[row],
                    "Qwen3.8 MTP suppressed premature EOS inside an open reasoning block"
                );
            }
            let accepted = mtp_accepted_rows(&predictions[lane_index], ids);
            cycle.accepted += accepted.saturating_sub(1);
            cycle.drafted += predictions[lane_index].len();
            let already_emitted = usize::from(pending_emitted);
            anyhow::ensure!(
                already_emitted <= accepted
                    && (!pending_emitted
                        || (work.prompt.last() == Some(&feeds[lane_index][old_len]))),
                "MTP lane {} lost its emitted frontier before VERIFY",
                work.slot
            );
            let available = already_emitted
                + (accepted - already_emitted).min(work.max_new.saturating_sub(work.generated));
            let mut committed = already_emitted;
            let mut stopped = false;
            for &token in &feeds[lane_index][old_len + already_emitted..old_len + available] {
                let eos =
                    !ec.sampling.ignore_eos && (cfg.eos_ids.contains(&token) || token == cfg.eos);
                if eos {
                    committed += 1;
                    stopped = true;
                    break;
                }
                let progress = self.batch_decode_progress(
                    work.prompt_end,
                    work.prefill_start,
                    work.generated + committed - already_emitted + 1,
                );
                let Some(channels) = work.channels.as_ref() else {
                    stopped = true;
                    break;
                };
                if channels
                    .events
                    .send(BatchEvent::Token {
                        id: token,
                        progress,
                    })
                    .is_err()
                {
                    stopped = true;
                    break;
                }
                committed += 1;
                if !channels.acknowledgements.recv().unwrap_or(false) {
                    stopped = true;
                    break;
                }
            }
            drop(timeline_accept);
            let restore_t0 = Instant::now();
            let timeline_restore = infr_core::timeline::span("mtp_restore");
            {
                let timeline_gate_wait = infr_core::timeline::span("mtp_restore_gate_wait");
                let _gate = req.gate_pass();
                drop(timeline_gate_wait);
                let kv = work.kv.as_mut().expect("MTP lane has target KV");
                if committed < rows {
                    kv.mtp_restore_delta_row(self.vk.as_ref(), committed)?;
                }
                kv.mtp_snapshot_delta(self.vk.as_ref(), cfg)?;
                let mut heads = self
                    .mtp_heads
                    .as_ref()
                    .expect("MTP scheduler has heads")
                    .lock()
                    .expect("MTP heads poisoned");
                let lane = &mut heads[work.slot];
                lane.head.settle_draft(self.vk.as_ref(), committed)?;
                lane.cached = kv.cached_tokens().to_vec();
                lane.live_valid = true;
                if committed > 0 {
                    lane.pending_id = Some(ids[committed - 1]);
                    lane.pending_emitted = false;
                    lane.target_snapshot_dirty = false;
                    lane.last_h
                        .copy_from_slice(&hidden[(committed - 1) * h_width..committed * h_width]);
                }
            }
            drop(timeline_restore);
            work.prompt.extend_from_slice(
                &feeds[lane_index][old_len + already_emitted..old_len + committed],
            );
            for &token in &feeds[lane_index][old_len + already_emitted..old_len + committed] {
                work.reasoning_guard.observe(cfg, token);
            }
            let emitted = committed - already_emitted;
            work.generated += emitted;
            work.stats.n_gen += emitted;
            let restore_secs = restore_t0.elapsed().as_secs_f64();
            work.stats.decode_secs += compute_secs + restore_secs;
            work.finished = stopped || work.generated >= work.max_new;
            if first_cycle {
                tracing::info!(
                    slot = work.slot,
                    context = old_len,
                    rows,
                    accepted = accepted - 1,
                    published = emitted,
                    tail_kv_only = ec.spec.mtp_tail_kv_only,
                    "Qwen3.8 MTP first Verify cycle"
                );
            }
            tracing::debug!(
                slot = work.slot,
                lanes = indices.len(),
                context = old_len,
                rows,
                drafted = predictions[lane_index].len(),
                accepted = accepted - 1,
                committed,
                published = emitted,
                snapshot_ms = snapshot_secs * 1000.0,
                draft_ms = draft_secs * 1000.0,
                verify_ms = (compute_secs - draft_secs).max(0.0) * 1000.0,
                restore_ms = restore_secs * 1000.0,
                "Qwen3.8 MTP verified cycle"
            );
        }
        Ok(cycle)
    }

    fn publish_lookup_token(&self, work: &mut BatchWork, token: u32) -> LookupTokenDelivery {
        let cfg = self.model.config();
        let eos = !self.model.engine_cfg().sampling.ignore_eos
            && (token == cfg.eos || cfg.eos_ids.contains(&token));
        let mut keep_going = !eos;
        if !eos {
            let timeline_output = infr_core::timeline::span("decode_output_progress");
            let progress =
                self.batch_decode_progress(work.prompt_end, work.prefill_start, work.generated + 1);
            drop(timeline_output);
            let Some(channels) = work.channels.as_ref() else {
                work.finished = true;
                return LookupTokenDelivery::Disconnected;
            };
            let timeline_send = infr_core::timeline::span("decode_output_send");
            let sent = channels
                .events
                .send(BatchEvent::Token {
                    id: token,
                    progress,
                })
                .is_ok();
            drop(timeline_send);
            if !sent {
                work.finished = true;
                return LookupTokenDelivery::Disconnected;
            }
            let _timeline_ack = infr_core::timeline::span("decode_output_ack_wait");
            keep_going = channels.acknowledgements.recv().unwrap_or(false);
        }
        // A successful send is delivered even when its ACK requests cancellation. EOS is
        // counted once without a text event, exactly as in the ordinary token runner.
        work.prompt.push(token);
        work.reasoning_guard.observe(cfg, token);
        work.sampler
            .as_mut()
            .expect("Lookup work owns a sampler")
            .observe(token);
        work.generated += 1;
        work.stats.n_gen += 1;
        work.lookup
            .as_mut()
            .expect("Lookup work owns an index")
            .append_confirmed(&[token]);
        work.finished = !keep_going || work.generated >= work.max_new;
        if work.finished {
            LookupTokenDelivery::Stop
        } else {
            LookupTokenDelivery::Continue
        }
    }

    fn try_lookup_cycle(&self, work: &mut BatchWork, req: &RequestCtx) -> Result<bool> {
        if work
            .lookup
            .as_ref()
            .is_none_or(|lookup| lookup.ordinary_token_secs.is_none())
            || self.mtp_heads.is_some()
            || work.finished
            || work.phase != BatchPhase::Decode
            || work.generated == 0
            || work.max_new.saturating_sub(work.generated) < 2
        {
            return Ok(false);
        }
        let result = self.run_lookup_cycle(work, req);
        if result.is_err() {
            // Any error after arming/Verify/restore may leave recurrent state ahead of
            // published tokens. Never return that slot as a reusable apparent prefix.
            if let Some(kv) = work.kv.as_mut() {
                kv.reset();
            }
            if let Some(lookup) = work.lookup.as_mut() {
                lookup.pending_proposal = None;
                lookup.proposer.clear();
                lookup.indexed_len = 0;
            }
        }
        result
    }

    fn run_lookup_cycle(&self, work: &mut BatchWork, req: &RequestCtx) -> Result<bool> {
        let cfg = self.model.config();
        let ec = self.model.engine_cfg();
        let base = work.kv().cached_len();
        anyhow::ensure!(
            work.prompt.len() == base + 1,
            "Lookup requires exactly one previously emitted pending token"
        );
        let budget = lookup_draft_budget(
            ec.spec.lookup_max_draft,
            work.max_new.saturating_sub(work.generated),
            work.prompt.len(),
            self.max_ctx,
            self.qsa_threshold(),
        );
        if budget == 0 {
            return Ok(false);
        }
        let cycle_t0 = Instant::now();
        let lookup = work.lookup.as_mut().expect("Lookup work owns an index");
        if lookup.cost_cooldown > 0 {
            return Ok(false);
        }
        if lookup.indexed_len != work.prompt.len() {
            // Known ordinary/speculative appends normally maintain this exactly. A caller
            // changing the frontier must synchronize the bounded retained window instead.
            lookup.sync_history(&work.prompt);
        }
        // The emitted pending is already in this confirmed logical history. Passing None
        // includes it exactly once; it has not yet been materialized in the target KV.
        let (candidates, cached_search_secs, search_secs) =
            if let Some((candidates, seconds)) = lookup.take_pending_proposal(budget) {
                (Some(candidates), seconds, 0.0)
            } else {
                let (candidates, seconds) = lookup.search(budget);
                (candidates, 0.0, seconds)
            };
        let Some(candidates) = candidates else {
            work.stats.decode_secs += search_secs;
            return Ok(false);
        };
        let rows = 1 + candidates.tokens.len();
        anyhow::ensure!(
            (2..=crate::speculative_verify::MAX_VERIFY_ROWS).contains(&rows),
            "Lookup proposer returned invalid Verify width"
        );
        let _timeline_cycle = infr_core::timeline::begin_step(&[base]);
        let mut feed = work.kv().cached_tokens().to_vec();
        feed.push(work.prompt[base]);
        feed.extend_from_slice(&candidates.tokens);
        let snapshot_t0 = Instant::now();
        {
            // Keep the existing timing series; Lookup now arms traces without a base copy.
            let _timeline_snapshot = infr_core::timeline::span("lookup_snapshot");
            let _gate = req.gate_pass();
            let kv = work.kv.as_mut().expect("Lookup owns target KV");
            kv.lookup_arm_delta_trace(base, rows)?;
        }
        let snapshot_secs = snapshot_t0.elapsed().as_secs_f64();
        let verify_t0 = Instant::now();
        let (mut ids, mut logits, _hidden) = {
            let _timeline_verify = infr_core::timeline::span("lookup_verify");
            let _gate = req.gate_pass();
            crate::seam::generate_dense_vulkan_parallel_verify_session(
                self.vk.as_ref(),
                self.model.gguf(),
                cfg,
                ec,
                self.model.embd(),
                std::slice::from_ref(&feed),
                &[work.mrope_plan.as_ref()],
                &mut work.kv,
                &mut [],
                self.max_ctx,
                false,
            )?
        };
        let verify_secs = verify_t0.elapsed().as_secs_f64();
        anyhow::ensure!(
            ids.len() == 1 && logits.len() == 1 && ids[0].len() == rows,
            "Lookup Verify returned inconsistent rows"
        );
        let ids = &mut ids[0];
        let logits = &mut logits[0];
        repair_lookup_predictions(
            cfg,
            work.reasoning_guard,
            &candidates.tokens,
            ids,
            logits,
            ec.sampling.ignore_eos,
        )?;
        anyhow::ensure!(
            work.kv().cached_len() == base + rows,
            "Lookup Verify did not materialize its input rows"
        );
        let delivery_t0 = Instant::now();
        let delivery = deliver_lookup_prefix(
            &candidates.tokens,
            ids,
            work.max_new.saturating_sub(work.generated),
            |token| {
                if crate::sampling::abort_requested(Some(req)) {
                    work.finished = true;
                    LookupTokenDelivery::Disconnected
                } else {
                    self.publish_lookup_token(work, token)
                }
            },
        )?;
        let delivery_secs = delivery_t0.elapsed().as_secs_f64();
        let restore_t0 = Instant::now();
        {
            let _timeline_restore = infr_core::timeline::span("lookup_restore");
            let _gate = req.gate_pass();
            let kv = work.kv.as_mut().expect("Lookup owns target KV");
            kv.lookup_settle_delta_trace(self.vk.as_ref(), delivery.keep_inputs)?;
            anyhow::ensure!(
                kv.cached_len() == base + delivery.keep_inputs,
                "Lookup row restore lost the retained input prefix"
            );
        }
        let restore_secs = restore_t0.elapsed().as_secs_f64();
        let cycle_secs = (cycle_t0.elapsed().as_secs_f64() - delivery_secs).max(0.0);
        work.stats.decode_secs += cycle_secs;
        work.finished |= delivery.stopped;
        anyhow::ensure!(
            work.finished || work.prompt.len() == work.kv().cached_len() + 1,
            "Lookup must leave its bonus as the next unmaterialized pending token"
        );
        let lookup = work.lookup.as_mut().expect("Lookup work owns an index");
        lookup.stats.snapshot_secs += snapshot_secs;
        lookup.stats.verify_secs += verify_secs;
        lookup.stats.restore_secs += restore_secs;
        lookup.record_cycle(
            delivery.accepted_drafts,
            candidates.tokens.len(),
            delivery.new_tokens,
            cycle_secs + cached_search_secs,
        );
        if lookup.stats.cycles == 1 {
            tracing::info!(
                slot = work.slot,
                rows,
                matched = candidates.matched_tokens,
                accepted = delivery.accepted_drafts,
                kept_inputs = delivery.keep_inputs,
                delivered = delivery.new_tokens,
                verify_ms = verify_secs * 1000.0,
                restore_ms = restore_secs * 1000.0,
                cycle_ms = cycle_secs * 1000.0,
                trace_only = true,
                "independent Lookup first Verify cycle"
            );
        }
        tracing::debug!(
            slot = work.slot,
            rows,
            matched = candidates.matched_tokens,
            accepted = delivery.accepted_drafts,
            kept_inputs = delivery.keep_inputs,
            delivered = delivery.new_tokens,
            search_ms = search_secs * 1000.0,
            verify_ms = verify_secs * 1000.0,
            restore_ms = restore_secs * 1000.0,
            cycle_ms = cycle_secs * 1000.0,
            trace_only = true,
            "independent Lookup Verify cycle"
        );
        Ok(true)
    }

    fn run_unified_batch(&self, work: Vec<BatchWork>, req: &RequestCtx) {
        let mut active = Vec::with_capacity(MAX_DECODE_BATCH);
        active.extend(work);
        let mut long_prefill_coalesced = false;

        loop {
            let capacity = MAX_DECODE_BATCH.saturating_sub(active.len());
            active.extend(self.take_pending_batch_work(capacity, false));
            if active.is_empty() {
                match self.refill_batch() {
                    Some(work) => active = work,
                    None => return,
                }
            }

            for work in &mut active {
                if let Err(error) = self.prepare_unified_work(work, req) {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
            }

            let mode = scheduler_mode(active.iter().map(|work| work.phase))
                .expect("the active scheduler cohort is non-empty");
            if mode == SchedulerMode::LongPrefill {
                if !long_prefill_coalesced {
                    let capacity = MAX_DECODE_BATCH.saturating_sub(active.len());
                    let peers = self.take_prefill_cohort_peers(capacity);
                    long_prefill_coalesced = true;
                    if !peers.is_empty() {
                        tracing::debug!(
                            lanes = peers.len(),
                            "coalescing work before the exclusive long-prefill phase"
                        );
                        active.extend(peers);
                        continue;
                    }
                }
                if let Err(error) = self.run_long_prefill_phase(&mut active, req) {
                    self.fail_unified_scheduler(&mut active, error);
                    return;
                }
                let cohort_len = active.len();
                self.retire_finished_work(&mut active);
                if active.len() < cohort_len {
                    self.vk.release_primary_runtime_after_cohort_shrink();
                }
                long_prefill_coalesced = false;
                continue;
            }
            long_prefill_coalesced = false;

            if active.len() == 1 && active[0].lookup.is_some() {
                match self.try_lookup_cycle(&mut active[0], req) {
                    Ok(true) => {
                        self.retire_finished_work(&mut active);
                        if active.is_empty() {
                            self.vk.release_primary_runtime_after_cohort_shrink();
                        }
                        continue;
                    }
                    Ok(false) => {}
                    Err(error) => {
                        self.fail_unified_scheduler(&mut active, error);
                        return;
                    }
                }
            }

            // QSA attention remains lane-specific, but the surrounding projections and MoE are
            // shared across dense- and sparse-QSA rows. Keep the whole token cohort together;
            // `run_token_group` still stops exactly at the next dense-to-sparse boundary so the
            // following graph is rebuilt with the correct per-lane QSA geometry.
            let mut indices = (0..active.len()).collect::<Vec<_>>();
            // The primary lane decides whether the shared graph carries MRoPE history.
            sort_token_lane_indices(&active, &mut indices);
            // After calibration, only proposals/admission/QSA boundaries end ordinary work.
            let quantum = lookup_plain_quantum(
                active.len(),
                active[0].generated,
                active[0].remaining_prefill(),
                active[0].lookup.as_ref(),
            );
            if let Err(error) = self.run_token_group(&mut active, &indices, quantum, false) {
                self.fail_unified_scheduler(&mut active, error);
                return;
            }
            let cohort_len = active.len();
            self.retire_finished_work(&mut active);
            if active.len() < cohort_len {
                // Paged Decode deliberately retains graph scratch across tokens. A wider cohort
                // raises that high-water mark; without this request-boundary reset, its retired
                // expert slots remain unavailable after lanes leave and the surviving request
                // stays permanently in the wider batch's slow cache state.
                self.vk.release_primary_runtime_after_cohort_shrink();
            }
        }
    }

    pub fn render_chat_messages(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.model.render_chat_messages(messages)
    }

    /// Generate one sequence: check out a slot, enqueue its work, stream scheduler events, and
    /// return the slot as soon as this request completes.
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

    /// Generate one turn through the authoritative Qwen3.8 scheduler. Eligible requests register
    /// before tokenization, then stay owned by the dedicated worker from KV reconciliation through
    /// their final decode token; they never fall back to the legacy per-request StepGate path.
    pub fn generate_turn(
        &self,
        prompt: &str,
        stable_prefix: Option<&str>,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: &RequestCtx,
        on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_turn_with_checkpoints(
            prompt,
            crate::seam::TurnCheckpointPrefixes::edit(stable_prefix),
            max_new,
            constraint,
            req,
            on_piece,
        )
    }

    pub fn generate_turn_with_checkpoints(
        &self,
        prompt: &str,
        checkpoint_prefixes: crate::seam::TurnCheckpointPrefixes<'_>,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: &RequestCtx,
        on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_turn_inner(
            prompt,
            checkpoint_prefixes,
            max_new,
            constraint,
            req,
            on_piece,
            true,
        )
    }

    /// Benchmark token sampling skips text decoding; scheduling, KV and sampling stay identical.
    pub fn generate_benchmark_turn(
        &self,
        prompt: &str,
        max_new: usize,
        req: &RequestCtx,
    ) -> Result<GenStats> {
        self.generate_turn_inner(
            prompt,
            crate::seam::TurnCheckpointPrefixes::benchmark_frontier(),
            max_new,
            None,
            req,
            |_| {},
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_turn_inner(
        &self,
        prompt: &str,
        checkpoint_prefixes: crate::seam::TurnCheckpointPrefixes<'_>,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: &RequestCtx,
        mut on_piece: impl FnMut(&str),
        stream_text: bool,
    ) -> Result<GenStats> {
        let mtp_eligible = self.mtp_heads.as_ref().is_none_or(|_| {
            let base = crate::sampling::Sampler::from_cfg(&self.model.engine_cfg().sampling);
            let effective =
                crate::sampling::Sampler::resolve(Some(req), &self.model.engine_cfg().sampling);
            (base.temp <= 0.0 || base.top_k == 1)
                && (effective.temp <= 0.0 || effective.top_k == 1)
                && !req.sampling().penalties_active()
        });
        let initially_eligible = self.model.config().supports_parallel_token_rows()
            && constraint.is_none()
            && self.scheduler_admission_eligible()
            && mtp_eligible
            && max_new > 0;
        let mut registration = initially_eligible.then(|| BatchRegistration::new(self));
        let prompt_tokens = self.model.encode(prompt)?;
        let turn_checkpoint = self
            .model
            .turn_checkpoints(&prompt_tokens, checkpoint_prefixes)?;
        let max_new = max_new.min(self.max_ctx.saturating_sub(prompt_tokens.len() + 1));
        let batch_candidate = initially_eligible && max_new > 0;
        if !batch_candidate {
            drop(registration.take());
        }

        let mut guard = self.checkout(&prompt_tokens, req)?;
        let mut acc = Vec::new();
        let mut printed = 0usize;
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        if !batch_candidate {
            let reason = if constraint.is_some() {
                "grammar_constraint"
            } else if !self.model.config().supports_parallel_token_rows() {
                "scheduler_architecture_unsupported"
            } else if self.gate.is_none() {
                "scheduler_unavailable"
            } else if !mtp_eligible {
                "mtp_request_ineligible"
            } else {
                "output_budget_zero"
            };
            self.report_lookup_policy(req, guard.idx, "legacy_text", Some(reason));
            if let Some(heads) = &self.mtp_heads {
                let mut heads = heads.lock().expect("MTP heads poisoned");
                heads[guard.idx].invalidate_live();
            }
            let result = crate::seam::generate_dense_vulkan_session(
                self.vk.as_ref(),
                self.model.gguf(),
                self.model.config(),
                self.model.engine_cfg(),
                self.model.embd(),
                self.model.per_layer_embd(),
                &prompt_tokens,
                max_new,
                |id| {
                    if stream_text {
                        crate::stream_token(
                            self.model.tokenizer(),
                            &mut acc,
                            &mut printed,
                            id,
                            &mut on_piece,
                        );
                    }
                },
                &mut guard.kv,
                self.max_ctx,
                turn_checkpoint,
                constraint,
                Some(req),
                None,
            );
            if result
                .as_ref()
                .err()
                .is_some_and(crate::sampling::is_numerical_corruption)
            {
                if let Some(kv) = guard.kv.as_mut() {
                    kv.reset();
                }
            }
            let (_, stats) = result?;
            return Ok(stats);
        }

        let sampler = ParallelSampler::new(req, &self.model.engine_cfg().sampling);
        let mut queue = self
            .decode_batch
            .lock()
            .expect("decode batch queue poisoned");
        registration
            .as_mut()
            .expect("eligible request registered before tokenization")
            .arrive(&mut queue);
        drop(registration.take());
        let (event_tx, event_rx) = mpsc::sync_channel(0);
        let (ack_tx, ack_rx) = mpsc::sync_channel(0);
        queue.waiting.push_back(BatchWork {
            slot: guard.idx,
            kv: Some(guard.detach()),
            reasoning_guard: crate::sampling::Qwen4ReasoningEosGuard::from_prompt(
                self.model.config(),
                &prompt_tokens,
            ),
            prompt_end: prompt_tokens.len(),
            lookup: self.new_lookup_work(req, &prompt_tokens, guard.idx, "scheduled_text"),
            prompt: prompt_tokens,
            max_new,
            generated: 0,
            stats: GenStats::default(),
            phase: BatchPhase::Unprepared,
            prefill_start: 0,
            checkpoint_boundaries: [None; crate::seam::TURN_CHECKPOINT_COUNT],
            paged_frontier: None,
            turn_checkpoint,
            mrope_plan: None,
            multimodal_key: None,
            finished: false,
            sampling: req.sampling().clone(),
            sampler: Some(sampler),
            mtp_ready: false,
            channels: Some(BatchChannels {
                events: event_tx,
                acknowledgements: ack_rx,
            }),
        });
        self.batch_interrupt.store(true, Ordering::Release);
        self.decode_ready.notify_all();
        drop(queue);
        self.wait_for_decode_batch(
            &mut guard,
            event_rx,
            ack_tx,
            req,
            &mut acc,
            &mut printed,
            &mut on_piece,
            stream_text,
        )
    }

    /// Generate one Qwen hybrid vision turn. Projector execution is completed by the caller before
    /// entering here, so its request-scoped weights have already returned to the unified arena.
    pub fn generate_multimodal_turn(
        &self,
        prompt: &str,
        stable_prefix: Option<&str>,
        images: Vec<MultimodalEmbedding>,
        max_new: usize,
        req: &RequestCtx,
        mut on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        if !(self.model.config().qwen4exp || self.model.config().qwen35) {
            return Err(anyhow!(
                "vision text integration currently supports qwen4exp and qwen35 models only"
            ));
        }
        if images.is_empty() {
            return self.generate_turn(prompt, stable_prefix, max_new, None, req, on_piece);
        }
        let image_pad_id = self
            .model
            .tokenizer()
            .token_to_id("<|image_pad|>")
            .ok_or_else(|| anyhow!("model tokenizer has no <|image_pad|> token"))?;
        let base_tokens = self.model.encode(prompt)?;
        let image_count = images.len();
        let key = multimodal_key(&images);
        let checkpoint_images = stable_prefix.map(|_| images.clone());
        let (prompt_tokens, plan) = expand_multimodal_prompt(
            &base_tokens,
            image_pad_id,
            images,
            self.model.config().n_embd,
        )?;
        if prompt_tokens.len().saturating_add(1) > self.max_ctx {
            return Err(anyhow!(
                "multimodal prompt expands to {} tokens, exceeding this slot's {}-token context",
                prompt_tokens.len(),
                self.max_ctx
            ));
        }
        let turn_checkpoint = stable_prefix.map(|stable| {
            let boundary = self
                .model
                .encode(stable)
                .ok()
                .and_then(|tokens| {
                    expand_multimodal_prompt(
                        &tokens,
                        image_pad_id,
                        checkpoint_images
                            .clone()
                            .expect("stable prefix retained checkpoint images"),
                        self.model.config().n_embd,
                    )
                    .ok()
                    .map(|(tokens, _)| tokens)
                })
                .filter(|tokens| {
                    !tokens.is_empty()
                        && tokens.len() < prompt_tokens.len()
                        && prompt_tokens.starts_with(tokens)
                })
                .map(|tokens| tokens.len());
            crate::seam::TurnCheckpoint::new(None, boundary)
        });
        let mut guard = self.checkout_multimodal(&prompt_tokens, key, req);
        let max_new = max_new.min(self.max_ctx.saturating_sub(prompt_tokens.len() + 1));
        let mut acc = Vec::new();
        let mut printed = 0usize;
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        tracing::info!(
            images = image_count,
            base_tokens = base_tokens.len(),
            expanded_tokens = prompt_tokens.len(),
            decode_base = plan.decode_base,
            "multimodal prompt prepared"
        );
        let mtp_eligible = self.mtp_heads.as_ref().is_some_and(|_| {
            let base = crate::sampling::Sampler::from_cfg(&self.model.engine_cfg().sampling);
            let effective =
                crate::sampling::Sampler::resolve(Some(req), &self.model.engine_cfg().sampling);
            self.gate.is_some()
                && max_new > 0
                && (base.temp <= 0.0 || base.top_k == 1)
                && (effective.temp <= 0.0 || effective.top_k == 1)
                && !req.sampling().penalties_active()
        });
        if mtp_eligible {
            let sampler = ParallelSampler::new(req, &self.model.engine_cfg().sampling);
            let mut queue = self
                .decode_batch
                .lock()
                .expect("decode batch queue poisoned");
            let (event_tx, event_rx) = mpsc::sync_channel(0);
            let (ack_tx, ack_rx) = mpsc::sync_channel(0);
            queue.waiting.push_back(BatchWork {
                slot: guard.idx,
                kv: Some(guard.detach()),
                reasoning_guard: crate::sampling::Qwen4ReasoningEosGuard::from_prompt(
                    self.model.config(),
                    &prompt_tokens,
                ),
                lookup: self.new_lookup_work(
                    req,
                    &prompt_tokens,
                    guard.idx,
                    "scheduled_mtp_multimodal",
                ),
                prompt: prompt_tokens,
                prompt_end: plan.prompt_pos4.len() / 4,
                max_new,
                generated: 0,
                stats: GenStats::default(),
                phase: BatchPhase::Unprepared,
                prefill_start: 0,
                checkpoint_boundaries: [None; crate::seam::TURN_CHECKPOINT_COUNT],
                paged_frontier: None,
                turn_checkpoint,
                mrope_plan: Some(plan),
                multimodal_key: Some(key),
                finished: false,
                sampling: req.sampling().clone(),
                sampler: Some(sampler),
                mtp_ready: false,
                channels: Some(BatchChannels {
                    events: event_tx,
                    acknowledgements: ack_rx,
                }),
            });
            self.batch_interrupt.store(true, Ordering::Release);
            self.decode_ready.notify_all();
            drop(queue);
            return self.wait_for_decode_batch(
                &mut guard,
                event_rx,
                ack_tx,
                req,
                &mut acc,
                &mut printed,
                &mut on_piece,
                true,
            );
        }
        if let Some(heads) = &self.mtp_heads {
            let mut heads = heads.lock().expect("MTP heads poisoned");
            heads[guard.idx].invalidate_live();
        }
        let frontier = prompt_tokens.len().saturating_sub(1);
        let frontier_is_image = plan.spans.iter().any(|span| {
            frontier >= span.start && frontier < span.start.saturating_add(span.n_tokens)
        });
        // A one-slot non-MTP engine has no scheduler worker, so its visual turn must stay on the
        // legacy runner. Multi-slot serving retains the handoff that batches visual and text decode.
        let handoff = should_handoff_multimodal_to_scheduler(
            self.scheduler_admission_eligible(),
            self.mtp_heads.is_some(),
            frontier_is_image,
            max_new,
        );
        let legacy_max_new = if handoff { 0 } else { max_new };
        if !handoff {
            let reason = if self.mtp_heads.is_some() {
                "mtp_request_ineligible"
            } else if frontier_is_image {
                "image_frontier_requires_legacy_decode"
            } else if max_new == 0 {
                "output_budget_zero"
            } else {
                "scheduler_unavailable"
            };
            self.report_lookup_policy(req, guard.idx, "legacy_multimodal", Some(reason));
        }
        let result = crate::seam::generate_dense_vulkan_session(
            &self.vk,
            self.model.gguf(),
            self.model.config(),
            self.model.engine_cfg(),
            self.model.embd(),
            self.model.per_layer_embd(),
            &prompt_tokens,
            legacy_max_new,
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
            None,
            Some(req),
            Some(&plan),
        );
        if result.is_err() {
            if handoff {
                self.report_lookup_policy(
                    req,
                    guard.idx,
                    "legacy_multimodal",
                    Some("prefill_failed_before_handoff"),
                );
            }
            if let Some(kv) = guard.kv.as_mut() {
                kv.reset();
            }
        }
        let (legacy_ids, stats) = result?;
        if !handoff || crate::sampling::abort_requested(Some(req)) {
            if handoff {
                self.report_lookup_policy(
                    req,
                    guard.idx,
                    "legacy_multimodal",
                    Some("cancelled_before_handoff"),
                );
            }
            return Ok(stats);
        }
        debug_assert!(legacy_ids.is_empty());

        let prompt_end = prompt_tokens.len();
        let generated = 0;
        let prefill_start = stats.n_cached;
        let remaining_prefill = prompt_end.saturating_sub(1).saturating_sub(
            guard
                .kv
                .as_ref()
                .map(crate::seam::SeamKv::cached_len)
                .unwrap_or(0),
        );
        let sampler = ParallelSampler::new(req, &self.model.engine_cfg().sampling);
        let mut queue = self
            .decode_batch
            .lock()
            .expect("decode batch queue poisoned");
        let (event_tx, event_rx) = mpsc::sync_channel(0);
        let (ack_tx, ack_rx) = mpsc::sync_channel(0);
        queue.waiting.push_back(BatchWork {
            slot: guard.idx,
            kv: Some(guard.detach()),
            reasoning_guard: crate::sampling::Qwen4ReasoningEosGuard::from_prompt(
                self.model.config(),
                &prompt_tokens,
            ),
            lookup: self.new_lookup_work(req, &prompt_tokens, guard.idx, "scheduled_multimodal"),
            prompt: prompt_tokens,
            prompt_end,
            max_new,
            generated,
            stats,
            phase: phase_for_remaining_prefill(remaining_prefill),
            prefill_start,
            checkpoint_boundaries: [None; crate::seam::TURN_CHECKPOINT_COUNT],
            paged_frontier: None,
            turn_checkpoint: None,
            mrope_plan: Some(plan),
            multimodal_key: Some(key),
            finished: false,
            sampling: req.sampling().clone(),
            sampler: Some(sampler),
            mtp_ready: false,
            channels: Some(BatchChannels {
                events: event_tx,
                acknowledgements: ack_rx,
            }),
        });
        self.batch_interrupt.store(true, Ordering::Release);
        self.decode_ready.notify_all();
        drop(queue);
        self.wait_for_decode_batch(
            &mut guard,
            event_rx,
            ack_tx,
            req,
            &mut acc,
            &mut printed,
            &mut on_piece,
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        concurrent_mtp_profitable, concurrent_mtp_ready, deliver_lookup_prefix,
        expand_multimodal_prompt, long_prefill_cohort_limit, long_prefill_phase_prompt_end,
        lookup_fallback_reason, lookup_greedy_eligible, lookup_scheduler_supported,
        mtp_accepted_rows, mtp_context_profitable, multimodal_key, multimodal_token_position,
        phase_for_remaining_prefill, pick_continuation, repair_lookup_predictions, scheduler_mode,
        scheduler_requested, should_handoff_multimodal_to_scheduler, token_lane_prefill_fallback,
        token_lane_sort_key, BatchPhase, LookupBatchState, LookupTokenDelivery,
        MultimodalEmbedding, SchedulerMode, SHORT_PREFILL_TOKENS,
    };
    use std::sync::Arc;

    #[test]
    fn lookup_delivery_exhausts_widths_caps_and_disconnect_prefixes() {
        for width in 2..=4 {
            let candidates = &([11, 12, 13])[..width - 1];
            for accepted_drafts in 0..width {
                let mut ids = vec![99; width];
                ids[..accepted_drafts].copy_from_slice(&candidates[..accepted_drafts]);
                let mut expected = candidates[..accepted_drafts].to_vec();
                expected.push(ids[accepted_drafts]);
                for remaining in 0..=width + 1 {
                    for disconnect_after in 0..=expected.len() {
                        let mut sent = Vec::new();
                        let delivery =
                            deliver_lookup_prefix(candidates, &ids, remaining, |token| {
                                if sent.len() == disconnect_after {
                                    LookupTokenDelivery::Disconnected
                                } else {
                                    sent.push(token);
                                    LookupTokenDelivery::Continue
                                }
                            })
                            .unwrap();
                        let delivered = remaining.min(disconnect_after).min(expected.len());
                        assert_eq!(sent, expected[..delivered]);
                        assert_eq!(delivery.accepted_drafts, accepted_drafts);
                        assert_eq!(delivery.new_tokens, delivered);
                        assert_eq!(delivery.keep_inputs, 1 + delivered.min(accepted_drafts));
                        assert_eq!(
                            delivery.stopped,
                            delivered == remaining
                                || (disconnect_after < expected.len()
                                    && disconnect_after < remaining)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lookup_ack_cancel_retains_the_sent_candidate_but_never_a_bonus_row() {
        for stop_after in 1..=4 {
            let mut sent = Vec::new();
            let delivery = deliver_lookup_prefix(&[11, 12, 13], &[11, 12, 13, 99], 10, |token| {
                sent.push(token);
                if sent.len() == stop_after {
                    LookupTokenDelivery::Stop
                } else {
                    LookupTokenDelivery::Continue
                }
            })
            .unwrap();
            assert!(delivery.stopped);
            assert_eq!(delivery.new_tokens, stop_after);
            assert_eq!(delivery.keep_inputs, 1 + stop_after.min(3));
            assert_eq!(sent, [11, 12, 13, 99][..stop_after]);
        }
        let delivery = deliver_lookup_prefix(&[11], &[11, 99], 0, |_| {
            panic!("pending is already emitted")
        })
        .unwrap();
        assert_eq!((delivery.keep_inputs, delivery.new_tokens), (1, 0));
    }

    #[test]
    fn lookup_eos_stops_candidate_or_bonus_delivery_without_extra_rows() {
        const EOS: u32 = 2;
        for (candidates, ids, expected_rows) in [
            (vec![11, EOS, 13], vec![11, EOS, 13, 99], 3),
            (vec![11], vec![11, EOS], 2),
        ] {
            let mut sent = Vec::new();
            let delivery = deliver_lookup_prefix(&candidates, &ids, 10, |token| {
                sent.push(token);
                if token == EOS {
                    LookupTokenDelivery::Stop
                } else {
                    LookupTokenDelivery::Continue
                }
            })
            .unwrap();
            assert_eq!(sent, [11, EOS]);
            assert_eq!(delivery.keep_inputs, expected_rows);
            assert_eq!(delivery.new_tokens, 2);
            assert!(delivery.stopped);
        }
    }

    #[test]
    fn lookup_delivery_rejects_invalid_results_before_publishing() {
        for ids in [
            &[][..],
            &[99][..],
            &[11, 12, 13][..],
            &[11, 12, 13, 14, 99][..],
        ] {
            assert!(
                deliver_lookup_prefix(&[11], ids, 5, |_| panic!("invalid Verify delivery"))
                    .is_err()
            );
        }
        let mut sent = Vec::new();
        let delivery = deliver_lookup_prefix(&[11, 12, 13], &[11, 99, 13, 77], 10, |token| {
            sent.push(token);
            LookupTokenDelivery::Continue
        })
        .unwrap();
        assert_eq!(sent, [11, 99]); // a later matching row cannot extend the rejected prefix
        assert_eq!(delivery.keep_inputs, 2);
    }

    #[test]
    fn lookup_eligibility_is_greedy_and_penalty_free() {
        let cfg = infr_core::config::SamplingCfg::default();
        for (temp, top_k, presence_penalty, expected) in [
            (0.0, 20, 0.0, true),
            (0.8, 1, 0.0, true),
            (0.8, 20, 0.0, false),
            (0.0, 1, 1.0, false),
            (f32::NAN, 20, 0.0, false),
        ] {
            let req = crate::sampling::RequestCtx::new(crate::sampling::RequestSampling {
                temp: Some(temp),
                top_k: Some(top_k),
                presence_penalty,
                ..Default::default()
            });
            assert_eq!(lookup_greedy_eligible(&req, &cfg), expected);
        }
    }

    #[test]
    fn lookup_only_scheduler_requires_device_embedding_for_admission() {
        let cfg = crate::Config {
            qwen35: true,
            n_layer: 4,
            n_layer_kv_from_start: 4,
            n_embd: 256,
            full_attn_interval: 4,
            n_kv: 4,
            head_dim: 128,
            ssm_d_conv: 4,
            ssm_d_state: 128,
            ssm_d_inner: 1024,
            ssm_n_group: 8,
            ssm_dt_rank: 16,
            ..Default::default()
        };
        assert!(cfg.supports_parallel_token_rows());
        assert!(crate::seam::target_verify_supported(&cfg));
        let mut ec = crate::EngineConfig::default();
        assert!(!ec.spec.lookup);
        ec.spec.lookup = true;
        ec.spec.mtp = false;
        for gpu_embed in [false, true] {
            ec.spec.gpu_embed = gpu_embed;
            for embed_gather in [false, true] {
                for dtype in [
                    None,
                    Some(infr_core::DType::F32),
                    Some(infr_core::DType::F16),
                    Some(infr_core::DType::Q8_0),
                ] {
                    let supported = lookup_scheduler_supported(&cfg, &ec, embed_gather, dtype);
                    let expected = gpu_embed
                        && embed_gather
                        && matches!(dtype, Some(infr_core::DType::F16 | infr_core::DType::Q8_0));
                    let admitted = scheduler_requested(1, false, ec.spec.lookup, || supported);
                    assert_eq!(supported, expected);
                    assert_eq!(admitted, expected);
                    assert_eq!(
                        should_handoff_multimodal_to_scheduler(admitted, false, false, 16),
                        expected,
                    );
                    // These pre-existing scheduler routes are not narrowed by Lookup policy.
                    assert!(scheduler_requested(2, false, ec.spec.lookup, || supported));
                    assert!(scheduler_requested(1, true, ec.spec.lookup, || supported));
                }
            }
        }
        ec.spec.gpu_embed = true;
        ec.spec.lookup = false;
        assert!(!lookup_scheduler_supported(
            &cfg,
            &ec,
            true,
            Some(infr_core::DType::F16)
        ));
        ec.spec.lookup = true;
        let mut unsupported = cfg;
        unsupported.n_layer = 0;
        assert!(!lookup_scheduler_supported(
            &unsupported,
            &ec,
            true,
            Some(infr_core::DType::F16),
        ));
    }

    #[test]
    fn scheduler_preserves_legacy_decisions_without_probing_lookup() {
        for slots in [1, 2] {
            for heads in [false, true] {
                assert_eq!(
                    scheduler_requested(slots, heads, false, || {
                        panic!("disabled Lookup must not inspect embedding metadata")
                    }),
                    slots > 1 || heads,
                );
                if slots > 1 || heads {
                    assert!(scheduler_requested(slots, heads, true, || {
                        panic!("legacy multi-slot/MTP scheduling must not depend on Lookup")
                    }));
                }
            }
        }
    }

    #[test]
    fn lookup_flash_metadata_shape_is_eligible_and_reports_each_fallback() {
        // CPU-only fixture from the actual Flash GGUF header; no tensor payload is loaded.
        let mut cfg = crate::Config {
            qwen4exp: true,
            n_layer: 48,
            n_layer_kv_from_start: 48,
            n_embd: 2560,
            n_head: 24,
            n_kv: 2,
            hc_mult: 4,
            full_attn_interval: 4,
            recurrent_layers: (0..48).map(|layer| (layer + 1) % 4 != 0).collect(),
            ssm_d_conv: 4,
            ssm_d_state: 128,
            ssm_d_inner: 6144,
            ssm_n_group: 16,
            ssm_dt_rank: 48,
            ple_layers: vec![false; 48],
            ple_ngram_size: 3,
            ple_heads_per_ngram: 8,
            ple_head_dim: 160,
            ple_conv_kernel: 4,
            ..Default::default()
        };
        cfg.ple_layers[1] = true;
        let mut ec = crate::EngineConfig::default();
        ec.spec.lookup = true;
        ec.spec.mtp = false;
        ec.spec.gpu_embed = true;
        ec.sampling.temp = 1.0;
        let req = crate::sampling::RequestCtx::new(crate::sampling::RequestSampling {
            temp: Some(0.0),
            ..Default::default()
        });
        let dtype = Some(infr_core::DType::Q8_0);
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, dtype),
            None
        );
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, true, true, dtype),
            Some("mtp_head_loaded")
        );
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, false, dtype),
            Some("backend_embed_gather_unsupported")
        );
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, None),
            Some("token_embedding_missing")
        );
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, Some(infr_core::DType::F32)),
            Some("token_embedding_dtype_unsupported")
        );
        ec.spec.lookup = false;
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, dtype),
            Some("lookup_disabled")
        );
        ec.spec.lookup = true;
        ec.spec.gpu_embed = false;
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, dtype),
            Some("gpu_embedding_disabled")
        );
        ec.spec.gpu_embed = true;
        ec.kernels.vulkan.dn_chunk = false;
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, dtype),
            Some("gdn_chunk_disabled")
        );
        ec.kernels.vulkan.dn_chunk = true;
        ec.kernels.vulkan.dn_chunk_scan = false;
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, dtype),
            Some("gdn_chunk_scan_disabled")
        );
        ec.kernels.vulkan.dn_chunk_scan = true;
        ec.kernels.vulkan.dn_split = false;
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, dtype),
            Some("gdn_split_disabled")
        );
        ec.kernels.vulkan.dn_split = true;
        ec.spec.lookup_max_draft = 0;
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, dtype),
            Some("zero_draft_budget")
        );
        ec.spec.lookup_max_draft = 3;
        let inherited = crate::sampling::RequestCtx::new(Default::default());
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &inherited, false, true, dtype),
            Some("request_sampling_not_greedy")
        );
        let penalized = crate::sampling::RequestCtx::new(crate::sampling::RequestSampling {
            temp: Some(0.0),
            presence_penalty: 1.0,
            ..Default::default()
        });
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &penalized, false, true, dtype),
            Some("request_penalties_active")
        );
        cfg.ple_layers[0] = true;
        assert_eq!(
            lookup_fallback_reason(&cfg, &ec, &req, false, true, dtype),
            Some("unsupported_target_state")
        );
    }

    #[test]
    fn lookup_request_greedy_override_works_with_nongreedy_model_defaults() {
        let cfg = infr_core::config::SamplingCfg {
            temp: 1.0,
            top_k: 20,
            ..Default::default()
        };
        let inherited = crate::sampling::RequestCtx::new(Default::default());
        assert!(!lookup_greedy_eligible(&inherited, &cfg));
        for sampling in [
            crate::sampling::RequestSampling {
                temp: Some(0.0),
                ..Default::default()
            },
            crate::sampling::RequestSampling {
                top_k: Some(1),
                ..Default::default()
            },
        ] {
            let req = crate::sampling::RequestCtx::new(sampling);
            assert!(lookup_greedy_eligible(&req, &cfg));
            // Ordinary token rows use this same request-local resolution.
            let sampler = crate::sampling::ParallelSampler::new(&req, &cfg);
            assert!(sampler.can_gpu_argmax());
        }
    }

    #[test]
    fn lookup_shared_target_gate_requires_gdn_flags_only_with_recurrent_layers() {
        let attention = crate::Config {
            n_layer: 4,
            n_layer_kv_from_start: 4,
            n_embd: 256,
            n_head: 4,
            n_kv: 2,
            head_dim: 64,
            ..Default::default()
        };
        let hybrid = crate::Config {
            qwen35: true,
            n_layer: 4,
            n_layer_kv_from_start: 4,
            n_embd: 256,
            full_attn_interval: 4,
            n_kv: 4,
            head_dim: 128,
            ssm_d_conv: 4,
            ssm_d_state: 128,
            ssm_d_inner: 1024,
            ssm_n_group: 8,
            ssm_dt_rank: 16,
            ..Default::default()
        };
        let mut ec = crate::EngineConfig::default();
        ec.spec.lookup = true;
        ec.spec.mtp = false;
        ec.spec.gpu_embed = true;
        for mask in 0..8 {
            ec.kernels.vulkan.dn_chunk = mask & 1 != 0;
            ec.kernels.vulkan.dn_chunk_scan = mask & 2 != 0;
            ec.kernels.vulkan.dn_split = mask & 4 != 0;
            assert!(crate::seam::target_verify_enabled(&attention, &ec));
            assert_eq!(crate::seam::target_verify_enabled(&hybrid, &ec), mask == 7);
        }
    }

    #[test]
    fn lookup_history_sync_handles_same_length_edits_rewind_and_ordinary_handoff() {
        let mut state = LookupBatchState::new(Default::default(), &[1; 8]).unwrap();
        state.sync_history(&[2; 8]);
        assert_eq!(state.indexed_len, 8);
        assert!(state.proposer.propose(None, 3).is_none());
        state.sync_history(&[2; 4]);
        assert_eq!(state.indexed_len, 4);
        state.append_confirmed(&[99]); // pending is indexed exactly once, not materialized here
        assert_eq!(state.indexed_len, 5);
        state.record_ordinary(0.16, 16);
        state.append_confirmed(&[100, 101]);
        assert_eq!(state.indexed_len, 7);
        assert_eq!(state.ordinary_token_secs, Some(0.01));
        state.sync_history(&[3; 8]);
        assert_eq!(state.indexed_len, 8);
        assert!(state.proposer.propose(None, 3).is_none());
    }

    #[test]
    fn lookup_cost_backoff_applies_even_at_full_acceptance_and_recovers() {
        let mut state = LookupBatchState::new(Default::default(), &[]).unwrap();
        state.record_ordinary(0.16, 16); // ordinary wall time = 10 ms/token
        state.record_cycle(3, 3, 4, 0.06); // full acceptance, but 15 ms/token
        assert_eq!(state.cost_cooldown, 0);
        state.record_cycle(3, 3, 4, 0.06);
        let interrupt = std::sync::atomic::AtomicBool::new(false);
        for remaining in (0..super::LOOKUP_COST_COOLDOWN_TOKENS).rev() {
            state.observe_ordinary_token(42, true, true, 3, &interrupt);
            assert_eq!(state.cost_cooldown, remaining);
            assert!(!interrupt.load(std::sync::atomic::Ordering::Acquire));
        }
        assert_eq!(state.cost_cooldown, 0);
        state.record_cycle(3, 3, 4, 0.02);
        assert_eq!(state.slow_cycles, 0);
        assert_eq!(state.stats.cost_backoffs, 1);
        state.missed_search_secs = 0.05; // unsuccessful searches also cost wall time
        state.record_cycle(3, 3, 4, 0.02);
        assert_eq!(state.slow_cycles, 1);
        state.record_ordinary(0.0, 0);
        assert_eq!(state.ordinary_token_secs, Some(0.01));
    }

    #[test]
    fn lookup_short_reentry_cannot_inflate_continuous_decode_reference() {
        let mut state = LookupBatchState::new(Default::default(), &[]).unwrap();
        state.record_ordinary(0.16, 16);
        for tokens in 1..super::LOOKUP_CALIBRATION_TOKENS {
            state.record_ordinary(tokens as f64, tokens);
            assert_eq!(state.ordinary_token_secs, Some(0.01));
        }
        state.record_cycle(3, 3, 4, 0.06);
        state.record_cycle(3, 3, 4, 0.06);
        assert_eq!(state.cost_cooldown, super::LOOKUP_COST_COOLDOWN_TOKENS);
        state.record_ordinary(0.32, 16);
        assert!((state.ordinary_token_secs.unwrap() - 0.0125).abs() < 1e-12);
    }

    #[test]
    fn lookup_cost_baseline_excludes_prefill_handoffs_and_concurrent_groups() {
        use super::{lookup_ordinary_sample_eligible, BatchPhase};
        assert!(!lookup_ordinary_sample_eligible(
            BatchPhase::ShortPrefill,
            0,
            1
        ));
        assert!(!lookup_ordinary_sample_eligible(
            BatchPhase::LongPrefill,
            0,
            1
        ));
        assert!(!lookup_ordinary_sample_eligible(BatchPhase::Decode, 0, 1));
        assert!(!lookup_ordinary_sample_eligible(BatchPhase::Decode, 16, 2));
        assert!(lookup_ordinary_sample_eligible(BatchPhase::Decode, 16, 1));
    }

    #[test]
    fn lookup_quantum_only_bounds_initial_single_lane_calibration() {
        use super::lookup_plain_quantum;
        let mut state = LookupBatchState::new(Default::default(), &[]).unwrap();
        assert_eq!(lookup_plain_quantum(1, 0, 7, Some(&state)), 8);
        assert_eq!(lookup_plain_quantum(1, 1, 0, Some(&state)), 16);
        assert_eq!(lookup_plain_quantum(2, 0, 7, Some(&state)), usize::MAX);
        assert_eq!(lookup_plain_quantum(2, 1, 0, Some(&state)), usize::MAX);
        assert_eq!(lookup_plain_quantum(1, 1, 0, None), usize::MAX);
        state.record_ordinary(0.0, 1); // short-prefill handoff does not warm the baseline
        assert_eq!(state.calibration_tokens, 0);
        state.record_ordinary(0.04, 4); // admission can interrupt calibration
        assert_eq!(state.ordinary_token_secs, None);
        assert_eq!(lookup_plain_quantum(1, 5, 0, Some(&state)), 12);
        state.record_ordinary(0.12, 12);
        assert_eq!(state.ordinary_token_secs, Some(0.01));
        assert_eq!(lookup_plain_quantum(1, 17, 0, Some(&state)), usize::MAX);
        state.cost_cooldown = super::LOOKUP_COST_COOLDOWN_TOKENS;
        assert_eq!(lookup_plain_quantum(1, 17, 0, Some(&state)), usize::MAX);
    }

    #[test]
    fn lookup_callback_no_match_keeps_contiguous_ordinary_and_indexes_once() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let history = (0..64).collect::<Vec<_>>();
        let mut state = LookupBatchState::new(Default::default(), &history).unwrap();
        state.record_ordinary(0.16, 16);
        let interrupt = AtomicBool::new(false);
        let outputs = (64..320).collect::<Vec<_>>();
        for &token in &outputs {
            state.observe_ordinary_token(token, true, true, 3, &interrupt);
            assert!(!interrupt.load(Ordering::Acquire));
            assert!(state.pending_proposal.is_none());
        }
        assert_eq!(state.stats.searches, outputs.len());
        assert_eq!(state.indexed_len, history.len() + outputs.len());
        state
            .finish_ordinary_outputs(history.len(), &outputs)
            .unwrap();
        assert_eq!(state.indexed_len, history.len() + outputs.len());
        assert!(state.proposer.propose(None, 3).is_none());
    }

    #[test]
    fn lookup_callback_ack_proposal_yields_and_is_consumed_without_research() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7];
        let mut state = LookupBatchState::new(Default::default(), &history).unwrap();
        state.record_ordinary(0.16, 16);
        let interrupt = AtomicBool::new(false);
        state.observe_ordinary_token(8, true, true, 3, &interrupt);
        assert!(interrupt.load(Ordering::Acquire));
        assert_eq!(state.indexed_len, history.len() + 1);
        state.finish_ordinary_outputs(history.len(), &[8]).unwrap();
        let searches = state.stats.searches;
        let (proposal, _) = state.take_pending_proposal(2).unwrap();
        assert_eq!(proposal.tokens, [9, 10]);
        assert_eq!(state.stats.searches, searches);
        assert!(state.take_pending_proposal(2).is_none());
        // The proposal consumed no candidate tokens: only the emitted pending was appended.
        assert_eq!(state.indexed_len, history.len() + 1);
        state.append_confirmed(&[9, 99]); // Verify publishes one match plus its target correction
        assert_eq!(state.indexed_len, history.len() + 3);
    }

    #[test]
    fn lookup_callback_preserves_the_selected_width_feedback_policy() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7];
        for adaptive_width in [false, true] {
            let mut state = LookupBatchState::new(
                crate::speculative_lookup::LookupConfig {
                    adaptive_width,
                    ..Default::default()
                },
                &history,
            )
            .unwrap();
            state.record_ordinary(0.16, 16);
            state.proposer.record_result(1, 3);
            let interrupt = AtomicBool::new(false);
            state.observe_ordinary_token(8, true, true, 3, &interrupt);
            assert!(interrupt.load(Ordering::Acquire));
            let (proposal, _) = state.take_pending_proposal(3).unwrap();
            let width = if adaptive_width { 1 } else { 3 };
            assert_eq!(proposal.tokens, [9, 10, 11][..width]);
            assert_eq!(state.indexed_len, history.len() + 1);
        }
    }

    #[test]
    fn lookup_callback_cancel_cohort_and_admission_do_not_request_proposal_yields() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let history = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7];
        for (ack, single, admission) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            let mut state = LookupBatchState::new(Default::default(), &history).unwrap();
            state.record_ordinary(0.16, 16);
            let interrupt = AtomicBool::new(admission);
            state.observe_ordinary_token(8, ack, single, 3, &interrupt);
            assert_eq!(interrupt.load(Ordering::Acquire), admission);
            assert!(state.pending_proposal.is_none());
            assert_eq!(state.stats.searches, 0);
            state.finish_ordinary_outputs(history.len(), &[8]).unwrap();
            assert_eq!(state.indexed_len, history.len() + 1);
        }
        let mut state = LookupBatchState::new(Default::default(), &history).unwrap();
        state.record_ordinary(0.16, 16);
        let interrupt = AtomicBool::new(false);
        state.observe_ordinary_token(8, true, true, 3, &interrupt);
        assert!(state.pending_proposal.is_some());
        // A new cohort can advance ordinary decode before Lookup gets another single-lane turn.
        state.observe_ordinary_token(9, true, false, 0, &interrupt);
        assert!(state.take_pending_proposal(3).is_none());
        state
            .finish_ordinary_outputs(history.len(), &[8, 9, 2])
            .unwrap(); // EOS bypassed callback
        assert_eq!(state.indexed_len, history.len() + 3);
        state.sync_history(&history); // rewind invalidates even a same-frontier cached proposal
        assert!(state.take_pending_proposal(3).is_none());
        assert_eq!(state.indexed_len, history.len());
        interrupt.store(false, Ordering::Release);
        state.observe_ordinary_token(8, true, true, 3, &interrupt);
        assert!(state.pending_proposal.is_some());
        let mut edited = history.to_vec();
        edited.push(8);
        edited[0] = 99;
        state.sync_history(&edited); // equal length is not proof of the same proposal frontier
        assert!(state.take_pending_proposal(3).is_none());
        assert_eq!(state.indexed_len, edited.len());
    }

    #[test]
    fn lookup_cost_cooldown_is_128_ordinary_tokens_without_intermediate_yields() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut state = LookupBatchState::new(Default::default(), &[1; 32]).unwrap();
        state.record_ordinary(0.16, 16);
        state.record_cycle(3, 3, 4, 0.06);
        state.record_cycle(3, 3, 4, 0.06);
        let interrupt = AtomicBool::new(false);
        for _ in 0..128 {
            state.observe_ordinary_token(1, true, true, 3, &interrupt);
            assert!(!interrupt.load(Ordering::Acquire));
        }
        assert_eq!(state.stats.searches, 0);
        assert_eq!(state.cost_cooldown, 0);
        state.observe_ordinary_token(1, true, true, 3, &interrupt);
        assert!(interrupt.load(Ordering::Acquire));
        assert!(state.take_pending_proposal(3).is_some());
    }

    #[test]
    fn lookup_callback_and_verify_share_output_context_and_qsa_budget() {
        use super::lookup_draft_budget;
        assert_eq!(lookup_draft_budget(3, 1, 20, 128, 64), 0);
        assert_eq!(lookup_draft_budget(3, 2, 20, 128, 64), 1);
        assert_eq!(lookup_draft_budget(9, 10, 20, 128, 64), 3);
        assert_eq!(lookup_draft_budget(3, 10, 128, 128, 64), 0);
        assert_eq!(lookup_draft_budget(3, 10, 127, 128, 64), 1);
        assert_eq!(lookup_draft_budget(3, 10, 63, 128, 64), 1);
        assert_eq!(lookup_draft_budget(3, 10, 64, 128, 64), 0);
        assert_eq!(lookup_draft_budget(3, 10, 65, 128, 64), 3);
        assert_eq!(lookup_draft_budget(0, 10, 65, 128, 64), 0);
    }

    #[test]
    fn mtp_verify_width_matches_standalone_limits() {
        for (configured, rows) in [(0, 2), (1, 2), (2, 2), (3, 3), (4, 4), (6, 4)] {
            assert_eq!(super::mtp_verify_rows(configured), rows);
        }
    }

    #[test]
    fn lookup_indexes_each_known_append_including_pending_exactly_once() {
        let mut state = LookupBatchState::new(
            Default::default(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],
        )
        .unwrap();
        assert!(state.proposer.propose(None, 3).is_none());
        state.append_confirmed(&[8]); // ordinary decode's already-emitted pending
        assert_eq!(state.indexed_len, 20);
        let found = state.proposer.propose(None, 3).unwrap();
        assert_eq!(found.tokens, [9, 10, 11]);
        assert_eq!(found.matched_tokens, 8);
        state.proposer.record_result(1, 3);
        state.append_confirmed(&[9, 99]); // accepted candidate plus target correction
        assert_eq!(state.indexed_len, 22);
        assert!(state.proposer.propose(None, 3).is_none());
    }

    #[test]
    fn lookup_eos_repair_uses_full_logits_and_only_the_candidate_guard_branch() {
        let cfg = crate::Config {
            qwen4exp: true,
            vocab: 16,
            eos: 2,
            eos_ids: vec![2, 3],
            qwen4_think_start: Some(10),
            qwen4_think_end: Some(11),
            ..Default::default()
        };
        let guard = crate::sampling::Qwen4ReasoningEosGuard::from_prompt(&cfg, &[10, 7]);
        let mut logits = vec![0.0; 2 * cfg.vocab];
        logits[2] = 100.0;
        logits[3] = 90.0;
        logits[7] = 10.0;
        let mut ids = vec![2, 2];
        repair_lookup_predictions(&cfg, guard, &[11], &mut ids, &mut logits, false).unwrap();
        assert_eq!(ids, [7, 2]); // candidate closes reasoning only for its following row
        assert!(guard.blocks(&cfg, 2, false)); // persistent guard remains unchanged
        assert!(repair_lookup_predictions(&cfg, guard, &[7], &mut [2, 2], &mut [], false).is_err());
        let mut ids = [2, 2];
        repair_lookup_predictions(&cfg, guard, &[7], &mut ids, &mut [], true).unwrap();
        assert_eq!(ids, [2, 2]);
        let mut corrupt = vec![0.0; 2 * cfg.vocab];
        corrupt[7] = f32::NAN;
        let error = repair_lookup_predictions(&cfg, guard, &[7], &mut [2, 2], &mut corrupt, false)
            .unwrap_err();
        assert!(crate::sampling::is_numerical_corruption(&error));
    }

    #[test]
    fn short_prefill_boundary_is_inclusive_at_96_tokens() {
        assert_eq!(phase_for_remaining_prefill(0), BatchPhase::Decode);
        assert_eq!(phase_for_remaining_prefill(1), BatchPhase::ShortPrefill);
        assert_eq!(
            phase_for_remaining_prefill(SHORT_PREFILL_TOKENS),
            BatchPhase::ShortPrefill
        );
        assert_eq!(
            phase_for_remaining_prefill(SHORT_PREFILL_TOKENS + 1),
            BatchPhase::LongPrefill
        );
    }

    #[test]
    fn long_prefill_hands_a_short_checkpoint_tail_to_token_steps() {
        assert_eq!(
            long_prefill_phase_prompt_end(0, 30_000, &[Some(29_950), None]),
            29_951
        );
        assert_eq!(
            long_prefill_phase_prompt_end(0, 30_000, &[Some(29_900), None]),
            30_000
        );
        assert_eq!(
            long_prefill_phase_prompt_end(29_950, 30_000, &[Some(29_950), None]),
            30_000
        );
    }

    #[test]
    fn qwen35_batches_decode_but_serializes_long_prefill() {
        let qwen35 = crate::Config {
            qwen35: true,
            ..Default::default()
        };
        assert!(qwen35.supports_parallel_token_rows());
        assert_eq!(long_prefill_cohort_limit(&qwen35), 1);

        let qwen4exp = crate::Config {
            qwen4exp: true,
            ..Default::default()
        };
        assert!(qwen4exp.supports_parallel_token_rows());
        assert_eq!(
            long_prefill_cohort_limit(&qwen4exp),
            super::MAX_DECODE_BATCH
        );

        assert!(!crate::Config::default().supports_parallel_token_rows());
    }

    #[test]
    fn multimodal_handoff_requires_a_scheduler_worker() {
        assert!(!should_handoff_multimodal_to_scheduler(
            false, false, false, 16
        ));
        assert!(should_handoff_multimodal_to_scheduler(
            true, false, false, 16
        ));
        assert!(!should_handoff_multimodal_to_scheduler(
            true, true, false, 16
        ));
        assert!(!should_handoff_multimodal_to_scheduler(
            true, false, true, 16
        ));
        assert!(!should_handoff_multimodal_to_scheduler(
            true, false, false, 0
        ));
    }

    #[test]
    fn token_lanes_put_longer_prefill_before_decode() {
        let mut lanes = [(0usize, false, 0usize), (1, false, 37)];
        lanes.sort_by_key(|&(slot, has_mrope, remaining)| {
            token_lane_sort_key(has_mrope, remaining, slot)
        });
        assert_eq!(lanes.map(|lane| lane.0), [1, 0]);
    }

    #[test]
    fn token_lanes_drain_text_prefill_before_visual_decode() {
        for remaining in [1, 37, SHORT_PREFILL_TOKENS] {
            let lanes = [(true, 0), (false, remaining)];
            assert_eq!(token_lane_prefill_fallback(lanes), Some((1, remaining)));
            // At the fallback boundary no token was sampled; both lanes can now Decode together.
            assert_eq!(token_lane_prefill_fallback([(true, 0), (false, 0)]), None);
        }
    }

    #[test]
    fn token_lanes_preserve_compatible_batches() {
        for lanes in [
            vec![],
            vec![(false, 37)],
            vec![(true, 0)],
            vec![(false, 37), (false, 0)],
            vec![(true, 37), (false, 0)],
            vec![(true, 37), (true, 5), (false, 0)],
            vec![(true, 0), (false, 0), (false, 0), (false, 0)],
        ] {
            assert_eq!(token_lane_prefill_fallback(lanes), None);
        }
    }

    #[test]
    fn token_lanes_drain_conflicting_suffixes_and_rejoin() {
        let mut lanes = [(0, true, 5), (1, true, 0), (2, false, 96), (3, false, 37)];
        for expected_slot in [2, 3] {
            lanes.sort_by_key(|&(slot, has_mrope, remaining)| {
                token_lane_sort_key(has_mrope, remaining, slot)
            });
            let (lane, steps) = token_lane_prefill_fallback(
                lanes
                    .iter()
                    .map(|&(_, has_mrope, remaining)| (has_mrope, remaining)),
            )
            .expect("text suffix must run before this mixed cohort");
            assert_eq!(lanes[lane].0, expected_slot);
            assert!(!lanes[lane].1);
            assert_eq!(steps, lanes[lane].2);
            lanes[lane].2 -= steps;
        }
        assert_eq!(
            token_lane_prefill_fallback(
                lanes.map(|(_, has_mrope, remaining)| (has_mrope, remaining))
            ),
            None
        );
        assert!(lanes.windows(2).all(|pair| pair[0].2 >= pair[1].2));
    }

    #[test]
    fn token_lanes_exhaustive_mixed_cohorts_remain_executable() {
        const REMAINING: [usize; 3] = [0, 1, SHORT_PREFILL_TOKENS];
        for count in 1..=4u32 {
            for mut encoded in 0..6usize.pow(count) {
                let mut lanes = (0..count as usize)
                    .map(|slot| {
                        let state = encoded % 6;
                        encoded /= 6;
                        (slot, state >= 3, REMAINING[state % 3])
                    })
                    .collect::<Vec<_>>();
                let mut fallbacks = 0;
                loop {
                    lanes.sort_by_key(|&(slot, has_mrope, remaining)| {
                        token_lane_sort_key(has_mrope, remaining, slot)
                    });
                    let fallback = token_lane_prefill_fallback(
                        lanes
                            .iter()
                            .map(|&(_, has_mrope, remaining)| (has_mrope, remaining)),
                    );
                    let descending = lanes.windows(2).all(|pair| pair[0].2 >= pair[1].2);
                    if let Some((lane, steps)) = fallback {
                        assert!(!descending);
                        assert!(!lanes[lane].1);
                        assert!(steps > 0);
                        assert_eq!(steps, lanes[lane].2);
                        lanes[lane].2 -= steps;
                        fallbacks += 1;
                        assert!(fallbacks < lanes.len());
                    } else {
                        assert!(descending);
                        assert!(!lanes.iter().any(|lane| lane.1) || lanes[0].1);
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn mtp_accepts_each_lane_independently() {
        let candidates = [11, 12, 13];
        assert_eq!(mtp_accepted_rows(&candidates, &[99, 0, 0, 0]), 1);
        assert_eq!(mtp_accepted_rows(&candidates, &[11, 99, 0, 0]), 2);
        assert_eq!(mtp_accepted_rows(&candidates, &[11, 12, 99, 0]), 3);
        assert_eq!(mtp_accepted_rows(&candidates, &[11, 12, 13, 99]), 4);
        assert_eq!(mtp_accepted_rows(&[5, 6, 7], &[5, 6, 7, 8]), 4);
    }

    #[test]
    fn concurrent_mtp_requires_eighty_percent_acceptance() {
        assert!(concurrent_mtp_profitable(24, 30));
        assert!(concurrent_mtp_profitable(25, 30));
        assert!(!concurrent_mtp_profitable(23, 30));
        assert!(!concurrent_mtp_profitable(0, 0));
    }

    #[test]
    fn concurrent_mtp_requires_every_lane_to_be_ready() {
        assert!(concurrent_mtp_ready(true, false, true));
        assert!(!concurrent_mtp_ready(false, false, true));
        assert!(!concurrent_mtp_ready(true, true, true));
        assert!(!concurrent_mtp_ready(true, false, false));
    }

    #[test]
    fn mtp_context_cutoff_can_be_bounded_or_disabled() {
        assert!(mtp_context_profitable(48 * 1024, 48 * 1024));
        assert!(!mtp_context_profitable(48 * 1024, 48 * 1024 + 1));
        assert!(mtp_context_profitable(0, usize::MAX));
    }

    #[test]
    fn scheduler_routes_the_four_supported_work_mixes() {
        assert_eq!(
            scheduler_mode([BatchPhase::Decode, BatchPhase::Decode]),
            Some(SchedulerMode::DecodeRows)
        );
        assert_eq!(
            scheduler_mode([BatchPhase::ShortPrefill, BatchPhase::ShortPrefill]),
            Some(SchedulerMode::ShortPrefillRows)
        );
        assert_eq!(
            scheduler_mode([BatchPhase::Decode, BatchPhase::ShortPrefill]),
            Some(SchedulerMode::DecodeAndShortPrefillRows)
        );
        assert_eq!(
            scheduler_mode([BatchPhase::Decode, BatchPhase::LongPrefill]),
            Some(SchedulerMode::LongPrefill)
        );
    }

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

    #[test]
    fn multimodal_expansion_preserves_order_and_grid_positions() {
        let images = vec![
            MultimodalEmbedding {
                values: Arc::new(vec![1.0; 2 * 2 * 3]),
                grid_nx: 2,
                grid_ny: 2,
                fingerprint: [1; 32],
            },
            MultimodalEmbedding {
                values: Arc::new(vec![2.0; 3 * 3]),
                grid_nx: 3,
                grid_ny: 1,
                fingerprint: [2; 32],
            },
        ];
        let (tokens, plan) = expand_multimodal_prompt(&[10, 99, 11, 99, 12], 99, images, 3)
            .expect("valid synthetic multimodal prompt");
        assert_eq!(tokens, [10, 99, 99, 99, 99, 11, 99, 99, 99, 12]);
        assert_eq!(plan.spans[0].start, 1);
        assert_eq!(plan.spans[0].n_tokens, 4);
        assert_eq!(plan.spans[1].start, 6);
        assert_eq!(plan.spans[1].n_tokens, 3);
        assert_eq!(
            &plan.prompt_pos4[4..20],
            &[1, 1, 1, 0, 1, 1, 2, 0, 1, 2, 1, 0, 1, 2, 2, 0]
        );
        assert_eq!(plan.decode_base, 8);
        assert_eq!(multimodal_token_position(Some(&plan), 9).unwrap(), 7);
        assert_eq!(multimodal_token_position(Some(&plan), 10).unwrap(), 8);
        assert_eq!(multimodal_token_position(Some(&plan), 13).unwrap(), 11);
    }

    #[test]
    fn multimodal_expansion_rejects_marker_count_mismatch() {
        let image = MultimodalEmbedding {
            values: Arc::new(vec![0.0; 4]),
            grid_nx: 1,
            grid_ny: 1,
            fingerprint: [0; 32],
        };
        assert!(expand_multimodal_prompt(&[1, 2], 99, vec![image], 4).is_err());
    }

    #[test]
    fn multimodal_key_covers_image_identity_order_and_grid() {
        let image = |fingerprint, grid_nx, grid_ny| MultimodalEmbedding {
            values: Arc::new(Vec::new()),
            grid_nx,
            grid_ny,
            fingerprint,
        };
        let a = image([1; 32], 2, 3);
        let b = image([2; 32], 4, 5);
        assert_eq!(
            multimodal_key(&[a.clone(), b.clone()]),
            multimodal_key(&[a.clone(), b.clone()])
        );
        assert_ne!(
            multimodal_key(&[a.clone(), b.clone()]),
            multimodal_key(&[b, a.clone()])
        );
        assert_ne!(
            multimodal_key(std::slice::from_ref(&a)),
            multimodal_key(&[image([1; 32], 3, 2)])
        );
    }
}
