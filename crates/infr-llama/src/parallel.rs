//! [`ParallelSeam`] — the N-slot concurrent generation engine behind `infr serve --parallel N`.
//!
//! # What this is (and what it is not)
//!
//! Every sequence owns one KV slot. Qwen3.8 text requests register with one persistent compute
//! worker before tokenization, then enqueue their prepared task and remain under that scheduler
//! through completion; no request thread doubles as the cohort leader. Decode rows run
//! layer-synchronously. Up to 96 uncached prompt tokens advance as teacher-forced rows in the same
//! Decode-LRU graphs; longer prompts share the configured ubatch in one exclusive Prefill-ring
//! phase before decode resumes. Stateless work and paged expert traffic are shared while every
//! sequence retains independent positions, KV/recurrent state and sampling. Dense/sparse QSA rows
//! form compatible subgroups but never fall back to independent request loops. Other architectures
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

use crate::sampling::{ParallelSampler, RequestCtx, RequestSampling, StepGate};
use crate::seam::SeamKv;
use crate::session_cache::{MtpCacheSource, SessionCache};
use crate::{Config, GenStats, SeamModel};
use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const DECODE_BATCH_WAIT: Duration = Duration::from_secs(1);
const PREFILL_COHORT_WAIT: Duration = Duration::from_millis(50);
const MAX_DECODE_BATCH: usize = 8;
const SHORT_PREFILL_TOKENS: usize = 96;
const MTP_PLAIN_SYNC_QUANTUM: usize = 1024;

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

/// Whether a visual turn may hand its decode off to the batched scheduler.
///
/// `scheduler_worker_present` must be the LIVE WORKER HANDLE, never `gate.is_some()` — the gate
/// exists on every `--parallel N>1` engine but `ParallelSeam::start_scheduler_worker` only spawns
/// the draining thread for qwen4exp. Passing `true` with no worker queues a batch nobody drains and
/// wedges the request thread in `wait_for_decode_batch`.
fn should_handoff_multimodal_to_scheduler(
    scheduler_worker_present: bool,
    mtp_heads_present: bool,
    frontier_is_image: bool,
    max_new: usize,
) -> bool {
    scheduler_worker_present && !mtp_heads_present && !frontier_is_image && max_new > 0
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

fn mtp_accepted_rows(candidates: &[u32], verified: &[u32]) -> usize {
    debug_assert_eq!(verified.len(), candidates.len() + 1);
    1 + candidates
        .iter()
        .zip(verified)
        .take_while(|(candidate, target)| candidate == target)
        .count()
}

fn mtp_prime_chunk_end(
    start: usize,
    prompt_len: usize,
    chunk: usize,
    checkpoint_boundaries: &[Option<usize>; crate::seam::TURN_CHECKPOINT_COUNT],
) -> usize {
    let natural_end = start.saturating_add(chunk).min(prompt_len);
    checkpoint_boundaries
        .iter()
        .flatten()
        .copied()
        .filter(|&boundary| start < boundary && boundary <= natural_end)
        .min()
        .unwrap_or(natural_end)
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
    turn_checkpoint: Option<crate::seam::TurnCheckpoint>,
    mrope_plan: Option<crate::seam::MropePlan>,
    multimodal_key: Option<MultimodalKey>,
    finished: bool,
    sampling: RequestSampling,
    sampler: Option<ParallelSampler>,
    reasoning_guard: crate::sampling::Qwen4ReasoningEosGuard,
    mtp_ready: bool,
    channels: Option<BatchChannels>,
}

struct MtpSlotState {
    head: crate::mtp::Qwen4MtpSession,
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
            .find(|checkpoint| checkpoint.tokens == tokens)
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

struct BatchChannels {
    events: SyncSender<BatchEvent>,
    acknowledgements: Receiver<bool>,
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

        let _gate = gate.as_deref().map(StepGate::enter);
        let mut heads = mtp_heads.as_ref().map(|heads| {
            heads
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        let mut cache = match cache.lock() {
            Ok(cache) => cache,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mtp_state = heads
            .as_ref()
            .and_then(|heads| heads[index].cache_state_for(kv.cached_tokens()));
        let mtp_source = heads
            .as_ref()
            .zip(mtp_state.as_ref())
            .map(|(heads, state)| MtpCacheSource {
                head: &heads[index].head,
                state,
            });
        let spill_failed = match cache.spill(&mut kv, backend.as_ref(), &model_cfg, mtp_source) {
            Ok(true) => {
                if let Some(heads) = heads.as_mut() {
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
                    "cold KV idle spill failed; keeping the resident state: {error}"
                );
                true
            }
        };
        if let Err(error) = cache.gc() {
            tracing::warn!("cold KV cache maintenance failed: {error}");
        }
        drop(cache);
        drop(heads);
        drop(_gate);

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
        if model.engine_cfg().spec.mtp && model.config().qwen4exp && n_slots > 2 {
            return Err(anyhow!("Qwen3.8 concurrent MTP supports at most two slots"));
        }
        // This engine's own placement pins, entered as the current scope for the whole
        // placement phase (the clamp inside `vulkan_slot_ctx` + the `init_slots` warmup, which is
        // where the binder pins the prefill chunk / auto-q8 KV). See `PlacementPins`.
        let pins = Arc::new(crate::seam::PlacementPins::for_slots(n_slots));
        let scope = crate::seam::PlacementScope::enter(pins.clone());
        let max_ctx = model.vulkan_slot_ctx(&vk, n_slots, want_ctx)?;
        let mtp_heads = if model.engine_cfg().spec.mtp && model.config().qwen4exp {
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
            let fixed =
                crate::mtp::Qwen4MtpSession::load_fixed_vulkan(&vk, sidecar, model.config())?;
            let catch = crate::mtp::Qwen4MtpSession::shared_catch_workspace(
                &vk,
                &fixed,
                max_ctx,
                crate::seam::ubatch_rows(model.engine_cfg()),
            )?;
            let mut heads = Vec::with_capacity(n_slots);
            for _ in 0..n_slots {
                heads.push(MtpSlotState {
                    head: crate::mtp::Qwen4MtpSession::with_fixed_and_catch(
                        &vk,
                        Arc::clone(&fixed),
                        max_ctx,
                        Arc::clone(&catch),
                    )?,
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
        let scheduler_enabled = n_slots > 1 || mtp_heads.is_some();
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
            // Ordinary one-slot serving has no scheduler. MTP needs the gate even with one slot
            // because the request thread hands ownership to the persistent speculative worker.
            gate: scheduler_enabled.then(|| Arc::new(StepGate::new())),
            session_cache: None,
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

    fn init_session_cache(&mut self) -> Result<()> {
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
            return Err(anyhow!(
                "kv.session_cache_dir requires dynamic segmented KV; use Qwen3.5/3.6/3.8 with Q8 KV and leave kv.dynamic enabled"
            ));
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
            session_idle: self.session_idle,
            max_ctx: self.max_ctx,
            pins: Arc::clone(&self.pins),
            mtp_heads: self.mtp_heads.as_ref().map(Arc::clone),
        }
    }

    fn start_scheduler_worker(&mut self) -> Result<()> {
        if self.gate.is_none() || !self.model.config().qwen4exp {
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
        if self.session_cache.is_some() {
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
                        let mtp_head = heads.as_ref().map(|heads| &heads[target].head);
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
                        crate::stream_token(self.model.tokenizer(), acc, printed, id, on_piece);
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

    fn prepare_unified_work(&self, work: &mut BatchWork, req: &RequestCtx) -> Result<()> {
        if work.phase != BatchPhase::Unprepared {
            return Ok(());
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
            "prepared work for the unified Qwen3.8 scheduler"
        );
        Ok(())
    }

    fn run_long_prefill_phase(&self, active: &mut Vec<BatchWork>, req: &RequestCtx) -> Result<()> {
        let mut long = Vec::new();
        let mut token_work = Vec::with_capacity(active.len());
        for work in active.drain(..) {
            if work.phase == BatchPhase::LongPrefill {
                long.push(work);
            } else {
                token_work.push(work);
            }
        }
        *active = token_work;
        if long.is_empty() {
            return Ok(());
        }

        let prompts = long
            .iter()
            .map(|work| work.prompt[..work.prompt_end].to_vec())
            .collect::<Vec<_>>();
        let prepared = long
            .iter()
            .map(|work| crate::seam::PreparedParallelPrompt {
                start: work.prefill_start,
                checkpoint_boundaries: work.checkpoint_boundaries,
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
            .any(|work| work.kv().cached_len() + 1 != work.prompt_end)
        {
            active.extend(long);
            return Err(anyhow!(
                "parallel long prefill stopped before every lane reached its decode frontier"
            ));
        }
        for (work, stats) in long.iter_mut().zip(stats) {
            work.stats.prompt_secs += stats.prompt_secs;
            work.phase = BatchPhase::Decode;
            work.checkpoint_boundaries = [None; crate::seam::TURN_CHECKPOINT_COUNT];
            let progress = infr_core::GenerationProgress {
                phase: infr_core::GenerationPhase::Prefill,
                prompt_tokens: work.prompt_end as u64,
                cached_prompt_tokens: work.prefill_start as u64,
                prefill_tokens: work.prompt_end.saturating_sub(work.prefill_start) as u64,
                completion_tokens: 0,
                context_tokens: work.prompt_end.saturating_sub(1) as u64,
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
        let mut stream = |lane: usize, id: u32| {
            streamed[lane] += 1;
            let (prompt_tokens, cached_prompt_tokens, generated) = progress_bases[lane];
            let progress = self.batch_decode_progress(
                prompt_tokens,
                cached_prompt_tokens,
                generated.saturating_add(streamed[lane]),
            );
            let keep_going =
                channels
                    .get_mut(lane)
                    .and_then(Option::as_mut)
                    .is_some_and(|channels| {
                        channels
                            .events
                            .send(BatchEvent::Token { id, progress })
                            .is_ok()
                            && channels.acknowledgements.recv().unwrap_or(false)
                    });
            accepted[lane] = keep_going;
            keep_going
        };
        let group_req = match &self.gate {
            Some(gate) => {
                RequestCtx::with_gate(active[indices[0]].sampling.clone(), Arc::clone(gate))
            }
            None => RequestCtx::new(active[indices[0]].sampling.clone()),
        };
        let result = crate::seam::generate_dense_vulkan_parallel_sampled_session(
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
            &mut stream,
            Some(&self.batch_interrupt),
            Some(&group_req),
            sync_mtp,
        );

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
        drop(mrope_plans);
        for (&index, plan) in indices.iter().zip(owned_mrope_plans.drain(..)) {
            active[index].mrope_plan = plan;
        }

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
            work.stats.prompt_secs += prompt_secs[lane];
            work.stats.decode_secs += decode_secs[lane];
            work.generated += outputs[lane].len();
            work.stats.n_gen += outputs[lane].len();
            work.prompt.extend_from_slice(&outputs[lane]);
            let cached_len = work.kv().cached_len();
            for boundary in &mut work.checkpoint_boundaries {
                if boundary.is_some_and(|boundary| cached_len >= boundary) {
                    *boundary = None;
                }
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

    fn fail_unified_scheduler(&self, active: &mut Vec<BatchWork>, error: anyhow::Error) {
        let message = error.to_string();
        if let Some(heads) = &self.mtp_heads {
            let mut heads = heads.lock().expect("MTP heads poisoned");
            for work in active.iter() {
                heads[work.slot].invalidate_live();
            }
        }
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
        let h_width = cfg.hc_mult * cfg.n_embd;
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
        let chunk = crate::seam::ubatch_rows(ec).max(1);
        let checkpoint_boundaries = prepared.checkpoint_boundaries;
        let mut chunk_start = start;
        while chunk_start < prompt.len() {
            let chunk_end =
                mtp_prime_chunk_end(chunk_start, prompt.len(), chunk, &checkpoint_boundaries);
            let rows = chunk_end - chunk_start;
            let (mut pending, mut pending_logits, hidden) =
                crate::mtp::run_qwen4_prime_frontier_with_finish(
                    self.vk.as_ref(),
                    &*bind,
                    self.model.gguf(),
                    cfg,
                    ec,
                    self.model.embd(),
                    &prompt[..chunk_end],
                    &mut work.kv,
                    self.max_ctx,
                    work.mrope_plan.as_ref(),
                    finish.as_deref(),
                )?;
            if chunk_end == prompt.len()
                && work
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
                pending = crate::mtp::argmax_row(&pending_logits);
                tracing::warn!(
                    slot = work.slot,
                    blocked_token = blocked,
                    replacement_token = pending,
                    "Qwen3.8 MTP suppressed premature EOS inside an open reasoning block"
                );
            }
            anyhow::ensure!(
                hidden.len() == rows * h_width,
                "MTP prime returned {} hidden values for {rows} rows",
                hidden.len()
            );
            let mut shifted = vec![0.0f32; rows * h_width];
            shifted[..h_width].copy_from_slice(&lane.last_h);
            if rows > 1 {
                shifted[h_width..].copy_from_slice(&hidden[..(rows - 1) * h_width]);
            }
            lane.head.catch_up(
                self.vk.as_ref(),
                &prompt[chunk_start..chunk_end],
                &shifted,
                chunk_start,
                work.mrope_plan.as_ref(),
                work.kv().mtp_shared_weights(),
            )?;
            for (index, boundary) in checkpoint_boundaries.iter().copied().enumerate() {
                let Some(boundary) =
                    boundary.filter(|&boundary| boundary > chunk_start && boundary <= chunk_end)
                else {
                    continue;
                };
                let hidden_row = boundary - chunk_start - 1;
                work.kv
                    .as_mut()
                    .expect("MTP work has target KV")
                    .capture_turn_recurrent(self.vk.as_ref(), cfg, index, &prompt[..boundary])?;
                lane.turn_checkpoints[index] = Some(crate::mtp::Qwen4MtpCheckpointState {
                    tokens: prompt[..boundary].to_vec(),
                    last_h: hidden[hidden_row * h_width..(hidden_row + 1) * h_width].to_vec(),
                });
                if ec.debug.state_trace {
                    let recurrent = work
                        .kv()
                        .session_state_meta()
                        .checkpoint_tokens
                        .map(|tokens| tokens.map(|tokens| tokens.len()));
                    tracing::warn!(
                        "[MTP state trace] slot={} captured_head_checkpoint={} tokens={} recurrent_checkpoints={:?}",
                        work.slot,
                        index,
                        boundary,
                        recurrent,
                    );
                }
            }
            lane.last_h.copy_from_slice(&hidden[(rows - 1) * h_width..]);
            lane.pending_id = Some(pending);
            lane.pending_emitted = false;
            lane.cached
                .extend_from_slice(&prompt[chunk_start..chunk_end]);
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
            chunk_start = chunk_end;
        }
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
        let mut concurrent_plain = false;
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
                let was_unprepared = work.phase == BatchPhase::Unprepared;
                if let Err(error) = self.prime_mtp_work(work, req) {
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
            if active.len() > 1 {
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
                .filter(|&index| {
                    active[index].kv().cached_len() + crate::mtp::DRAFT_TOKENS > self.max_ctx
                })
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
            if let Err(error) = self.run_mtp_verify_cycle(&mut active, &indices, req) {
                self.fail_unified_scheduler(&mut active, error);
                return;
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
    ) -> Result<()> {
        let cfg = self.model.config();
        let ec = self.model.engine_cfg();
        let h_width = cfg.hc_mult * cfg.n_embd;
        let rows = crate::mtp::DRAFT_TOKENS;
        self.refresh_mtp_snapshots(active, indices)?;
        let t0 = Instant::now();
        let (feeds, predictions) = {
            let _gate = req.gate_pass();
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
                let pending = lane.pending_id.expect("primed MTP lane has a frontier");
                let physical_position = work.kv().cached_len();
                let rope_position =
                    multimodal_token_position(work.mrope_plan.as_ref(), physical_position)?;
                let candidates = lane.head.draft(
                    self.vk.as_ref(),
                    pending,
                    &lane.last_h,
                    physical_position,
                    rope_position,
                    rows,
                    work.kv().mtp_shared_weights(),
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

        let (mut verify_ids, mut verify_logits, verify_hidden) = {
            let _gate = req.gate_pass();
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
        anyhow::ensure!(
            verify_ids.len() == indices.len()
                && verify_logits.len() == indices.len()
                && verify_hidden.len() == indices.len(),
            "parallel MTP VERIFY returned inconsistent lane counts"
        );
        let compute_secs = t0.elapsed().as_secs_f64();

        for (lane_index, &index) in indices.iter().enumerate() {
            let work = &mut active[index];
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
                ids[row] = crate::mtp::argmax_row(row_logits);
                tracing::warn!(
                    slot = work.slot,
                    row,
                    blocked_token = blocked,
                    replacement_token = ids[row],
                    "Qwen3.8 MTP suppressed premature EOS inside an open reasoning block"
                );
            }
            let accepted = mtp_accepted_rows(&predictions[lane_index], ids);
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
            let restore_t0 = Instant::now();
            {
                let _gate = req.gate_pass();
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
            work.prompt.extend_from_slice(
                &feeds[lane_index][old_len + already_emitted..old_len + committed],
            );
            for &token in &feeds[lane_index][old_len + already_emitted..old_len + committed] {
                work.reasoning_guard.observe(cfg, token);
            }
            let emitted = committed - already_emitted;
            work.generated += emitted;
            work.stats.n_gen += emitted;
            work.stats.decode_secs += compute_secs + restore_t0.elapsed().as_secs_f64();
            work.finished = stopped || work.generated >= work.max_new;
        }
        Ok(())
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

            // QSA attention remains lane-specific, but the surrounding projections and MoE are
            // shared across dense- and sparse-QSA rows. Keep the whole token cohort together;
            // `run_token_group` still stops exactly at the next dense-to-sparse boundary so the
            // following graph is rebuilt with the correct per-lane QSA geometry.
            let mut indices = (0..active.len()).collect::<Vec<_>>();
            // The primary lane decides whether the shared graph carries MRoPE history.
            sort_token_lane_indices(&active, &mut indices);
            if let Err(error) = self.run_token_group(&mut active, &indices, usize::MAX, false) {
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
        mut on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        let mtp_eligible = self.mtp_heads.as_ref().is_none_or(|_| {
            let base = crate::sampling::Sampler::from_cfg(&self.model.engine_cfg().sampling);
            let effective =
                crate::sampling::Sampler::resolve(Some(req), &self.model.engine_cfg().sampling);
            (base.temp <= 0.0 || base.top_k == 1)
                && (effective.temp <= 0.0 || effective.top_k == 1)
                && !req.sampling().penalties_active()
        });
        let initially_eligible = self.model.config().qwen4exp
            && constraint.is_none()
            && self.gate.is_some()
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
            if let Some(heads) = &self.mtp_heads {
                let mut heads = heads.lock().expect("MTP heads poisoned");
                heads[guard.idx].invalidate_live();
            }
            let (_, stats) = crate::seam::generate_dense_vulkan_session(
                self.vk.as_ref(),
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
                None,
            )?;
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
            prompt: prompt_tokens,
            max_new,
            generated: 0,
            stats: GenStats::default(),
            phase: BatchPhase::Unprepared,
            prefill_start: 0,
            checkpoint_boundaries: [None; crate::seam::TURN_CHECKPOINT_COUNT],
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
        )
    }

    /// Generate one Qwen vision turn (qwen4exp, or the qwen35/qwen35moe hybrid family).
    /// Projector execution is completed by the caller before entering here, so its
    /// request-scoped weights have already returned to the unified arena.
    pub fn generate_multimodal_turn(
        &self,
        prompt: &str,
        stable_prefix: Option<&str>,
        images: Vec<MultimodalEmbedding>,
        max_new: usize,
        req: &RequestCtx,
        mut on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        // qwen35 (dense) and qwen35moe share qwen4exp's M-RoPE / hybrid-attention skeleton, and
        // the Vulkan `QkNormMrope` kernel is written for exactly their geometry (partial rope
        // `rope_dim=64 < hd=256`, interleaved q+gate via `x_stride`). Any other arch carries no
        // `rope.dimension_sections` and is rejected by the empty-sections check downstream.
        if !(self.model.config().qwen35 || self.model.config().qwen4exp) {
            return Err(anyhow!(
                "vision text integration requires a Qwen3.5/3.6 hybrid or Qwen3.8 model"
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
                prompt: prompt_tokens,
                prompt_end: plan.prompt_pos4.len() / 4,
                max_new,
                generated: 0,
                stats: GenStats::default(),
                phase: BatchPhase::Unprepared,
                prefill_start: 0,
                checkpoint_boundaries: [None; crate::seam::TURN_CHECKPOINT_COUNT],
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
        //
        // The predicate is the WORKER HANDLE, not `gate.is_some()`: `--parallel N>1` allocates the
        // gate for every arch, but `start_scheduler_worker` starts a thread only for qwen4exp (its
        // own parallel prefill additionally refuses multimodal position rows). Handing a batch to a
        // queue that no thread drains blocks `wait_for_decode_batch` forever — the event channel's
        // sender stays alive inside the queued `BatchWork`, so it never observes a disconnect. A
        // qwen35moe image turn under `--parallel N>1` would therefore HANG rather than error, which
        // is why this checks for the worker itself.
        let handoff = should_handoff_multimodal_to_scheduler(
            self.scheduler_worker.is_some(),
            self.mtp_heads.is_some(),
            frontier_is_image,
            max_new,
        );
        let legacy_max_new = if handoff { 0 } else { max_new };
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
            if let Some(kv) = guard.kv.as_mut() {
                kv.reset();
            }
        }
        let (legacy_ids, stats) = result?;
        if !handoff || crate::sampling::abort_requested(Some(req)) {
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
            prompt: prompt_tokens,
            prompt_end,
            max_new,
            generated,
            stats,
            phase: phase_for_remaining_prefill(remaining_prefill),
            prefill_start,
            checkpoint_boundaries: [None; crate::seam::TURN_CHECKPOINT_COUNT],
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
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        expand_multimodal_prompt, mtp_accepted_rows, mtp_prime_chunk_end, multimodal_key,
        multimodal_token_position, phase_for_remaining_prefill, pick_continuation, scheduler_mode,
        should_handoff_multimodal_to_scheduler, token_lane_sort_key, BatchPhase,
        MultimodalEmbedding, SchedulerMode, SHORT_PREFILL_TOKENS,
    };
    use std::sync::Arc;

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
    fn mtp_accepts_each_lane_independently() {
        let candidates = [11, 12, 13];
        assert_eq!(mtp_accepted_rows(&candidates, &[99, 0, 0, 0]), 1);
        assert_eq!(mtp_accepted_rows(&candidates, &[11, 99, 0, 0]), 2);
        assert_eq!(mtp_accepted_rows(&candidates, &[11, 12, 99, 0]), 3);
        assert_eq!(mtp_accepted_rows(&candidates, &[11, 12, 13, 99]), 4);
        assert_eq!(mtp_accepted_rows(&[5, 6, 7], &[5, 6, 7, 8]), 4);
    }

    #[test]
    fn mtp_prime_stops_exactly_at_each_turn_checkpoint() {
        let checkpoints = [Some(789), Some(2405)];
        assert_eq!(mtp_prime_chunk_end(0, 4064, 2048, &checkpoints), 789);
        assert_eq!(mtp_prime_chunk_end(789, 4064, 2048, &checkpoints), 2405);
        assert_eq!(mtp_prime_chunk_end(2405, 4064, 2048, &checkpoints), 4064);
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
