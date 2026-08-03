//! Generic, backend-agnostic block-paging bookkeeping: a fixed-slot LRU cache mapping an opaque
//! `BlockId` to a slot index, for any backend that wants to keep a working set of uniform-ish
//! byte blocks resident in a budget smaller than the full set.
//!
//! This module holds ONLY the host-side residency/eviction *decision* (no GPU types, no bytes) —
//! [`Pager::touch`] is pure bookkeeping, cheap to unit test without a device. A backend wraps this
//! with the actual VRAM arena + device LUT buffer + upload machinery (see `infr-vulkan`'s
//! `GpuPager`) and drives it by calling `touch` for every block a dispatch is about to read,
//! before recording that dispatch.
//!
//! ## Today: MoE expert paging
//! The Vulkan MoE lowering plugs in `BlockId = (layer, role, expert_id)` packed into a `u32` (see
//! `infr-vulkan`'s `pager` module) with a demand/LRU policy — an expert is paged in on first use
//! by a layer and evicted least-recently-used when the arena is full.
//!
//! ## Dense layer streaming
//! The second policy reuses the SAME slot bookkeeping for `BlockId = layer_idx` (or a per-tensor
//! block within a layer) in a dense model whose weights don't fit VRAM: a dense forward visits
//! layers in a fixed, known order every step, so residency is schedule-driven via
//! [`Pager::schedule`] (exact cyclic-sweep eviction, Belady-parity — see its doc) rather than
//! demand/LRU. The PREFETCH side (staging layer `l+1`'s bytes while `l` executes) is the
//! backend's job — `infr-vulkan`'s `DensePagerSession` records uploads ahead of GPU execution
//! through its pipelined ring; this module stays pure residency/eviction bookkeeping.
use std::collections::{HashMap, VecDeque};

/// Opaque identifier for one pageable block. The pager never interprets this — callers pack
/// whatever key space they need (an expert id, a `(layer, role, expert)` tuple, a layer index for
/// the planned dense-streaming policy, ...) into it.
pub type BlockId = u32;

/// Sentinel LUT value meaning "not resident" — mirrors what a device-side LUT buffer should hold
/// for any block the pager hasn't (yet) admitted, so an accidental stale read is loud (an
/// out-of-range slot index) rather than silently aliasing slot 0.
pub const NOT_RESIDENT: u32 = u32::MAX;

/// Outcome of [`Pager::touch`]: whether the block was already resident (no upload needed) or had
/// to be paged in (possibly evicting another block first).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// Already resident at `slot` — no upload needed, LUT already correct.
    Hit { slot: u32 },
    /// Now resident at `slot` after an upload the caller must perform. `evicted` is the block
    /// that previously occupied `slot`, if any (so the caller can invalidate its LUT entry, or
    /// just leave it — it's never read again unless `evicted` is re-touched, which re-admits it
    /// through the normal miss path).
    Miss { slot: u32, evicted: Option<BlockId> },
}

/// Cumulative pager activity — the `INFR_PAGER_STATS` hit-rate counter the task's validation step
/// asks for rides this.
#[derive(Debug, Clone, Copy, Default)]
pub struct PagerStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl PagerStats {
    /// Hit rate over all `touch` calls so far; `1.0` (vacuously) before any calls.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            1.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Fixed-`n_slots` LRU residency map. `n_slots` slots are handed out first-come (no eviction)
/// until exhausted, then every further miss evicts the least-recently-touched resident block.
///
/// # Within-batch safety
/// A caller MUST resolve every block a single dispatch batch needs (e.g. all `n_used` experts a
/// decode step's top-k picked, or every distinct expert a prefill ubatch's bucket counts named)
/// via `touch` BEFORE recording/dispatching anything that reads the result — and must do so as one
/// uninterrupted sequence of `touch` calls. `touch` marks a block most-recently-used the instant
/// it resolves, and eviction only ever pops the LEAST-recently-used entry, so a block touched
/// earlier in the same batch can never be evicted by a later touch in that SAME batch. This only
/// holds if `n_slots >= ` the number of DISTINCT blocks one batch touches — [`Pager::new`]'s doc
/// repeats this invariant; violating it (a cache budget too small to even hold one batch) is a
/// configuration error the caller should surface, not silently thrash.
pub struct Pager {
    n_slots: usize,
    /// block_id -> slot index for every currently-resident block.
    resident: HashMap<BlockId, u32>,
    /// LRU order, oldest (least-recently-used) at the front. A block appears at most once;
    /// `touch` on an existing entry removes-then-repushes it (O(n_slots), fine at the slot counts
    /// this cache runs at — tens to low hundreds; an intrusive doubly-linked list is the upgrade
    /// path if that ever isn't true, per the module doc's SLRU note).
    lru: VecDeque<BlockId>,
    /// Slot indices never yet handed out (drained before any eviction kicks in).
    free: Vec<u32>,
    /// block_id -> the [`Self::begin_batch`] epoch of its last touch. Eviction skips blocks
    /// touched in the CURRENT batch — the explicit form of the within-batch safety invariant
    /// (MRU ordering used to carry it implicitly; the scan-resistant [`Self::touch_cold`] path
    /// inserts at the COLD end, so it needs the epoch guard instead).
    epoch: HashMap<BlockId, u64>,
    cur_epoch: u64,
    /// block_id -> outstanding pin count, for blocks a caller is READING right now
    /// ([`Self::resolve_and_pin`]). Only present while non-zero. A pinned block is never an
    /// eviction victim: the epoch guard above cannot express this, because it is scoped to one
    /// dispatch batch while a pin lasts as long as the borrow — a CPU kernel reads a weight for a
    /// whole op, spanning batches.
    pinned: HashMap<BlockId, u32>,
    stats: PagerStats,
}

/// Where a miss inserts into the LRU order — the two policies this pager already had, named so a
/// caller can pick one at the call site instead of through three near-identical entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Insert {
    /// Most-recently-used end: recency (`touch`). Right when reuse follows use.
    Mru,
    /// Cold end: scan-resistant / cyclic-sweep (`touch_cold`, `schedule`). Right when the block
    /// just used is the one whose next use is furthest away.
    Cold,
}

impl Pager {
    /// A pager with `n_slots` uniform slots and nothing resident. `n_slots` must be at least the
    /// largest number of distinct blocks any single dispatch batch will `touch` (see the
    /// within-batch safety note) — the pager can't check that itself (it doesn't know about
    /// batches), so a caller sizing `n_slots` from a VRAM budget must clamp it to at least that
    /// floor and error out earlier if the budget can't cover even one batch.
    ///
    /// # Panics
    /// If `n_slots == 0`. A zero-slot pager cannot satisfy even a one-block batch, so there is no
    /// correct behaviour to fall back to: the first `touch` would find `free` empty, scan an empty
    /// `lru` for a victim, and die inside [`Self::take_slot`] on an `expect` about a `position()`
    /// index — a panic that describes the wrong thing entirely, several frames away from the
    /// mistake. Clamping to 1 instead would be worse than either: it manufactures a pager that
    /// silently misses on EVERY touch and evicts the block the caller resolved a moment ago,
    /// violating the within-batch safety invariant above for any batch wider than one block —
    /// precisely the coherent-but-wrong failure this module refuses elsewhere (see `take_slot`'s
    /// batch-overflow assert). The contract quoted above already makes the FLOOR the caller's job,
    /// so a 0 here is a caller bug that must surface at the mistake. The one in-tree caller
    /// (`infr-vulkan`'s `validate_pager_dims`) already rejects a 0-slot budget with a clean `Err`
    /// before constructing anything, so this can only fire for a NEW caller that skipped that step
    /// — never on a shipping path. `new` also runs once per pool at model load, so a check here
    /// costs nothing measurable.
    pub fn new(n_slots: usize) -> Self {
        assert!(
            n_slots >= 1,
            "pager: n_slots must be at least 1 (a zero-slot pager can never satisfy a touch) — \
             size the slot count from the paging budget and reject a budget that can't hold one \
             dispatch batch BEFORE constructing the pager"
        );
        Self {
            n_slots,
            resident: HashMap::with_capacity(n_slots),
            lru: VecDeque::with_capacity(n_slots),
            free: (0..n_slots as u32).rev().collect(), // pop() hands out slot 0 first
            epoch: HashMap::with_capacity(n_slots),
            cur_epoch: 0,
            pinned: HashMap::new(),
            stats: PagerStats::default(),
        }
    }

    /// Open a new touch batch (one dispatch's worth of `touch`/`touch_cold` calls). Blocks
    /// touched under the current epoch are never eviction victims, which makes the within-batch
    /// safety invariant explicit for BOTH insertion orders (classic MRU and `touch_cold`'s
    /// cold-end scan inserts, where LRU position alone no longer protects batch siblings).
    pub fn begin_batch(&mut self) {
        self.cur_epoch += 1;
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn stats(&self) -> PagerStats {
        self.stats
    }

    /// Number of currently-resident blocks (== `n_slots` once the cache is full).
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Number of entries in the `epoch` map — test-only, to assert eviction keeps it bounded by
    /// `n_slots` rather than growing per distinct BlockId ever touched.
    #[cfg(test)]
    pub(crate) fn epoch_len(&self) -> usize {
        self.epoch.len()
    }

    pub fn slot_of(&self, id: BlockId) -> Option<u32> {
        self.resident.get(&id).copied()
    }

    /// Move `id` to the most-recently-used end without changing residency (internal to `touch`,
    /// exposed for the doubly-recorded case: a batch that touches the same id twice).
    fn mark_mru(&mut self, id: BlockId) {
        if let Some(pos) = self.lru.iter().position(|&x| x == id) {
            self.lru.remove(pos);
        }
        self.lru.push_back(id);
    }

    /// Ensure `id` is resident, evicting the least-recently-used unprotected block if the cache
    /// is full. Returns whether it was already resident (caller skips the upload) or had to be
    /// paged in (caller must upload the block's bytes into `slot` and write the device LUT entry
    /// `id -> slot` before any dispatch reads it).
    pub fn touch(&mut self, id: BlockId) -> Resolution {
        self.epoch.insert(id, self.cur_epoch);
        if let Some(&slot) = self.resident.get(&id) {
            self.stats.hits += 1;
            self.mark_mru(id);
            return Resolution::Hit { slot };
        }
        self.stats.misses += 1;
        let (slot, evicted) = self.take_slot();
        self.resident.insert(id, slot);
        self.lru.push_back(id);
        Resolution::Miss { slot, evicted }
    }

    /// Ensure `id` is resident AND pin it, so it cannot be evicted until [`Self::unpin`].
    ///
    /// This is the entry point for a tier whose slots are read DIRECTLY (the host DRAM arena: a
    /// CPU kernel dereferences the slot for a whole op, and a staging copy reads it until the copy
    /// is recorded), as opposed to the VRAM pagers, whose callers upload a slot's bytes and then
    /// let residency ride the batch epoch alone.
    ///
    /// `None` means every resident block is protected right now — all pinned, or all touched by
    /// the current batch — so there is no slot to give. Nothing is evicted, nothing is inserted and
    /// no counter moves in that case: the caller has to release pins (or was sized below its own
    /// working set, which is a configuration error it should report). This is a `None` and not a
    /// panic precisely because it IS reachable at runtime, unlike the unpinned callers' floor.
    ///
    /// On a `Miss` the caller must fill the slot before reading it; on a `Hit` it is already valid.
    /// Either way the block comes back pinned exactly once.
    pub fn resolve_and_pin(&mut self, id: BlockId, insert: Insert) -> Option<Resolution> {
        if let Some(&slot) = self.resident.get(&id) {
            self.epoch.insert(id, self.cur_epoch);
            self.stats.hits += 1;
            if insert == Insert::Mru {
                self.mark_mru(id);
            }
            *self.pinned.entry(id).or_insert(0) += 1;
            return Some(Resolution::Hit { slot });
        }
        let (slot, evicted) = self.take_slot_opt()?;
        self.epoch.insert(id, self.cur_epoch);
        self.stats.misses += 1;
        self.resident.insert(id, slot);
        match insert {
            Insert::Mru => self.lru.push_back(id),
            Insert::Cold => self.lru.push_front(id),
        }
        *self.pinned.entry(id).or_insert(0) += 1;
        Some(Resolution::Miss { slot, evicted })
    }

    /// Pin `id` only if it is ALREADY resident — never reads, never evicts, never inserts.
    ///
    /// The hit-only probe a tier above this one uses to decide between "copy from here" and "go to
    /// the tier below". A hit counts as a hit; a miss counts as nothing, because no residency
    /// decision was made.
    pub fn pin_if_resident(&mut self, id: BlockId) -> Option<u32> {
        let slot = *self.resident.get(&id)?;
        self.epoch.insert(id, self.cur_epoch);
        self.stats.hits += 1;
        *self.pinned.entry(id).or_insert(0) += 1;
        Some(slot)
    }

    /// Release one pin taken by [`Self::resolve_and_pin`] / [`Self::pin_if_resident`].
    ///
    /// # Panics
    /// If `id` holds no pin. An unbalanced release is not a state to recover from: it would let a
    /// slot be evicted while a reader still holds bytes from it, which is silent wrong output
    /// rather than a failure. The RAII guard in the host tier is what makes this balanced.
    pub fn unpin(&mut self, id: BlockId) {
        match self.pinned.get_mut(&id) {
            Some(n) if *n > 1 => *n -= 1,
            Some(_) => {
                self.pinned.remove(&id);
            }
            None => panic!("pager: unpin of block {id}, which holds no pin"),
        }
    }

    /// How many distinct blocks are pinned right now — the number a caller sizes `n_slots`
    /// against (see [`Self::resolve_and_pin`]'s `None`).
    pub fn pinned_blocks(&self) -> usize {
        self.pinned.len()
    }

    /// Schedule-driven residency for a DETERMINISTIC cyclic sweep — the dense layer-streaming
    /// policy the module doc names (`BlockId = layer`, every forward pass visits blocks in the
    /// same fixed order). This is NOT demand/LRU: under a cyclic sweep the block whose next use
    /// is FURTHEST away is precisely the one the sweep just passed, so a miss inserts at the COLD
    /// end (it won't be needed again until every other block has been visited) and a hit keeps
    /// its position (promotion would let the sweep evict the stable set). The cache converges to
    /// a stable resident prefix of `n_slots - 1` blocks re-hit every pass plus one churn slot the
    /// remaining blocks cycle through — the same hit count Belady's offline-optimal policy
    /// achieves on a cyclic sweep (see `schedule_cyclic_sweep_is_belady_optimal`).
    ///
    /// Mechanically this shares [`Self::touch_cold`]'s cold-end insertion; it is a separate entry
    /// point because the CONTRACT differs: `touch_cold` promises scan-resistance for full-set
    /// sweeps MIXED with recency (`touch`) traffic on the same pager, while `schedule` promises
    /// Belady-parity for a pager driven EXCLUSIVELY by one fixed cyclic order (dense layer
    /// streaming never mixes policies on a pool). Within-batch safety still rides the batch
    /// epoch ([`Self::begin_batch`]), not insertion order.
    pub fn schedule(&mut self, id: BlockId) -> Resolution {
        self.touch_cold(id)
    }

    /// [`Self::touch`]'s SCAN-RESISTANT twin, for full-set sweeps (a batched-prefill layer
    /// pre-ensuring all `n_expert` experts, layer 0..n cyclically): hits keep their LRU position
    /// (no MRU promotion) and misses insert at the COLD end. Under plain LRU a cyclic sweep
    /// larger than the cache is the pathological worst case — every block evicted exactly before
    /// its next use, 0% steady-state reuse (measured on Scout pp512: 768/768 blocks re-uploaded
    /// per rep). Cold insertion makes the sweep churn a single victim region instead and
    /// converges to a stable resident prefix of ~`n_slots` blocks — near the offline-optimal
    /// hit count for a repeated sweep — while recency-driven [`Self::touch`] users (decode's
    /// routed-only path) keep their hot set at the MRU end, out of the sweep's victim zone.
    /// Within-batch safety comes from the batch epoch (see [`Self::begin_batch`]), NOT ordering.
    pub fn touch_cold(&mut self, id: BlockId) -> Resolution {
        self.epoch.insert(id, self.cur_epoch);
        if let Some(&slot) = self.resident.get(&id) {
            self.stats.hits += 1;
            return Resolution::Hit { slot };
        }
        self.stats.misses += 1;
        let (slot, evicted) = self.take_slot();
        self.resident.insert(id, slot);
        self.lru.push_front(id);
        Resolution::Miss { slot, evicted }
    }

    /// A free slot, evicting if none remain, or `None` when every resident block is protected
    /// (pinned, or touched by the current batch). [`Self::take_slot`] is the infallible form the
    /// unpinned callers use.
    fn take_slot_opt(&mut self) -> Option<(u32, Option<BlockId>)> {
        if let Some(s) = self.free.pop() {
            return Some((s, None));
        }
        // A block is protected by its PIN always, and by the batch epoch only once batches are in
        // use: a caller that never opens one leaves `cur_epoch` at 0, which every block's default
        // epoch also is, so an unconditional epoch test would mark the whole cache as in-batch and
        // find no victim at all. (The infallible `take_slot` expresses the same exemption as its
        // `cur_epoch == 0` assert.)
        let batches = self.cur_epoch != 0;
        let idx = self.lru.iter().position(|b| {
            !(batches && self.epoch.get(b) == Some(&self.cur_epoch)) && !self.pinned.contains_key(b)
        })?;
        let victim = self.lru.remove(idx).expect("index from position()");
        let vslot = self
            .resident
            .remove(&victim)
            .expect("every lru entry has a resident mapping");
        // Drop the victim's epoch entry too — otherwise `epoch` is the one pager structure not
        // bounded by `n_slots`, accumulating a stale entry per distinct BlockId ever touched. A
        // stale entry could also mask an id as "current batch" across a `cur_epoch` wraparound.
        self.epoch.remove(&victim);
        self.stats.evictions += 1;
        Some((vslot, Some(victim)))
    }

    /// A free slot, evicting if none remain. The victim is the coldest block NOT touched in the
    /// current batch (epoch guard — see [`Self::begin_batch`]); legacy callers that never open a
    /// batch share epoch 0 everywhere, for which the guard degrades to plain LRU `pop_front`.
    fn take_slot(&mut self) -> (u32, Option<BlockId>) {
        if self.free.is_empty() {
            // Only reachable through the infallible entry points, whose callers hold no pins.
            debug_assert!(
                self.pinned.is_empty(),
                "pins are for the fallible resolve_and_pin path; touch/schedule cannot see them"
            );
        }
        if let Some(s) = self.free.pop() {
            return (s, None);
        }
        let idx = self
            .lru
            .iter()
            .position(|b| self.epoch.get(b) != Some(&self.cur_epoch))
            // Every resident block touched by the CURRENT batch means the batch exceeded
            // n_slots — the sizing floor (`Pager::new`'s doc) was violated upstream. Falling
            // back to pop_front would silently evict a batch sibling mid-flight (the
            // coherent-but-wrong class); fail loudly instead when batches are in use.
            .unwrap_or_else(|| {
                // Actionable in the voice of the Vulkan VRAM guard (`check_vram_budget`): say what
                // was asked for, say why it cannot be tolerated rather than degraded, and name the
                // knob to turn. The knob is `INFR_CACHE` — the paging budget (`paging.cache`) the
                // seam splits proportionally into per-pool slot counts, so raising it is what makes
                // `n_slots` clear the batch's width. NOT reachable on a default configuration: the
                // MoE seam floors every pool at `min(n_expert, n_blocks)` slots (a batch is one
                // (layer, role) resolution, at most `n_expert` distinct experts) and the dense
                // streaming path opens a fresh batch around each single `schedule` call, so both
                // shipping callers satisfy the floor by construction — reaching this means an
                // explicit `INFR_CACHE` too small for one dispatch, or a new caller that skipped
                // the floor.
                assert!(
                    self.cur_epoch == 0,
                    "pager: a single dispatch batch touched more distinct blocks than the \
                     n_slots={} this cache holds — within-batch eviction safety is unsatisfiable. \
                     Refusing to degrade: evicting a block this same batch already resolved \
                     doesn't fail cleanly, it feeds the dispatch another block's bytes (silent \
                     wrong output, not an error). Raise the paging budget (INFR_CACHE) so every \
                     pool gets at least one slot per block a batch touches, or leave it unset to \
                     take the auto budget, which is sized to that floor.",
                    self.n_slots
                );
                0 // epoch never used (legacy caller): plain LRU
            });
        let victim = self.lru.remove(idx).expect("index from position()");
        let vslot = self
            .resident
            .remove(&victim)
            .expect("every lru entry has a resident mapping");
        // Drop the victim's epoch entry too — otherwise `epoch` is the one pager structure not
        // bounded by `n_slots`, accumulating a stale entry per distinct BlockId ever touched. A
        // stale entry could also mask an id as "current batch" across a `cur_epoch` wraparound.
        self.epoch.remove(&victim);
        self.stats.evictions += 1;
        (vslot, Some(victim))
    }
}

/// Upload-ring sizing policy: the configured ring size (`paging.ring`, from `INFR_PAGER_RING` or a
/// config file, in the shared size grammar) wins; otherwise an eighth of the pager budget, clamped
/// to [256 MiB, 2 GiB].
///
/// Bigger halves = fewer pipeline rotations, and each rotation stalls the CPU on the other half's
/// fence — measured on Scout pp512 (miss-heavy steady state, ~22 GB staged/rep): 256 MiB →
/// 224 t/s, 1 GiB → 324, 2 GiB → 404, flat beyond. The budget fraction keeps small explicit
/// `INFR_CACHE` runs from spending most of their grant on staging instead of arena slots.
///
/// Pure budget arithmetic — no device types — so the seam's placement math can price the ring
/// without reaching into a backend crate, and a second paging backend gets the same policy rather
/// than a fourth copy of the clamp. The staging ring ITSELF (the buffers, the fences, the half
/// rotation) stays per-backend.
pub fn ring_bytes(pager_budget: u64, ring: Option<crate::SizeSpec>) -> usize {
    const MIB: u64 = 1024 * 1024;
    // A zero (or absent) override can never be honoured — a 0-byte ring means "never stage a
    // slot" — so it falls through to the budget fraction, exactly as an unparseable value does.
    if let Some(b) = ring.map(|s| s.resolve(0) as usize).filter(|&b| b > 0) {
        return b;
    }
    (pager_budget / 8).clamp(256 * MIB, 2048 * MIB) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring clamp's boundaries, pinning the pre-hoist inline expression
    /// `(budget / 8).clamp(256 MiB, 2048 MiB)` — the seam subtracts this from the pager budget
    /// before splitting arena shares, so a moved edge silently re-sizes every MoE arena — plus the
    /// override's fall-through rule.
    ///
    /// Driven through [`ring_bytes`] itself, over resolved [`SizeSpec`] values. It used to go
    /// through a `&str` façade (`ring_bytes_from`) so the sweep could avoid the process
    /// environment, which the value-taking function already allows; the SPELLINGS it exercised
    /// (`1g`, `512m`, a bare byte count, and the unparseable `banana` / `1GiB`) are
    /// `crate::parse_size`'s grammar and are pinned by `parse_size_grammar` and by
    /// `config`'s `mib_and_size_string_spellings` — everything unparseable arrives here as `None`.
    #[test]
    fn ring_bytes_clamp_boundaries_and_override_grammar() {
        const MIB: u64 = 1024 * 1024;
        // Below the floor's crossover (8 x 256 MiB = 2 GiB of budget) the floor wins — including
        // the `0` budget the pager passes when it has no budget figure at all.
        assert_eq!(ring_bytes(0, None), 256 * MIB as usize);
        assert_eq!(ring_bytes(2048 * MIB, None), 256 * MIB as usize);
        // Exactly at the crossover, and just past it, the eighth wins.
        assert_eq!(ring_bytes(2048 * MIB + 8, None), (256 * MIB + 1) as usize);
        // At the ceiling's crossover (8 x 2 GiB) and beyond, the ceiling wins.
        assert_eq!(ring_bytes(16 * 1024 * MIB, None), (2048 * MIB) as usize);
        assert_eq!(ring_bytes(64 * 1024 * MIB, None), (2048 * MIB) as usize);

        // An explicit override wins outright; a ZERO one (or an absent/unparseable one, which the
        // config layer turns into `None`) falls through to the policy — a `0` ring would mean
        // "never stage a slot".
        let bytes = crate::SizeSpec::Bytes;
        for (spec, want) in [
            (Some(bytes(1024 * MIB)), Some(1024 * MIB as usize)),
            (Some(bytes(512 * MIB)), Some(512 * MIB as usize)),
            (Some(bytes(3_221_225_472)), Some(3 * 1024 * MIB as usize)),
            (Some(bytes(0)), None),
            (None, None),
        ] {
            let got = ring_bytes(64 * 1024 * MIB, spec);
            match want {
                Some(w) => assert_eq!(got, w, "spec={spec:?}"),
                // Falls through to the policy, which caps at 2 GiB on this budget.
                None => assert_eq!(got, (2048 * MIB) as usize, "spec={spec:?}"),
            }
        }
        // A PERCENT override resolves against a zero base (the ring has no base to scale), so it
        // is zero and falls through too — the same arm the `0` case above takes.
        assert_eq!(
            ring_bytes(64 * 1024 * MIB, Some(crate::SizeSpec::Percent(0.5))),
            (2048 * MIB) as usize
        );
    }

    /// The override as the read sites now take it: a `paging.ring` field off the backend's
    /// `Config`, already through the shared size grammar. `None` (unset, or a value the grammar
    /// rejects) falls through to the budget fraction.
    #[test]
    fn ring_bytes_takes_the_override_from_the_config() {
        use crate::config::Config;
        const MIB: u64 = 1024 * 1024;
        assert_eq!(Config::default().paging.ring, None);
        assert_eq!(
            ring_bytes(64 * 1024 * MIB, Config::default().paging.ring),
            (2048 * MIB) as usize
        );
        for (raw, want) in [
            ("1g", 1024 * MIB as usize),
            ("512m", 512 * MIB as usize),
            ("banana", (2048 * MIB) as usize), // rejected by the grammar ⇒ field stays None
            ("0", (2048 * MIB) as usize),      // a 0-byte ring is not a ring
        ] {
            let layer =
                crate::config::env::parse(&|k| (k == "INFR_PAGER_RING").then(|| raw.to_string()))
                    .expect("env layer");
            let cfg = Config::load_from_layers(&[layer]);
            assert_eq!(
                ring_bytes(64 * 1024 * MIB, cfg.paging.ring),
                want,
                "INFR_PAGER_RING={raw:?}"
            );
        }
    }

    #[test]
    fn fresh_pager_is_all_misses_until_full() {
        let mut p = Pager::new(3);
        assert_eq!(
            p.touch(10),
            Resolution::Miss {
                slot: 0,
                evicted: None
            }
        );
        assert_eq!(
            p.touch(11),
            Resolution::Miss {
                slot: 1,
                evicted: None
            }
        );
        assert_eq!(
            p.touch(12),
            Resolution::Miss {
                slot: 2,
                evicted: None
            }
        );
        assert_eq!(p.resident_count(), 3);
        assert_eq!(p.stats().misses, 3);
        assert_eq!(p.stats().hits, 0);
    }

    #[test]
    fn repeat_touch_is_a_hit_at_the_same_slot() {
        let mut p = Pager::new(2);
        let Resolution::Miss { slot, .. } = p.touch(5) else {
            panic!("expected miss")
        };
        assert_eq!(p.touch(5), Resolution::Hit { slot });
        assert_eq!(p.touch(5), Resolution::Hit { slot });
        assert_eq!(p.stats().hits, 2);
        assert_eq!(p.stats().misses, 1);
    }

    #[test]
    fn eviction_picks_least_recently_used() {
        let mut p = Pager::new(2);
        p.touch(1); // slot 0
        p.touch(2); // slot 1, full now
        p.touch(1); // hit, 1 is now MRU; 2 is LRU
                    // 3 must evict 2 (LRU), not 1 (just touched).
        let Resolution::Miss { slot, evicted } = p.touch(3) else {
            panic!("expected miss")
        };
        assert_eq!(evicted, Some(2));
        assert_eq!(p.slot_of(1), Some(0)); // 1 kept its original slot
        assert_eq!(p.slot_of(3), Some(slot));
        assert_eq!(p.slot_of(2), None);
        assert_eq!(p.stats().evictions, 1);
    }

    #[test]
    fn slot_reuse_after_eviction_is_exact() {
        // The freed slot from an eviction is the ONLY slot available for the next miss (n_slots
        // fixed) — assert the pager actually reuses it rather than growing.
        let mut p = Pager::new(1);
        let Resolution::Miss { slot: s0, .. } = p.touch(1) else {
            panic!()
        };
        let Resolution::Miss {
            slot: s1,
            evicted: e1,
        } = p.touch(2)
        else {
            panic!()
        };
        assert_eq!(s0, s1); // only one slot exists, must be reused
        assert_eq!(e1, Some(1));
        assert_eq!(p.resident_count(), 1);
    }

    #[test]
    fn within_batch_touch_order_protects_earlier_ids_from_later_ones() {
        // Simulates a decode step's top-k = 3 experts touched in a fixed order against a 3-slot
        // cache with 1 already resident from a PRIOR step — the prior step's expert should be the
        // only eviction victim; none of the 3 in-flight ids may evict each other.
        let mut p = Pager::new(3);
        p.touch(100); // prior step's expert, now LRU-oldest
        for id in [7u32, 8, 9] {
            p.touch(id);
        }
        // Cache is now full (100 evicted by whichever of 7/8/9 needed a slot first); crucially,
        // ALL of 7, 8, 9 must be resident simultaneously — that's the within-batch invariant.
        assert!(p.slot_of(7).is_some());
        assert!(p.slot_of(8).is_some());
        assert!(p.slot_of(9).is_some());
        assert_eq!(p.slot_of(100), None);
    }

    #[test]
    fn lut_coherence_across_a_simulated_token_sequence() {
        // Drives a small cache through a token sequence with a repeating expert-access pattern
        // (like a real MoE decode loop) and checks the (block_id -> slot) mapping stays exactly
        // what a host-mirrored LUT array would need at every step.
        let mut p = Pager::new(4);
        let mut lut = vec![NOT_RESIDENT; 16]; // host mirror of the device LUT buffer
        let apply = |p: &mut Pager, lut: &mut [u32], id: BlockId| match p.touch(id) {
            Resolution::Hit { slot } => assert_eq!(lut[id as usize], slot, "hit must match LUT"),
            Resolution::Miss { slot, evicted } => {
                if let Some(e) = evicted {
                    lut[e as usize] = NOT_RESIDENT;
                }
                lut[id as usize] = slot;
            }
        };
        // A token sequence revisiting a hot set {0,1} plus a cold long tail.
        let seq = [0u32, 1, 2, 0, 1, 3, 4, 0, 1, 5, 0, 1, 6, 7, 0, 1];
        for &id in &seq {
            apply(&mut p, &mut lut, id);
        }
        // The hot set must still be resident (never evicted across the whole run) and every
        // resident block's LUT entry must exactly match the pager's own view.
        for id in [0u32, 1] {
            let slot = p.slot_of(id).expect("hot id stays resident");
            assert_eq!(lut[id as usize], slot);
        }
        for id in 0..16u32 {
            match p.slot_of(id) {
                Some(slot) => assert_eq!(lut[id as usize], slot),
                None => assert_eq!(lut[id as usize], NOT_RESIDENT),
            }
        }
        assert!(p.stats().hits > 0);
        assert!(p.stats().evictions > 0);
    }

    #[test]
    fn cold_touch_sweep_converges_to_a_stable_prefix() {
        // A cyclic sweep of 6 blocks through 3 slots: plain LRU would miss EVERY touch from the
        // second pass on (the pathological case touch_cold exists for); cold insertion must
        // instead keep a stable resident prefix and only churn the cold end.
        let mut p = Pager::new(3);
        for _pass in 0..4 {
            for id in 0..6u32 {
                p.begin_batch(); // one "layer" per id in this reduced shape
                p.touch_cold(id);
            }
        }
        // Steady state: passes 2..4 must hit the stable prefix — strictly more than plain LRU's
        // zero. (Exact count: 2 hits/pass here — n_slots-1 stable + 1 churn slot.)
        assert!(
            p.stats().hits >= 6,
            "cold sweep should retain a stable prefix (hits={})",
            p.stats().hits
        );
        // And the prefix blocks are still resident for the next pass.
        assert!(p.slot_of(0).is_some());
        assert!(p.slot_of(1).is_some());
    }

    #[test]
    fn cold_touch_batch_cannot_evict_its_own_siblings() {
        // Within-batch safety under COLD insertion: LRU position no longer protects batch
        // siblings (they sit at the cold end), so the epoch guard must. A 3-slot cache fully
        // occupied by an old batch, then one batch cold-touching 3 fresh blocks: all 3 must be
        // simultaneously resident at the end — no sibling may evict another.
        let mut p = Pager::new(3);
        p.begin_batch();
        for id in [100u32, 101, 102] {
            p.touch_cold(id);
        }
        p.begin_batch();
        for id in [7u32, 8, 9] {
            p.touch_cold(id);
        }
        for id in [7u32, 8, 9] {
            assert!(p.slot_of(id).is_some(), "batch sibling {id} was evicted");
        }
    }

    #[test]
    #[should_panic(expected = "within-batch eviction safety")]
    fn batch_larger_than_cache_fails_loudly() {
        // The sizing-floor violation must panic (silent sibling eviction = the
        // coherent-but-wrong output class), not degrade.
        let mut p = Pager::new(2);
        p.begin_batch();
        for id in 0..3u32 {
            p.touch_cold(id);
        }
    }

    #[test]
    #[should_panic(expected = "n_slots must be at least 1")]
    fn zero_slot_pager_is_rejected_at_construction() {
        // A 0-slot pager used to build fine and then die on the FIRST touch, inside `take_slot`,
        // with `expect("index from position()")` — a message about a `VecDeque` index that says
        // nothing about the actual mistake (a paging budget that couldn't buy a single slot). The
        // guard must fire in the constructor, at the mistake.
        let _ = Pager::new(0);
    }

    #[test]
    fn one_slot_pager_is_the_smallest_legal_one() {
        // The guard's boundary: 1 slot is a legitimate (if maximally thrashy) configuration and
        // must keep working — `slot_reuse_after_eviction_is_exact` relies on it.
        let mut p = Pager::new(1);
        assert_eq!(p.n_slots(), 1);
        assert_eq!(
            p.touch(1),
            Resolution::Miss {
                slot: 0,
                evicted: None
            }
        );
    }

    #[test]
    fn classic_touches_keep_their_hot_set_against_cold_sweeps() {
        // Decode's recency set (classic touch → MRU end) must survive a prefill-style cold
        // sweep, which churns the COLD end only.
        let mut p = Pager::new(4);
        p.begin_batch();
        p.touch(1); // decode-hot
        p.touch(2); // decode-hot
        for id in 10..16u32 {
            p.begin_batch();
            p.touch_cold(id); // sweep bigger than the remaining capacity
        }
        assert!(
            p.slot_of(1).is_some(),
            "hot block 1 evicted by a cold sweep"
        );
        assert!(
            p.slot_of(2).is_some(),
            "hot block 2 evicted by a cold sweep"
        );
    }

    #[test]
    fn schedule_cyclic_sweep_is_belady_optimal() {
        // Dense layer streaming's access shape: N blocks visited 0..N in order, repeated every
        // forward pass, through C < N slots. Belady (evict the furthest-next-use block) yields
        // exactly C-1 hits per steady-state pass on this trace; `schedule` must match it —
        // plain LRU would yield ZERO (every block evicted right before its reuse).
        let (n_blocks, n_slots, passes) = (10u32, 4usize, 5u64);
        let mut p = Pager::new(n_slots);
        for _ in 0..passes {
            for id in 0..n_blocks {
                p.begin_batch(); // one layer's weights = one batch
                p.schedule(id);
            }
        }
        // Pass 1 is all misses; every later pass hits the stable prefix of n_slots-1 blocks.
        let want_hits = (passes - 1) * (n_slots as u64 - 1);
        assert_eq!(
            p.stats().hits,
            want_hits,
            "schedule policy fell short of the Belady hit count on a cyclic sweep"
        );
    }

    #[test]
    fn schedule_stable_prefix_keeps_its_slots_across_passes() {
        // The stable prefix must not only stay resident but keep the SAME slot assignment pass
        // over pass — the property that makes the streamed dispatch's per-pass offsets cheap to
        // reason about (only the churn slot's occupant changes).
        let mut p = Pager::new(3);
        let record = |p: &Pager| (0..3u32).map(|id| p.slot_of(id)).collect::<Vec<_>>();
        for id in 0..8u32 {
            p.begin_batch();
            p.schedule(id);
        }
        let first = record(&p);
        for _ in 0..3 {
            for id in 0..8u32 {
                p.begin_batch();
                p.schedule(id);
            }
            assert_eq!(
                record(&p),
                first,
                "stable-prefix slots moved between passes"
            );
        }
        // And the prefix really is resident (blocks 0..n_slots-1 minus the churn slot).
        assert!(p.slot_of(0).is_some());
        assert!(p.slot_of(1).is_some());
    }

    #[test]
    fn schedule_batch_siblings_are_eviction_safe() {
        // A layer whose weights span several blocks in ONE pool touches them as one batch; the
        // epoch guard must keep earlier siblings resident while later ones evict (same invariant
        // as touch_cold, restated for the schedule entry point).
        let mut p = Pager::new(3);
        p.begin_batch();
        for id in [100u32, 101, 102] {
            p.schedule(id);
        }
        p.begin_batch();
        for id in [7u32, 8, 9] {
            p.schedule(id);
        }
        for id in [7u32, 8, 9] {
            assert!(p.slot_of(id).is_some(), "batch sibling {id} was evicted");
        }
    }

    /// Eviction must drop the victim's `epoch` entry, so `epoch` stays bounded by `n_slots` (never
    /// growing an entry per distinct BlockId ever touched). Exercised via the test-only
    /// `epoch_len()` accessor.
    #[test]
    fn eviction_drops_the_victim_epoch_entry() {
        let mut p = Pager::new(2);
        p.touch(1); // slot 0
        p.touch(2); // slot 1, full
                    // 3 evicts 1 (LRU); 1's epoch entry must go with it.
        assert_eq!(
            p.touch(3),
            Resolution::Miss {
                slot: 0,
                evicted: Some(1)
            }
        );
        assert!(p.slot_of(1).is_none());
        // epoch tracks only the two currently-resident blocks (2 and 3), not the evicted 1.
        assert_eq!(p.epoch_len(), 2);
        assert!(p.epoch_len() <= p.n_slots());
        // Churn many more distinct ids through the 2-slot cache: epoch stays bounded, not linear
        // in the number of distinct ids ever seen.
        for id in 100..200u32 {
            p.touch(id);
        }
        assert!(
            p.epoch_len() <= p.n_slots(),
            "epoch grew past n_slots ({} > {})",
            p.epoch_len(),
            p.n_slots()
        );
    }

    /// The pin's whole purpose: a pinned block is not a victim even when it is the COLDEST thing
    /// in the cache. Without the guard, LRU order alone would hand slot 0 straight back.
    #[test]
    fn a_pinned_block_is_never_the_victim() {
        let mut p = Pager::new(2);
        let Some(Resolution::Miss { slot: s1, .. }) = p.resolve_and_pin(1, Insert::Mru) else {
            panic!("expected miss")
        };
        p.touch(2); // unpinned, and now the MRU end — but 1 is pinned, so 2 is the only victim
        let Some(Resolution::Miss { evicted, .. }) = p.resolve_and_pin(3, Insert::Mru) else {
            panic!("expected miss")
        };
        assert_eq!(evicted, Some(2), "the unpinned block must be the victim");
        assert_eq!(p.slot_of(1), Some(s1), "pinned block moved or was evicted");
    }

    /// Exhaustion is a `None`, not a panic and not a silent eviction — and it must leave the cache
    /// exactly as it was, so a caller that releases a pin and retries gets a working pager.
    #[test]
    fn every_slot_pinned_yields_none_and_changes_nothing() {
        let mut p = Pager::new(2);
        p.resolve_and_pin(1, Insert::Mru).expect("slot 1");
        p.resolve_and_pin(2, Insert::Mru).expect("slot 2");
        let before = p.stats();
        assert!(
            p.resolve_and_pin(3, Insert::Mru).is_none(),
            "a fully pinned cache cannot admit a third block"
        );
        assert_eq!(p.resident_count(), 2);
        assert!(p.slot_of(1).is_some() && p.slot_of(2).is_some());
        assert_eq!(
            p.slot_of(3),
            None,
            "the rejected block must not be resident"
        );
        let after = p.stats();
        assert_eq!(
            (after.hits, after.misses, after.evictions),
            (before.hits, before.misses, before.evictions),
            "a refused resolve must not move any counter"
        );

        // Releasing one pin makes exactly one slot available again.
        p.unpin(1);
        let Some(Resolution::Miss { evicted, .. }) = p.resolve_and_pin(3, Insert::Mru) else {
            panic!("expected the freed slot")
        };
        assert_eq!(evicted, Some(1));
    }

    /// A pin outlives the batch epoch — the reason it exists at all. The epoch guard expires the
    /// moment the next batch opens; the pin does not.
    #[test]
    fn a_pin_outlives_its_batch() {
        let mut p = Pager::new(2);
        p.begin_batch();
        p.resolve_and_pin(1, Insert::Cold).expect("resident");
        p.begin_batch(); // block 1's epoch protection is now stale; only its pin remains
        p.touch_cold(2);
        p.begin_batch(); // ...and now block 2's is stale too, so it is the only legal victim
        let Some(Resolution::Miss { evicted, .. }) = p.resolve_and_pin(3, Insert::Cold) else {
            panic!("expected miss")
        };
        assert_eq!(evicted, Some(2), "the pin must still protect block 1");
        assert!(p.slot_of(1).is_some());
    }

    /// Nesting: two pins on one block need two releases. A single `unpin` must NOT make it
    /// evictable while the second reader still holds it.
    #[test]
    fn pins_are_counted_not_boolean() {
        let mut p = Pager::new(1);
        p.resolve_and_pin(1, Insert::Mru).expect("resident");
        p.resolve_and_pin(1, Insert::Mru).expect("hit, second pin");
        assert_eq!(p.pinned_blocks(), 1);
        p.unpin(1);
        assert!(
            p.resolve_and_pin(2, Insert::Mru).is_none(),
            "one release of two must leave the block pinned"
        );
        p.unpin(1);
        assert!(p.resolve_and_pin(2, Insert::Mru).is_some());
    }

    /// `pin_if_resident` is a probe: a miss must not read, insert, evict or count a miss.
    #[test]
    fn pin_if_resident_never_admits_a_block() {
        let mut p = Pager::new(2);
        assert_eq!(p.pin_if_resident(9), None);
        assert_eq!(p.resident_count(), 0);
        assert_eq!(p.stats().misses, 0, "a probe miss is not a residency miss");

        let Some(Resolution::Miss { slot, .. }) = p.resolve_and_pin(9, Insert::Mru) else {
            panic!("expected miss")
        };
        assert_eq!(p.pin_if_resident(9), Some(slot));
        // Two pins now (resolve + probe): both must be released before 9 can be evicted.
        p.unpin(9);
        p.unpin(9);
        assert_eq!(p.pinned_blocks(), 0);
    }

    #[test]
    #[should_panic(expected = "holds no pin")]
    fn unpinning_an_unpinned_block_panics() {
        // An unbalanced release would let a slot be evicted under a live reader — wrong bytes with
        // no error, which is worse than the panic.
        let mut p = Pager::new(1);
        p.touch(1);
        p.unpin(1);
    }

    #[test]
    fn hit_rate_reports_sane_values() {
        let mut p = Pager::new(2);
        assert_eq!(p.stats().hit_rate(), 1.0); // vacuous
        p.touch(1);
        p.touch(1);
        assert!((p.stats().hit_rate() - 0.5).abs() < 1e-9);
    }
}
