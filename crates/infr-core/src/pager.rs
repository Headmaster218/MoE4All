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
    stats: PagerStats,
}

impl Pager {
    /// A pager with `n_slots` uniform slots and nothing resident. `n_slots` must be at least the
    /// largest number of distinct blocks any single dispatch batch will `touch` (see the
    /// within-batch safety note) — the pager can't check that itself (it doesn't know about
    /// batches), so a caller sizing `n_slots` from a VRAM budget must clamp it to at least that
    /// floor and error out earlier if the budget can't cover even one batch.
    pub fn new(n_slots: usize) -> Self {
        Self {
            n_slots,
            resident: HashMap::with_capacity(n_slots),
            lru: VecDeque::with_capacity(n_slots),
            free: (0..n_slots as u32).rev().collect(), // pop() hands out slot 0 first
            epoch: HashMap::with_capacity(n_slots),
            cur_epoch: 0,
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

    /// A free slot, evicting if none remain. The victim is the coldest block NOT touched in the
    /// current batch (epoch guard — see [`Self::begin_batch`]); legacy callers that never open a
    /// batch share epoch 0 everywhere, for which the guard degrades to plain LRU `pop_front`.
    fn take_slot(&mut self) -> (u32, Option<BlockId>) {
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
                assert!(
                    self.cur_epoch == 0,
                    "pager: a single batch touched more blocks than n_slots={} — within-batch \
                     eviction safety is unsatisfiable (sizing floor violated)",
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

/// [`ring_bytes`] over the raw `INFR_PAGER_RING` STRING — the env spelling of the same policy
/// (`crate::parse_size`, then the clamp), pinned here so the grammar has one home.
pub fn ring_bytes_from(pager_budget: u64, raw: Option<&str>) -> usize {
    ring_bytes(pager_budget, raw.and_then(crate::parse_size))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring clamp's boundaries, pinning the pre-hoist inline expression
    /// `(budget / 8).clamp(256 MiB, 2048 MiB)` — the seam subtracts this from the pager budget
    /// before splitting arena shares, so a moved edge silently re-sizes every MoE arena — plus the
    /// `INFR_PAGER_RING` override grammar. Driven through [`ring_bytes_from`], which takes the
    /// override as a plain string: no process-environment mutation, so this cannot race another
    /// test in the shared binary.
    #[test]
    fn ring_bytes_clamp_boundaries_and_override_grammar() {
        const MIB: u64 = 1024 * 1024;
        // Below the floor's crossover (8 x 256 MiB = 2 GiB of budget) the floor wins — including
        // the `0` budget the pager passes when it has no budget figure at all.
        assert_eq!(ring_bytes_from(0, None), 256 * MIB as usize);
        assert_eq!(ring_bytes_from(2048 * MIB, None), 256 * MIB as usize);
        // Exactly at the crossover, and just past it, the eighth wins.
        assert_eq!(
            ring_bytes_from(2048 * MIB + 8, None),
            (256 * MIB + 1) as usize
        );
        // At the ceiling's crossover (8 x 2 GiB) and beyond, the ceiling wins.
        assert_eq!(
            ring_bytes_from(16 * 1024 * MIB, None),
            (2048 * MIB) as usize
        );
        assert_eq!(
            ring_bytes_from(64 * 1024 * MIB, None),
            (2048 * MIB) as usize
        );

        // The override in the shared size grammar; anything unparseable or zero falls through to
        // the policy (a `0` ring would mean "never stage a slot").
        for (raw, want) in [
            ("1g", Some(1024 * MIB as usize)),
            ("512m", Some(512 * MIB as usize)),
            ("3221225472", Some(3 * 1024 * MIB as usize)), // plain bytes
            ("0", None),
            ("", None),
            ("banana", None),
            ("1GiB", None), // not the shared grammar's spelling — treated as unset
        ] {
            let got = ring_bytes_from(64 * 1024 * MIB, Some(raw));
            match want {
                Some(w) => assert_eq!(got, w, "raw={raw:?}"),
                // Falls through to the policy, which caps at 2 GiB on this budget.
                None => assert_eq!(got, (2048 * MIB) as usize, "raw={raw:?}"),
            }
        }
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

    #[test]
    fn hit_rate_reports_sane_values() {
        let mut p = Pager::new(2);
        assert_eq!(p.stats().hit_rate(), 1.0); // vacuous
        p.touch(1);
        p.touch(1);
        assert!((p.stats().hit_rate() - 0.5).abs() < 1e-9);
    }
}
