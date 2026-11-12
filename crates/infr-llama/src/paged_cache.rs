//! SGLang-style paged radix prefix cache: the tree holds FRAME CHAINS, not byte snapshots.
//!
//! Phase B of `.trae/documents/paged-hierarchical-kv-cache.md`. Where [`crate::ram_cache`] stores
//! one self-contained host-RAM blob per conversation (copy out on demote, copy in on restore), this
//! tier keeps the conversation's KV *in place*: a demoted session publishes its page->frame chain
//! (one [`FrameId`] per fixed-size frame, frames shared zero-copy by refcount), and a restoring
//! session pins the shared prefix frames and points its page table at them. No bytes move for a
//! prefix hit — reuse is a table write. Divergence inside a shared page is recomputed from the
//! page boundary: a frame is the smallest shareable unit, so the tail page of any match is never
//! half-reused (SGLang's rule, and the only one the kernel's page-shift addressing can honor).
//!
//! # Tree semantics (inherited from the blob tier)
//!
//! Same compressed-radix skeleton: nodes keyed by token runs, edges split on partial matches,
//! entries live at the node whose path equals the conversation's tokens, and publishing a longer
//! conversation supersedes (and unpins) strict-ancestor entries — their chains are prefixes of the
//! new chain, so the frames survive via the new entry's owners. Matching is *positional*: a chain
//! covers tokens from 0, so whatever prefix the prompt walk consumes is reusable at page
//! granularity — `floor(consumed / page_tokens) * page_tokens` — including divergence inside an
//! entry's edge, which the blob tier's whole-snapshot rule could not share.
//!
//! # Hybrid recurrent aux
//!
//! DeltaNet/linear-attention models keep conv/S state and PLE history OUTSIDE the page table, so
//! frames alone cannot resume their prefill: re-prefilling a shared prefix would overwrite state
//! the resumed positions depend on. A published entry may therefore carry host-resident
//! [`PagedCkpt`] snapshots keyed to a page-aligned token boundary; a recurrent session attaches
//! ONLY at exactly such a boundary, uploads the snapshot into its live buffers, and resumes from
//! there ([`PagedKv::entry_ckpt`]). Aux bytes live under their own budget — an eighth of the RAM
//! tier. What exhausting that budget costs depends on the model: a dense entry ignores its
//! snapshots, so the coldest one is simply dropped and its chain still serves as an attention
//! prefix. A recurrent entry resumes ONLY through them, so dropping its last snapshot would leave
//! a chain that matches and then hands back zero tokens — dead cache holding frames. Such an entry
//! is evicted whole instead, and a recurrent conversation that arrives with no usable snapshot is
//! refused at publish time ([`PagedKv::publish_with_ckpts`], counted as `aux_refusals`) rather than
//! cached unreachable.
//!
//! # Ownership and eviction
//!
//! Every frame's refcount = tree entry owners + session owners (a resident conversation attached
//! from the tree keeps both). Eviction drops a whole entry (unpin its chain); frames still owned
//! by a live session survive the unpin, so eviction never needs an "in use" guard beyond the
//! refcount itself. The corridor is the cap: when [`PagedKv::alloc_frame`] finds no free slot it
//! evicts leaf-first (an entry with entry-bearing descendants loses the least matching power per
//! frame freed), then oldest access — the "coldest leaf tip first" policy, at frame granularity.
//! Phase C adds the RAM tier in front of true deletion.
//!
//! # Locks
//!
//! One `Mutex<PagedKv>` per engine guards pool + tree together: every operation here is synchronous
//! under that single lock, and the scheduler's discipline is unchanged — tree mutation only happens
//! at request boundaries (checkout/demote) while the GPU baton is held, so no dispatch can observe
//! a half-written page table. There is deliberately no idle worker (PR #86 black-hole conviction).
//!
//! # Phase C: the RAM tier
//!
//! `kv.paged_cache_max` sizes a host-RAM frame pool behind the VRAM corridor. When the corridor
//! runs dry, the coldest tree entry SINKS (one frame DMA down per frame, the entry's chain
//! rewritten to RAM ids) before any entry is truly evicted; a hit on a chain with RAM frames
//! PROMOTES them back ([`PagedKv::entry_chain_promoted`]) before the attaching session's page
//! table is written — RAM addresses never enter a table (bandwidth-wall + BDA convictions).
//! Promotion claims may themselves sink other cold entries; the hot chain that just hit is
//! skipped by the sinker because its LRU stamp was refreshed by [`PagedKv::best_match`].
//! The host backing is `Vec<u8>` indexed by RAM slot — a RAM `FrameId`'s `slot() * frame_bytes`
//! IS its backing offset, mirroring how a VRAM slot maps into its corridor.
//!
//! # Phase D: the disk tier
//!
//! `kv.disk_cache_max` sizes a third pool behind the RAM tier, backed by ONE sparse file (disk
//! slot `s` owns bytes `s * frame_bytes..`, grown lazily to the high-water mark). Only whole
//! RAM-resident chains sink: when the RAM pool runs dry, the coldest all-RAM entry's frames are
//! written to the file and its chain rewritten to Disk ids, freeing RAM slots for the VRAM sink
//! that triggered the cascade. A hit on a Disk frame reads it back into a fresh RAM slot (making
//! room by sinking another cold RAM entry if needed) before the normal RAM→VRAM promotion uploads
//! it. Synchronous `std::fs` I/O runs inside the GPU baton like every other tree mutation — an
//! NVMe frame read is ~ms, and the alternative (an idle worker) is the PR #86 black hole. The
//! backing is process-scoped: slots recycle without wiping, and a restart starts with an empty
//! disk tier (content-addressed cross-process reuse stays out of scope).
//!
//! # The elastic corridor
//!
//! A fixed corridor leases `n_slots * ceil(max_ctx / page_tokens)` frames at init — the worst case
//! every session could ever need — and holds that lease for the process's life. At a long context
//! that freezes gigabytes of the shared expert arena while the resident conversations own a few
//! percent of it, and the experts that lost those bytes pay for it in decode throughput. The
//! blob/segmented KV path never had this problem: it reserves coordinates and commits only the
//! segments a request actually reaches.
//!
//! `kv.paged_elastic` gives the frame pool the same discipline. Init commits ONE chunk; when the
//! free list runs dry [`PagedKv::claim_frame`] claims another chunk from the arena (reusing the
//! expert-victim reclaim every other KV claim goes through) before it sinks or evicts anything;
//! and at each request boundary [`PagedKv::release_idle_corridors`] hands every fully-free segment
//! above the retained chunk back, so the expert pager refills the bytes an idle conversation was
//! squatting on. Growth is capped at the same worst-case ceiling the fixed corridor used, so the
//! arena's KV corridor geometry — and therefore every other budget in the VRAM plan — is unchanged.
//! Slot indices are never recycled across a shrink: retired slots stay as tombstones, so a live
//! handle can never alias a released address.

use crate::seam::kv_frame_pool::{FrameId, FramePool, Tier};
use anyhow::{anyhow, Result};
use infr_core::backend::{Backend, Buffer, KvFrameCorridor};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Arc;

/// Frame-granularity DMA between the VRAM corridor and host memory, factored out of [`Backend`]
/// so the pool/tree logic tests against an in-memory stand-in. Blanket-impl'd for every backend.
pub(crate) trait FrameDma {
    /// Copy one frame's corridor bytes into host memory (`dst.len() == frame_bytes`).
    fn download_frame(&self, frame_addr: u64, dst: &mut [u8]) -> Result<()>;
    /// Copy one frame's host bytes into its corridor slot (`src.len() == frame_bytes`).
    fn upload_frame(&self, frame_addr: u64, src: &[u8]) -> Result<()>;
    /// Claim `count` more corridor frames from the shared device arena. An empty vector means the
    /// arena has nothing left to give, so the tree falls back to its sink/evict ladder. The default
    /// keeps a fixed corridor (host stand-ins, backends without an elastic arena).
    fn grow_frames(&self, frame_bytes: u64, count: usize) -> Result<Vec<KvFrameCorridor>> {
        let _ = (frame_bytes, count);
        Ok(Vec::new())
    }
    /// Hand fully-free corridors back to the arena so experts can refill the same bytes. Every
    /// frame must already be unowned and no dispatch may still name the segment.
    fn release_frames(&self, corridors: &[KvFrameCorridor]) -> Result<()> {
        let _ = corridors;
        Ok(())
    }
    /// Whether the expert cache sharing this arena is starving for VRAM (see
    /// [`Backend::expert_vram_pressure`]). Default `false`: host stand-ins and backends without a
    /// paged expert cache never ask the tree to yield.
    fn expert_vram_pressure(&self) -> bool {
        false
    }
}

impl<T: Backend + ?Sized> FrameDma for T {
    fn download_frame(&self, frame_addr: u64, dst: &mut [u8]) -> Result<()> {
        self.kv_frame_download(frame_addr, dst)
            .map_err(|error| anyhow!("{error}"))
    }
    fn upload_frame(&self, frame_addr: u64, src: &[u8]) -> Result<()> {
        self.kv_frame_upload(frame_addr, src)
            .map_err(|error| anyhow!("{error}"))
    }
    fn grow_frames(&self, frame_bytes: u64, count: usize) -> Result<Vec<KvFrameCorridor>> {
        Ok(self
            .alloc_kv_frames(frame_bytes, count)?
            .unwrap_or_default())
    }
    fn release_frames(&self, corridors: &[KvFrameCorridor]) -> Result<()> {
        self.free_kv_frames(corridors)
            .map_err(|error| anyhow!("{error}"))
    }
    fn expert_vram_pressure(&self) -> bool {
        Backend::expert_vram_pressure(self)
    }
}

/// The disk tier's backing store: one open sparse file, `slot * frame_bytes` per frame.
/// `high_water` tracks the logical file length so growth is one `set_len` per new max slot,
/// never a metadata round-trip per frame.
struct DiskBacking {
    file: File,
    high_water: u64,
}

/// One cached conversation: the frames covering every COMPLETE page of its token prefix
/// (`chain.len() == tokens.len() / page_tokens`). The token spine itself lives in the node edges,
/// so the entry needs no copy of them. The sub-page tail is recomputed on reuse.
struct ChainEntry {
    chain: Vec<FrameId>,
    /// Host snapshots of the recurrent/PLE state at page-aligned boundaries of this conversation,
    /// sorted by `boundary_tokens`. A session that resumes on a hybrid recurrent model (DeltaNet,
    /// KDA, PLE) must attach at exactly one of these lengths; a pure attention model ignores them.
    ckpts: Vec<Arc<PagedCkpt>>,
    /// Frames the last attach actually reused ([`PagedKv::entry_chain_promoted`], after
    /// [`PagedKv::hit_prefix`] cut the match to a resumable boundary). The sinker keeps this prefix
    /// resident and moves the tail behind it: those are the frames a next hit would have to DMA
    /// straight back, so sinking them pays the copy twice to free bytes nothing else wanted.
    /// Zero until the first attach, which is what makes a fresh chain sink whole.
    warm_frames: usize,
}

/// Host-resident recurrent state at one stable conversation boundary of a published chain. The
/// bytes mirror [`crate::seam::weights::TurnRecurrentCkpt`]'s device buffers (one conv/`S` pair per
/// recurrent layer plus the model-level PLE history) for a model whose per-token KV lives in the
/// frames: the page table resumes attention, this snapshot resumes everything the page table does
/// not cover. Snapshots are captured at publish time and restored into the live buffers BEFORE the
/// attaching session's page table is written, so a tree hit never replays the prefill.
///
/// The storage is a backend `Readback` buffer, not a `Vec<u8>`: that class is host-visible system
/// RAM outside the device-local budget, so the aux costs exactly the host bytes `bytes()` reports
/// and no VRAM. It also lets the prefill graph WRITE the snapshot directly (see
/// `seam::runner::StateExport`) instead of downloading device state after the fact.
pub(crate) struct PagedCkpt {
    /// Token boundary (a multiple of `page_tokens`) this snapshot describes: a session attaching
    /// it may reuse exactly this many tokens, never more (the state stops here).
    pub(crate) boundary_tokens: usize,
    /// Recurrent layer indices; `layers[i]` owns `kbufs[i]`/`vbufs[i]`.
    pub(crate) layers: Vec<usize>,
    pub(crate) kbufs: Vec<Box<dyn Buffer>>,
    pub(crate) vbufs: Vec<Box<dyn Buffer>>,
    /// Qwen3.8's model-level PLE convolution history, when the model has one.
    pub(crate) ple: Option<Box<dyn Buffer>>,
}

impl PagedCkpt {
    pub(crate) fn bytes(&self) -> usize {
        self.kbufs
            .iter()
            .chain(self.vbufs.iter())
            .map(|buffer| buffer.len_bytes())
            .sum::<usize>()
            + self.ple.as_ref().map_or(0, |buffer| buffer.len_bytes())
    }
}

fn ckpts_bytes(ckpts: &[Arc<PagedCkpt>]) -> usize {
    ckpts.iter().map(|ckpt| ckpt.bytes()).sum()
}

struct Node {
    parent: Option<usize>,
    edge: Vec<u32>,
    children: HashMap<u32, usize>,
    path_len: usize,
    entry: Option<ChainEntry>,
    last_access: u64,
}

impl Node {
    fn new(parent: Option<usize>, edge: Vec<u32>, path_len: usize) -> Self {
        Self {
            parent,
            edge,
            children: HashMap::new(),
            path_len,
            entry: None,
            last_access: 0,
        }
    }
}

/// The frame pool, the radix tree over published chains, and the shared geometry. Wrapped in a
/// `Mutex` by the scheduler; cloned `Arc` reaches each session slot's `SegmentedKvState`.
pub(crate) struct PagedKv {
    pool: FramePool,
    nodes: Vec<Node>,
    /// Recycled indices of detached tree nodes (see `free_node`): churn reuses slots so the
    /// tree's per-token metadata stays bounded by the live peak, matching the blob tier.
    free: Vec<usize>,
    page_tokens: usize,
    /// Byte offset of each layout plane inside one frame, in the SAME order the session collects
    /// its plane buffers (`SegmentedKvLayout::planes`): prefix sums of `plane.segment_bytes()`.
    /// A page-table entry for plane `p` on frame `f` is `plane_addr(f, p)`.
    plane_offsets: Vec<u64>,
    /// Host backing for the RAM tier: slot `s` owns bytes `s * frame_bytes..(s + 1) *
    /// frame_bytes`. Grown lazily to the high-water mark of demoted slots so a server that never
    /// fills its RAM budget never commits the full host allocation.
    ram_backing: Vec<u8>,
    /// Phase D: the disk tier's sparse backing file, `None` when the tier is closed (budget 0 or
    /// the file could not be opened — a cache layer must never fail engine init).
    disk: Option<DiskBacking>,
    /// Host bytes currently held by every entry's `ckpts` (see [`PagedCkpt`]).
    aux_bytes: usize,
    /// Cap on `aux_bytes`: an eighth of the RAM-tier frame backing, 0 when that tier is off. Aux
    /// is a restore accelerator, not a second KV copy: on a dense tree it only ever trims itself,
    /// and on a recurrent tree it evicts the entry it can no longer make resumable (see
    /// [`Self::recurrent`]) rather than keep an unreachable chain.
    aux_budget: usize,
    /// Whether the model this tree serves summarizes its prefix into recurrent/PLE state
    /// ([`crate::config::Config::has_recurrent_state`]). Decides what an entry's snapshots MEAN:
    /// for a dense model they are an accelerator that may be trimmed freely, for a recurrent model
    /// they are the only way back into the chain, so an entry that loses its last one is evicted
    /// rather than left holding frames no hit can resume.
    recurrent: bool,
    /// Elastic-corridor ceiling in frames; 0 keeps the corridor fixed at its init size.
    vram_ceiling: usize,
    /// Frames per growth transaction, and the floor below which idle segments are never handed
    /// back: one chunk stays committed so a new request never starts from an empty corridor.
    grow_chunk: usize,
    /// A growth attempt that came back short or failed. Re-planning the arena per page would cost
    /// more than the sink ladder it replaces, so the tree stops asking until a request boundary
    /// ([`Self::release_idle_corridors`]) clears this.
    grow_stalled: bool,
    /// Whether a starving expert cache may ask this tree to sink cold tails and return the emptied
    /// segments (`kv.paged_yield_under_pressure`). Off by default: yielding costs a DMA out now and
    /// a DMA back on the next hit, so it is only worth paying when the other owner of the arena is
    /// actually evicting.
    yield_under_pressure: bool,
    clock: u64,
    hits: u64,
    misses: u64,
    demotions: u64,
    evictions: u64,
    /// Recurrent publishes refused for want of a resumable snapshot (none captured, a device
    /// download failed, or the deepest one cannot fit the aux budget). A rising count means the
    /// tree is caching nothing for this model and every request re-prefills.
    aux_refusals: u64,
    /// Snapshots a deeper publish inherited from the ancestor entries it superseded. Zero on a
    /// workload whose every request continues the same conversation at its own frontier; a rising
    /// count is the branch-point reuse that supersession used to delete.
    ckpts_inherited: u64,
    tier_demotions: u64,
    tier_promotions: u64,
    /// Sinks that left a warm prefix in VRAM and moved only the tail behind it. Zero means every
    /// sink still moved whole chains — either nothing is being reused, or RAM is so tight the
    /// floor keeps getting overridden.
    tail_sinks: u64,
    disk_demotions: u64,
    disk_promotions: u64,
    corridor_grows: u64,
    corridor_shrinks: u64,
}

impl PagedKv {
    /// One root per projection setting: unprojected, OFF, ON (see [`Self::root`]).
    const PROJECTION_ROOTS: usize = 3;

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        frame_bytes: u64,
        page_tokens: usize,
        plane_offsets: Vec<u64>,
        vram_frames: usize,
        ram_frames: usize,
        disk_frames: usize,
        disk_path: Option<PathBuf>,
        recurrent: bool,
    ) -> Self {
        assert!(
            page_tokens.is_power_of_two(),
            "page_tokens must be a power of two"
        );
        assert!(
            !plane_offsets.is_empty(),
            "a paged KV cache needs at least one plane"
        );
        // Open (lazily-sparse) the disk backing up front so a misconfigured path surfaces once at
        // init instead of silently disabling every later sink. A failure is a WARN + closed tier:
        // the VRAM+RAM behavior of Phase C must survive its own cache layer being unavailable.
        let disk = match (disk_frames > 0, disk_path) {
            (true, Some(path)) => match OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                // The backing is process-scoped bookkeeping, never restored across restarts
                // (content-addressed cross-process reuse is out of scope). `truncate` makes an
                // explicit shared `kv.disk_cache_path` start empty and reclaims the prior run's
                // disk space; the default per-pid temp path never collides anyway.
                .truncate(true)
                .open(&path)
            {
                Ok(file) => {
                    tracing::info!(
                        path = %path.display(),
                        disk_frames,
                        "paged KV disk tier backing opened"
                    );
                    Some(DiskBacking {
                        file,
                        high_water: 0,
                    })
                }
                Err(error) => {
                    tracing::warn!(
                        path = %path.display(),
                        "paged KV disk tier disabled: could not open the backing file: {error}"
                    );
                    None
                }
            },
            _ => None,
        };
        Self {
            pool: FramePool::new(frame_bytes, vram_frames, ram_frames, disk_frames),
            // One root per projection setting — see [`Self::root`]. Root 0 is the unprojected
            // tree, so a session with no direction file addresses exactly the keys it always did.
            nodes: (0..Self::PROJECTION_ROOTS)
                .map(|_| Node::new(None, Vec::new(), 0))
                .collect(),
            free: Vec::new(),
            page_tokens,
            plane_offsets,
            ram_backing: Vec::new(),
            disk,
            aux_bytes: 0,
            // Derived from the RAM tier so the aux cap needs no new config surface: snapshots are
            // an eighth-scale appendix to the prefix bytes they accelerate, never a peer tier. A
            // corridor-only tree (`kv.paged` with no pool) has nothing to divide, so its caller
            // overrides this — see [`Self::set_aux_budget`].
            aux_budget: (ram_frames * frame_bytes as usize) / 8,
            recurrent,
            vram_ceiling: 0,
            grow_chunk: 0,
            grow_stalled: false,
            yield_under_pressure: false,
            clock: 0,
            hits: 0,
            misses: 0,
            demotions: 0,
            evictions: 0,
            aux_refusals: 0,
            ckpts_inherited: 0,
            tier_demotions: 0,
            tier_promotions: 0,
            tail_sinks: 0,
            disk_demotions: 0,
            disk_promotions: 0,
            corridor_grows: 0,
            corridor_shrinks: 0,
        }
    }

    /// The root of the tree addressed under `projection`: `None` when the session loaded no
    /// direction file, `Some(on)` for the setting its rows were actually computed under.
    ///
    /// One token prefix has two different bodies of activations, so the setting has to be part of
    /// the address. Partitioning at the ROOT is what keeps the key injective: stamping a reserved
    /// id into the first position instead would overwrite the first token — and a first image's
    /// content id — so two prompts differing only there would be handed each other's KV rows.
    /// Roots share the frame pool, the aux budget and the eviction clock; only the spine is split.
    fn root(&self, projection: Option<bool>) -> usize {
        match projection {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    /// Tokens per page (callers recompute a truncated reuse length after a partial promotion).
    pub(crate) fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    /// Install the VRAM corridors backing frames `0..vram_frames`. Usually one segment; the
    /// unified arena hands back several when the request exceeds a single shard's largest gap.
    pub(crate) fn set_vram_segments(&mut self, corridors: &[KvFrameCorridor]) {
        self.pool.set_segments(Tier::Vram, corridors);
    }

    /// Let the corridor breathe between `chunk_frames` (already committed at init, and retained
    /// forever) and `ceiling_frames` (the worst case a fixed corridor would have leased outright).
    /// Call once at init, after [`Self::set_vram_segments`]. The ceiling never exceeds the arena's
    /// reserved KV corridor, so growth cannot steal bytes the VRAM plan gave to another owner.
    pub(crate) fn set_elastic_corridor(&mut self, ceiling_frames: usize, chunk_frames: usize) {
        assert!(
            chunk_frames > 0,
            "an elastic KV corridor needs a non-zero chunk"
        );
        self.vram_ceiling = ceiling_frames.max(self.pool.capacity_frames(Tier::Vram));
        self.grow_chunk = chunk_frames.min(self.vram_ceiling);
    }

    /// Let a starving expert cache reclaim this tree's cold tails (see
    /// [`Self::yield_vram_under_expert_pressure`]). Call once at init; meaningless without an
    /// elastic corridor and a RAM tier, both of which the yield path checks itself.
    pub(crate) fn set_yield_under_pressure(&mut self, enabled: bool) {
        self.yield_under_pressure = enabled;
    }

    /// Replace the aux (recurrent/PLE snapshot) budget implied by the RAM frame pool. Only the
    /// corridor-only tree needs it: `aux_budget` derives from a pool that is not there, and a zero
    /// budget does not merely shrink reuse — a recurrent publish is refused outright (see
    /// [`Self::publish_with_ckpts`]), so the tree caches nothing for a hybrid model. Call once at
    /// init, before anything publishes. Host bytes are still claimed per captured snapshot, never
    /// reserved here.
    pub(crate) fn set_aux_budget(&mut self, bytes: usize) {
        self.aux_budget = bytes;
    }

    /// Device address to write into plane `plane`'s table for one VRAM frame. RAM-tier frames must
    /// never reach a table (bandwidth-wall + BDA stale convictions) — Phase C writes 0 instead.
    pub(crate) fn plane_addr(&self, frame: FrameId, plane: usize) -> u64 {
        debug_assert!(
            matches!(frame.tier(), Tier::Vram),
            "RAM frames stay out of tables"
        );
        self.pool.frame_addr(frame) + self.plane_offsets[plane]
    }

    /// One-line console report. Besides the tree counters it spells the corridor in the units an
    /// operator reads on the serve console: frames (allocated/total/free), the token capacity they
    /// buy, and the GiB claimed — `held` counts every allocated frame (tree chains AND live
    /// sessions), so `free` is exactly what a growing request can still claim before eviction.
    pub(crate) fn stats_line(&self) -> String {
        let gib =
            |frames: usize| frames as f64 * self.pool.frame_bytes() as f64 / (1u64 << 30) as f64;
        let held = self.pool.held_frames(Tier::Vram);
        let total = self.pool.capacity_frames(Tier::Vram);
        format!(
            "entries={} frames_held={}/{} free={} tokens_held={}/{} bytes={:.2}GiB/{:.2}GiB ram={}/{} ram_bytes={:.2}GiB disk={}/{} disk_bytes={:.2}GiB aux_bytes={:.2}MiB/{:.2}MiB hits={} misses={} demotions={} evictions={} aux_refusals={} ckpts_inherited={} sunk={} promoted={} tail_sinks={} disk_sunk={} disk_promoted={} corridor_grows={} corridor_shrinks={} corridor_ceiling={}",
            self.nodes
                .iter()
                .filter(|node| node.entry.is_some())
                .count(),
            held,
            total,
            self.pool.free_frames(Tier::Vram),
            held * self.page_tokens,
            total * self.page_tokens,
            gib(held),
            gib(total),
            self.pool.held_frames(Tier::Ram),
            self.pool.capacity_frames(Tier::Ram),
            self.pool.held_bytes(Tier::Ram) as f64 / (1u64 << 30) as f64,
            self.pool.held_frames(Tier::Disk),
            self.pool.capacity_frames(Tier::Disk),
            self.pool.held_bytes(Tier::Disk) as f64 / (1u64 << 30) as f64,
            self.aux_bytes as f64 / (1u64 << 20) as f64,
            self.aux_budget as f64 / (1u64 << 20) as f64,
            self.hits,
            self.misses,
            self.demotions,
            self.evictions,
            self.aux_refusals,
            self.ckpts_inherited,
            self.tier_demotions,
            self.tier_promotions,
            self.tail_sinks,
            self.disk_demotions,
            self.disk_promotions,
            self.corridor_grows,
            self.corridor_shrinks,
            self.vram_ceiling,
        )
    }

    // ---- session-side frame operations ----------------------------------------------------

    /// Claim a fresh frame for a growing session. The frame arrives ACTIVE: the caller's page
    /// table points at it immediately, and only active frames are excluded from tier migration.
    /// On an exhausted corridor, cold tree entries first SINK to RAM (Phase C) and only then are
    /// evicted; `None` when every frame is owned by a live session's chain that the tree cannot
    /// drop (all entries gone or every remaining frame still ref-pinned by sessions — the caller
    /// then recycles the slot the old way).
    pub(crate) fn alloc_frame<M: FrameDma + ?Sized>(&mut self, be: &M) -> Option<FrameId> {
        self.claim_frame(be, true, &[])
    }

    /// Claim one VRAM frame, marking it active (a session table will point at it) or not.
    /// `protect` names frames a caller is already carrying (e.g. a promotion in flight): the
    /// sinker must not touch any entry whose chain holds one, because a tier migration REWRITES
    /// that id in every chain and would strand the caller's copy against a freed slot.
    fn claim_frame<M: FrameDma + ?Sized>(
        &mut self,
        be: &M,
        active: bool,
        protect: &[FrameId],
    ) -> Option<FrameId> {
        if let Some(frame) = self.pool.alloc(Tier::Vram) {
            self.pool.set_active(frame, active);
            return Some(frame);
        }
        // Sink the coldest tree entry to RAM before deleting anything: the prefix survives the
        // pressure and a later hit pays one DMA instead of a full re-prefill. Phase D adds the
        // cascade: when RAM is full, sinking the coldest all-RAM entry to disk frees RAM slots
        // for the VRAM sink that needs them. The loop interleaves sinks and eviction so an
        // evicted entry's freed host slot immediately re-enables a sink (evict one, then try
        // sinking again before evicting the next) — bytes always move down a tier before an
        // entry is dropped. Every successful step moves bytes strictly DOWN or deletes an
        // entry, so the loop cannot spin.
        loop {
            if let Some(frame) = self.pool.alloc(Tier::Vram) {
                self.pool.set_active(frame, active);
                return Some(frame);
            }
            // An elastic corridor asks the shared arena for another chunk before it demotes
            // anything: growing keeps the prefix in VRAM, where a sink would cost a DMA per frame
            // now and a DMA per frame back on the next hit.
            if self.grow_corridor(be) {
                continue;
            }
            if self.demote_coldest_entry(be, protect) || self.demote_ram_coldest_entry(protect) {
                continue;
            }
            if self.evict_cold_one() {
                continue;
            }
            return None;
        }
    }

    /// Claim one more chunk of corridor from the arena. Returns whether the slot space grew, so
    /// [`Self::claim_frame`]'s loop only continues on strict progress. A short or failed grant
    /// latches `grow_stalled`: the arena is out of KV space, and re-planning it for every page of
    /// a long prefill would cost more than the sink ladder below.
    fn grow_corridor<M: FrameDma + ?Sized>(&mut self, be: &M) -> bool {
        if self.grow_chunk == 0 || self.grow_stalled {
            return false;
        }
        let capacity = self.pool.capacity_frames(Tier::Vram);
        if capacity >= self.vram_ceiling {
            return false;
        }
        let want = self.grow_chunk.min(self.vram_ceiling - capacity);
        let corridors = match be.grow_frames(self.pool.frame_bytes(), want) {
            Ok(corridors) => corridors,
            Err(error) => {
                tracing::warn!(
                    requested_frames = want,
                    "elastic KV corridor growth failed, keeping {capacity} frames: {error}"
                );
                self.grow_stalled = true;
                return false;
            }
        };
        let added = self.pool.grow_vram(&corridors);
        if added == 0 {
            self.grow_stalled = true;
            return false;
        }
        if added < want {
            self.grow_stalled = true;
        }
        self.corridor_grows += 1;
        tracing::debug!(
            added_frames = added,
            capacity_frames = capacity + added,
            ceiling_frames = self.vram_ceiling,
            "elastic KV corridor grew"
        );
        true
    }

    /// Hand every fully-free corridor segment above the retained chunk back to the arena, so the
    /// expert pager can refill bytes an idle conversation was squatting on. Runs at a request
    /// boundary under the GPU baton: no dispatch may still name a frame in a released segment.
    ///
    /// Highest segment first — growth appends, so the newest chunk is the one most likely to be
    /// entirely free, and releasing top-down keeps the low addresses (where the hot prefix of a
    /// long-lived conversation lives) resident. Pool slots are retired BEFORE the backend lease is
    /// dropped: a failed release then leaks arena bytes, which is recoverable, instead of leaving
    /// live frames pointing at memory the arena already handed to an expert.
    pub(crate) fn release_idle_corridors<M: FrameDma + ?Sized>(&mut self, be: &M) {
        self.grow_stalled = false;
        if self.grow_chunk == 0 {
            return;
        }
        while self.pool.capacity_frames(Tier::Vram) > self.grow_chunk {
            let Some(index) = (0..self.pool.vram_segment_count())
                .rev()
                .find(|&index| self.pool.vram_segment_all_free(index))
            else {
                return;
            };
            let Some(corridor) = self.pool.take_free_vram_segment(index) else {
                return;
            };
            if let Err(error) = be.release_frames(&[corridor]) {
                tracing::warn!(
                    base = corridor.base,
                    frames = corridor.frames,
                    "elastic KV corridor release failed, its arena bytes stay leased: {error}"
                );
                return;
            }
            self.corridor_shrinks += 1;
            tracing::debug!(
                released_frames = corridor.frames,
                capacity_frames = self.pool.capacity_frames(Tier::Vram),
                "elastic KV corridor returned to the expert arena"
            );
        }
    }

    /// Sink cold tails until the segments above the retained base chunk are empty, so
    /// [`Self::release_idle_corridors`] can hand them to an expert cache that is starving. Returns
    /// how many frames left VRAM.
    ///
    /// The tree and the expert arena sub-allocate ONE physical pool, but each only sees its own
    /// ceiling: a conversation below `vram_ceiling` keeps every page in VRAM while the expert pager
    /// evicts on every admission, and the host tier sits empty. [`FrameDma::expert_vram_pressure`]
    /// is the cross-owner signal that closes that gap.
    ///
    /// Two limits keep this from being a pure loss. The last attach's reuse depth is a HARD floor
    /// here (unlike the exhaustion ladder, which offers a fully warm run rather than let the evictor
    /// delete it): nothing is about to be evicted, so sinking bytes a next hit would DMA straight
    /// back buys nothing. And only the highest segments are emptied — the base chunk always stays,
    /// and a hot conversation that reuses its whole prefix every turn pins itself and yields zero.
    pub(crate) fn yield_vram_under_expert_pressure<M: FrameDma + ?Sized>(
        &mut self,
        be: &M,
    ) -> usize {
        if self.grow_chunk == 0
            || !self.yield_under_pressure
            || self.pool.capacity_frames(Tier::Ram) == 0
            || self.pool.vram_segment_count() < 2
            || !be.expert_vram_pressure()
        {
            return 0;
        }
        let mut sunk = 0usize;
        while let Some(index) = (1..self.pool.vram_segment_count())
            .rev()
            .find(|&index| !self.pool.vram_segment_all_free(index))
        {
            let Some((first, frames)) = self.pool.vram_segment_range(index) else {
                break;
            };
            let reach = first..first + frames;
            let mut best: Option<(usize, usize, usize)> = None;
            let mut best_key: Option<(bool, u64, usize)> = None;
            for (node, item) in self.nodes.iter().enumerate() {
                let Some(entry) = item.entry.as_ref() else {
                    continue;
                };
                let (keep, end) = self.sink_region(entry, Tier::Vram, &[]);
                // Floor at the entry's first frame inside the target segment. `first` is a POOL
                // slot while `keep` indexes this entry's own chain, so the two only coincide for a
                // single conversation that owns the corridor from slot 0; comparing them directly
                // skips every shorter chain and the yield silently does nothing. Sinking below the
                // segment is wasted DMA anyway: the take runs to the end of the whole VRAM run, and
                // bytes in the retained base chunk are never handed back to the arena.
                let keep = entry
                    .chain
                    .iter()
                    .position(|frame| frame.slot() >= first)
                    .unwrap_or(entry.chain.len())
                    .max(entry.warm_frames)
                    .max(keep);
                if keep >= end
                    || !entry.chain[keep..end]
                        .iter()
                        .any(|frame| frame.tier() == Tier::Vram && reach.contains(&frame.slot()))
                {
                    continue;
                }
                let key = (self.has_entry_below(node), item.last_access, node);
                if best_key.is_none_or(|current| key < current) {
                    best_key = Some(key);
                    best = Some((node, keep, end));
                }
            }
            // No entry can yield a frame in that segment (all active, all warm, or already sunk):
            // stop rather than spin. Every iteration below moves at least one frame out of VRAM.
            let Some((node, keep, end)) = best else {
                break;
            };
            let moved = self.sink_entry_tail(be, node, keep, end);
            if moved == 0 {
                break;
            }
            if keep > 0 {
                self.tail_sinks += 1;
            }
            sunk += moved;
        }
        if sunk > 0 {
            tracing::debug!(
                sunk_frames = sunk,
                capacity_frames = self.pool.capacity_frames(Tier::Vram),
                "paged KV yielded cold tails to the expert arena under device pressure"
            );
        }
        sunk
    }

    /// Rewrite every entry-chain occurrence of `source` to `fresh` after a tier migration.
    /// Entries share positional frames: an ancestor's chain and a descendant's chain carry the
    /// SAME id at the SAME index (attach reuses the matched chain, publish pins it), and the
    /// frame's refcount counts exactly those occurrences. `migrate_into` moves the full refcount,
    /// so ALL chains naming the frame must be rewritten, not just the entry being sunk/promoted —
    /// otherwise an ancestor would keep a dangling id for a freed slot.
    fn rewrite_frame_in_entries(&mut self, source: FrameId, fresh: FrameId) {
        for node in &mut self.nodes {
            if let Some(entry) = node.entry.as_mut() {
                for f in entry.chain.iter_mut() {
                    if *f == source {
                        *f = fresh;
                    }
                }
            }
        }
    }

    /// The suffix of an entry's `tier` run that may move one tier down, as `[keep, end)`.
    ///
    /// Two rules shape it. A chain stays a monotone gradient — device prefix, then RAM, then disk —
    /// because [`Self::entry_chain_promoted`] truncates reuse at the first frame it has to fetch: a
    /// host frame in the middle of the device prefix would throw away every page behind it. And the
    /// last hit's reuse depth is a floor on the VRAM run, since those are the frames a next hit
    /// would DMA straight back.
    ///
    /// The floor is a preference, not a veto: when it covers the whole run the run is offered
    /// anyway. Bytes move down a tier before an entry is deleted, and a sinker that refused here
    /// would hand a fully warm chain to the evictor instead.
    fn sink_region(&self, entry: &ChainEntry, tier: Tier, protect: &[FrameId]) -> (usize, usize) {
        fn rank(tier: Tier) -> u8 {
            match tier {
                Tier::Vram => 0,
                Tier::Ram => 1,
                Tier::Disk => 2,
            }
        }
        let want = rank(tier);
        let start = entry
            .chain
            .iter()
            .take_while(|frame| rank(frame.tier()) < want)
            .count();
        let mut end = start;
        while end < entry.chain.len() {
            let frame = entry.chain[end];
            if rank(frame.tier()) != want || self.pool.is_active(frame) || protect.contains(&frame)
            {
                break;
            }
            end += 1;
        }
        let keep = if tier == Tier::Vram {
            entry.warm_frames.min(end)
        } else {
            start
        };
        if keep >= end {
            (start, end)
        } else {
            (keep, end)
        }
    }

    /// DMA the coldest tree entry's cold TAIL into the RAM tier and rewrite those frames to RAM
    /// ids, leaving its hot prefix in VRAM. Skips a run with any ACTIVE frame (a session table
    /// points there — migration would corrupt it) and stops at the first frame already sunk, so
    /// the chain keeps its gradient. Returns whether at least one frame left VRAM.
    ///
    /// The tail is bounded by free RAM, not by the chain length: a long conversation whose whole
    /// chain cannot fit used to be skipped here and then DELETED by the evictor, losing a prefix
    /// that a partial sink keeps addressable. A DMA failure rolls back frame by frame — whatever
    /// already sank stays sunk (its bytes are safe in the backing), the rest stay in VRAM.
    fn demote_coldest_entry<M: FrameDma + ?Sized>(&mut self, be: &M, protect: &[FrameId]) -> bool {
        if self.pool.capacity_frames(Tier::Ram) == 0 || self.pool.free_frames(Tier::Ram) == 0 {
            return false;
        }
        let mut best: Option<(usize, usize, usize)> = None; // (node, keep, end)
        let mut best_key: Option<(bool, u64, usize)> = None;
        for (index, node) in self.nodes.iter().enumerate() {
            let Some(entry) = node.entry.as_ref() else {
                continue;
            };
            let (keep, end) = self.sink_region(entry, Tier::Vram, protect);
            if keep >= end {
                continue;
            }
            let key = (self.has_entry_below(index), node.last_access, index);
            if best_key.is_none_or(|current| key < current) {
                best_key = Some(key);
                best = Some((index, keep, end));
            }
        }
        let Some((node, keep, end)) = best else {
            return false;
        };
        let moved = self.sink_entry_tail(be, node, keep, end);
        if moved > 0 && keep > 0 {
            self.tail_sinks += 1;
        }
        moved > 0
    }

    /// Move the tail of one entry's VRAM run `[keep, end)` into the RAM tier, highest frame first.
    /// Returns how many frames left VRAM.
    ///
    /// Taking from the TAIL is what keeps the chain a gradient: sinking from `keep` forward would
    /// move the hot end and leave device frames behind host ones, and promotion truncates at the
    /// first host frame. The take is bounded by free RAM, not by the run length, so a conversation
    /// longer than the host tier keeps a mixed chain instead of being skipped and then deleted. A
    /// DMA failure rolls back frame by frame — whatever already sank stays sunk (its bytes are safe
    /// in the backing), the rest stay in VRAM.
    fn sink_entry_tail<M: FrameDma + ?Sized>(
        &mut self,
        be: &M,
        node: usize,
        keep: usize,
        end: usize,
    ) -> usize {
        let take = (end - keep).min(self.pool.free_frames(Tier::Ram));
        let frame_bytes = self.pool.frame_bytes() as usize;
        let mut moved = 0usize;
        for i in (end - take)..end {
            let source = self.nodes[node].entry.as_ref().unwrap().chain[i];
            let Some(fresh) = self.pool.alloc(Tier::Ram) else {
                break; // RAM filled mid-tail: the entry keeps a mixed chain, still consistent
            };
            let backing_off = fresh.slot() * frame_bytes;
            if self.ram_backing.len() < backing_off + frame_bytes {
                self.ram_backing.resize(backing_off + frame_bytes, 0);
            }
            if let Err(error) = be.download_frame(
                self.pool.frame_addr(source),
                &mut self.ram_backing[backing_off..backing_off + frame_bytes],
            ) {
                self.pool.unpin(fresh);
                tracing::warn!(
                    "paged KV could not DMA frame {} to RAM; keeping it in VRAM: {error}",
                    source.slot()
                );
                break;
            }
            self.rewrite_frame_in_entries(source, fresh);
            self.pool.migrate_into(source, fresh);
            self.tier_demotions += 1;
            moved += 1;
        }
        moved
    }

    /// Phase D: write one frame's bytes to its disk slot, growing the sparse file to cover it.
    /// Field-disjoint borrows (`disk` vs `ram_backing`) let the caller keep the backing slice
    /// alive while the file handle is mutated — no copy through a temp buffer.
    fn disk_write(
        disk: &mut DiskBacking,
        frame_bytes: usize,
        slot: usize,
        src: &[u8],
    ) -> Result<()> {
        let off = slot as u64 * frame_bytes as u64;
        let need = off + frame_bytes as u64;
        if disk.high_water < need {
            disk.file
                .set_len(need)
                .map_err(|error| anyhow!("disk backing set_len {need}: {error}"))?;
            disk.high_water = need;
        }
        disk.file
            .seek(SeekFrom::Start(off))
            .map_err(|error| anyhow!("disk backing seek {off}: {error}"))?;
        disk.file
            .write_all(src)
            .map_err(|error| anyhow!("disk backing write slot {slot}: {error}"))
    }

    /// Phase D: read one frame's bytes from its disk slot. A short read (file truncated under an
    /// external tool) is an error: the caller keeps the frame where it is rather than DMA holes.
    fn disk_read(
        disk: &mut DiskBacking,
        frame_bytes: usize,
        slot: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        let off = slot as u64 * frame_bytes as u64;
        if disk.high_water < off + frame_bytes as u64 {
            return Err(anyhow!(
                "disk backing has no bytes for slot {slot} (high_water {})",
                disk.high_water
            ));
        }
        disk.file
            .seek(SeekFrom::Start(off))
            .map_err(|error| anyhow!("disk backing seek {off}: {error}"))?;
        disk.file
            .read_exact(dst)
            .map_err(|error| anyhow!("disk backing read slot {slot}: {error}"))
    }

    /// Sink the coldest tree entry's RAM TAIL to the disk tier, one tier down the same gradient.
    /// The device prefix stays in RAM/VRAM and the frames behind it move to disk, so a chain that
    /// only partly fit RAM can still relieve RAM pressure instead of blocking the cascade until
    /// it is evicted. A write failure keeps the chain consistent — whatever already sank stays
    /// sunk.
    fn demote_ram_coldest_entry(&mut self, protect: &[FrameId]) -> bool {
        if self.disk.is_none()
            || self.pool.capacity_frames(Tier::Disk) == 0
            || self.pool.free_frames(Tier::Disk) == 0
        {
            return false;
        }
        let mut best: Option<(usize, usize, usize)> = None; // (node, keep, end)
        let mut best_key: Option<(bool, u64, usize)> = None;
        for (index, node) in self.nodes.iter().enumerate() {
            let Some(entry) = node.entry.as_ref() else {
                continue;
            };
            let (keep, end) = self.sink_region(entry, Tier::Ram, protect);
            if keep >= end {
                continue;
            }
            let key = (self.has_entry_below(index), node.last_access, index);
            if best_key.is_none_or(|current| key < current) {
                best_key = Some(key);
                best = Some((index, keep, end));
            }
        }
        let Some((node, keep, end)) = best else {
            return false;
        };
        let take = (end - keep).min(self.pool.free_frames(Tier::Disk));
        let frame_bytes = self.pool.frame_bytes() as usize;
        let mut moved = 0usize;
        for i in (end - take)..end {
            let source = self.nodes[node].entry.as_ref().unwrap().chain[i];
            let Some(fresh) = self.pool.alloc(Tier::Disk) else {
                break; // disk filled mid-chain: mixed chain, still consistent
            };
            let backing_off = source.slot() * frame_bytes;
            if backing_off + frame_bytes > self.ram_backing.len() {
                // Bookkeeping can only reach here if the backing shrank under a bug.
                self.pool.unpin(fresh);
                tracing::warn!(
                    "paged KV found no RAM backing for sunk frame {}; leaving the entry in RAM",
                    source.slot()
                );
                break;
            }
            let write_result = match self.disk.as_mut() {
                Some(disk) => Self::disk_write(
                    disk,
                    frame_bytes,
                    fresh.slot(),
                    &self.ram_backing[backing_off..backing_off + frame_bytes],
                ),
                None => Err(anyhow!("disk tier closed")),
            };
            if let Err(error) = write_result {
                self.pool.unpin(fresh);
                tracing::warn!(
                    "paged KV could not write frame {} to disk; keeping it in RAM: {error}",
                    source.slot()
                );
                break;
            }
            self.rewrite_frame_in_entries(source, fresh);
            self.pool.migrate_into(source, fresh);
            self.disk_demotions += 1;
            moved += 1;
        }
        moved > 0
    }

    /// A restoring session takes tree-owned frames: gain one owner each and mark them active
    /// (Phase C: active frames are never DMA candidates). The tree keeps its own owner — the
    /// entry stays published, so a second session can share the same prefix frames concurrently.
    pub(crate) fn session_attach(&mut self, frames: &[FrameId]) {
        for &frame in frames {
            self.pool.pin(frame);
            self.pool.set_active(frame, true);
        }
    }

    /// A session's table stopped pointing at these frames: drop its owners. Frames the tree still
    /// owns stay held; the last owner's release returns the slot to the corridor free list.
    pub(crate) fn session_release(&mut self, frames: &[FrameId]) {
        for &frame in frames {
            self.pool.set_active(frame, false);
            self.pool.unpin(frame);
        }
    }

    // ---- tree operations --------------------------------------------------------------------

    /// Longest page-aligned prefix any entry's chain can serve for `prompt`. Unlike the blob
    /// tier's whole-snapshot continuation rule, a chain is positional from token 0, so ANY walked
    /// prefix is shareable: the walk consumes `prompt` along the spine, `floor(consumed / page)`
    /// complete pages are reusable, and the deepest candidate node whose entry chain covers that
    /// many pages hands them out. Candidates, deepest first: the partially-matched edge child (the
    /// prompt diverges *inside* its edge — its chain still covers the shared prefix), the spine
    /// path nodes, and — when the prompt was fully consumed — entries deeper in the subtree (the
    /// superseded-ancestor case). Counts one hit/miss and stamps the chosen node + its frames.
    pub(crate) fn best_match(
        &mut self,
        prompt: &[u32],
        projection: Option<bool>,
    ) -> Option<(usize, usize)> {
        let (path, consumed, partial) = self.walk(self.root(projection), prompt);
        let pages = consumed / self.page_tokens;
        if pages == 0 {
            self.misses += 1;
            return None;
        }
        let mut candidates: Vec<usize> = Vec::new();
        candidates.extend(partial);
        candidates.extend(path.iter().rev().copied());
        // Every entry at or below the walk's frontier shares the query's first `consumed` tokens, so
        // its first `pages` frames cover a prefix of the query — including when the query diverges
        // INSIDE an edge, or stops at a node with no matching child. Scanning only the fully
        // consumed case left a fork that edits history matching nothing while the frames it needed
        // sat one node deeper, behind the entry-less node the split created.
        if let Some(base) = partial.or_else(|| path.last().copied()) {
            let mut stack = vec![base];
            while let Some(index) = stack.pop() {
                if self.nodes[index].entry.is_some() {
                    candidates.push(index);
                }
                stack.extend(self.nodes[index].children.values().copied());
            }
        }
        let depth = pages * self.page_tokens;
        let n_candidates = candidates.len();
        let detail = candidates
            .iter()
            .map(|&node| {
                let entry = self.nodes[node].entry.as_ref();
                (
                    node,
                    self.nodes[node].path_len,
                    entry.map(|entry| entry.chain.len()),
                    entry.map(|entry| {
                        entry
                            .ckpts
                            .iter()
                            .map(|ckpt| ckpt.boundary_tokens)
                            .collect::<Vec<_>>()
                    }),
                )
            })
            .collect::<Vec<_>>();
        // A recurrent entry resumes only at a snapshot boundary, so the longest chain is not the
        // best candidate: one whose snapshots all sit above `depth` matches and then hands back
        // zero tokens. Rank by the depth actually resumable, first-wins on ties, stop at a full one.
        let mut chosen = None;
        let mut best_resumable = 0usize;
        for node in candidates {
            let Some(entry) = self.nodes[node].entry.as_ref() else {
                continue;
            };
            if entry.chain.len() < pages {
                continue;
            }
            let resumable = if self.recurrent {
                entry
                    .ckpts
                    .iter()
                    .map(|ckpt| ckpt.boundary_tokens)
                    .filter(|&boundary| boundary <= depth)
                    .max()
                    .unwrap_or(0)
            } else {
                depth
            };
            if chosen.is_none() || resumable > best_resumable {
                chosen = Some(node);
                best_resumable = resumable;
                if resumable == depth {
                    break;
                }
            }
        }
        let Some(node) = chosen else {
            self.misses += 1;
            tracing::debug!(
                query = prompt.len(),
                consumed,
                pages,
                candidates = n_candidates,
                detail = ?detail,
                "paged KV match miss"
            );
            return None;
        };
        self.hits += 1;
        self.clock += 1;
        self.nodes[node].last_access = self.clock;
        let entry = self.nodes[node].entry.as_mut().unwrap();
        for &frame in &entry.chain[..pages] {
            self.pool.touch(frame);
        }
        Some((node, depth))
    }

    /// Read-only form of the chain accessor below: the entry node's frames covering `reusable`
    /// tokens, tiers untouched. Only the unit tests read raw (possibly RAM-tier) chains — every
    /// production attach path must go through [`Self::entry_chain_promoted`] so a RAM address can
    /// never reach a page table.
    #[cfg(test)]
    fn entry_chain(&self, node: usize, reusable: usize) -> Vec<FrameId> {
        let entry = self.nodes[node]
            .entry
            .as_ref()
            .expect("matched node carries an entry");
        entry.chain[..reusable / self.page_tokens].to_vec()
    }

    /// The production chain accessor: every RAM frame in the matched prefix is DMA'd
    /// back into a freshly claimed VRAM slot BEFORE returning, so the caller only ever handles
    /// device-addressable frames (the page table never sees a RAM address). VRAM claims go
    /// through [`Self::claim_frame`], so a full corridor sinks/evicts other cold entries first —
    /// the just-hit chain is skipped by the sinker because [`Self::best_match`] refreshed its
    /// LRU stamp. If VRAM cannot yield a slot for frame `k`, the chain truncates at `k` frames:
    /// the caller recomputes the reuse length from the returned length (`len * page_tokens()`),
    /// and the un-promoted tail stays sunk in RAM for a later hit. A mid-chain DMA failure
    /// truncates the same way; already-promoted frames are committed to the tree chain.
    pub(crate) fn entry_chain_promoted<M: FrameDma + ?Sized>(
        &mut self,
        be: &M,
        node: usize,
        reusable: usize,
    ) -> Vec<FrameId> {
        let pages = reusable / self.page_tokens;
        // Stamp the depth the caller will actually attach, which is NOT the depth `best_match`
        // matched: a recurrent entry only resumes at a snapshot boundary, so `hit_prefix` can cut
        // a 200-page token match down to nothing. Keeping the matched depth would pin a prefix no
        // request can read and force eviction of bytes that WOULD have been reused.
        self.nodes[node]
            .entry
            .as_mut()
            .expect("matched node carries an entry")
            .warm_frames = pages;
        let frame_bytes = self.pool.frame_bytes() as usize;
        let mut out = Vec::with_capacity(pages);
        for i in 0..pages {
            let frame = self.nodes[node]
                .entry
                .as_ref()
                .expect("matched node carries an entry")
                .chain[i];
            if matches!(frame.tier(), Tier::Vram) {
                out.push(frame);
                continue;
            }
            // Phase D: a disk frame must first land in a RAM slot — the upload below reads from
            // the host backing, and the disk->RAM read may itself cascade (RAM full: sink other
            // cold RAM entries to disk).
            let ram = match frame.tier() {
                Tier::Ram => frame,
                Tier::Disk => {
                    let Some(ram) = self.disk_to_ram(frame, &out) else {
                        break; // no RAM room / read failure: truncate, the tail stays on disk
                    };
                    ram
                }
                Tier::Vram => unreachable!("handled above"),
            };
            let backing_off = ram.slot() * frame_bytes;
            if backing_off + frame_bytes > self.ram_backing.len() {
                // Bookkeeping can only reach here if the backing shrank under a bug; refuse the
                // promotion rather than DMA garbage bytes into the corridor.
                tracing::warn!(
                    "paged KV promotion found no backing for RAM frame {}",
                    ram.slot()
                );
                break;
            }
            // The in-flight RAM frame is protected alongside the already-promoted VRAM ids: the
            // VRAM claim below can cascade a RAM->disk sink, and if that sink moved an entry
            // sharing THIS frame, `rewrite_frame_in_entries` would strand `backing_off`'s owner.
            let mut protect = out.clone();
            protect.push(ram);
            let Some(fresh) = self.claim_frame(be, false, &protect) else {
                break; // corridor cannot yield: truncate the reuse here
            };
            if let Err(error) = be.upload_frame(
                self.pool.frame_addr(fresh),
                &self.ram_backing[backing_off..backing_off + frame_bytes],
            ) {
                self.pool.unpin(fresh);
                tracing::warn!(
                    "paged KV could not DMA frame {} back to VRAM; re-prefilling from here: {error}",
                    ram.slot()
                );
                break;
            }
            self.rewrite_frame_in_entries(ram, fresh);
            self.pool.migrate_into(ram, fresh);
            self.tier_promotions += 1;
            out.push(fresh);
        }
        out
    }

    /// Phase D: bring one disk frame up into a fresh RAM slot (bytes read from the sparse
    /// backing). RAM pressure is relieved by the same cascade the VRAM claim uses — sink the
    /// coldest all-RAM entry to disk — and the just-converted frame is NOT yet in any chain the
    /// sinker can see as a victim candidate problem: `rewrite_frame_in_entries` keeps every
    /// chain naming the disk id consistent across the conversion. `None` = no RAM slot even
    /// after cascading, or a read failure; the caller truncates the promotion there.
    fn disk_to_ram(&mut self, disk_id: FrameId, protect: &[FrameId]) -> Option<FrameId> {
        let frame_bytes = self.pool.frame_bytes() as usize;
        loop {
            let Some(ram) = self.pool.alloc(Tier::Ram) else {
                // RAM full: make room the cheap way first (sink a cold all-RAM entry to disk).
                if !self.demote_ram_coldest_entry(protect) {
                    return None;
                }
                continue;
            };
            let backing_off = ram.slot() * frame_bytes;
            if self.ram_backing.len() < backing_off + frame_bytes {
                self.ram_backing.resize(backing_off + frame_bytes, 0);
            }
            let read_result = match self.disk.as_mut() {
                Some(disk) => Self::disk_read(
                    disk,
                    frame_bytes,
                    disk_id.slot(),
                    &mut self.ram_backing[backing_off..backing_off + frame_bytes],
                ),
                None => Err(anyhow!("disk tier closed")),
            };
            if let Err(error) = read_result {
                self.pool.unpin(ram);
                tracing::warn!(
                    "paged KV could not read disk frame {} back into RAM: {error}",
                    disk_id.slot()
                );
                return None;
            }
            self.rewrite_frame_in_entries(disk_id, ram);
            self.pool.migrate_into(disk_id, ram);
            self.disk_promotions += 1;
            return Some(ram);
        }
    }

    /// Publish a demoted session's conversation with no recurrent snapshots. Gated with its only
    /// callers (the tree's own tests): every production publish goes through
    /// [`Self::publish_with_ckpts`], because a hybrid recurrent entry without a snapshot is a
    /// conversation the next request has to replay.
    #[cfg(test)]
    pub(crate) fn publish(&mut self, tokens: Vec<u32>, chain: Vec<FrameId>) -> Result<()> {
        self.publish_with_ckpts(tokens, chain, Vec::new(), None)
    }

    /// [`Self::publish`] plus host snapshots of the conversation's recurrent state at
    /// page-aligned boundaries (hybrid recurrent models — see [`PagedCkpt`]). Every snapshot must
    /// describe a page-aligned prefix of the published tokens; anything else is a capture bug and
    /// is refused BEFORE any frame is pinned, so a rejected publish never leaks ownership.
    ///
    /// On a recurrent tree a snapshot is not an accelerator but the entry's only resume point, so
    /// a conversation arriving without a usable one is refused on the same terms: publishing it
    /// would pin a chain that every later hit resolves to zero reusable tokens.
    ///
    /// `projection` selects the tree the chain is published into (see [`Self::root`]); a lookup
    /// under a different setting cannot reach it.
    ///
    /// A publish that lands DEEPER than an existing entry supersedes it, and inherits the snapshot
    /// that entry resumed at: the branch point is still a prefix of these tokens, so a later request
    /// forking there attaches instead of re-prefilling. Inheritance moves aux bytes between entries
    /// rather than adding them — the ancestor's bytes leave the budget before admission re-adds them.
    pub(crate) fn publish_with_ckpts(
        &mut self,
        tokens: Vec<u32>,
        chain: Vec<FrameId>,
        mut ckpts: Vec<Arc<PagedCkpt>>,
        projection: Option<bool>,
    ) -> Result<()> {
        let pages = tokens.len() / self.page_tokens;
        if chain.len() != pages {
            return Err(anyhow!(
                "paged publish: {} tokens cover {pages} pages but the chain carries {}",
                tokens.len(),
                chain.len()
            ));
        }
        for ckpt in &ckpts {
            anyhow::ensure!(
                ckpt.boundary_tokens > 0
                    && ckpt.boundary_tokens % self.page_tokens == 0
                    && ckpt.boundary_tokens <= tokens.len(),
                "paged publish: checkpoint boundary {} is not a positive page-aligned prefix of {} tokens",
                ckpt.boundary_tokens,
                tokens.len()
            );
        }
        if self.recurrent {
            match ckpts.iter().max_by_key(|ckpt| ckpt.boundary_tokens) {
                None => {
                    self.aux_refusals += 1;
                    return Err(anyhow!(
                        "paged publish: a recurrent conversation is resumable only at a snapshot \
                         boundary, and none was captured"
                    ));
                }
                // Admission never trims the deepest snapshot, so it has to fit on its own.
                Some(anchor) if anchor.bytes() > self.aux_budget => {
                    self.aux_refusals += 1;
                    return Err(anyhow!(
                        "paged publish: the deepest recurrent snapshot needs {} B but the aux budget is {} B",
                        anchor.bytes(),
                        self.aux_budget
                    ));
                }
                Some(_) => {}
            }
        }
        for &frame in &chain {
            self.pool.pin(frame);
        }
        let node = self.descend_to(self.root(projection), &tokens);
        let mut inherited = self.drop_superseded_ancestors(node);
        inherited.sort_by_key(|ckpt| ckpt.boundary_tokens);
        inherited.dedup_by_key(|ckpt| ckpt.boundary_tokens);
        // A boundary this publish captured itself wins: its state belongs to the chain being
        // published, and counting it twice would charge the aux budget for one snapshot twice.
        inherited.retain(|ckpt| {
            !ckpts
                .iter()
                .any(|own| own.boundary_tokens == ckpt.boundary_tokens)
        });
        self.ckpts_inherited += inherited.len() as u64;
        let inherited_now = inherited.len();
        ckpts.extend(inherited);
        self.clock += 1;
        // The replaced entry leaves the aux budget NOW, before admission: its frames stay pinned
        // by the new chain (or by their session owners), and counting its snapshots against the
        // incoming ones would refuse a re-publish of the same conversation that trivially fits.
        let old = self.nodes[node].entry.take();
        if let Some(old) = &old {
            self.aux_bytes = self.aux_bytes.saturating_sub(ckpts_bytes(&old.ckpts));
        }
        let ckpts = self.admit_ckpts(ckpts, node);
        tracing::debug!(
            node,
            depth = tokens.len(),
            pages = chain.len(),
            inherited = inherited_now,
            snapshots = ?ckpts.iter().map(|ckpt| ckpt.boundary_tokens).collect::<Vec<_>>(),
            "paged KV publish"
        );
        self.nodes[node].entry = Some(ChainEntry {
            chain,
            ckpts,
            warm_frames: 0,
        });
        if let Some(old) = old {
            for frame in old.chain {
                self.pool.unpin(frame);
            }
        }
        self.nodes[node].last_access = self.clock;
        self.demotions += 1;
        Ok(())
    }

    /// Drop one entry's tree ownership: frames unpinned, aux bytes subtracted. Callers bump their
    /// own counters (evictions vs supersessions) around it.
    fn unpin_entry(&mut self, entry: ChainEntry) {
        for frame in entry.chain {
            self.pool.unpin(frame);
        }
        self.aux_bytes = self.aux_bytes.saturating_sub(ckpts_bytes(&entry.ckpts));
    }

    /// Fit `ckpts` (sorted ascending by boundary) under the aux budget by yielding the coldest
    /// OTHER entries first, then the incoming snapshots shallowest-first. Returns the snapshots
    /// that fit and adds their bytes to `aux_bytes`. `incoming` is the node being published: never
    /// a victim, because pruning it could free the index the caller still holds.
    ///
    /// What yielding costs depends on the model. A dense entry ignores its snapshots, so they leave
    /// alone and its chain stays matchable. A recurrent entry resumes ONLY through them, so taking
    /// its last one would leave a chain that matches and then hands back zero tokens — the whole
    /// entry is evicted instead, returning its frames to the pool.
    fn admit_ckpts(
        &mut self,
        mut ckpts: Vec<Arc<PagedCkpt>>,
        incoming: usize,
    ) -> Vec<Arc<PagedCkpt>> {
        ckpts.sort_by_key(|ckpt| ckpt.boundary_tokens);
        while !ckpts.is_empty() && self.aux_bytes + ckpts_bytes(&ckpts) > self.aux_budget {
            let Some(node) = self.coldest_ckpt_entry(incoming) else {
                break;
            };
            if self.recurrent {
                let entry = self.nodes[node]
                    .entry
                    .take()
                    .expect("victim carries an entry");
                self.unpin_entry(entry);
                self.evictions += 1;
                self.prune_upward(node);
            } else {
                let dropped = std::mem::take(
                    &mut self.nodes[node]
                        .entry
                        .as_mut()
                        .expect("victim carries an entry")
                        .ckpts,
                );
                self.aux_bytes = self.aux_bytes.saturating_sub(ckpts_bytes(&dropped));
            }
        }
        // Still over after every other entry yielded: the shallowest incoming snapshots go. The
        // deepest one is what a same-conversation resume attaches, and on a recurrent tree it is
        // the entry's only way in — it fits the budget alone, either because the publish pre-flight
        // proved it or because an inherited snapshot was admitted under this same budget.
        let keep = usize::from(self.recurrent && !ckpts.is_empty());
        while self.aux_bytes + ckpts_bytes(&ckpts) > self.aux_budget && ckpts.len() > keep {
            ckpts.remove(0);
        }
        self.aux_bytes += ckpts_bytes(&ckpts);
        ckpts
    }

    /// The coldest entry that still carries snapshots, in [`Self::evict_cold_one`]'s order,
    /// ignoring `skip` (the node a publish is about to replace).
    fn coldest_ckpt_entry(&self, skip: usize) -> Option<usize> {
        let mut best: Option<(bool, u64, usize)> = None;
        for (index, node) in self.nodes.iter().enumerate() {
            if index == skip
                || node
                    .entry
                    .as_ref()
                    .is_none_or(|entry| entry.ckpts.is_empty())
            {
                continue;
            }
            let key = (self.has_entry_below(index), node.last_access, index);
            if best.is_none_or(|current| key < current) {
                best = Some(key);
            }
        }
        best.map(|(_, _, index)| index)
    }

    /// Deepest host snapshot at or below `reusable` tokens. A hybrid recurrent session must attach
    /// at EXACTLY `boundary_tokens`: the snapshot describes the recurrent/PLE state there, so
    /// reusing fewer frames wastes a resumable prefix and reusing more would resume on state that
    /// does not exist. `None` = the entry's chain is a pure attention prefix for this prompt.
    pub(crate) fn entry_ckpt(&self, node: usize, reusable: usize) -> Option<Arc<PagedCkpt>> {
        self.nodes[node]
            .entry
            .as_ref()?
            .ckpts
            .iter()
            .filter(|ckpt| ckpt.boundary_tokens <= reusable)
            .max_by_key(|ckpt| ckpt.boundary_tokens)
            .cloned()
    }

    /// The deepest prefix of `node`'s chain a request may actually resume, plus the snapshot to
    /// restore there.
    ///
    /// A dense-attention entry resumes at whatever depth it matched: its pages hold every K/V row
    /// the continuation reads. A hybrid recurrent entry is pinned to its snapshot boundaries,
    /// because the conv window, the recurrent matrix and the PLE history summarize the whole prefix
    /// into buffers no page table addresses. `depth` is in tokens, and callers evaluate this twice —
    /// once for the match, once for the chain that survived promotion to VRAM — so a short promotion
    /// re-picks a shallower snapshot instead of resuming ahead of the KV.
    ///
    /// `0` from a recurrent entry says the chain cannot be resumed at all: the caller's strict
    /// comparison against the resident continuation then fails and the prompt is re-prefilled.
    pub(crate) fn hit_prefix(
        &self,
        node: usize,
        recurrent: bool,
        depth: usize,
    ) -> (usize, Option<Arc<PagedCkpt>>) {
        if recurrent {
            let ckpt = self.entry_ckpt(node, depth);
            let out = (ckpt.as_ref().map_or(0, |ckpt| ckpt.boundary_tokens), ckpt);
            tracing::debug!(
                node,
                matched = depth,
                resumable = out.0,
                entry_depth = self.nodes[node].path_len,
                chain = ?self.nodes[node].entry.as_ref().map(|entry| entry.chain.len()),
                snapshots = ?self.nodes[node].entry.as_ref().map(|entry| {
                    entry.ckpts.iter().map(|ckpt| ckpt.boundary_tokens).collect::<Vec<_>>()
                }),
                "paged KV hit"
            );
            out
        } else {
            (depth, None)
        }
    }

    /// Drop the coldest unpublishable... evict one entry leaf-first (see module doc). Returns
    /// whether an entry was dropped. Frames still owned by live sessions survive the unpin; the
    /// freed slots surface through the pool's refcount, not here.
    fn evict_cold_one(&mut self) -> bool {
        let mut best: Option<(bool, u64, usize)> = None;
        for (index, node) in self.nodes.iter().enumerate() {
            if node.entry.is_none() {
                continue;
            }
            let has_entry_child = self.has_entry_below(index);
            let key = (has_entry_child, node.last_access, index);
            if best.is_none_or(|current| key < current) {
                best = Some(key);
            }
        }
        let Some((_, _, node)) = best else {
            return false;
        };
        if let Some(entry) = self.nodes[node].entry.take() {
            self.unpin_entry(entry);
            self.evictions += 1;
        }
        self.prune_upward(node);
        true
    }

    fn has_entry_below(&self, node: usize) -> bool {
        self.nodes[node]
            .children
            .values()
            .copied()
            .any(|child| self.nodes[child].entry.is_some() || self.has_entry_below(child))
    }

    /// Walk `tokens` from the root: the fully-consumed spine `path` (always includes the root),
    /// the number of leading `tokens` matched against the tree (`consumed`), and the child whose
    /// edge matched partially — the node where the query diverges *inside* an edge, whose entry
    /// chain still covers the shared prefix.
    fn walk(&self, root: usize, tokens: &[u32]) -> (Vec<usize>, usize, Option<usize>) {
        let mut path = vec![root];
        let mut node = root;
        let mut index = 0usize;
        while index < tokens.len() {
            let Some(&child) = self.nodes[node].children.get(&tokens[index]) else {
                return (path, index, None);
            };
            let edge = &self.nodes[child].edge;
            let mut offset = 0usize;
            while offset < edge.len()
                && index + offset < tokens.len()
                && edge[offset] == tokens[index + offset]
            {
                offset += 1;
            }
            if offset == edge.len() {
                path.push(child);
                node = child;
                index += offset;
                continue;
            }
            // Partial edge: divergence (or query end) inside it.
            return (path, index + offset, Some(child));
        }
        (path, tokens.len(), None)
    }

    /// Claim a node slot: a recycled index when the free list has one, else a fresh push. Tree
    /// node indices never escape a locked operation (unlike the blob tier's `TakenSnapshot`),
    /// so reuse needs no generation guard.
    fn alloc_node(&mut self, parent: usize, edge: Vec<u32>, path_len: usize) -> usize {
        if let Some(index) = self.free.pop() {
            let node = &mut self.nodes[index];
            node.parent = Some(parent);
            node.edge = edge;
            node.children.clear();
            node.path_len = path_len;
            node.entry = None;
            node.last_access = 0;
            index
        } else {
            self.nodes.push(Node::new(Some(parent), edge, path_len));
            self.nodes.len() - 1
        }
    }

    /// Return a detached node's index to the free list. Callers must have already unlinked it
    /// from its parent's children and proved `entry` gone and children empty (or re-homed).
    fn free_node(&mut self, node: usize) {
        self.free.push(node);
        let node = &mut self.nodes[node];
        node.parent = None;
        node.edge = Vec::new();
        node.children.clear();
        node.entry = None;
    }

    fn descend_to(&mut self, root: usize, tokens: &[u32]) -> usize {
        let mut node = root;
        let mut index = 0usize;
        while index < tokens.len() {
            let first = tokens[index];
            match self.nodes[node].children.get(&first).copied() {
                Some(child) => {
                    let edge = &self.nodes[child].edge;
                    let mut offset = 0usize;
                    while offset < edge.len()
                        && index + offset < tokens.len()
                        && edge[offset] == tokens[index + offset]
                    {
                        offset += 1;
                    }
                    if offset == edge.len() {
                        node = child;
                        index += offset;
                        continue;
                    }
                    let shared = edge[..offset].to_vec();
                    let remainder = edge[offset..].to_vec();
                    let path_len = self.nodes[node].path_len + offset;
                    let mid = self.alloc_node(node, shared, path_len);
                    let remainder_len = remainder.len();
                    self.nodes[child].edge = remainder;
                    self.nodes[child].parent = Some(mid);
                    self.nodes[child].path_len = path_len + remainder_len;
                    let remainder_first = self.nodes[child].edge[0];
                    self.nodes[mid].children.insert(remainder_first, child);
                    self.nodes[node].children.insert(first, mid);
                    node = mid;
                    index += offset;
                }
                None => {
                    let path_len = self.nodes[node].path_len + (tokens.len() - index);
                    let leaf = self.alloc_node(node, tokens[index..].to_vec(), path_len);
                    self.nodes[node].children.insert(first, leaf);
                    node = leaf;
                    break;
                }
            }
        }
        node
    }

    /// Unpin ancestor entry chains superseded by the chain just published at `node`, and hand back
    /// the snapshot each of them resumes at. The new entry already holds owners on those frames, so
    /// the ancestors' owners leaving cannot free anything the new chain still references. Their
    /// deepest snapshot describes a prefix of `node`'s tokens — the radix path guarantees it — so it
    /// is still resumable, and it is the ONLY state that can resume a fork branching at that depth:
    /// deleting it left a chain that matches and then hands back zero tokens. Cleared ancestors are
    /// swept AFTER the walk: pruning mid-walk could free a node whose index the walk still has to
    /// visit.
    fn drop_superseded_ancestors(&mut self, node: usize) -> Vec<Arc<PagedCkpt>> {
        let mut cleared = Vec::new();
        let mut parent = self.nodes[node].parent;
        while let Some(index) = parent {
            if self.nodes[index].entry.is_some() {
                cleared.push(index);
            }
            parent = self.nodes[index].parent;
        }
        let mut inherited = Vec::new();
        for index in cleared {
            if let Some(entry) = self.nodes[index].entry.take() {
                // The DEEPEST snapshot only. Shallower ones are either re-captured by this publish
                // (the shared instruction prefix) or describe a branch point two turns back, and
                // inheriting whole sets would grow every entry by its ancestry until the aux budget
                // started evicting entire conversations to pay for it.
                if let Some(anchor) = entry.ckpts.iter().max_by_key(|ckpt| ckpt.boundary_tokens) {
                    inherited.push(anchor.clone());
                }
                self.unpin_entry(entry);
            }
            self.prune_upward(index);
        }
        inherited
    }

    /// Detach entry-less, child-less nodes upward and merge pass-through nodes into their only
    /// child, so evicted conversations neither accumulate token spines nor node indices. The
    /// merge keeps every path's token string (and therefore every match) exactly as it was.
    fn prune_upward(&mut self, mut node: usize) {
        loop {
            let Some(parent) = self.nodes[node].parent else {
                return; // root, or already detached on an earlier pass
            };
            let detach = {
                let node = &self.nodes[node];
                node.entry.is_none() && node.children.is_empty()
            };
            if detach {
                if let Some(first) = self.nodes[node].edge.first().copied() {
                    self.nodes[parent].children.remove(&first);
                }
                self.free_node(node);
                node = parent; // the parent may have just degenerated
                continue;
            }
            match self.merge_single_child(node) {
                Some(grandparent) => node = grandparent,
                None => return,
            }
        }
    }

    /// Merge `node` into its only child when it is a pure pass-through (no entry, exactly one
    /// child, not the root): the child's edge grows by `node`'s and the child takes `node`'s
    /// place under the grandparent — same children key, because the merged edge still starts
    /// with `node`'s first token. Returns the grandparent to continue pruning from.
    fn merge_single_child(&mut self, node: usize) -> Option<usize> {
        let (parent, child) = {
            let node = &self.nodes[node];
            if node.entry.is_some() || node.children.len() != 1 {
                return None;
            }
            let parent = node.parent?;
            let child = *node.children.values().next()?;
            (parent, child)
        };
        let key = self.nodes[node].edge.first().copied()?;
        let edge = std::mem::take(&mut self.nodes[node].edge);
        self.nodes[child].edge.splice(0..0, edge);
        self.nodes[child].parent = Some(parent);
        self.nodes[parent].children.insert(key, child);
        self.free_node(node);
        Some(parent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 4;
    const FB: u64 = 1 << 20;

    fn cache(vram_frames: usize) -> PagedKv {
        // Two planes so plane_addr exercises the offset prefix sums. RAM tier off: the tree
        // behaves exactly like Phase B (cold entries evict instead of sink).
        cache_ram(vram_frames, 0)
    }

    fn cache_ram(vram_frames: usize, ram_frames: usize) -> PagedKv {
        let mut kv = PagedKv::new(
            FB,
            PAGE,
            vec![0, FB / 2],
            vram_frames,
            ram_frames,
            0,
            None,
            false,
        );
        kv.set_vram_segments(&[KvFrameCorridor {
            base: 0x1_0000_0000,
            frames: vram_frames,
        }]);
        kv
    }

    /// A tree serving a hybrid recurrent model: an entry resumes ONLY at a snapshot boundary, so
    /// its aux is a correctness constraint rather than a tuning knob.
    fn cache_recurrent(vram_frames: usize, ram_frames: usize) -> PagedKv {
        let mut kv = cache_ram(vram_frames, ram_frames);
        kv.recurrent = true;
        kv
    }

    /// Phase D: a disk-backed tree on a real temp file (the backing is plain `std::fs`, so the
    /// tests exercise the actual sparse-file path — no mock needed below the RAM tier).
    fn cache_disk(vram_frames: usize, ram_frames: usize, disk_frames: usize) -> (PagedKv, PathBuf) {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "infr-paged-kv-disk-test-{}-{}.bin",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&path);
        let mut kv = PagedKv::new(
            FB,
            PAGE,
            vec![0, FB / 2],
            vram_frames,
            ram_frames,
            disk_frames,
            Some(path.clone()),
            false,
        );
        kv.set_vram_segments(&[KvFrameCorridor {
            base: 0x1_0000_0000,
            frames: vram_frames,
        }]);
        (kv, path)
    }

    /// Every chain must stay a monotone gradient: device frames, then RAM, then disk. Promotion
    /// truncates reuse at the first frame it has to fetch, so a host frame sitting in front of a
    /// device one silently discards every page behind it. A sink that takes the wrong end of a
    /// tier run produces exactly that, and no byte round-trip test would notice.
    fn assert_gradient(kv: &PagedKv) {
        fn rank(tier: Tier) -> u8 {
            match tier {
                Tier::Vram => 0,
                Tier::Ram => 1,
                Tier::Disk => 2,
            }
        }
        for (index, node) in kv.nodes.iter().enumerate() {
            let Some(entry) = node.entry.as_ref() else {
                continue;
            };
            for pair in entry.chain.windows(2) {
                assert!(
                    rank(pair[0].tier()) <= rank(pair[1].tier()),
                    "node {index} chain is not a gradient: {:?}",
                    entry.chain.iter().map(|f| f.tier()).collect::<Vec<_>>()
                );
            }
        }
    }

    /// Deterministic "corridor content" for one frame byte: [`MockDma`] writes it on download
    /// and demands to see exactly it on upload, so a sunk-then-promoted frame proves the bytes
    /// round-tripped through the host backing untouched.
    fn pattern(addr: u64, i: usize) -> u8 {
        let x =
            addr.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64).wrapping_mul(0x0100_0000_01B3);
        (x >> 32) as u8
    }

    /// Host-side DMA stand-in: downloads stamp a frame's corridor bytes with [`pattern`],
    /// uploads must replay the pattern of some frame this mock ever downloaded — content
    /// that vanished from VRAM can only come back through the backing it wrote.
    #[derive(Default)]
    struct MockDma {
        downloaded: std::sync::Mutex<Vec<u64>>,
    }

    impl FrameDma for MockDma {
        fn download_frame(&self, frame_addr: u64, dst: &mut [u8]) -> Result<()> {
            for (i, byte) in dst.iter_mut().enumerate() {
                *byte = pattern(frame_addr, i);
            }
            self.downloaded.lock().unwrap().push(frame_addr);
            Ok(())
        }
        fn upload_frame(&self, frame_addr: u64, src: &[u8]) -> Result<()> {
            let replayed = self.downloaded.lock().unwrap().iter().any(|&addr| {
                src.iter()
                    .enumerate()
                    .all(|(i, byte)| *byte == pattern(addr, i))
            });
            if replayed {
                Ok(())
            } else {
                Err(anyhow!(
                    "mock DMA upload of frame slot {frame_addr} matches no downloaded frame"
                ))
            }
        }
    }

    /// A session-shaped chain: `pages` freshly allocated frames (one session owner each). The
    /// throwaway DMA is fine here — growth only claims free corridor frames (demotion needs
    /// pressure a free corridor cannot offer), tests that verify DMA carry their own handle.
    fn grow(kv: &mut PagedKv, pages: usize) -> Vec<FrameId> {
        grow_with(kv, &MockDma::default(), pages)
    }

    fn grow_with<M: FrameDma + ?Sized>(kv: &mut PagedKv, dma: &M, pages: usize) -> Vec<FrameId> {
        (0..pages)
            .map(|_| kv.alloc_frame(dma).expect("corridor has room"))
            .collect()
    }

    /// Shared-arena stand-in for the elastic corridor: grants at most `left` frames of fresh space
    /// per call, re-hands released corridors before carving new ones, and records every release.
    struct ElasticDma {
        dma: MockDma,
        arena: std::sync::Mutex<ElasticArena>,
    }

    #[derive(Default)]
    struct ElasticArena {
        next_base: u64,
        left: usize,
        idle: Vec<KvFrameCorridor>,
        released: Vec<KvFrameCorridor>,
    }

    impl ElasticDma {
        fn new(left: usize) -> Self {
            Self {
                dma: MockDma::default(),
                arena: std::sync::Mutex::new(ElasticArena {
                    next_base: 0x2_0000_0000,
                    left,
                    idle: Vec::new(),
                    released: Vec::new(),
                }),
            }
        }
    }

    impl FrameDma for ElasticDma {
        fn download_frame(&self, frame_addr: u64, dst: &mut [u8]) -> Result<()> {
            self.dma.download_frame(frame_addr, dst)
        }
        fn upload_frame(&self, frame_addr: u64, src: &[u8]) -> Result<()> {
            self.dma.upload_frame(frame_addr, src)
        }
        fn grow_frames(&self, frame_bytes: u64, count: usize) -> Result<Vec<KvFrameCorridor>> {
            let mut arena = self.arena.lock().unwrap();
            let mut out = Vec::new();
            let mut want = count;
            while want > 0 {
                let Some(index) = arena.idle.iter().position(|c| c.frames <= want) else {
                    break;
                };
                let corridor = arena.idle.remove(index);
                want -= corridor.frames;
                out.push(corridor);
            }
            if want > 0 && arena.left > 0 {
                let granted = want.min(arena.left);
                arena.left -= granted;
                let base = arena.next_base;
                arena.next_base += granted as u64 * frame_bytes;
                out.push(KvFrameCorridor {
                    base,
                    frames: granted,
                });
            }
            Ok(out)
        }
        fn release_frames(&self, corridors: &[KvFrameCorridor]) -> Result<()> {
            let mut arena = self.arena.lock().unwrap();
            arena.idle.extend_from_slice(corridors);
            arena.released.extend_from_slice(corridors);
            Ok(())
        }
    }

    /// The cross-owner signal on top of the elastic stand-in: a test flips whether the expert cache
    /// sharing the arena is starving without changing the arena itself.
    struct PressureDma {
        inner: ElasticDma,
        pressure: std::cell::Cell<bool>,
    }

    impl PressureDma {
        fn new(left: usize, pressure: bool) -> Self {
            Self {
                inner: ElasticDma::new(left),
                pressure: std::cell::Cell::new(pressure),
            }
        }
    }

    impl FrameDma for PressureDma {
        fn download_frame(&self, frame_addr: u64, dst: &mut [u8]) -> Result<()> {
            self.inner.download_frame(frame_addr, dst)
        }
        fn upload_frame(&self, frame_addr: u64, src: &[u8]) -> Result<()> {
            self.inner.upload_frame(frame_addr, src)
        }
        fn grow_frames(&self, frame_bytes: u64, count: usize) -> Result<Vec<KvFrameCorridor>> {
            self.inner.grow_frames(frame_bytes, count)
        }
        fn release_frames(&self, corridors: &[KvFrameCorridor]) -> Result<()> {
            self.inner.release_frames(corridors)
        }
        fn expert_vram_pressure(&self) -> bool {
            self.pressure.get()
        }
    }

    #[test]
    fn an_elastic_corridor_grows_before_it_sinks_or_evicts() {
        let mut kv = cache_ram(4, 8);
        kv.set_elastic_corridor(16, 4);
        let dma = ElasticDma::new(12);
        let chain = grow_with(&mut kv, &dma, 12);
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 12);
        assert_eq!(kv.corridor_grows, 2);
        assert_eq!(kv.tier_demotions, 0);
        assert_eq!(kv.evictions, 0);
        // Growth appends segments, so frame 4 is the first byte of the second corridor and frame 8
        // of the third: a handle's address must follow the segment table, not a single base.
        assert_eq!(kv.plane_addr(chain[0], 0), 0x1_0000_0000);
        assert_eq!(kv.plane_addr(chain[4], 0), 0x2_0000_0000);
        assert_eq!(kv.plane_addr(chain[8], 0), 0x2_0000_0000 + 4 * FB);
        assert_eq!(kv.plane_addr(chain[8], 1), 0x2_0000_0000 + 4 * FB + FB / 2);
    }

    #[test]
    fn growth_stops_at_the_ceiling_and_the_sink_ladder_takes_over() {
        let mut kv = cache_ram(4, 8);
        kv.set_elastic_corridor(8, 4);
        let dma = ElasticDma::new(64);
        let cold = grow_with(&mut kv, &dma, 4);
        kv.publish((1..=16).collect(), cold.clone()).unwrap();
        kv.session_release(&cold);
        let _hot = grow_with(&mut kv, &dma, 5);
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 8);
        assert_eq!(kv.corridor_grows, 1);
        assert_eq!(
            kv.tier_demotions, 4,
            "the cold chain sinks instead of growing"
        );
        // An empty arena grant latches: the tree must not re-plan it for every later page.
        let mut kv = cache_ram(4, 0);
        kv.set_elastic_corridor(64, 4);
        let dry = ElasticDma::new(0);
        let _ = grow_with(&mut kv, &dry, 4);
        assert!(!kv.grow_stalled);
        assert!(
            kv.alloc_frame(&dry).is_none(),
            "a dry arena leaves the corridor at its init size"
        );
        assert!(kv.grow_stalled);
        assert_eq!(kv.corridor_grows, 0);
    }

    #[test]
    fn idle_chunks_go_back_to_the_arena_and_a_live_frame_keeps_its_address() {
        let mut kv = cache_ram(4, 0);
        kv.set_elastic_corridor(16, 4);
        let dma = ElasticDma::new(8);
        let chain = grow_with(&mut kv, &dma, 12);
        kv.publish((1..=16).collect(), chain[..4].to_vec()).unwrap();
        // One session still owns a frame in the middle chunk: that segment must stay.
        kv.session_release(&chain[..5]);
        kv.session_release(&chain[6..]);
        kv.release_idle_corridors(&dma);
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 8);
        assert_eq!(kv.corridor_shrinks, 1);
        assert_eq!(
            dma.arena.lock().unwrap().released,
            vec![KvFrameCorridor {
                base: 0x2_0000_0000 + 4 * FB,
                frames: 4
            }]
        );
        let addr_before = kv.plane_addr(chain[0], 0);
        kv.session_release(&chain[5..6]);
        kv.release_idle_corridors(&dma);
        // The retained chunk is the floor even though its frames are tree-owned, not free.
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 4);
        assert_eq!(kv.corridor_shrinks, 2);
        assert_eq!(kv.plane_addr(chain[0], 0), addr_before);
        let (node, reusable) = kv.best_match(&(1..=16).collect::<Vec<_>>(), None).unwrap();
        assert_eq!(reusable, 16);
        assert_eq!(kv.entry_chain(node, reusable), chain[..4]);

        // Regrowth reuses the returned bytes but never the retired slot indices, so the published
        // chain cannot alias a frame the new session owns.
        let fresh = grow_with(&mut kv, &dma, 4);
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 8);
        assert!(fresh.iter().all(|f| f.slot() >= 12));
        assert_eq!(kv.plane_addr(fresh[0], 0), 0x2_0000_0000 + 4 * FB);
        assert_eq!(kv.entry_chain(node, reusable), chain[..4]);
    }

    #[test]
    fn a_fixed_corridor_never_grows_or_shrinks() {
        let mut kv = cache_ram(4, 0);
        let dma = ElasticDma::new(64);
        let chain = grow_with(&mut kv, &dma, 4);
        kv.publish((1..=16).collect(), chain.clone()).unwrap();
        kv.session_release(&chain);
        kv.release_idle_corridors(&dma);
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 4);
        assert_eq!(kv.corridor_shrinks, 0);
        assert!(dma.arena.lock().unwrap().released.is_empty());
        // Exhaustion falls straight through to the Phase B eviction ladder, never to the arena.
        assert!(kv.alloc_frame(&dma).is_some());
        assert_eq!(kv.evictions, 1);
        assert_eq!(kv.corridor_grows, 0);
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 4);
    }

    #[test]
    fn expert_pressure_sinks_cold_tails_and_the_corridor_goes_back() {
        // The host tier is deliberately roomier than the 8 frames that should sink: sized to exactly
        // 8 it fills up, and a base chunk the yield wrongly tried to empty would be refused by the
        // take bound instead of by the segment floor — the `== 8` assertion could not tell them apart.
        let mut kv = cache_ram(4, 16);
        kv.set_elastic_corridor(16, 4);
        kv.set_yield_under_pressure(true);
        let dma = PressureDma::new(8, false);
        let chain = grow_with(&mut kv, &dma, 12);
        kv.publish((1..=48).collect(), chain.clone()).unwrap();
        kv.session_release(&chain);
        // No pressure: every page stays in VRAM even though the host tier sits empty.
        assert_eq!(kv.yield_vram_under_expert_pressure(&dma), 0);
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 12);

        dma.pressure.set(true);
        assert_eq!(kv.yield_vram_under_expert_pressure(&dma), 8);
        assert_gradient(&kv);
        // The sink rewrites the TREE's chain; the caller's FrameId copies still name the old slots.
        let sunk_chain = kv.entry_chain(node_of(&kv, chain[0]), 48);
        assert!(sunk_chain[..4]
            .iter()
            .all(|frame| frame.tier() == Tier::Vram));
        assert!(sunk_chain[4..]
            .iter()
            .all(|frame| frame.tier() == Tier::Ram));
        assert_eq!(
            kv.evictions, 0,
            "bytes move down a tier, they are not deleted"
        );
        // Only the two emptied segments go back; the base chunk is the floor.
        kv.release_idle_corridors(&dma);
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 4);
        assert_eq!(kv.corridor_shrinks, 2);
        assert_eq!(
            dma.inner.arena.lock().unwrap().released,
            vec![
                KvFrameCorridor {
                    base: 0x2_0000_0000 + 4 * FB,
                    frames: 4
                },
                KvFrameCorridor {
                    base: 0x2_0000_0000,
                    frames: 4
                },
            ]
        );
        // The prefix is still addressable, and the sunk tail comes back through the host backing:
        // MockDma refuses an upload whose bytes match no frame it ever downloaded.
        let (node, reusable) = kv.best_match(&(1..=48).collect::<Vec<_>>(), None).unwrap();
        assert_eq!(reusable, 48);
        let back = kv.entry_chain_promoted(&dma, node, reusable);
        assert_eq!(back.len(), 12);
        assert!(back.iter().all(|frame| frame.tier() == Tier::Vram));
        assert_eq!(kv.tier_demotions, 8);
        assert_eq!(kv.tier_promotions, 8);
    }

    #[test]
    fn expert_pressure_yields_for_short_chains_that_do_not_start_at_slot_zero() {
        // Two conversations side by side, neither as long as the corridor. The segment floor is a
        // POOL slot but `keep` indexes an entry's OWN chain, so a floor compared against the raw
        // slot number skips both of these (keep >= end) and the yield returns 0 — the feature is a
        // no-op for exactly the multi-session case it exists to serve.
        let mut kv = cache_ram(4, 16);
        kv.set_elastic_corridor(16, 4);
        kv.set_yield_under_pressure(true);
        let dma = PressureDma::new(8, true);
        let a = grow_with(&mut kv, &dma, 6); // pool slots 0..5
        kv.publish((1..=24).collect(), a.clone()).unwrap();
        let b = grow_with(&mut kv, &dma, 6); // pool slots 6..11
        kv.publish((101..=124).collect(), b.clone()).unwrap();
        assert_eq!(a[0].slot(), 0);
        assert_eq!(b[0].slot(), 6);
        // Resolve the nodes BEFORE the sink: it rewrites the tree's FrameIds, so the caller's
        // copies no longer name the same frames afterwards.
        let na = node_of(&kv, a[0]);
        let nb = node_of(&kv, b[0]);
        kv.session_release(&a);
        kv.session_release(&b);

        assert_eq!(kv.yield_vram_under_expert_pressure(&dma), 8);
        assert_gradient(&kv);
        let ca = kv.entry_chain(na, 24);
        let cb = kv.entry_chain(nb, 24);
        // A keeps the base chunk it lives in; B owns no base-chunk frame, so all of it goes.
        assert!(ca[..4].iter().all(|f| f.tier() == Tier::Vram));
        assert!(ca[4..].iter().all(|f| f.tier() == Tier::Ram));
        assert!(cb.iter().all(|f| f.tier() == Tier::Ram));
        assert_eq!(kv.evictions, 0);
        kv.release_idle_corridors(&dma);
        assert_eq!(kv.pool.capacity_frames(Tier::Vram), 4);
        assert_eq!(kv.corridor_shrinks, 2);
        // Both prefixes are still addressable and round-trip through the host backing.
        for (tokens, node) in [
            ((1..=24).collect::<Vec<_>>(), na),
            ((101..=124).collect(), nb),
        ] {
            let (found, reusable) = kv.best_match(&tokens, None).unwrap();
            assert_eq!(found, node);
            assert_eq!(reusable, 24);
            let back = kv.entry_chain_promoted(&dma, found, reusable);
            assert!(back.iter().all(|f| f.tier() == Tier::Vram));
        }
    }

    #[test]
    fn a_warm_prefix_a_dry_host_tier_or_the_off_knob_never_yields() {
        let dma = PressureDma::new(64, true);
        // Opt-in: an existing deployment keeps its VRAM residency until the knob is set.
        let mut kv = cache_ram(4, 8);
        kv.set_elastic_corridor(16, 4);
        let chain = grow_with(&mut kv, &dma, 12);
        kv.publish((1..=48).collect(), chain.clone()).unwrap();
        kv.session_release(&chain);
        assert_eq!(kv.yield_vram_under_expert_pressure(&dma), 0);

        // The last attach's reuse depth is a HARD floor here, unlike the exhaustion ladder (which
        // offers a fully warm run rather than let the evictor delete it): nothing is about to be
        // evicted, so sinking bytes the next hit would DMA straight back buys nothing.
        kv.set_yield_under_pressure(true);
        let (node, reusable) = kv.best_match(&(1..=48).collect::<Vec<_>>(), None).unwrap();
        let _ = kv.entry_chain_promoted(&dma, node, reusable);
        assert_eq!(kv.yield_vram_under_expert_pressure(&dma), 0);
        assert_eq!(kv.tier_demotions, 0);

        // No host tier: yielding would mean deleting the prefix, so the ladder keeps it instead.
        let mut dry = cache_ram(4, 0);
        dry.set_elastic_corridor(16, 4);
        dry.set_yield_under_pressure(true);
        let cold = grow_with(&mut dry, &dma, 12);
        dry.publish((1..=48).collect(), cold.clone()).unwrap();
        dry.session_release(&cold);
        assert_eq!(dry.yield_vram_under_expert_pressure(&dma), 0);
        assert_eq!(dry.evictions, 0);

        // A fixed corridor is one segment: there is nothing above the base chunk to empty.
        let mut fixed = cache_ram(12, 8);
        fixed.set_yield_under_pressure(true);
        let one = grow_with(&mut fixed, &dma, 12);
        fixed.publish((1..=48).collect(), one.clone()).unwrap();
        fixed.session_release(&one);
        assert_eq!(fixed.yield_vram_under_expert_pressure(&dma), 0);
    }

    #[test]
    fn publish_then_match_returns_the_page_aligned_chain_prefix() {
        let mut kv = cache(16);
        let a = grow(&mut kv, 3); // 12 tokens, exactly 3 pages
        kv.publish((1..=12).collect(), a.clone()).unwrap();
        // Exact replay: full chain.
        let (node, reusable) = kv.best_match(&(1..=12).collect::<Vec<_>>(), None).unwrap();
        assert_eq!(reusable, 12);
        assert_eq!(kv.entry_chain(node, reusable), a);
        // Longer prompt: truncated continuation at the entry length.
        let (node, reusable) = kv.best_match(&(1..=20).collect::<Vec<_>>(), None).unwrap();
        assert_eq!(reusable, 12);
        assert_eq!(kv.entry_chain(node, reusable), a);
    }

    #[test]
    fn reuse_is_floor_page_aligned() {
        let mut kv = cache(16);
        let a = grow(&mut kv, 3); // tokens 1..=12
        kv.publish((1..=12).collect(), a.clone()).unwrap();
        kv.session_release(&a); // demotion complete: the tree owns the chain
                                // Prompt shares 10 tokens with the entry: only 2 complete pages (8 tokens) are reusable.
        let mut prompt: Vec<u32> = (1..=10).collect();
        prompt.push(99);
        let (node, reusable) = kv.best_match(&prompt, None).unwrap();
        assert_eq!(reusable, 8);
        assert_eq!(kv.entry_chain(node, reusable), a[..2]);
    }

    #[test]
    fn shared_prefix_frames_survive_ancestor_supersession() {
        let mut kv = cache(16);
        let short = grow(&mut kv, 2); // tokens 1..=8
        kv.publish((1..=8).collect(), short.clone()).unwrap();
        kv.session_release(&short); // tree-only owners
                                    // Session B restores the prefix (pin), grows one divergence page, then demotes the chain.
        let (node, reusable) = kv
            .best_match(&(1..=10).collect::<Vec<u32>>(), None)
            .unwrap();
        assert_eq!(reusable, 8);
        let mut long = kv.entry_chain(node, reusable);
        kv.session_attach(&long);
        long.push(grow(&mut kv, 1)[0]);
        kv.publish((1..=12).collect(), long.clone()).unwrap();
        kv.session_release(&long);
        // The ancestor entry was superseded but its frames live through the new chain's owner.
        assert_eq!(kv.pool.refcount(long[0]), 1, "tree owns each frame once");
        assert_eq!(kv.pool.held_frames(Tier::Vram), 3, "two shared + one new");
        // The deeper entry truncates to continue the shorter prompt.
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<u32>>(), None).unwrap();
        assert_eq!(reusable, 8);
        assert_eq!(kv.entry_chain(node, reusable), long[..2]);
    }

    #[test]
    fn session_attach_and_release_keep_tree_and_session_owners_separate() {
        let mut kv = cache(16);
        let a = grow(&mut kv, 2);
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        kv.session_release(&a); // demotion complete: tree-only
        assert_eq!(kv.pool.refcount(a[0]), 1);
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<_>>(), None).unwrap();
        let chain = kv.entry_chain(node, reusable);
        kv.session_attach(&chain); // restoring session gains owners
        assert_eq!(kv.pool.refcount(a[0]), 2);
        assert!(kv.pool.is_active(a[0]));
        // Eviction of a still-used entry cannot free its frames out from under the session.
        let before = kv.pool.held_frames(Tier::Vram);
        assert!(kv.evict_cold_one());
        assert_eq!(
            kv.pool.held_frames(Tier::Vram),
            before,
            "session owners keep the frames"
        );
        assert!(
            kv.pool.is_active(a[0]),
            "still live after the tree owner left"
        );
        kv.session_release(&chain);
        // The last owner is gone: the slot is freed, so only aggregate counters remain valid.
        assert_eq!(
            kv.pool.held_frames(Tier::Vram),
            before - 2,
            "last owners left"
        );
    }

    #[test]
    fn alloc_evicts_cold_leaf_then_frees_the_frame_for_reuse() {
        let mut kv = cache(2);
        let a = grow(&mut kv, 2);
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        kv.session_release(&a); // tree-only owners; corridor now fully held by the entry
        assert_eq!(kv.pool.free_frames(Tier::Vram), 0);
        // A new conversation needs a frame: the cold leaf entry is evicted, its frames freed.
        let b = kv
            .alloc_frame(&MockDma::default())
            .expect("eviction made room");
        assert_eq!(kv.evictions, 1);
        assert_eq!(
            kv.best_match(&(1..=8).collect::<Vec<_>>(), None),
            None,
            "entry gone"
        );
        let _ = b;
    }

    #[test]
    fn edge_split_keeps_both_branches_matchable() {
        let mut kv = cache(16);
        let a = grow(&mut kv, 2); // tokens 1..=8
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        kv.session_release(&a);
        // Branch B shares the first page only (diverges at token 5): the edge splits at 4.
        let mut b_frames = vec![a[0]];
        b_frames.push(grow(&mut kv, 1)[0]);
        kv.publish(vec![1, 2, 3, 4, 9, 10, 11, 12], b_frames.clone())
            .unwrap();
        kv.session_release(&b_frames);
        // A's entry sits on one side of the split, B's on the other; both stay matchable.
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<_>>(), None).unwrap();
        assert_eq!(reusable, 8);
        assert_eq!(kv.entry_chain(node, reusable), a);
        let (node, reusable) = kv.best_match(&[1, 2, 3, 4, 9, 10, 11, 12], None).unwrap();
        assert_eq!(reusable, 8);
        assert_eq!(kv.entry_chain(node, reusable), b_frames);
        // A prompt diverging inside the split edge still shares the complete first page.
        let (node, reusable) = kv.best_match(&[1, 2, 3, 4, 5, 99], None).unwrap();
        assert_eq!(reusable, 4);
        assert_eq!(kv.entry_chain(node, reusable), vec![a[0]]);
    }

    /// The projection setting partitions the tree at the ROOT, so the key's first position keeps
    /// its own identity. Stamping a reserved setting id over it instead would make these two
    /// prompts one chain, and the second reader would be handed the first one's activations.
    #[test]
    fn a_projected_key_still_addresses_its_own_first_token() {
        let mut kv = cache(4);
        let chain = grow(&mut kv, 1);
        kv.publish_with_ckpts(vec![1, 2, 3, 4], chain, Vec::new(), Some(true))
            .unwrap();
        let (node, reusable) = kv.best_match(&[1, 2, 3, 4], Some(true)).unwrap();
        assert_eq!(reusable, 4);
        let (depth, ckpt) = kv.hit_prefix(node, false, reusable);
        assert_eq!(depth, 4);
        assert!(ckpt.is_none());
        assert!(
            kv.best_match(&[9, 2, 3, 4], Some(true)).is_none(),
            "a different first token is a different conversation"
        );
    }

    /// Same rule for a prompt that OPENS with an image: position 0 carries that picture's content
    /// id, and a setting stamp would erase it — two different first pictures would share one chain.
    #[test]
    fn a_projected_key_still_addresses_its_own_first_image() {
        use crate::seam::IMAGE_CACHE_TOKEN_BIT;
        let mut kv = cache(4);
        let picture_a = 0x1234_5678 | IMAGE_CACHE_TOKEN_BIT;
        let picture_b = 0x8765_4321 | IMAGE_CACHE_TOKEN_BIT;
        let chain = grow(&mut kv, 1);
        kv.publish_with_ckpts(vec![picture_a, 2, 3, 4], chain, Vec::new(), Some(true))
            .unwrap();
        assert_eq!(
            kv.best_match(&[picture_a, 2, 3, 4], Some(true)).unwrap().1,
            4
        );
        assert!(
            kv.best_match(&[picture_b, 2, 3, 4], Some(true)).is_none(),
            "a different first picture is a different conversation"
        );
    }

    /// One token prefix has three bodies of activations — unprojected, OFF and ON — so all three
    /// chains coexist and each lookup reaches only its own. A session with no direction file keeps
    /// addressing the unprojected root, exactly as it did before the setting existed.
    #[test]
    fn each_projection_setting_addresses_a_disjoint_chain() {
        let mut kv = cache(8);
        let tokens = vec![1, 2, 3, 4];
        let mut nodes = Vec::new();
        for projection in [None, Some(false), Some(true)] {
            let chain = grow(&mut kv, 1);
            kv.publish_with_ckpts(tokens.clone(), chain, Vec::new(), projection)
                .unwrap();
            let (node, reusable) = kv.best_match(&tokens, projection).unwrap();
            assert_eq!(reusable, 4, "projection={projection:?}");
            nodes.push(node);
        }
        assert_eq!(nodes.len(), 3);
        assert!(
            nodes
                .iter()
                .all(|&a| nodes.iter().filter(|&&b| a == b).count() == 1),
            "three settings must be three entries, not one overwritten twice"
        );
        // Every chain is still reachable after all three publishes.
        for (projection, &node) in [None, Some(false), Some(true)].into_iter().zip(&nodes) {
            let (found, reusable) = kv.best_match(&tokens, projection).unwrap();
            assert_eq!(found, node);
            assert_eq!(reusable, 4);
        }
    }

    /// Partitioning must not cost prefix reuse INSIDE one setting: a shared head still hits, and
    /// only the divergent tail is replayed.
    #[test]
    fn a_shared_prefix_still_hits_inside_one_projection() {
        let mut kv = cache(4);
        let chain = grow(&mut kv, 2);
        kv.publish_with_ckpts((1..=8).collect(), chain, Vec::new(), Some(false))
            .unwrap();
        let (node, reusable) = kv
            .best_match(&[1, 2, 3, 4, 9, 10, 11, 12], Some(false))
            .unwrap();
        assert_eq!(reusable, 4);
        let (depth, ckpt) = kv.hit_prefix(node, false, reusable);
        assert_eq!(depth, 4);
        assert!(ckpt.is_none());
        assert!(kv
            .best_match(&[1, 2, 3, 4, 9, 10, 11, 12], Some(true))
            .is_none());
    }

    #[test]
    fn publish_rejects_a_misaligned_chain() {
        let mut kv = cache(16);
        let a = grow(&mut kv, 2);
        assert!(
            kv.publish((1..=7).collect(), a.clone()).is_err(),
            "7 tokens need 1 page"
        );
    }

    #[test]
    fn churn_recycles_nodes_and_frames_together() {
        // Two frames live in the corridor: each new conversation evicts the LRU entry, whose
        // spine AND frames must both come back for reuse (the metadata-growth bound, mirrored
        // from the blob tier). `alloc_frame` already grants the session's owner, so publish +
        // release alone mirror the checkout demotion: after it the tree owns the chain alone.
        let mut kv = cache(2);
        for seed in 0..20u32 {
            let chain = kv
                .alloc_frame(&MockDma::default())
                .expect("frames recycle as the LRU leaves");
            let chain = vec![chain];
            let tokens = [seed * 10 + 1, seed * 10 + 2, seed * 10 + 3, seed * 10 + 4].to_vec();
            kv.publish(tokens, chain.clone()).unwrap();
            kv.session_release(&chain);
        }
        let steady = kv.nodes.len();
        for seed in 20..40u32 {
            let chain = kv
                .alloc_frame(&MockDma::default())
                .expect("frames recycle as the LRU leaves");
            let chain = vec![chain];
            let tokens = [seed * 10 + 1, seed * 10 + 2, seed * 10 + 3, seed * 10 + 4].to_vec();
            kv.publish(tokens, chain.clone()).unwrap();
            kv.session_release(&chain);
        }
        assert_eq!(kv.nodes.len(), steady, "churn grew the arena: {steady}");
        assert!(
            steady <= PagedKv::PROJECTION_ROOTS + 3,
            "steady arena should stay small, got {steady}"
        );
        let (_, reusable) = kv
            .best_match(&[381, 382, 383, 384], None)
            .expect("the second-newest conversation survived");
        assert_eq!(reusable, 4);
        assert!(
            kv.best_match(&[1, 2, 3, 4], None).is_none(),
            "evicted prompt is gone"
        );
    }

    #[test]
    fn superseded_pass_through_spine_merges_into_the_deeper_entry() {
        let mut kv = cache(4);
        let f1 = kv.alloc_frame(&MockDma::default()).unwrap();
        kv.publish(vec![1, 2, 3, 4], vec![f1]).unwrap();
        kv.session_release(&[f1]);
        let spine = node_of(&kv, f1);
        let f2 = kv.alloc_frame(&MockDma::default()).unwrap();
        let f3 = kv.alloc_frame(&MockDma::default()).unwrap();
        kv.publish(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![f2, f3])
            .unwrap();
        kv.session_release(&[f2, f3]);
        // The 4-token spine node lost its entry to the supersede and became a pass-through with
        // one child: it must merge into that child, not linger.
        assert!(kv.free.contains(&spine), "pass-through spine was not freed");
        let (node, reusable) = kv
            .best_match(&[1, 2, 3, 4], None)
            .expect("the merged spine still matches");
        assert_eq!(reusable, 4);
        assert_eq!(kv.entry_chain(node, 4), vec![f2]);
        let (_, reusable) = kv
            .best_match(&[1, 2, 3, 4, 5, 6, 7, 8], None)
            .expect("the full conversation matches");
        assert_eq!(reusable, 8);
    }

    #[test]
    fn cold_chain_sinks_to_ram_and_promotes_back_on_hit() {
        let mut kv = cache_ram(2, 4);
        let dma = MockDma::default();
        let a = grow(&mut kv, 2);
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        kv.session_release(&a); // tree-only owners; corridor fully held by the entry
        assert_eq!(kv.pool.free_frames(Tier::Vram), 0);
        // Corridor pressure sinks the published chain instead of evicting it: the prefix
        // survives in RAM and the fresh request gets its frame.
        let fresh = kv.alloc_frame(&dma).expect("sinking freed a VRAM frame");
        assert_eq!(kv.tier_demotions, 2, "both frames left the corridor");
        assert_eq!(kv.evictions, 0, "sinking precedes eviction");
        assert_eq!(kv.pool.held_frames(Tier::Ram), 2);
        // A fully sunk chain still matches; the entry is not gone, only resting.
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<u32>>(), None).unwrap();
        assert_eq!(reusable, 8);
        // Give the corridor its frame back so the whole chain can promote at once.
        kv.session_release(&[fresh]);
        // Promotion hands out device-addressable frames only (the page table never sees a RAM
        // id) and the mock's pattern check proves the bytes came back through the backing.
        let back = kv.entry_chain_promoted(&dma, node, reusable);
        assert_eq!(back.len(), 2);
        assert!(
            back.iter().all(|f| matches!(f.tier(), Tier::Vram)),
            "the attach path only ever sees VRAM frames"
        );
        assert_eq!(kv.tier_promotions, 2);
        assert_eq!(kv.pool.held_frames(Tier::Ram), 0, "backing slots recycled");
        assert_eq!(kv.pool.held_frames(Tier::Vram), 2);
        // The entry chain was rewritten in place: a second pass promotes nothing.
        let again = kv.entry_chain_promoted(&dma, node, reusable);
        assert_eq!(again, back);
        assert_eq!(kv.tier_promotions, 2, "VRAM frames pass straight through");
    }

    #[test]
    fn active_chains_are_never_sunk() {
        let mut kv = cache_ram(2, 4);
        let dma = MockDma::default();
        let a = grow(&mut kv, 2); // still active: a session table points here
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        // Session AND tree own the frames and the session is live: nothing may DMA out, and
        // eviction cannot free bytes out from under the session either.
        assert!(
            kv.alloc_frame(&dma).is_none(),
            "a corridor of only-active frames yields nothing"
        );
        assert_eq!(kv.tier_demotions, 0, "active frames are not DMA candidates");
        assert_eq!(kv.pool.held_frames(Tier::Ram), 0);
        assert_eq!(
            kv.pool.held_frames(Tier::Vram),
            2,
            "the session owner keeps the bytes"
        );
        assert!(kv.pool.is_active(a[0]));
    }

    #[test]
    fn shared_frames_follow_their_migration_across_both_chains() {
        // Branch A (tokens 1..=8, frames a0+a1) and branch B (1,2,3,4,9..=12, frames a0+b1)
        // share a0. Sinking either leaf must rewrite a0 in BOTH chains (refcount travels with
        // `migrate_into`), and a promotion must hand both chains the same fresh id again.
        let mut kv = cache_ram(3, 4);
        let dma = MockDma::default();
        let a = grow(&mut kv, 2);
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        kv.session_release(&a);
        // Session B restores the shared first page (attach gains its owner), grows one divergence
        // page, publishes, then demotes — the full checkout shape.
        kv.session_attach(&[a[0]]);
        let mut b = vec![a[0]];
        b.push(grow(&mut kv, 1)[0]);
        kv.publish(vec![1, 2, 3, 4, 9, 10, 11, 12], b.clone())
            .unwrap();
        kv.session_release(&b);
        assert_eq!(
            kv.pool.refcount(a[0]),
            2,
            "both entries own the shared frame"
        );
        // Pressure sinks the coldest leaf (whichever it picks); both branches must survive.
        let c = kv
            .alloc_frame(&dma)
            .expect("sinking a leaf freed a VRAM frame");
        assert!(kv.tier_demotions >= 1);
        assert!(kv
            .best_match(&(1..=8).collect::<Vec<u32>>(), None)
            .is_some());
        assert!(
            kv.best_match(&[1, 2, 3, 4, 9, 10, 11, 12], None).is_some(),
            "sinking never deletes an entry"
        );
        kv.session_release(&[c]);
        // Promote branch A fully, then branch B: the shared frame must land on one id owned
        // by exactly two chains again.
        let (node_a, reusable_a) = kv.best_match(&(1..=8).collect::<Vec<u32>>(), None).unwrap();
        let back_a = kv.entry_chain_promoted(&dma, node_a, reusable_a);
        assert_eq!(back_a.len(), 2, "the corridor can host the whole chain");
        let (node_b, reusable_b) = kv.best_match(&[1, 2, 3, 4, 9, 10, 11, 12], None).unwrap();
        let back_b = kv.entry_chain_promoted(&dma, node_b, reusable_b);
        assert_eq!(back_b.len(), 2);
        assert_eq!(back_a[0], back_b[0], "shared frame id agrees again");
        assert_eq!(
            kv.pool.refcount(back_a[0]),
            2,
            "shared ownership survived the trip"
        );
        assert_eq!(kv.pool.held_frames(Tier::Ram), 0, "the tier is empty again");
    }

    #[test]
    fn ram_pressure_cascades_to_disk_and_a_hit_walks_back_up() {
        // VRAM 2 / RAM 2 / disk 4: the cascade must move A all the way down (VRAM->RAM->disk)
        // and back up (disk->RAM->VRAM) with the MockDma pattern proving byte integrity across
        // the real temp-file round-trip.
        let (mut kv, path) = cache_disk(2, 2, 4);
        let dma = MockDma::default();
        let a = grow(&mut kv, 2);
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        kv.session_release(&a); // tree-only; VRAM fully held by A
        let b = grow(&mut kv, 2); // claim sinks A to RAM, then claims both frames
        kv.publish((21..=28).collect(), b.clone()).unwrap();
        kv.session_release(&b); // VRAM now fully held by B, RAM fully held by A
        assert_eq!(kv.tier_demotions, 2, "A sank VRAM->RAM");
        assert_eq!(kv.pool.held_frames(Tier::Disk), 0);
        let c = kv
            .alloc_frame(&dma)
            .expect("cascade freed a VRAM frame via RAM->disk");
        assert_eq!(
            kv.disk_demotions, 2,
            "A sank RAM->disk, freeing its RAM slots"
        );
        assert_eq!(kv.tier_demotions, 4, "then B sank VRAM->RAM");
        assert_eq!(kv.evictions, 0, "the cascade precedes eviction");
        assert_eq!(kv.pool.held_frames(Tier::Disk), 2);
        assert_eq!(kv.pool.held_frames(Tier::Ram), 2);
        kv.session_release(&[c]); // give the corridor one frame back for the promotion
                                  // A is a Disk chain now: still matchable, still resting.
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<u32>>(), None).unwrap();
        assert_eq!(reusable, 8);
        // Promotion walks A back up: disk->RAM (cascading B to disk for the RAM slot) then
        // RAM->VRAM. The mock's pattern check can only pass if the file round-trip was exact.
        let back = kv.entry_chain_promoted(&dma, node, reusable);
        assert_eq!(back.len(), 2, "the whole chain came back");
        assert!(
            back.iter().all(|f| matches!(f.tier(), Tier::Vram)),
            "the attach path only ever sees VRAM frames"
        );
        assert_eq!(kv.disk_promotions, 2);
        assert_eq!(kv.tier_promotions, 2, "RAM->VRAM counted separately");
        assert_eq!(kv.pool.held_frames(Tier::Disk), 2, "B took A's disk slots");
        assert_eq!(kv.pool.held_frames(Tier::Ram), 0);
        drop(kv);
        let _ = std::fs::remove_file(&path);
    }

    /// One disk slot cannot hold a two-frame chain, and that used to mean the entry was DELETED:
    /// the sinker's whole-chain capacity check skipped it, the cascade yielded nothing, and the
    /// evictor took it. A tail sink keeps every byte instead — the slot takes the coldest frame
    /// and the rest stays one tier up. Eviction is still the last resort, but only once no tier
    /// below can take anything at all.
    #[test]
    fn an_undersized_disk_takes_the_tail_and_eviction_waits_its_turn() {
        let (mut kv, path) = cache_disk(2, 2, 1);
        let dma = MockDma::default();
        let a = grow(&mut kv, 2);
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        kv.session_release(&a);
        let b = grow(&mut kv, 2);
        kv.publish((21..=28).collect(), b.clone()).unwrap();
        kv.session_release(&b);
        assert_eq!(kv.tier_demotions, 2, "A sank to RAM");

        let c = kv.alloc_frame(&dma).expect("the tail sink freed a slot");
        assert_eq!(
            kv.disk_demotions, 1,
            "the one disk slot took A's coldest frame"
        );
        assert_eq!(
            kv.evictions, 0,
            "nothing was deleted while a tier could still take bytes"
        );
        assert_eq!(
            kv.tier_demotions, 3,
            "B's tail sank into the RAM slot A left"
        );
        assert_eq!(kv.pool.held_frames(Tier::Disk), 1);
        assert_eq!(kv.pool.held_frames(Tier::Ram), 2);
        assert_gradient(&kv);
        let a_node = node_at(&kv, &(1..=8).collect::<Vec<u32>>());
        let b_node = node_at(&kv, &(21..=28).collect::<Vec<u32>>());
        assert_eq!(
            kv.nodes[a_node]
                .entry
                .as_ref()
                .unwrap()
                .chain
                .iter()
                .map(|f| f.tier())
                .collect::<Vec<_>>(),
            vec![Tier::Ram, Tier::Disk],
            "A is a gradient, not a hole"
        );
        assert_eq!(
            kv.nodes[b_node]
                .entry
                .as_ref()
                .unwrap()
                .chain
                .iter()
                .map(|f| f.tier())
                .collect::<Vec<_>>(),
            vec![Tier::Vram, Tier::Ram],
            "B kept its device prefix"
        );

        // Every tier is now full, so the next claim has nothing to sink into and eviction is the
        // only move left. It drops the coldest entry (A), whose freed host slots then let B's
        // remaining device frame sink — bytes still move down before anything else is deleted.
        let d = kv
            .alloc_frame(&dma)
            .expect("eviction reclaimed the corridor");
        assert_eq!(kv.evictions, 1, "only the coldest entry was dropped");
        assert_eq!(kv.tier_demotions, 4);
        assert_eq!(
            kv.pool.held_frames(Tier::Disk),
            0,
            "A's disk slot came back"
        );
        assert_eq!(kv.pool.held_frames(Tier::Ram), 2, "B rests in RAM");
        assert_gradient(&kv);
        kv.session_release(&[c, d]);
        drop(kv);
        let _ = std::fs::remove_file(&path);
    }

    /// The point of frame-level tiering: a hit that reused only the head of a long chain leaves
    /// that head in VRAM and sinks the tail behind it, so the next full hit pays a DMA for the
    /// tail alone instead of for every page. Whole-chain sinking moved the head too and bought it
    /// straight back.
    #[test]
    fn a_hit_keeps_its_reused_prefix_resident_and_sinks_only_the_tail() {
        let mut kv = cache_ram(4, 4);
        let dma = MockDma::default();
        let a = grow(&mut kv, 4);
        kv.publish((1..=16).collect(), a.clone()).unwrap();
        kv.session_release(&a);

        // A hit that consumes two of the four pages, through the production accessor.
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<u32>>(), None).unwrap();
        assert_eq!(reusable, 8);
        let head = kv.entry_chain_promoted(&dma, node, reusable);
        assert_eq!(head.len(), 2);
        assert_eq!(kv.nodes[node].entry.as_ref().unwrap().warm_frames, 2);

        let c1 = kv.alloc_frame(&dma).expect("the tail sink freed a slot");
        let c2 = kv
            .alloc_frame(&dma)
            .expect("the second slot was already free");
        assert_eq!(kv.tail_sinks, 1, "the sink kept a prefix behind");
        assert_eq!(kv.tier_demotions, 2, "only the two cold frames moved");
        assert_eq!(kv.evictions, 0);
        assert_gradient(&kv);
        assert_eq!(
            kv.nodes[node]
                .entry
                .as_ref()
                .unwrap()
                .chain
                .iter()
                .map(|f| f.tier())
                .collect::<Vec<_>>(),
            vec![Tier::Vram, Tier::Vram, Tier::Ram, Tier::Ram],
            "the reused head stayed addressable without a DMA"
        );
        kv.session_release(&[c1, c2]);

        // The full hit now promotes the tail alone, and MockDma checks every byte round-tripped.
        let (node, reusable) = kv
            .best_match(&(1..=16).collect::<Vec<u32>>(), None)
            .unwrap();
        assert_eq!(reusable, 16);
        let back = kv.entry_chain_promoted(&dma, node, reusable);
        assert_eq!(back.len(), 4);
        assert!(back.iter().all(|f| matches!(f.tier(), Tier::Vram)));
        assert_eq!(
            kv.tier_promotions, 2,
            "the resident prefix was not copied back"
        );
        assert_gradient(&kv);
    }

    /// A chain longer than the whole RAM tier used to be skipped by the sinker and then deleted.
    /// Now it sinks as far down as the host pool reaches and keeps the rest addressable.
    #[test]
    fn a_chain_longer_than_the_ram_tier_sinks_its_tail_instead_of_dying() {
        let mut kv = cache_ram(6, 2);
        let dma = MockDma::default();
        let a = grow(&mut kv, 6);
        kv.publish((1..=24).collect(), a.clone()).unwrap();
        kv.session_release(&a);
        assert_eq!(kv.pool.free_frames(Tier::Ram), 2);

        let c = kv.alloc_frame(&dma).expect("a partial sink freed a slot");
        assert_eq!(kv.evictions, 0, "the long chain was not deleted");
        assert_eq!(kv.tier_demotions, 2, "exactly as much as RAM could host");
        assert_gradient(&kv);
        let node = node_of(&kv, a[0]);
        let tiers: Vec<_> = kv.nodes[node]
            .entry
            .as_ref()
            .unwrap()
            .chain
            .iter()
            .map(|f| f.tier())
            .collect();
        assert_eq!(
            tiers,
            vec![
                Tier::Vram,
                Tier::Vram,
                Tier::Vram,
                Tier::Vram,
                Tier::Ram,
                Tier::Ram
            ]
        );

        // The resident prefix still serves a hit with no DMA at all.
        let (node, reusable) = kv
            .best_match(&(1..=16).collect::<Vec<u32>>(), None)
            .unwrap();
        assert_eq!(reusable, 16);
        let back = kv.entry_chain_promoted(&dma, node, reusable);
        assert_eq!(back.len(), 4);
        assert_eq!(
            kv.tier_promotions, 0,
            "a prefix hit inside the resident head is free"
        );
        kv.session_release(&[c]);
    }

    /// A recurrent entry only resumes where it carries a snapshot, so a deep TOKEN match can attach
    /// nothing at all. The warm floor must follow the attach depth, not the match depth: stamping
    /// the match would pin a prefix no request can read, and the sinker would then evict bytes that
    /// a real hit WOULD have reused. This is the shape every Qwen3.5/3.6/3.8 conversation has —
    /// one snapshot at the published end, nothing shallower.
    #[test]
    fn a_match_that_cannot_resume_does_not_pin_a_warm_prefix() {
        let mut kv = cache_ram(4, 4);
        let dma = MockDma::default();
        let a = grow(&mut kv, 4);
        kv.publish_with_ckpts((1..=16).collect(), a.clone(), vec![ckpt(16, 64)], None)
            .unwrap();
        kv.session_release(&a);

        // Two pages of tokens match, but the only snapshot sits at four pages.
        let (node, matched) = kv.best_match(&(1..=8).collect::<Vec<u32>>(), None).unwrap();
        assert_eq!(matched, 8);
        let (reusable, ckpt) = kv.hit_prefix(node, true, matched);
        assert_eq!((reusable, ckpt.is_none()), (0, true));
        assert_eq!(
            kv.nodes[node].entry.as_ref().unwrap().warm_frames,
            0,
            "an unusable match must not reserve VRAM"
        );

        // So the whole chain is sinkable, exactly like a never-hit chain.
        let c1 = kv.alloc_frame(&dma).expect("the sink freed a slot");
        let c2 = kv.alloc_frame(&dma).expect("the second slot too");
        assert_eq!(kv.tail_sinks, 0, "nothing was held back");
        assert_eq!(kv.tier_demotions, 4, "the whole chain sank");
        assert_gradient(&kv);
        kv.session_release(&[c1, c2]);

        // A snapshot-backed attach DOES stamp its depth, and the next sink keeps that head.
        let (node, matched) = kv
            .best_match(&(1..=16).collect::<Vec<u32>>(), None)
            .unwrap();
        let (reusable, _) = kv.hit_prefix(node, true, matched);
        assert_eq!(reusable, 16);
        let back = kv.entry_chain_promoted(&dma, node, reusable);
        assert_eq!(back.len(), 4);
        assert_eq!(kv.nodes[node].entry.as_ref().unwrap().warm_frames, 4);
    }

    #[test]
    fn disk_closed_behaves_exactly_like_phase_c() {
        // The regression guard for the default-off path: with disk_frames=0 the cascade helper
        // short-circuits and every Phase C assertion (sink, promote, evict) still holds.
        let mut kv = cache_ram(2, 4);
        let dma = MockDma::default();
        let a = grow(&mut kv, 2);
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        kv.session_release(&a);
        let _b = kv.alloc_frame(&dma).expect("sinking freed a frame");
        assert_eq!(kv.disk_demotions, 0);
        assert_eq!(kv.pool.capacity_frames(Tier::Disk), 0);
    }

    /// A snapshot-shaped aux whose byte size is deterministic (the tree never inspects content).
    /// The storage is a real backend buffer, so the tree's byte accounting is exercised against the
    /// same `len_bytes()` the production capture path reports.
    fn ckpt(boundary: usize, bytes: usize) -> Arc<PagedCkpt> {
        static BE: std::sync::OnceLock<infr_cpu::CpuBackend> = std::sync::OnceLock::new();
        let be = BE.get_or_init(|| {
            infr_cpu::CpuBackend::new_with(Arc::new(crate::EngineConfig::default()))
        });
        Arc::new(PagedCkpt {
            boundary_tokens: boundary,
            layers: vec![0],
            kbufs: vec![be
                .alloc(bytes, infr_core::backend::BufferUsage::Readback)
                .unwrap()],
            vbufs: Vec::new(),
            ple: None,
        })
    }

    /// The node a token path lands on. Unlike [`node_of`] this survives a sink: migration replaces
    /// frame ids, so a handle captured before the bytes moved no longer names anything.
    fn node_at(kv: &PagedKv, tokens: &[u32]) -> usize {
        let (path, matched, _) = kv.walk(kv.root(None), tokens);
        assert_eq!(matched, tokens.len(), "the path is not fully published");
        *path.last().expect("the root is always on the path")
    }

    /// The node whose entry chain owns `frame` (tests only — never stamps the LRU clock).
    fn node_of(kv: &PagedKv, frame: FrameId) -> usize {
        kv.nodes
            .iter()
            .position(|node| {
                node.entry
                    .as_ref()
                    .is_some_and(|entry| entry.chain.contains(&frame))
            })
            .expect("some entry owns the frame")
    }

    #[test]
    fn entry_ckpt_returns_the_deepest_boundary_at_or_below_reuse() {
        let mut kv = cache_ram(16, 8);
        let a = grow(&mut kv, 4); // 16 tokens
        kv.publish_with_ckpts(
            (1..=16).collect(),
            a.clone(),
            vec![ckpt(12, 64), ckpt(4, 64)], // insertion order must not matter
            None,
        )
        .unwrap();
        let node = node_of(&kv, a[0]);
        assert_eq!(kv.entry_ckpt(node, 16).unwrap().boundary_tokens, 12);
        assert_eq!(kv.entry_ckpt(node, 13).unwrap().boundary_tokens, 12);
        assert_eq!(kv.entry_ckpt(node, 11).unwrap().boundary_tokens, 4);
        assert_eq!(kv.entry_ckpt(node, 4).unwrap().boundary_tokens, 4);
        assert!(
            kv.entry_ckpt(node, 3).is_none(),
            "no snapshot covers 3 tokens"
        );
        assert_eq!(kv.aux_bytes, 128);
        assert_eq!(kv.aux_budget, FB as usize, "an eighth of the RAM tier");
    }

    #[test]
    fn a_recurrent_branch_shallower_than_its_only_snapshot_reuses_nothing() {
        // The sub-agent shape: several requests share a long prefix and diverge early. The tree
        // holds that prefix in VRAM and counts a HIT, but a recurrent entry resumes only at a
        // snapshot boundary and the publish captured one at the conversation's own frontier.
        let mut kv = cache_recurrent(16, 8);
        let a = grow(&mut kv, 4); // 16 tokens
        kv.publish_with_ckpts((1..=16).collect(), a.clone(), vec![ckpt(16, 64)], None)
            .unwrap();
        let branch: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 900, 901];
        let (node, matched) = kv
            .best_match(&branch, None)
            .expect("the shared prefix is a hit");
        assert_eq!(matched, 8);
        assert_eq!(
            kv.entry_chain(node, 8).len(),
            2,
            "two VRAM pages already cover the branch point"
        );
        let (reusable, ckpt) = kv.hit_prefix(node, true, matched);
        assert_eq!(
            (reusable, ckpt.is_some()),
            (0, false),
            "a hit that hands back zero tokens: the whole prompt re-prefills"
        );
        // Same tree, same prompt, dense attention: the pages alone are the resume point.
        assert_eq!(kv.hit_prefix(node, false, matched).0, 8);
    }

    #[test]
    fn a_deeper_continuation_publish_inherits_the_branch_point_it_supersedes() {
        // t1 -> t2 -> t3 of the measured conversation probe: t2 continues t1 and publishes deeper,
        // t3 branches off at exactly t1's frontier. A continuation captures no new snapshot AT its
        // start (the runner's `boundary > start` filter), so the deeper entry carries only its own
        // frontier — and supersession deletes the shallower entry that held the branch point. The
        // frames survive either way; inheriting the deleted entry's anchor is what keeps the STATE
        // that can resume them, instead of a hit that hands back zero tokens.
        let mut kv = cache_recurrent(16, 8);
        let t1 = grow(&mut kv, 4); // 16 tokens
        kv.publish_with_ckpts(
            (1..=16).collect(),
            t1.clone(),
            vec![ckpt(8, 64), ckpt(16, 64)],
            None,
        )
        .unwrap();
        kv.session_release(&t1);
        let branch: Vec<u32> = (1..=16).chain([900, 901]).collect();
        let (node, matched) = kv.best_match(&branch, None).unwrap();
        assert_eq!(
            kv.hit_prefix(node, true, matched).0,
            16,
            "t1 resumes the branch"
        );

        // t2 continues t1 to 24 tokens and publishes; its only snapshot is its own frontier.
        let t2 = grow(&mut kv, 6);
        kv.publish_with_ckpts((1..=24).collect(), t2.clone(), vec![ckpt(24, 64)], None)
            .unwrap();
        kv.session_release(&t2);
        // The pass-through node merges into its only child, so the deeper entry stays matchable and
        // its chain physically covers all 16 branch tokens. The superseded entry's deepest snapshot
        // moved onto it, so the branch resumes there rather than re-prefilling.
        let (node, matched) = kv
            .best_match(&branch, None)
            .expect("the merged chain still matches");
        assert_eq!(matched, 16);
        assert_eq!(
            kv.entry_chain(node, 16).len(),
            4,
            "four VRAM pages still cover the branch point"
        );
        let (reusable, ckpt) = kv.hit_prefix(node, true, matched);
        assert_eq!(
            (reusable, ckpt.unwrap().boundary_tokens),
            (16, 16),
            "the fork attaches at the frontier it branched from"
        );
        assert_eq!(kv.ckpts_inherited, 1);
        // The shallow 8-token snapshot was NOT inherited: only the branch point was. Aux moved
        // between entries instead of growing — 64 B left with the ancestor, 128 B sit on t2.
        assert_eq!(kv.aux_bytes, 128);
        assert_eq!(
            kv.evictions, 0,
            "inheritance must not cost another conversation"
        );
        assert_gradient(&kv);
    }

    /// Two conversations sharing a 12-token prefix and diverging there, so the split leaves an
    /// entry-less node at 12 above two entries at 16. The left one can resume at 8 and 16, the
    /// right one only at 16 — chain length alone cannot tell them apart.
    fn split_recurrent_tree() -> (PagedKv, usize, usize) {
        let mut kv = cache_recurrent(16, 8);
        let a = grow(&mut kv, 4);
        kv.publish_with_ckpts(
            (1..=16).collect(),
            a.clone(),
            vec![ckpt(8, 64), ckpt(16, 64)],
            None,
        )
        .unwrap();
        kv.session_release(&a);
        let b = grow(&mut kv, 4);
        kv.publish_with_ckpts(
            (1..=12).chain([900, 901, 902, 903]).collect(),
            b.clone(),
            vec![ckpt(16, 64)],
            None,
        )
        .unwrap();
        kv.session_release(&b);
        let (left, right) = (node_of(&kv, a[0]), node_of(&kv, b[0]));
        assert_ne!(left, right, "the divergence split the edge");
        (kv, left, right)
    }

    #[test]
    fn a_fork_diverging_inside_a_split_edge_reuses_the_deeper_entry() {
        // The measured sub-agent shape: a third request diverges EARLIER than the split, inside the
        // entry-less node's own edge. The walk's partial candidate carries no entry, and the frames
        // it needs sit one node deeper.
        let (mut kv, left, right) = split_recurrent_tree();
        let fork: Vec<u32> = (1..=10).chain([800, 801]).collect();
        let (node, matched) = kv
            .best_match(&fork, None)
            .expect("the split node's subtree holds the frames");
        assert_eq!(matched, 8);
        assert_eq!(node, left, "the entry whose snapshot is usable wins");
        let (reusable, ckpt) = kv.hit_prefix(node, true, matched);
        assert_eq!((reusable, ckpt.unwrap().boundary_tokens), (8, 8));
        assert!(
            kv.entry_ckpt(right, matched).is_none(),
            "the sibling's only snapshot sits above the branch point"
        );
        assert_eq!(kv.hit_prefix(node, false, matched).0, 8);
    }

    #[test]
    fn a_fork_stopping_at_a_node_with_no_matching_child_reuses_the_subtree() {
        let (mut kv, left, _right) = split_recurrent_tree();
        // The shared prefix ends exactly ON the split node: no child matches the next token, so
        // there is no partial candidate either.
        let fork: Vec<u32> = (1..=12).chain([800, 801]).collect();
        let (node, matched) = kv
            .best_match(&fork, None)
            .expect("the subtree continues it");
        assert_eq!((matched, node), (12, left));
        assert_eq!(kv.hit_prefix(node, true, matched).0, 8);
        // Fully consumed at the split node: the subtree was already scanned here, but the first
        // entry with a long enough chain won, which could be the one that resumes at zero.
        let (node, matched) = kv
            .best_match(&(1..=12).collect::<Vec<u32>>(), None)
            .unwrap();
        assert_eq!((matched, node), (12, left));
        assert_eq!(kv.hit_prefix(node, true, matched).0, 8);
    }

    #[test]
    fn a_dense_fork_inside_a_split_edge_reuses_the_pages_it_matches() {
        let mut kv = cache(16);
        let a = grow(&mut kv, 4);
        kv.publish((1..=16).collect(), a.clone()).unwrap();
        kv.session_release(&a);
        let b = grow(&mut kv, 4);
        kv.publish((1..=12).chain([900, 901, 902, 903]).collect(), b.clone())
            .unwrap();
        kv.session_release(&b);
        let fork: Vec<u32> = (1..=10).chain([800, 801]).collect();
        let (node, matched) = kv
            .best_match(&fork, None)
            .expect("dense pages are their own resume point");
        assert_eq!(matched, 8);
        assert_eq!(kv.hit_prefix(node, false, matched).0, 8);
        assert_eq!(kv.entry_chain(node, 8).len(), 2);
    }

    #[test]
    fn a_usable_snapshot_outranks_a_longer_chain_on_the_matched_path() {
        // A shallow publish splits the deep entry off and leaves an entry ON the walk's path. That
        // entry's chain covers the branch point but its only snapshot sits above it, so taking the
        // first long-enough chain — the old rule — matches and then hands back zero tokens.
        let mut kv = cache_recurrent(16, 8);
        let deep = grow(&mut kv, 4);
        kv.publish_with_ckpts(
            (1..=16).collect(),
            deep.clone(),
            vec![ckpt(8, 64), ckpt(16, 64)],
            None,
        )
        .unwrap();
        kv.session_release(&deep);
        let shallow = grow(&mut kv, 3);
        kv.publish_with_ckpts(
            (1..=12).collect(),
            shallow.clone(),
            vec![ckpt(12, 64)],
            None,
        )
        .unwrap();
        kv.session_release(&shallow);
        let fork: Vec<u32> = (1..=10).chain([800, 801]).collect();
        let (node, matched) = kv.best_match(&fork, None).unwrap();
        assert_eq!(matched, 8);
        assert_eq!(node, node_of(&kv, deep[0]));
        assert_eq!(kv.hit_prefix(node, true, matched).0, 8);
        assert_eq!(
            kv.hit_prefix(node_of(&kv, shallow[0]), true, matched).0,
            0,
            "the path entry is the one that would have resumed at zero"
        );
    }

    #[test]
    fn inheritance_keeps_an_entrys_snapshot_count_bounded_across_turns() {
        // Ten continuations of one conversation. Inheriting whole ancestor sets would accumulate
        // every turn's frontier on the newest entry until the aux budget started evicting OTHER
        // conversations to pay for it; inheriting the deepest keeps the count flat.
        let mut kv = cache_recurrent(64, 8);
        let mut tokens = 0usize;
        for turn in 0..10 {
            let mut chain = if tokens == 0 {
                Vec::new()
            } else {
                let published: Vec<u32> = (1..=tokens as u32).collect();
                kv.entry_chain(node_at(&kv, &published), tokens)
            };
            kv.session_attach(&chain);
            chain.extend(grow(&mut kv, 2)); // 8 more tokens per turn
            tokens += 8;
            kv.publish_with_ckpts(
                (1..=tokens as u32).collect(),
                chain.clone(),
                vec![ckpt(tokens - 4, 64), ckpt(tokens, 64)],
                None,
            )
            .unwrap();
            kv.session_release(&chain);
            let snapshots = kv
                .nodes
                .iter()
                .filter_map(|node| node.entry.as_ref())
                .map(|entry| entry.ckpts.len())
                .max()
                .expect("the conversation is in the tree");
            assert!(
                snapshots <= 3,
                "turn {turn} accumulated {snapshots} snapshots"
            );
            // Every turn's own frontier still resumes, and so does the previous one — the depth a
            // request editing the last user message branches at.
            let (node, matched) = kv
                .best_match(&(1..=tokens as u32).collect::<Vec<_>>(), None)
                .unwrap();
            assert_eq!(kv.hit_prefix(node, true, matched).0, tokens);
            if turn > 0 {
                let (node, matched) = kv
                    .best_match(&(1..=(tokens - 8) as u32).collect::<Vec<_>>(), None)
                    .unwrap();
                assert_eq!(kv.hit_prefix(node, true, matched).0, tokens - 8);
            }
        }
        assert_eq!(kv.aux_bytes, 3 * 64, "three snapshots, not one per turn");
        assert_eq!(kv.evictions, 0);
        assert_gradient(&kv);
    }

    #[test]
    fn a_corridor_only_tree_resumes_from_the_aux_cap_its_caller_sets() {
        // The shape of `kv.paged` without `kv.paged_cache_max`: zero host RAM frames, so the /8
        // derivation is zero and the init caller installs a standing cap. Without it a hybrid
        // recurrent model would silently reuse nothing.
        let mut kv = cache_ram(16, 0);
        assert_eq!(kv.aux_budget, 0, "no RAM pool means no implied aux budget");
        kv.set_aux_budget(128);
        let a = grow(&mut kv, 4); // 16 tokens
        kv.publish_with_ckpts((1..=16).collect(), a.clone(), vec![ckpt(12, 64)], None)
            .unwrap();
        let node = node_of(&kv, a[0]);
        let (reusable, ckpt) = kv.hit_prefix(node, true, 16);
        assert_eq!((reusable, ckpt.unwrap().boundary_tokens), (12, 12));
        assert_eq!(
            kv.pool.capacity_frames(Tier::Ram),
            0,
            "a corridor-only tree never claims a host frame"
        );
        assert_eq!(kv.aux_bytes, 64, "the snapshot fit under the standing cap");
    }

    #[test]
    fn a_recurrent_hit_resumes_only_at_its_snapshot_boundary() {
        let mut kv = cache_ram(16, 8);
        let a = grow(&mut kv, 4); // 16 tokens, 4 pages
        kv.publish_with_ckpts(
            (1..=16).collect(),
            a.clone(),
            vec![ckpt(12, 64), ckpt(4, 64)],
            None,
        )
        .unwrap();
        let node = node_of(&kv, a[0]);
        // Dense attention reuses the whole matched depth and carries no aux with it.
        let (reusable, ckpt) = kv.hit_prefix(node, false, 16);
        assert_eq!((reusable, ckpt.is_none()), (16, true));
        // Recurrent state is summarized, so the resume point is the DEEPEST snapshot, not the chain.
        let (reusable, ckpt) = kv.hit_prefix(node, true, 16);
        assert_eq!((reusable, ckpt.unwrap().boundary_tokens), (12, 12));
        // A promotion that fell short of the snapshot re-picks a shallower boundary rather than
        // resume ahead of the KV it actually has.
        let (reusable, ckpt) = kv.hit_prefix(node, true, 8);
        assert_eq!((reusable, ckpt.unwrap().boundary_tokens), (4, 4));
        // Nothing resumable is nothing reused: the caller re-prefills instead of attaching.
        let (reusable, ckpt) = kv.hit_prefix(node, true, 3);
        assert_eq!((reusable, ckpt.is_none()), (0, true));
    }

    #[test]
    fn aux_budget_trims_the_coldest_snapshots_before_any_frame() {
        let mut kv = cache_ram(8, 8); // aux budget = 8 MiB RAM backing / 8 = 1 MiB
        let a = grow(&mut kv, 2);
        kv.publish_with_ckpts((1..=8).collect(), a.clone(), vec![ckpt(4, 400_000)], None)
            .unwrap();
        let b = grow(&mut kv, 2);
        kv.publish_with_ckpts((21..=28).collect(), b.clone(), vec![ckpt(8, 400_000)], None)
            .unwrap();
        assert_eq!(kv.aux_bytes, 800_000, "both snapshots fit");
        // A third snapshot exceeds the budget: the COLDEST entry's aux leaves first...
        let c = grow(&mut kv, 2);
        kv.publish_with_ckpts((41..=48).collect(), c.clone(), vec![ckpt(8, 400_000)], None)
            .unwrap();
        assert_eq!(kv.aux_bytes, 800_000);
        assert!(
            kv.entry_ckpt(node_of(&kv, a[0]), 8).is_none(),
            "the coldest snapshot was trimmed"
        );
        // ...while its FRAMES and its match stay intact: on a DENSE tree aux is only an
        // accelerator, so trimming it never evicts cache content.
        assert!(kv.best_match(&(1..=8).collect::<Vec<_>>(), None).is_some());
        assert_eq!(kv.evictions, 0);
        // A single conversation may not monopolize the budget either.
        let d = grow(&mut kv, 2);
        kv.publish_with_ckpts(
            (61..=68).collect(),
            d.clone(),
            vec![ckpt(8, 3 * FB as usize)],
            None,
        )
        .unwrap();
        assert_eq!(
            kv.aux_bytes, 0,
            "every other snapshot was trimmed and the oversized incoming one dropped"
        );
        assert!(kv.entry_ckpt(node_of(&kv, d[0]), 8).is_none());
    }

    #[test]
    fn a_recurrent_publish_without_a_resumable_snapshot_is_refused() {
        let mut kv = cache_recurrent(8, 8);
        kv.set_aux_budget(0);
        let a = grow(&mut kv, 2); // 8 tokens
        let tokens = (1..=8).collect::<Vec<u32>>();
        // No snapshot at all — the shape a failed device download leaves behind. The chain would
        // match and then resume zero tokens, so it must never reach the tree.
        assert!(
            kv.publish_with_ckpts(tokens.clone(), a.clone(), Vec::new(), None)
                .is_err(),
            "a recurrent entry without a snapshot is unreachable cache"
        );
        // A snapshot the budget cannot hold is the same dead end.
        assert!(
            kv.publish_with_ckpts(tokens.clone(), a.clone(), vec![ckpt(4, 64)], None)
                .is_err(),
            "the anchor has to fit the aux budget on its own"
        );
        assert_eq!(kv.aux_refusals, 2);
        assert_eq!(kv.aux_bytes, 0);
        assert!(kv.best_match(&tokens, None).is_none());
        assert!(kv.nodes.iter().all(|node| node.entry.is_none()));
        // Refusal lands BEFORE any frame is pinned, so the session still owns them alone.
        assert_eq!(kv.pool.held_frames(Tier::Vram), 2);
        kv.session_release(&a);
        assert_eq!(kv.pool.held_frames(Tier::Vram), 0, "no tree owner leaked");
    }

    #[test]
    fn aux_pressure_evicts_a_recurrent_entry_instead_of_stranding_it() {
        let mut kv = cache_recurrent(8, 8);
        kv.set_aux_budget(400_000); // room for exactly one snapshot
        let a = grow(&mut kv, 2);
        kv.publish_with_ckpts((1..=8).collect(), a.clone(), vec![ckpt(4, 400_000)], None)
            .unwrap();
        kv.session_release(&a);
        assert_eq!(kv.pool.refcount(a[0]), 1, "tree-only owner");
        let b = grow(&mut kv, 2);
        kv.publish_with_ckpts((21..=28).collect(), b.clone(), vec![ckpt(8, 400_000)], None)
            .unwrap();
        kv.session_release(&b);
        // The colder entry left WHOLE: stripping its anchor would have kept a chain that matches
        // and resumes nothing, holding frames the pool could otherwise hand out.
        assert_eq!(kv.aux_bytes, 400_000);
        assert_eq!(kv.evictions, 1);
        assert!(kv.best_match(&(1..=8).collect::<Vec<_>>(), None).is_none());
        assert_eq!(
            kv.pool.held_frames(Tier::Vram),
            2,
            "the evicted entry's frames went back to the corridor"
        );
        let (node, matched) = kv.best_match(&(21..=28).collect::<Vec<_>>(), None).unwrap();
        let (reusable, ckpt) = kv.hit_prefix(node, true, matched);
        assert_eq!((reusable, ckpt.unwrap().boundary_tokens), (8, 8));
        for entry in kv.nodes.iter().filter_map(|node| node.entry.as_ref()) {
            assert!(!entry.ckpts.is_empty(), "every survivor is resumable");
        }
    }

    #[test]
    fn a_recurrent_republish_reuses_its_own_aux_and_keeps_the_anchor() {
        let mut kv = cache_recurrent(8, 8);
        kv.set_aux_budget(400_000);
        let a = grow(&mut kv, 4); // 16 tokens
        let tokens = (1..=16).collect::<Vec<u32>>();
        kv.publish_with_ckpts(
            tokens.clone(),
            a.clone(),
            vec![ckpt(4, 200_000), ckpt(12, 200_000)],
            None,
        )
        .unwrap();
        assert_eq!(kv.aux_bytes, 400_000);
        // The same conversation published again with one more boundary: the replaced entry's bytes
        // must not count against the incoming ones, and the shallow snapshot may go but the
        // deepest — the one a resume actually attaches — may not.
        kv.publish_with_ckpts(
            tokens.clone(),
            a.clone(),
            vec![ckpt(4, 200_000), ckpt(12, 200_000), ckpt(16, 200_000)],
            None,
        )
        .unwrap();
        assert_eq!(kv.aux_bytes, 400_000, "the budget is not exceeded");
        assert_eq!(kv.evictions, 0, "a conversation never evicts itself");
        let node = node_of(&kv, a[0]);
        let (reusable, ckpt) = kv.hit_prefix(node, true, 16);
        assert_eq!((reusable, ckpt.unwrap().boundary_tokens), (16, 16));
    }

    #[test]
    fn eviction_and_supersession_release_aux_bytes() {
        let mut kv = cache_ram(8, 8);
        let short = grow(&mut kv, 2);
        kv.publish_with_ckpts(
            (1..=8).collect(),
            short.clone(),
            vec![ckpt(4, 300_000)],
            None,
        )
        .unwrap();
        kv.session_release(&short);
        assert_eq!(kv.aux_bytes, 300_000);
        // Publishing a longer conversation supersedes the shared ancestor entry: its chain owners
        // leave, and its snapshot MOVES to the deeper entry rather than vanishing — the branch point
        // it describes is still a prefix of the new conversation. Eviction releases all of it.
        let mut long = kv.entry_chain(node_of(&kv, short[0]), 8);
        kv.session_attach(&long);
        long.push(grow(&mut kv, 1)[0]);
        kv.publish_with_ckpts(
            (1..=12).collect(),
            long.clone(),
            vec![ckpt(12, 500_000)],
            None,
        )
        .unwrap();
        assert_eq!(
            kv.aux_bytes, 800_000,
            "the ancestor's snapshot was inherited, not deleted"
        );
        assert_eq!(kv.ckpts_inherited, 1);
        assert_eq!(
            kv.nodes.iter().filter(|node| node.entry.is_some()).count(),
            1,
            "the superseded ancestor no longer holds an entry"
        );
        assert!(kv.entry_ckpt(node_of(&kv, long[0]), 4).is_some());
        kv.session_release(&long);
        assert!(kv.evict_cold_one());
        assert_eq!(kv.aux_bytes, 0, "eviction released the last snapshot");
        assert!(kv.nodes.iter().all(|node| node
            .entry
            .as_ref()
            .is_none_or(|entry| entry.ckpts.is_empty())));
    }

    #[test]
    fn publish_rejects_a_snapshot_that_is_not_a_page_aligned_prefix() {
        let mut kv = cache_ram(16, 8);
        let a = grow(&mut kv, 2);
        let tokens = (1..=8).collect::<Vec<u32>>();
        assert!(
            kv.publish_with_ckpts(tokens.clone(), a.clone(), vec![ckpt(6, 64)], None)
                .is_err(),
            "boundary is not page aligned"
        );
        assert!(
            kv.publish_with_ckpts(tokens.clone(), a.clone(), vec![ckpt(0, 64)], None)
                .is_err(),
            "empty boundary"
        );
        assert!(
            kv.publish_with_ckpts(tokens.clone(), a.clone(), vec![ckpt(16, 64)], None)
                .is_err(),
            "boundary lies beyond the chain"
        );
        // The rejected publishes pinned nothing: the tree is still empty.
        assert_eq!(kv.pool.refcount(a[0]), 1);
        assert!(kv.nodes.iter().all(|node| node.entry.is_none()));
    }
}
