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
//! entry's edge, which the blob tier's whole-snapshot rule could not share. Turn-recurrent
//! checkpoints are not paged (Phase D attaches them as node aux).
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

use crate::seam::kv_frame_pool::{FrameId, FramePool, Tier};
use anyhow::{anyhow, Result};
use infr_core::backend::{Backend, KvFrameCorridor};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Frame-granularity DMA between the VRAM corridor and host memory, factored out of [`Backend`]
/// so the pool/tree logic tests against an in-memory stand-in. Blanket-impl'd for every backend.
pub(crate) trait FrameDma {
    /// Copy one frame's corridor bytes into host memory (`dst.len() == frame_bytes`).
    fn download_frame(&self, frame_addr: u64, dst: &mut [u8]) -> Result<()>;
    /// Copy one frame's host bytes into its corridor slot (`src.len() == frame_bytes`).
    fn upload_frame(&self, frame_addr: u64, src: &[u8]) -> Result<()>;
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
    clock: u64,
    hits: u64,
    misses: u64,
    demotions: u64,
    evictions: u64,
    tier_demotions: u64,
    tier_promotions: u64,
    disk_demotions: u64,
    disk_promotions: u64,
}

impl PagedKv {
    pub(crate) fn new(
        frame_bytes: u64,
        page_tokens: usize,
        plane_offsets: Vec<u64>,
        vram_frames: usize,
        ram_frames: usize,
        disk_frames: usize,
        disk_path: Option<PathBuf>,
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
            nodes: vec![Node::new(None, Vec::new(), 0)],
            free: Vec::new(),
            page_tokens,
            plane_offsets,
            ram_backing: Vec::new(),
            disk,
            clock: 0,
            hits: 0,
            misses: 0,
            demotions: 0,
            evictions: 0,
            tier_demotions: 0,
            tier_promotions: 0,
            disk_demotions: 0,
            disk_promotions: 0,
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
            "entries={} frames_held={}/{} free={} tokens_held={}/{} bytes={:.2}GiB/{:.2}GiB ram={}/{} ram_bytes={:.2}GiB disk={}/{} disk_bytes={:.2}GiB hits={} misses={} demotions={} evictions={} sunk={} promoted={} disk_sunk={} disk_promoted={}",
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
            self.hits,
            self.misses,
            self.demotions,
            self.evictions,
            self.tier_demotions,
            self.tier_promotions,
            self.disk_demotions,
            self.disk_promotions,
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
            if self.demote_coldest_entry(be, protect) || self.demote_ram_coldest_entry(protect) {
                continue;
            }
            if self.evict_cold_one() {
                continue;
            }
            return None;
        }
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

    /// DMA the coldest RAM-eligible tree entry (leaf-first, LRU — the [`Self::evict_cold_one`]
    /// order) into the RAM tier and rewrite its chain to RAM ids. Skips entries with any ACTIVE
    /// frame (a session table points there — migration would corrupt it) and entries already sunk
    /// (a second sink is a no-op). Returns whether at least one frame left VRAM. A DMA failure
    /// rolls the entry back frame by frame: whatever already sank stays sunk (its bytes are safe
    /// in the backing), the rest stay in VRAM, and the entry keeps a consistent mixed chain.
    fn demote_coldest_entry<M: FrameDma + ?Sized>(&mut self, be: &M, protect: &[FrameId]) -> bool {
        if self.pool.capacity_frames(Tier::Ram) == 0 {
            return false;
        }
        let mut best: Option<(usize, usize)> = None; // (node, chain length)
        let mut best_key: Option<(bool, u64, usize)> = None;
        for (index, node) in self.nodes.iter().enumerate() {
            let Some(entry) = node.entry.as_ref() else {
                continue;
            };
            if entry.chain.is_empty()
                || entry.chain.iter().any(|f| !matches!(f.tier(), Tier::Vram))
                || entry.chain.iter().any(|f| self.pool.is_active(*f))
                || entry.chain.iter().any(|f| protect.contains(f))
            {
                continue;
            }
            if self.pool.free_frames(Tier::Ram) < entry.chain.len() {
                continue;
            }
            let key = (self.has_entry_below(index), node.last_access, index);
            if best_key.is_none_or(|current| key < current) {
                best_key = Some(key);
                best = Some((index, entry.chain.len()));
            }
        }
        let Some((node, len)) = best else {
            return false;
        };
        let frame_bytes = self.pool.frame_bytes() as usize;
        let mut moved = 0usize;
        for i in 0..len {
            let source = self.nodes[node].entry.as_ref().unwrap().chain[i];
            let Some(fresh) = self.pool.alloc(Tier::Ram) else {
                break; // RAM filled mid-chain: the entry keeps a mixed chain, still consistent
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
        moved > 0
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

    /// Sink the coldest ALL-RAM tree entry to the disk tier (the same leaf-first LRU order as
    /// [`Self::demote_coldest_entry`], one tier down). Mixed chains (some frames still VRAM) are
    /// skipped exactly like the VRAM sinker skips chains with host frames: a chain crosses tiers
    /// only in the direction the cascade needs, never partially back. A write failure keeps the
    /// entry's mixed RAM/disk chain consistent — whatever already sank stays sunk.
    fn demote_ram_coldest_entry(&mut self, protect: &[FrameId]) -> bool {
        if self.disk.is_none() || self.pool.capacity_frames(Tier::Disk) == 0 {
            return false;
        }
        let mut best: Option<(usize, usize)> = None; // (node, chain length)
        let mut best_key: Option<(bool, u64, usize)> = None;
        for (index, node) in self.nodes.iter().enumerate() {
            let Some(entry) = node.entry.as_ref() else {
                continue;
            };
            if entry.chain.is_empty()
                || entry.chain.iter().any(|f| !matches!(f.tier(), Tier::Ram))
                || entry.chain.iter().any(|f| protect.contains(f))
            {
                continue;
            }
            if self.pool.free_frames(Tier::Disk) < entry.chain.len() {
                continue;
            }
            let key = (self.has_entry_below(index), node.last_access, index);
            if best_key.is_none_or(|current| key < current) {
                best_key = Some(key);
                best = Some((index, entry.chain.len()));
            }
        }
        let Some((node, len)) = best else {
            return false;
        };
        let frame_bytes = self.pool.frame_bytes() as usize;
        let mut moved = 0usize;
        for i in 0..len {
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
    pub(crate) fn best_match(&mut self, prompt: &[u32]) -> Option<(usize, usize)> {
        let (path, consumed, partial) = self.walk(prompt);
        let pages = consumed / self.page_tokens;
        if pages == 0 {
            self.misses += 1;
            return None;
        }
        let mut candidates: Vec<usize> = Vec::new();
        candidates.extend(partial);
        candidates.extend(path.iter().rev().copied());
        if consumed == prompt.len() {
            // Fully consumed prompt at a node boundary: every subtree entry continues it.
            if let Some(&root) = path.last() {
                let mut stack = vec![root];
                while let Some(index) = stack.pop() {
                    if self.nodes[index].entry.is_some() {
                        candidates.push(index);
                    }
                    stack.extend(self.nodes[index].children.values().copied());
                }
            }
        }
        let chosen = candidates.into_iter().find(|&node| {
            self.nodes[node]
                .entry
                .as_ref()
                .is_some_and(|entry| entry.chain.len() >= pages)
        });
        let Some(node) = chosen else {
            self.misses += 1;
            return None;
        };
        self.hits += 1;
        self.clock += 1;
        self.nodes[node].last_access = self.clock;
        for &frame in &self.nodes[node].entry.as_ref().unwrap().chain[..pages] {
            self.pool.touch(frame);
        }
        Some((node, pages * self.page_tokens))
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

    /// Publish a demoted session's conversation. `chain` must cover exactly the complete pages of
    /// `tokens`. Pins every frame for the tree BEFORE the caller drops the session's owners, so
    /// shared prefix frames never hit refcount zero mid-swap (the black-hole-window lesson).
    pub(crate) fn publish(&mut self, tokens: Vec<u32>, chain: Vec<FrameId>) -> Result<()> {
        let pages = tokens.len() / self.page_tokens;
        if chain.len() != pages {
            return Err(anyhow!(
                "paged publish: {} tokens cover {pages} pages but the chain carries {}",
                tokens.len(),
                chain.len()
            ));
        }
        for &frame in &chain {
            self.pool.pin(frame);
        }
        let node = self.descend_to(&tokens);
        self.drop_superseded_ancestors(node);
        self.clock += 1;
        let target = &mut self.nodes[node];
        // A newer entry at the same node replaces the old chain; the old frames' tree owners
        // leave now (session owners keep any still in use alive).
        if let Some(old) = target.entry.replace(ChainEntry { chain }) {
            for frame in old.chain {
                self.pool.unpin(frame);
            }
        }
        self.nodes[node].last_access = self.clock;
        self.demotions += 1;
        Ok(())
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
            for frame in entry.chain {
                self.pool.unpin(frame);
            }
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
    fn walk(&self, tokens: &[u32]) -> (Vec<usize>, usize, Option<usize>) {
        let mut path = vec![0usize];
        let mut node = 0usize;
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

    fn descend_to(&mut self, tokens: &[u32]) -> usize {
        let mut node = 0usize;
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

    /// Unpin ancestor entry chains superseded by the chain just published at `node`. The new
    /// entry already holds owners on those frames, so the ancestors' owners leaving cannot free
    /// anything the new chain still references. Cleared ancestors are swept AFTER the walk:
    /// pruning mid-walk could free a node whose index the walk still has to visit.
    fn drop_superseded_ancestors(&mut self, node: usize) {
        let mut cleared = Vec::new();
        let mut parent = self.nodes[node].parent;
        while let Some(index) = parent {
            if self.nodes[index].entry.is_some() {
                cleared.push(index);
            }
            parent = self.nodes[index].parent;
        }
        for index in cleared {
            if let Some(entry) = self.nodes[index].entry.take() {
                for frame in entry.chain {
                    self.pool.unpin(frame);
                }
            }
            self.prune_upward(index);
        }
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
        let mut kv = PagedKv::new(FB, PAGE, vec![0, FB / 2], vram_frames, ram_frames, 0, None);
        kv.set_vram_segments(&[KvFrameCorridor {
            base: 0x1_0000_0000,
            frames: vram_frames,
        }]);
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
        );
        kv.set_vram_segments(&[KvFrameCorridor {
            base: 0x1_0000_0000,
            frames: vram_frames,
        }]);
        (kv, path)
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
        let dma = MockDma::default();
        (0..pages)
            .map(|_| kv.alloc_frame(&dma).expect("corridor has room"))
            .collect()
    }

    #[test]
    fn publish_then_match_returns_the_page_aligned_chain_prefix() {
        let mut kv = cache(16);
        let a = grow(&mut kv, 3); // 12 tokens, exactly 3 pages
        kv.publish((1..=12).collect(), a.clone()).unwrap();
        // Exact replay: full chain.
        let (node, reusable) = kv.best_match(&(1..=12).collect::<Vec<_>>()).unwrap();
        assert_eq!(reusable, 12);
        assert_eq!(kv.entry_chain(node, reusable), a);
        // Longer prompt: truncated continuation at the entry length.
        let (node, reusable) = kv.best_match(&(1..=20).collect::<Vec<_>>()).unwrap();
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
        let (node, reusable) = kv.best_match(&prompt).unwrap();
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
        let (node, reusable) = kv.best_match(&(1..=10).collect::<Vec<u32>>()).unwrap();
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
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<u32>>()).unwrap();
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
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<_>>()).unwrap();
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
            kv.best_match(&(1..=8).collect::<Vec<_>>()),
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
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<_>>()).unwrap();
        assert_eq!(reusable, 8);
        assert_eq!(kv.entry_chain(node, reusable), a);
        let (node, reusable) = kv.best_match(&[1, 2, 3, 4, 9, 10, 11, 12]).unwrap();
        assert_eq!(reusable, 8);
        assert_eq!(kv.entry_chain(node, reusable), b_frames);
        // A prompt diverging inside the split edge still shares the complete first page.
        let (node, reusable) = kv.best_match(&[1, 2, 3, 4, 5, 99]).unwrap();
        assert_eq!(reusable, 4);
        assert_eq!(kv.entry_chain(node, reusable), vec![a[0]]);
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
        assert!(steady <= 4, "steady arena should stay small, got {steady}");
        let (_, reusable) = kv
            .best_match(&[381, 382, 383, 384])
            .expect("the second-newest conversation survived");
        assert_eq!(reusable, 4);
        assert!(
            kv.best_match(&[1, 2, 3, 4]).is_none(),
            "evicted prompt is gone"
        );
    }

    #[test]
    fn superseded_pass_through_spine_merges_into_the_deeper_entry() {
        let mut kv = cache(4);
        let f1 = kv.alloc_frame(&MockDma::default()).unwrap();
        kv.publish(vec![1, 2, 3, 4], vec![f1]).unwrap();
        kv.session_release(&[f1]);
        let f2 = kv.alloc_frame(&MockDma::default()).unwrap();
        let f3 = kv.alloc_frame(&MockDma::default()).unwrap();
        kv.publish(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![f2, f3])
            .unwrap();
        kv.session_release(&[f2, f3]);
        // The 4-token spine node lost its entry to the supersede and became a pass-through with
        // one child: it must merge into that child, not linger.
        assert!(kv.free.contains(&1), "pass-through spine was not freed");
        let (node, reusable) = kv
            .best_match(&[1, 2, 3, 4])
            .expect("the merged spine still matches");
        assert_eq!(reusable, 4);
        assert_eq!(kv.entry_chain(node, 4), vec![f2]);
        let (_, reusable) = kv
            .best_match(&[1, 2, 3, 4, 5, 6, 7, 8])
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
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<u32>>()).unwrap();
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
        assert!(kv.best_match(&(1..=8).collect::<Vec<u32>>()).is_some());
        assert!(
            kv.best_match(&[1, 2, 3, 4, 9, 10, 11, 12]).is_some(),
            "sinking never deletes an entry"
        );
        kv.session_release(&[c]);
        // Promote branch A fully, then branch B: the shared frame must land on one id owned
        // by exactly two chains again.
        let (node_a, reusable_a) = kv.best_match(&(1..=8).collect::<Vec<u32>>()).unwrap();
        let back_a = kv.entry_chain_promoted(&dma, node_a, reusable_a);
        assert_eq!(back_a.len(), 2, "the corridor can host the whole chain");
        let (node_b, reusable_b) = kv.best_match(&[1, 2, 3, 4, 9, 10, 11, 12]).unwrap();
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
        let (node, reusable) = kv.best_match(&(1..=8).collect::<Vec<u32>>()).unwrap();
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

    #[test]
    fn disk_too_small_to_host_a_chain_falls_back_to_eviction() {
        // One disk slot cannot hold a two-frame chain: the RAM sinker skips the entry (capacity
        // check), the cascade yields nothing, and eviction reclaims the tier exactly like the
        // disk-closed Phase C path.
        let (mut kv, path) = cache_disk(2, 2, 1);
        let dma = MockDma::default();
        let a = grow(&mut kv, 2);
        kv.publish((1..=8).collect(), a.clone()).unwrap();
        kv.session_release(&a);
        let b = grow(&mut kv, 2);
        kv.publish((21..=28).collect(), b.clone()).unwrap();
        kv.session_release(&b);
        assert_eq!(kv.tier_demotions, 2, "A sank to RAM");
        let c = kv
            .alloc_frame(&dma)
            .expect("eviction freed the corridor when disk could not help");
        assert_eq!(
            kv.disk_demotions, 0,
            "the chain never half-sinks to an undersized disk"
        );
        // Interleaving sinks with eviction: evicting A frees the RAM slots that let B SINK, so
        // only the truly unpromotable cold entry (A) is lost.
        assert_eq!(
            kv.evictions, 1,
            "only the undersized-disk entry was dropped"
        );
        assert_eq!(kv.tier_demotions, 4, "B sank once A's RAM slots came back");
        assert_eq!(kv.pool.held_frames(Tier::Disk), 0);
        assert_eq!(kv.pool.held_frames(Tier::Ram), 2, "B rests in RAM");
        kv.session_release(&[c]);
        drop(kv);
        let _ = std::fs::remove_file(&path);
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
}
