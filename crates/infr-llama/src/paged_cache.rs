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

use crate::seam::kv_frame_pool::{FrameId, FramePool, Tier};
use anyhow::{anyhow, Result};
use infr_core::backend::KvFrameCorridor;
use std::collections::HashMap;

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
    clock: u64,
    hits: u64,
    misses: u64,
    demotions: u64,
    evictions: u64,
}

impl PagedKv {
    pub(crate) fn new(
        frame_bytes: u64,
        page_tokens: usize,
        plane_offsets: Vec<u64>,
        vram_frames: usize,
        ram_frames: usize,
    ) -> Self {
        assert!(
            page_tokens.is_power_of_two(),
            "page_tokens must be a power of two"
        );
        assert!(
            !plane_offsets.is_empty(),
            "a paged KV cache needs at least one plane"
        );
        Self {
            pool: FramePool::new(frame_bytes, vram_frames, ram_frames),
            nodes: vec![Node::new(None, Vec::new(), 0)],
            free: Vec::new(),
            page_tokens,
            plane_offsets,
            clock: 0,
            hits: 0,
            misses: 0,
            demotions: 0,
            evictions: 0,
        }
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

    pub(crate) fn stats_line(&self) -> String {
        format!(
            "entries={} frames_held={}/{} free={} hits={} misses={} demotions={} evictions={}",
            self.nodes
                .iter()
                .filter(|node| node.entry.is_some())
                .count(),
            self.pool.held_frames(Tier::Vram),
            self.pool.capacity_frames(Tier::Vram),
            self.pool.free_frames(Tier::Vram),
            self.hits,
            self.misses,
            self.demotions,
            self.evictions,
        )
    }

    // ---- session-side frame operations ----------------------------------------------------

    /// Claim a fresh frame for a growing session. On an exhausted corridor, evict cold entries
    /// until one slot frees; `None` when every frame is owned by a live session's chain that the
    /// tree cannot drop (all entries gone or every remaining frame still ref-pinned by sessions —
    /// the caller then recycles the slot the old way).
    pub(crate) fn alloc_frame(&mut self) -> Option<FrameId> {
        if let Some(frame) = self.pool.alloc(Tier::Vram) {
            return Some(frame);
        }
        while self.evict_cold_one() {
            if let Some(frame) = self.pool.alloc(Tier::Vram) {
                return Some(frame);
            }
        }
        None
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

    /// The entry node's frames covering `reusable` tokens (a chain prefix — frames are positional
    /// from token 0). Callers pair this with [`Self::session_attach`] under the same lock.
    pub(crate) fn entry_chain(&self, node: usize, reusable: usize) -> Vec<FrameId> {
        let entry = self.nodes[node]
            .entry
            .as_ref()
            .expect("matched node carries an entry");
        entry.chain[..reusable / self.page_tokens].to_vec()
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
        // Two planes so plane_addr exercises the offset prefix sums.
        let mut kv = PagedKv::new(FB, PAGE, vec![0, FB / 2], vram_frames, 0);
        kv.set_vram_segments(&[KvFrameCorridor {
            base: 0x1_0000_0000,
            frames: vram_frames,
        }]);
        kv
    }

    /// A session-shaped chain: `pages` freshly allocated frames (one session owner each).
    fn grow(kv: &mut PagedKv, pages: usize) -> Vec<FrameId> {
        (0..pages)
            .map(|_| kv.alloc_frame().expect("corridor has room"))
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
        let b = kv.alloc_frame().expect("eviction made room");
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
        let (node, reusable) = kv.best_match(&vec![1, 2, 3, 4, 9, 10, 11, 12]).unwrap();
        assert_eq!(reusable, 8);
        assert_eq!(kv.entry_chain(node, reusable), b_frames);
        // A prompt diverging inside the split edge still shares the complete first page.
        let (node, reusable) = kv.best_match(&vec![1, 2, 3, 4, 5, 99]).unwrap();
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
            let chain = kv.alloc_frame().expect("frames recycle as the LRU leaves");
            let chain = vec![chain];
            let tokens = [seed * 10 + 1, seed * 10 + 2, seed * 10 + 3, seed * 10 + 4].to_vec();
            kv.publish(tokens, chain.clone()).unwrap();
            kv.session_release(&chain);
        }
        let steady = kv.nodes.len();
        for seed in 20..40u32 {
            let chain = kv.alloc_frame().expect("frames recycle as the LRU leaves");
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
        let f1 = kv.alloc_frame().unwrap();
        kv.publish(vec![1, 2, 3, 4], vec![f1]).unwrap();
        kv.session_release(&[f1]);
        let f2 = kv.alloc_frame().unwrap();
        let f3 = kv.alloc_frame().unwrap();
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
}
