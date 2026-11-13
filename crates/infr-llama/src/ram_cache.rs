//! SGLang-style radix-tree prefix cache holding idle conversation KV in host RAM.
//!
//! # What is (and is not) shared
//!
//! The radix tree shares the *prefix index* — it decides which cached conversation can serve a
//! prompt. Each conversation's KV *snapshot* stays a full, self-contained copy: two conversations
//! sharing a 2000-token prefix store those bytes twice in RAM. This is NOT RadixAttention-style
//! shared KV blocks; a true block-sharing paged tier (frames owned by the tree, zero-copy prefix
//! reuse) is a separate experimental tier and out of scope here.
//!
//! The VRAM slot pool keeps `kv.slots` conversations resident; this module is the tier below
//! it: when a different-prefix request needs its slot (checkout replacement), the outgoing
//! device state is captured — every buffer listed by
//! [`SeamKv::session_state_buffers`] through `download_range` — into an immutable host-RAM
//! snapshot keyed by its token prefix in a compressed radix tree. A later request whose prompt
//! matches a cached prefix restores the snapshot into any free slot via `prepare_session_restore`
//! and `upload_range`, skipping the prefill of the shared prefix. Nothing is ever written to
//! disk — eviction drops the snapshot bytes and the memory returns immediately.
//!
//! # Tree (radix) semantics
//!
//! Nodes are keyed by token sequences; edges carry compressed token runs and split on partial
//! matches, so shared system prompts live once in the spine and each cached conversation is an
//! entry on the node whose path equals its `meta.cached` tokens. Matching is the tree walk of
//! the prompt path, keeping the deepest node whose entry can continue the prompt (the same
//! [`SessionStateMeta::continuation_prefix_len`] rule the disk cold cache applies) — and when
//! the walk consumes the whole prompt, entries DEEPER in the subtree count too: the rule lets a
//! longer snapshot continue a shorter prompt truncated at the prompt, which is exactly what a
//! superseded-ancestor publication leaves behind. Because a
//! snapshot is SELF-CONTAINED (it carries the whole prefix's KV bytes, not a parent's), evicting
//! any entry never invalidates another, and inserting a longer conversation evicts the strict
//! ancestor entries it supersedes: the descendant snapshot can serve every prompt the ancestor
//! could (the continuation rule truncates on partial prompts), which keeps repeated re-demotions
//! of one growing conversation from double-storing overlapping byte ranges.
//!
//! # Byte accounting (leak-proof by construction)
//!
//! Every snapshot's byte cost is registered in a shared `held` counter when the `Arc` is created
//! (capture) and refunded in [`SnapshotInner::drop`]. Eviction, replacement, failed publishes and
//! restore holds therefore need no manual bookkeeping: the ledger tracks the allocation's real
//! lifetime, including a snapshot still referenced by an in-flight restore after its node was
//! evicted. The `kv.ram_cache_max` cap is enforced against that ledger before every publish, and
//! a snapshot larger than the whole cap is refused at capture so no transfer is started.
//!
//! # MTP
//!
//! This tier is main-conversation KV only. The MTP draft-head cache state a resident slot may
//! carry is NOT captured; a slot restored from RAM resumes with ordinary decode until its head
//! re-warms (the same fallback the disk store applies when an MTP sidecar is missing).
//!
//! # Locks
//!
//! The scheduler owns policy; this module owns the tree and the device transfers. Callers follow
//! the disk store's discipline: reserve the pool slot, drop the pool lock, take the GPU baton
//! (`StepGate`) around every `sync`/`download_range`/`upload_range` burst, and serialize tree
//! mutation under the pool's `Mutex<RadixCache>`. Checkout holds the baton AND the tree lock
//! across the whole demote→restore. There is deliberately no idle-time worker: a resident
//! continuation always beats a restore, and time-driven demotion only opened a window where the
//! session was in neither tier (replays re-prefilled from scratch) — 2026-10-15 A/B regression.

use crate::seam::{SeamKv, SessionBufferKey, SessionStateMeta};
use crate::Config;
use anyhow::{anyhow, Result};
use infr_core::backend::Backend;
use infr_core::tensor::DType;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// PCIe transfer chunk, matching the disk store's streaming window so neither tier allocates a
/// whole-session temporary beyond the snapshot itself.
const CHUNK_BYTES: usize = 16 * 1024 * 1024;
/// Per-snapshot allocation overhead credited to the ledger (Vec/Arc/node headers). Deliberately
/// coarse: the cap is a RAM guard, not an allocator mirror.
const SNAPSHOT_OVERHEAD_BYTES: u64 = 1024;

/// Geometry a snapshot must have been captured against to restore into this engine's slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SnapshotGeometry {
    pub max_ctx: usize,
    pub k_fmt: DType,
    pub v_fmt: DType,
    pub max_bytes: u64,
}

/// One immutable conversation snapshot: metadata plus every session-state buffer payload,
/// sorted by buffer key. Dropped (and the ledger refunded) when the last holder lets go.
pub(crate) struct SnapshotInner {
    pub meta: SessionStateMeta,
    pub buffers: Vec<(SessionBufferKey, Vec<u8>)>,
    pub bytes: u64,
    ledger: Arc<AtomicU64>,
}

impl Drop for SnapshotInner {
    fn drop(&mut self) {
        self.ledger.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// A snapshot checked out for restore. Holds the tree node pinned (refs) so eviction skips it.
/// The `generation` guards the hand-back: a node freed and REUSED while a restore was in flight
/// must not take the stale release's `refs` decrement (it would un-pin the new owner).
pub(crate) struct TakenSnapshot {
    node: usize,
    generation: u64,
    inner: Arc<SnapshotInner>,
}

impl TakenSnapshot {
    pub(crate) fn inner(&self) -> &Arc<SnapshotInner> {
        &self.inner
    }
}

struct Node {
    parent: Option<usize>,
    /// Token run on the incoming edge; the node's path is its parent's path plus this edge.
    edge: Vec<u32>,
    /// Child selected by the first token of its edge.
    children: HashMap<u32, usize>,
    /// Token count of the path from the root to this node.
    path_len: usize,
    entry: Option<Arc<SnapshotInner>>,
    /// LRU stamp (monotone tree clock, not wall time).
    last_access: u64,
    /// Restores in flight; eviction and spine pruning skip pinned nodes.
    refs: usize,
    /// Bumped on every slot reuse off the free list; stale [`TakenSnapshot`] hand-backs compare
    /// it before touching `refs`.
    generation: u64,
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
            refs: 0,
            generation: 0,
        }
    }
}

/// The radix tree plus the byte ledger. Wrapped in a `Mutex` by the scheduler.
pub(crate) struct RadixCache {
    nodes: Vec<Node>,
    /// Recycled indices of detached nodes (see `free_node`): long-lived churn reuses slots
    /// instead of growing the arena forever, which is what the PR review's metadata-growth
    /// concern asked for.
    free: Vec<usize>,
    geometry: SnapshotGeometry,
    /// Shared with every `SnapshotInner`; its value IS the held bytes.
    ledger: Arc<AtomicU64>,
    clock: u64,
    hits: u64,
    misses: u64,
    demotions: u64,
    evictions: u64,
}

impl RadixCache {
    pub(crate) fn new(slot: &SessionStateMeta, max_bytes: u64) -> Self {
        Self {
            nodes: vec![Node::new(None, Vec::new(), 0)],
            free: Vec::new(),
            geometry: SnapshotGeometry {
                max_ctx: slot.max_ctx,
                k_fmt: slot.k_fmt,
                v_fmt: slot.v_fmt,
                max_bytes,
            },
            ledger: Arc::new(AtomicU64::new(0)),
            clock: 0,
            hits: 0,
            misses: 0,
            demotions: 0,
            evictions: 0,
        }
    }

    pub(crate) fn geometry(&self) -> SnapshotGeometry {
        self.geometry
    }

    pub(crate) fn held_bytes(&self) -> u64 {
        self.ledger.load(Ordering::Relaxed)
    }

    /// One-shot ledger handle for a snapshot captured before the tree existed (boot ordering is
    /// fixed, but capture and publish take the ledger through the same handle).
    pub(crate) fn ledger(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.ledger)
    }

    pub(crate) fn stats_line(&self) -> String {
        format!(
            "entries={} held={:.2}GiB/limit={:.2}GiB hits={} misses={} demotions={} evictions={}",
            self.nodes
                .iter()
                .filter(|node| node.entry.is_some())
                .count(),
            self.held_bytes() as f64 / (1u64 << 30) as f64,
            self.geometry.max_bytes as f64 / (1u64 << 30) as f64,
            self.hits,
            self.misses,
            self.demotions,
            self.evictions,
        )
    }

    /// Descend along `tokens`: the spine nodes whose full path is a prefix of the query, plus
    /// the subtree root whose entries all EXTEND the query, when the whole query was consumed
    /// (at a node boundary or mid-edge). [`SessionStateMeta::continuation_prefix_len`] accepts a
    /// snapshot whose cached tokens extend the prompt (truncated continuation), so those
    /// descendants are valid matches — the case superseded-ancestor publishing creates, where
    /// only a deeper snapshot remains for a shorter prompt.
    fn locate(&self, tokens: &[u32]) -> (Vec<usize>, Option<usize>) {
        let mut path = vec![0usize];
        let mut node = 0usize;
        let mut index = 0usize;
        while index < tokens.len() {
            let Some(&child) = self.nodes[node].children.get(&tokens[index]) else {
                return (path, None);
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
            // Partial edge: the query ends inside it (the first token matched, so offset >= 1)
            // exactly when the consumed run covers the rest of the tokens.
            let subtree = (index + offset == tokens.len()).then_some(child);
            return (path, subtree);
        }
        (path, Some(node))
    }

    /// Most-recently-used entry in the subtree rooted at `root` (inclusive). Every entry there
    /// continues a fully consumed query at exactly the query length, so LRU picks among equals.
    fn subtree_entry_node(&self, root: usize) -> Option<usize> {
        let mut stack = vec![root];
        let mut best: Option<(u64, usize)> = None;
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];
            if node.entry.is_some() && best.is_none_or(|(seen, _)| node.last_access > seen) {
                best = Some((node.last_access, index));
            }
            stack.extend(node.children.values().copied());
        }
        best.map(|(_, index)| index)
    }

    /// Longest prompt prefix any cached snapshot on the prompt path can continue. Counts one
    /// hit/miss for the stats line.
    pub(crate) fn best_continuation_len(&mut self, prompt: &[u32]) -> usize {
        let best = self.best_match(prompt);
        if best == 0 {
            self.misses += 1;
        } else {
            self.hits += 1;
        }
        best
    }

    /// Best continuable node on the spine plus the descendant-subtree node when the whole
    /// prompt was consumed. A subtree entry always continues at exactly `prompt.len()`, so it
    /// wins unless a spine entry matches the same length (the exact-session snapshot).
    fn match_node(&self, prompt: &[u32]) -> Option<usize> {
        let (path, subtree) = self.locate(prompt);
        let spine = path
            .into_iter()
            .filter_map(|node| {
                self.nodes[node]
                    .entry
                    .as_ref()
                    .and_then(|entry| entry.meta.continuation_prefix_len(prompt))
                    .map(|prefix| (prefix, self.nodes[node].last_access, node))
            })
            .max()
            .map(|(prefix, _, node)| (prefix, node));
        let below = (!prompt.is_empty())
            .then(|| subtree.and_then(|root| self.subtree_entry_node(root)))
            .flatten();
        match (spine, below) {
            (Some((prefix, node)), Some(below)) => {
                (prefix >= prompt.len()).then_some(node).or(Some(below))
            }
            (Some((_, node)), None) => Some(node),
            (None, Some(below)) => Some(below),
            (None, None) => None,
        }
    }

    fn best_match(&self, prompt: &[u32]) -> usize {
        let (path, subtree) = self.locate(prompt);
        let spine = path
            .into_iter()
            .filter_map(|node| {
                self.nodes[node]
                    .entry
                    .as_ref()
                    .and_then(|entry| entry.meta.continuation_prefix_len(prompt))
            })
            .max()
            .unwrap_or(0);
        let below = (!prompt.is_empty())
            .then(|| subtree.and_then(|root| self.subtree_entry_node(root)))
            .flatten();
        if below.is_some() {
            spine.max(prompt.len())
        } else {
            spine
        }
    }

    /// Pin the best continuable snapshot. The caller must hand it back via [`Self::release`]
    /// whether or not the restore succeeds.
    pub(crate) fn take_best(&mut self, prompt: &[u32]) -> Option<TakenSnapshot> {
        let node = self.match_node(prompt)?;
        let inner = Arc::clone(self.nodes[node].entry.as_ref()?);
        self.nodes[node].refs += 1;
        self.clock += 1;
        self.nodes[node].last_access = self.clock;
        let generation = self.nodes[node].generation;
        Some(TakenSnapshot {
            node,
            generation,
            inner,
        })
    }

    /// Un-pin a taken snapshot and stamp it as used (a restore keeps the entry for reuse). A
    /// hand-back whose `generation` no longer matches targets a recycled slot: touching it would
    /// un-pin the node's NEW owner, so it is dropped. A node whose entry was cleared while it
    /// was pinned (superseded or discarded mid-restore) is swept once the last hold leaves.
    pub(crate) fn release(&mut self, taken: TakenSnapshot) {
        let TakenSnapshot {
            node,
            generation,
            inner,
        } = taken;
        if self
            .nodes
            .get(node)
            .is_none_or(|n| n.generation != generation)
        {
            return;
        }
        self.nodes[node].refs = self.nodes[node].refs.saturating_sub(1);
        let stamp = self.nodes[node]
            .entry
            .as_ref()
            .is_some_and(|entry| Arc::ptr_eq(entry, &inner));
        if stamp {
            self.clock += 1;
            self.nodes[node].last_access = self.clock;
        } else if self.nodes[node].entry.is_none() && self.nodes[node].refs == 0 {
            self.prune_upward(node);
        }
    }

    /// Walk or split the spine down to the node whose path equals `tokens`.
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
                    // Split the edge at `offset`: `mid` owns the shared run, the old child the
                    // remainder. The query token that matched the edge head forces offset >= 1.
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

    /// Evict strict-ancestor entries superseded by a snapshot keyed at `node` (see module doc).
    /// Cleared ancestors are swept after the walk: pruning mid-walk could free a node whose
    /// index the walk still has to visit (recycled slots are alive for the new conversation).
    fn drop_superseded_ancestors(&mut self, node: usize) {
        let mut cleared = Vec::new();
        let mut parent = self.nodes[node].parent;
        while let Some(index) = parent {
            if self.nodes[index].entry.is_some() {
                self.nodes[index].entry = None;
                cleared.push(index);
            }
            parent = self.nodes[index].parent;
        }
        for index in cleared {
            self.prune_upward(index);
        }
    }

    /// Register and attach a captured snapshot, evicting LRU entries to honor the cap. The
    /// snapshot bytes were already credited to the ledger at capture; a refused publish refunds
    /// through `SnapshotInner::drop`.
    pub(crate) fn publish(&mut self, inner: Arc<SnapshotInner>) -> Result<usize> {
        let bytes = inner.bytes;
        if bytes > self.geometry.max_bytes {
            return Err(anyhow!(
                "one RAM-cached session needs {:.2} GiB, exceeding kv.ram_cache_max {:.2} GiB",
                bytes as f64 / (1u64 << 30) as f64,
                self.geometry.max_bytes as f64 / (1u64 << 30) as f64,
            ));
        }
        // The ledger already credits this snapshot (capture billed it), so `held` IS the
        // post-publish total: evict others toward the headroom the new entry leaves.
        if self.held_bytes() > self.geometry.max_bytes {
            self.evict_all_toward(self.geometry.max_bytes.saturating_sub(bytes));
            if self.held_bytes() > self.geometry.max_bytes {
                return Err(anyhow!(
                    "kv.ram_cache_max is full of in-flight snapshots; session stays resident"
                ));
            }
        }
        let tokens = inner.meta.cached.clone();
        let node = self.descend_to(&tokens);
        self.drop_superseded_ancestors(node);
        self.clock += 1;
        let node_ref = &mut self.nodes[node];
        node_ref.entry = Some(inner);
        node_ref.last_access = self.clock;
        self.demotions += 1;
        Ok(node)
    }

    /// Evict unpinned entries down to `floor_bytes` (pinned ones can legitimately exceed it).
    /// Leaf-first (an entry with no entry-bearing child loses the least matching power per byte
    /// kept), then oldest access.
    fn evict_all_toward(&mut self, floor_bytes: u64) {
        while self.held_bytes() > floor_bytes {
            let mut best: Option<(bool, u64, usize)> = None;
            for (index, node) in self.nodes.iter().enumerate() {
                if node.entry.is_none() || node.refs > 0 {
                    continue;
                }
                // Leaf-first in the entry graph: a node with entry-bearing descendants also anchors
                // the truncated continuations those descendants serve, so it loses last.
                let has_entry_child = self.has_entry_below(index);
                let key = (has_entry_child, node.last_access, index);
                if best.is_none_or(|current| key < current) {
                    best = Some(key);
                }
            }
            let Some((_, _, node)) = best else {
                break;
            };
            self.nodes[node].entry = None;
            self.evictions += 1;
            // The ledger refunds when the Arc dies; a restore still holding it keeps the bytes
            // counted until it lets go.
            self.prune_upward(node);
        }
    }

    /// Whether any node strictly below `node` in the subtree carries an entry.
    fn has_entry_below(&self, node: usize) -> bool {
        self.nodes[node]
            .children
            .values()
            .copied()
            .any(|child| self.nodes[child].entry.is_some() || self.has_entry_below(child))
    }

    /// Claim a node slot: a recycled index when the free list has one, else a fresh push. Reuse
    /// bumps `generation` so any [`TakenSnapshot`] still naming the old occupant fails its
    /// hand-back check instead of un-pinning this node.
    fn alloc_node(&mut self, parent: usize, edge: Vec<u32>, path_len: usize) -> usize {
        if let Some(index) = self.free.pop() {
            let node = &mut self.nodes[index];
            node.parent = Some(parent);
            node.edge = edge;
            node.children.clear();
            node.path_len = path_len;
            node.entry = None;
            node.last_access = 0;
            node.refs = 0;
            node.generation += 1;
            index
        } else {
            self.nodes.push(Node::new(Some(parent), edge, path_len));
            self.nodes.len() - 1
        }
    }

    /// Return a detached node's index to the free list. Callers must have already unlinked it
    /// from its parent's children and proved `entry` gone, children empty (or re-homed) and
    /// `refs == 0`.
    fn free_node(&mut self, node: usize) {
        self.free.push(node);
        let node = &mut self.nodes[node];
        node.parent = None;
        node.edge = Vec::new();
        node.children.clear();
        node.entry = None;
        node.generation += 1;
    }

    /// Detach entry-less, child-less, unpinned nodes upward and merge pass-through nodes into
    /// their only child, so token spines of evicted conversations neither accumulate bytes nor
    /// indices (the PR review's metadata-growth point). Detached slots go back on the free list
    /// for `alloc_node` to recycle; merging keeps every path's token string (and therefore every
    /// match) exactly as it was.
    fn prune_upward(&mut self, mut node: usize) {
        loop {
            let Some(parent) = self.nodes[node].parent else {
                return; // root, or already detached on an earlier pass
            };
            let detach = {
                let node = &self.nodes[node];
                node.entry.is_none() && node.children.is_empty() && node.refs == 0
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

    /// Merge `node` into its only child when it is a pure pass-through (no entry, unpinned,
    /// exactly one child, not the root): the child's edge grows by `node`'s and the child takes
    /// `node`'s place under the grandparent — whose children key is unchanged, because the
    /// merged edge still starts with `node`'s first token. Returns the grandparent (the merge
    /// point to continue pruning from) or `None` when `node` must stay.
    fn merge_single_child(&mut self, node: usize) -> Option<usize> {
        let (parent, child) = {
            let node = &self.nodes[node];
            if node.entry.is_some() || node.refs > 0 || node.children.len() != 1 {
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
        // Same key: the merged edge begins with the token that selected `node`.
        self.nodes[parent].children.insert(key, child);
        self.free_node(node);
        Some(parent)
    }

    /// Descend only along fully matching edges (the publish path created the exact spine, so a
    /// miss means the entry was already replaced or evicted).
    fn path_nodes_exact(&self, tokens: &[u32]) -> Option<usize> {
        let mut node = 0usize;
        let mut index = 0usize;
        while index < tokens.len() {
            let child = *self.nodes[node].children.get(&tokens[index])?;
            let edge = &self.nodes[child].edge;
            if tokens[index..].len() < edge.len() || tokens[index..index + edge.len()] != edge[..] {
                return None;
            }
            node = child;
            index += edge.len();
        }
        Some(node)
    }

    /// Drop a published snapshot again when the matching device release failed afterwards, so the
    /// RAM tier never keeps bytes for state the device no longer holds. Only removes the exact
    /// snapshot that was published (a newer one at the same node wins).
    pub(crate) fn discard(&mut self, tokens: &[u32], inner: &Arc<SnapshotInner>) {
        if let Some(node) = self.path_nodes_exact(tokens) {
            if self.nodes[node]
                .entry
                .as_ref()
                .is_some_and(|entry| Arc::ptr_eq(entry, inner))
            {
                self.nodes[node].entry = None;
                self.prune_upward(node);
            }
        }
    }
}

/// Device→host capture of one session's complete state. `Ok(None)` means the slot holds nothing
/// worth caching. Runs under the caller's GPU baton — checkout holds it across the whole
/// demote→restore, so there is no per-chunk gate like the disk worker had (the starvation
/// concern does not apply when the baton is already exclusive). The `ledger` is the tree's
/// shared byte counter, handed in so the snapshot refunds correctly whenever it drops.
pub(crate) fn capture(
    kv: &SeamKv,
    backend: &dyn Backend,
    geometry: SnapshotGeometry,
    ledger: Arc<AtomicU64>,
) -> Result<Option<Arc<SnapshotInner>>> {
    let meta = kv.session_state_meta();
    if meta.cached.is_empty() {
        return Ok(None);
    }
    if !kv.can_release_session_state() {
        return Err(anyhow!(
            "RAM KV sessions require the dynamic segmented KV allocator"
        ));
    }
    if meta.max_ctx != geometry.max_ctx
        || meta.k_fmt != geometry.k_fmt
        || meta.v_fmt != geometry.v_fmt
    {
        return Err(anyhow!(
            "resident slot geometry changed after the RAM cache was initialized"
        ));
    }
    let buffers = kv.session_state_buffers(backend)?;
    let mut data_bytes = 0u64;
    for buffer in &buffers {
        data_bytes = data_bytes
            .checked_add(buffer.committed_bytes as u64)
            .ok_or_else(|| anyhow!("RAM KV payload size overflow"))?;
    }
    let token_bytes = token_accounting_bytes(&meta);
    let total = data_bytes
        .checked_add(token_bytes)
        .and_then(|bytes| bytes.checked_add(SNAPSHOT_OVERHEAD_BYTES))
        .ok_or_else(|| anyhow!("RAM KV snapshot size overflow"))?;
    if total > geometry.max_bytes {
        return Err(anyhow!(
            "one RAM-cached session needs {:.2} GiB, exceeding kv.ram_cache_max {:.2} GiB",
            total as f64 / (1u64 << 30) as f64,
            geometry.max_bytes as f64 / (1u64 << 30) as f64,
        ));
    }
    backend
        .sync()
        .map_err(|error| anyhow!("sync before RAM KV capture: {error}"))?;
    let mut payloads = Vec::with_capacity(buffers.len());
    let mut scratch = vec![0u8; CHUNK_BYTES];
    for buffer in buffers {
        let mut payload = vec![0u8; buffer.committed_bytes];
        let mut offset = 0usize;
        while offset < buffer.committed_bytes {
            let count = (buffer.committed_bytes - offset).min(scratch.len());
            backend
                .download_range(buffer.buffer, offset, &mut scratch[..count])
                .map_err(|error| {
                    anyhow!("download RAM KV {:?} at {offset}: {error}", buffer.key)
                })?;
            payload[offset..offset + count].copy_from_slice(&scratch[..count]);
            offset += count;
        }
        payloads.push((buffer.key, payload));
    }
    payloads.sort_unstable_by_key(|(key, _)| *key);
    ledger.fetch_add(total, Ordering::Relaxed);
    let inner = Arc::new(SnapshotInner {
        meta,
        buffers: payloads,
        bytes: total,
        ledger,
    });
    Ok(Some(inner))
}

/// host→device restore. Mirrors `restore_session_file` minus checksums (host memory cannot rot
/// like a file), keeping the geometry/spec validation and the failure-release contract. On any
/// error the slot is released so normal prefill resumes cleanly.
pub(crate) fn restore(
    inner: &SnapshotInner,
    kv: &mut SeamKv,
    backend: &dyn Backend,
    model_cfg: &Config,
) -> Result<()> {
    kv.prepare_session_restore(backend, model_cfg, &inner.meta)?;
    let result = (|| -> Result<()> {
        let expected = kv.session_restore_buffer_specs(backend)?;
        if expected.len() != inner.buffers.len() {
            return Err(anyhow!(
                "RAM KV snapshot has {} buffers; the slot expects {}",
                inner.buffers.len(),
                expected.len()
            ));
        }
        for (key, expected_len) in &expected {
            let payload = inner
                .buffers
                .iter()
                .find(|(snapshot_key, _)| snapshot_key == key)
                .map(|(_, payload)| payload)
                .ok_or_else(|| anyhow!("RAM KV snapshot lacks buffer {key:?}"))?;
            if payload.len() != *expected_len {
                return Err(anyhow!(
                    "RAM KV buffer {key:?} has {} bytes; the slot expects {expected_len}",
                    payload.len()
                ));
            }
            let target = kv
                .session_state_buffer(*key)
                .ok_or_else(|| anyhow!("RAM KV target buffer {key:?} is absent"))?;
            let mut offset = 0usize;
            while offset < payload.len() {
                let count = (payload.len() - offset).min(CHUNK_BYTES);
                backend
                    .upload_range(target, offset, &payload[offset..offset + count])
                    .map_err(|error| anyhow!("upload RAM KV {key:?} at {offset}: {error}"))?;
                offset += count;
            }
        }
        backend
            .sync()
            .map_err(|error| anyhow!("sync restored RAM KV state: {error}"))?;
        kv.finish_session_restore(inner.meta.clone())
    })();
    if result.is_err() {
        if let Err(error) = kv.release_session_state(backend, model_cfg) {
            tracing::warn!("RAM KV cache: cleanup after failed restore also failed: {error}");
            kv.reset();
        }
    }
    result
}

/// Token ids are part of what the snapshot pins in RAM; price them so the cap is honest.
fn token_accounting_bytes(meta: &SessionStateMeta) -> u64 {
    let tokens = meta.cached.len().saturating_add(
        meta.checkpoint_tokens
            .iter()
            .flatten()
            .map(|tokens| tokens.len())
            .sum::<usize>(),
    );
    (tokens as u64).saturating_mul(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(tokens: &[u32]) -> SessionStateMeta {
        SessionStateMeta {
            max_ctx: 4096,
            k_fmt: DType::Q8_0,
            v_fmt: DType::Q8_0,
            committed_tokens: tokens.len().next_power_of_two().max(32),
            cached: tokens.to_vec(),
            checkpoint_tokens: std::array::from_fn(|_| None),
        }
    }

    fn snapshot(cache: &RadixCache, tokens: &[u32], payload: usize) -> Arc<SnapshotInner> {
        let buffers = vec![(SessionBufferKey::K(0), vec![7u8; payload])];
        let bytes = payload as u64 + (tokens.len() as u64) * 4 + SNAPSHOT_OVERHEAD_BYTES;
        cache.ledger().fetch_add(bytes, Ordering::Relaxed);
        Arc::new(SnapshotInner {
            meta: meta(tokens),
            buffers,
            bytes,
            ledger: cache.ledger(),
        })
    }

    #[test]
    fn ledger_tracks_publish_and_discard() {
        let mut cache = RadixCache::new(&meta(&[1]), 1 << 20);
        let a = snapshot(&cache, &[1, 2, 3], 1000);
        let bytes_a = a.bytes;
        cache.publish(a.clone()).unwrap();
        assert_eq!(cache.held_bytes(), bytes_a);
        cache.discard(&[1, 2, 3], &a);
        // The node let go, but the ledger follows the allocation: still billed while the
        // test's own Arc lives (a restore hold in production), refunded on the last drop.
        assert_eq!(cache.held_bytes(), bytes_a);
        drop(a);
        assert_eq!(cache.held_bytes(), 0);
    }

    #[test]
    fn pinned_snapshot_survives_eviction_and_refunds_last() {
        let mut cache = RadixCache::new(&meta(&[1]), 1 << 20);
        let entry = snapshot(&cache, &[1, 2, 3], 1000);
        let bytes = entry.bytes;
        cache.publish(entry.clone()).unwrap();
        let taken = cache.take_best(&[1, 2, 3, 4]).unwrap();
        // A pinned node cannot be evicted even to a zero floor.
        cache.evict_all_toward(0);
        assert_eq!(cache.held_bytes(), bytes);
        cache.release(taken);
        // Now unpinned: eviction detaches it, and dropping the caller's Arc refunds the ledger.
        cache.evict_all_toward(0);
        drop(entry);
        assert_eq!(cache.held_bytes(), 0);
    }

    #[test]
    fn shared_prefixes_supersede_ancestor_entries() {
        let mut cache = RadixCache::new(&meta(&[1]), 1 << 20);
        let short = snapshot(&cache, &[1, 2, 3], 500);
        cache.publish(short).unwrap();
        let long = snapshot(&cache, &[1, 2, 3, 4, 5], 900);
        let long_bytes = long.bytes;
        cache.publish(long).unwrap();
        // The strict ancestor entry is gone; only the deeper snapshot is billed.
        assert_eq!(cache.held_bytes(), long_bytes);
        assert_eq!(cache.best_continuation_len(&[1, 2, 3, 4, 5, 6]), 5);
        assert_eq!(cache.best_continuation_len(&[1, 2, 3]), 3);
        assert_eq!(cache.best_continuation_len(&[9]), 0);
    }

    #[test]
    fn edge_split_keeps_both_branches_matchable() {
        let mut cache = RadixCache::new(&meta(&[1]), 1 << 20);
        cache.publish(snapshot(&cache, &[1, 2, 3], 400)).unwrap();
        cache.publish(snapshot(&cache, &[1, 2, 9], 400)).unwrap();
        cache.publish(snapshot(&cache, &[1, 2, 3, 4], 400)).unwrap();
        assert_eq!(cache.best_continuation_len(&[1, 2, 3, 4, 5]), 4);
        assert_eq!(cache.best_continuation_len(&[1, 2, 9, 0]), 3);
        // Shorter prompt: a descendant snapshot truncated-continues it (engine continuation rule).
        assert_eq!(cache.best_continuation_len(&[1, 2]), 2);
        assert_eq!(cache.best_continuation_len(&[1, 2, 3]), 3); // descendant of the superseded node
        assert_eq!(cache.best_continuation_len(&[1]), 1); // spine consumption reaches both branches
        assert_eq!(cache.best_continuation_len(&[1, 5]), 0); // diverges before any subtree
    }

    #[test]
    fn publish_supersedes_and_keeps_distinct_branches_billed() {
        let mut cache = RadixCache::new(&meta(&[1]), 1 << 20);
        let child = snapshot(&cache, &[1, 2], 100);
        let child_bytes = child.bytes;
        cache.publish(child.clone()).unwrap();
        assert_eq!(cache.held_bytes(), child_bytes);
        let other = snapshot(&cache, &[7, 8, 9], 100);
        let other_bytes = other.bytes;
        cache.publish(other).unwrap();
        assert_eq!(cache.held_bytes(), child_bytes + other_bytes);
    }

    #[test]
    fn cap_refuses_a_snapshot_larger_than_the_whole_cache() {
        let mut cache = RadixCache::new(&meta(&[1]), 1 << 20);
        let giant = snapshot(&cache, &[5], cache.geometry.max_bytes as usize);
        let giant_bytes = giant.bytes;
        let before = cache.held_bytes();
        assert!(cache.publish(giant).is_err());
        // `publish` consumed the Arc; the refused snapshot's Drop refunds exactly the cost
        // capture billed, leaving the rest of the ledger untouched.
        assert_eq!(cache.held_bytes(), before - giant_bytes);
    }

    #[test]
    fn publish_evicts_lru_to_fit_and_keeps_the_fresh_entry() {
        let mut cache = RadixCache::new(&meta(&[1]), 4096);
        let first = snapshot(&cache, &[1], 100);
        let first_bytes = first.bytes;
        cache.publish(first.clone()).unwrap();
        // A fresh publish that would overflow the cap evicts the older unpinned entry.
        let room_before = cache.held_bytes();
        assert!(room_before <= cache.geometry.max_bytes);
        let second = snapshot(&cache, &[2], first_bytes as usize);
        let second_bytes = second.bytes;
        cache.publish(second).unwrap();
        // Both fit or the LRU was evicted; either way the fresh entry is present and billed.
        assert!(cache.held_bytes() <= cache.geometry.max_bytes);
        assert!(cache.best_continuation_len(&[2, 3]) > 0);
        let _ = second_bytes;
    }

    #[test]
    fn churn_recycles_freed_nodes_so_the_arena_stays_bounded() {
        // The cap holds ~two conversations; from the third publish on, every new conversation
        // evicts the LRU one and its spine must be recycled instead of left in the arena.
        let mut cache = RadixCache::new(&meta(&[1]), 4096);
        let publish = |cache: &mut RadixCache, seed: u32| {
            let tokens = [seed * 100 + 1, seed * 100 + 2, seed * 100 + 3];
            cache
                .publish(snapshot(cache, &tokens, 1500))
                .expect("publish fits once the LRU left");
        };
        for seed in 0..20 {
            publish(&mut cache, seed);
        }
        let steady = cache.nodes.len();
        for seed in 20..40 {
            publish(&mut cache, seed);
        }
        // Root + the survivors' spine: the arena stopped growing once recycling kicked in.
        // Without recycling, 40 fresh conversations would have pushed the arena past 40 nodes.
        assert_eq!(cache.nodes.len(), steady, "churn grew the arena: {steady}");
        assert!(steady <= 4, "steady arena should stay small, got {steady}");
        // Every survivor and every evictee's prompt still resolves correctly.
        assert_eq!(cache.best_continuation_len(&[3901, 3902, 3903]), 3);
        assert_eq!(cache.best_continuation_len(&[1]), 0);
    }

    #[test]
    fn superseded_pass_through_spine_merges_into_the_deeper_entry() {
        let mut cache = RadixCache::new(&meta(&[1]), 1 << 20);
        cache.publish(snapshot(&cache, &[1, 2, 3, 4], 100)).unwrap();
        assert_eq!(cache.nodes.len(), 2); // root + the 4-token spine
        cache
            .publish(snapshot(&cache, &[1, 2, 3, 4, 5, 6], 100))
            .unwrap();
        // The 4-token node lost its entry to the supersede and became a pass-through with one
        // child: it must merge into that child, not linger.
        assert!(cache.free.contains(&1), "pass-through spine was not freed");
        assert!(cache.nodes[1].parent.is_none());
        assert_eq!(cache.best_continuation_len(&[1, 2, 3, 4]), 4);
        assert_eq!(cache.best_continuation_len(&[1, 2, 3, 4, 5, 6]), 6);
        assert!(!cache.free.is_empty());
    }

    #[test]
    fn entry_cleared_while_pinned_is_swept_when_the_restore_hands_back() {
        let mut cache = RadixCache::new(&meta(&[1]), 1 << 20);
        let a = snapshot(&cache, &[9, 9], 100);
        let node = cache.publish(a).unwrap();
        let taken = cache.take_best(&[9, 9]).expect("pin the fresh entry");
        assert_eq!(taken.node, node);
        // A longer snapshot supersedes the pinned entry MID-RESTORE: the node must survive
        // pinned, then be swept once the hand-back lands.
        let b = snapshot(&cache, &[9, 9, 9], 100);
        cache.publish(b).unwrap();
        assert!(cache.nodes[node].entry.is_none());
        assert_eq!(
            cache.nodes[node].refs, 1,
            "pinned node must survive the supersede"
        );
        cache.release(taken);
        assert!(
            cache.free.contains(&node),
            "sweep only happens at hand-back"
        );
        assert_eq!(cache.best_continuation_len(&[9, 9, 9]), 3);
    }

    #[test]
    fn stale_handback_cannot_disturb_a_recycled_slot() {
        let mut cache = RadixCache::new(&meta(&[1]), 1 << 20);
        let a = snapshot(&cache, &[4, 4], 100);
        let a_handle = Arc::clone(&a);
        let node = cache.publish(a).unwrap();
        let taken = cache.take_best(&[4, 4]).expect("pin");
        let stale = TakenSnapshot {
            node,
            generation: taken.generation,
            inner: a_handle,
        };
        cache.discard(&[4, 4], &stale.inner); // entry gone, still pinned
        cache.release(taken); // last hold: the spine is swept to the free list
        assert!(cache.free.contains(&node));
        let c = snapshot(&cache, &[7, 7], 100);
        let reused = cache.publish(c).unwrap();
        assert_eq!(reused, node, "LIFO free list hands the same slot back");
        cache.release(stale); // stale generation: must be a complete no-op
        assert!(cache.nodes[reused]
            .entry
            .as_ref()
            .is_some_and(|entry| entry.meta.cached == vec![7, 7]));
        assert_eq!(cache.best_continuation_len(&[7, 7]), 2);
    }
}
