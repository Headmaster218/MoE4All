//! Bounded, model-independent prompt lookup. Every draft still requires target verification.
//!
//! Call `sync` with confirmed prompt/output tokens, then `propose(pending, max_drafts)`.
//! `pending` is an optional SINGLE token absent from the confirmed history; it participates
//! in the query exactly once and is never indexed. Returned tokens exclude pending.
//! After verification, report draft acceptance with `record_result` and append ONLY the
//! newly confirmed tokens (including pending/bonus tokens as appropriate). Result reporting
//! never commits tokens. No tokenizer, model, MTP, KV, or scheduler state is accessed here.
//!
//! For ordinary append-only decoding, use `append_confirmed` after the initial `sync`.
//! Use `sync` again after edits, rewind, compaction, or context replacement. It compares
//! the entire retained overlap exactly; detecting arbitrary edits cannot use an unchecked
//! append fast path. Edits strictly before the retained window cannot affect lookup.
//!
//! Let H = history_capacity, P = positions_per_key, M = max_match, D = max_drafts.
//! Appending N confirmed tokens costs expected O(N), including ring/index eviction;
//! `sync` costs O(H) worst case, even for a large replacement. A proposal costs
//! expected O(P*M + D), independent of total session length, and allocates only its
//! output. Storage is O(H) tokens/keys/positions plus O(D) output, with at most P recent
//! positions per exact trigram and at most H-2 positions in total. Hash-map operations
//! have their usual expected-time qualification; collisions cannot change outputs.
//!
//! Search is approximate only in coverage: evicted positions are deliberately forgotten.
//! Among retained candidates, the longest exact suffix wins, then the newest position.
//! Continuations may overlap the current suffix but contain only confirmed tokens; no
//! periodic extrapolation or confidence vote is made from overlapping occurrences.
//! Repeated calls are deterministic except for the documented cooldown counter.
//!
//! This file is standalone and uses only std. The parent must register the module and
//! connect its candidates to the shared verifier; nothing is enabled by adding this file.

use std::collections::{HashMap, VecDeque};

/// Per-request CPU memory, search, and rejection policy. Immutable after construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupConfig {
    pub min_match: usize,
    pub max_match: usize,
    /// Maximum draft tokens, excluding the optional pending token.
    pub max_drafts: usize,
    /// Adjust draft width from acceptance feedback. Rejection cooldown applies in either mode.
    pub adaptive_width: bool,
    pub history_capacity: usize,
    pub positions_per_key: usize,
    /// Consecutive completely rejected proposals before temporarily backing off.
    pub reject_threshold: usize,
    /// Number of nonzero-budget proposal calls to skip. Zero disables cooldown.
    pub cooldown_steps: usize,
}

impl Default for LookupConfig {
    fn default() -> Self {
        Self {
            min_match: 8,
            max_match: 64,
            max_drafts: 3,
            adaptive_width: true,
            history_capacity: 65_536,
            positions_per_key: 4,
            reject_threshold: 3,
            cooldown_steps: 8,
        }
    }
}

/// One ordered continuation for a shared target verification transaction, not branches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidates {
    pub tokens: Vec<u32>,
    pub matched_tokens: usize,
    /// Exclusive end of the matching source suffix in the current confirmed stream.
    /// The first draft token is the confirmed token at this absolute position.
    pub source_end: usize,
}

/// Own one proposer per independent confirmed token stream.
#[derive(Debug)]
pub struct LookupProposer {
    config: LookupConfig,
    history: VecDeque<u32>,
    /// Exact keys keep hash collisions from sharing the per-key eviction budget.
    index: HashMap<[u32; 3], VecDeque<usize>>,
    confirmed_len: usize,
    draft_limit: usize,
    rejected: usize,
    cooldown: usize,
}

impl Default for LookupProposer {
    fn default() -> Self {
        Self::new(LookupConfig::default()).expect("valid lookup defaults")
    }
}

impl LookupProposer {
    pub fn new(config: LookupConfig) -> Result<Self, &'static str> {
        if config.min_match < 3 || config.max_match < config.min_match {
            return Err("lookup requires 3 <= min_match <= max_match");
        }
        if config.history_capacity <= config.min_match {
            return Err("history_capacity must exceed min_match");
        }
        if config.max_drafts == 0 || config.positions_per_key == 0 || config.reject_threshold == 0 {
            return Err("max_drafts, positions_per_key, and reject_threshold must be nonzero");
        }
        Ok(Self {
            config,
            history: VecDeque::new(),
            index: HashMap::new(),
            confirmed_len: 0,
            draft_limit: config.max_drafts,
            rejected: 0,
            cooldown: 0,
        })
    }

    /// Synchronize a full, zero-based CONFIRMED history, excluding speculative tokens.
    /// Unchanged/extended retained prefixes keep their rejection policy. Large appends
    /// rebuild only the final window; small appends update the existing index incrementally.
    /// Rewinds or retained edits rebuild the bounded window and reset that policy.
    /// The caller's slice is borrowed only for this call; its discarded prefix is not kept.
    pub fn sync(&mut self, history: &[u32]) {
        let start = self.confirmed_len - self.history.len();
        if history.len() >= self.confirmed_len
            && self
                .history
                .iter()
                .copied()
                .eq(history[start..self.confirmed_len].iter().copied())
        {
            let appended = &history[self.confirmed_len..];
            if appended.len() < self.config.history_capacity {
                self.append_confirmed(appended);
                return;
            }
            self.history.clear();
            self.index.clear();
        } else {
            self.clear();
        }

        let start = history.len().saturating_sub(self.config.history_capacity);
        self.confirmed_len = start;
        self.append_confirmed(&history[start..]);
    }

    /// Incremental expected O(tokens.len()) path when the caller KNOWS the prefix is unchanged.
    /// Never pass unverified draft tokens. Use `sync` for arbitrary history changes.
    /// Panics if the session token count would overflow usize, before changing state.
    pub fn append_confirmed(&mut self, tokens: &[u32]) {
        self.confirmed_len
            .checked_add(tokens.len())
            .expect("lookup history length overflow");
        for &token in tokens {
            if self.history.len() == self.config.history_capacity {
                // Evict the only trigram that will cross the ring's new left boundary.
                let expired = self.confirmed_len - self.history.len() + 3;
                let key = [self.history[0], self.history[1], self.history[2]];
                let remove = if let Some(positions) = self.index.get_mut(&key) {
                    if positions.front() == Some(&expired) {
                        positions.pop_front();
                    }
                    positions.is_empty()
                } else {
                    false
                };
                if remove {
                    self.index.remove(&key);
                }
                self.history.pop_front();
            }
            self.history.push_back(token);
            self.confirmed_len += 1;
            let n = self.history.len();
            if n >= 3 {
                let key = [
                    self.history[n - 3],
                    self.history[n - 2],
                    self.history[n - 1],
                ];
                let positions = self.index.entry(key).or_default();
                if positions.len() == self.config.positions_per_key {
                    positions.pop_front();
                }
                positions.push_back(self.confirmed_len);
            }
        }
    }

    /// Forget this stream, including rejection/cooldown state. Allocations remain bounded
    /// by this instance's configuration and may be reused for another stream.
    pub fn clear(&mut self) {
        self.history.clear();
        self.index.clear();
        self.confirmed_len = 0;
        self.draft_limit = self.config.max_drafts;
        self.rejected = 0;
        self.cooldown = 0;
    }

    /// Query confirmed suffix + optional pending token. Does not commit either pending
    /// or drafts. The caller's budget and configured/adaptive limits all bound the output.
    /// A zero budget is a no-op; other calls consume one cooldown step when cooling down.
    pub fn propose(&mut self, pending: Option<u32>, max_drafts: usize) -> Option<Candidates> {
        let budget = max_drafts.min(self.config.max_drafts).min(self.draft_limit);
        if budget == 0 {
            return None;
        }
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return None;
        }
        let available = self
            .history
            .len()
            .saturating_add(usize::from(pending.is_some()));
        if available < self.config.min_match {
            return None;
        }
        let key = [
            self.query_back(pending, 2),
            self.query_back(pending, 1),
            self.query_back(pending, 0),
        ];
        let positions = self.index.get(&key)?;
        let start = self.confirmed_len - self.history.len();
        let mut best_end = 0;
        let mut best_len = 0;
        for &end in positions.iter().rev() {
            // This excludes the query itself and sources without a confirmed continuation,
            // including the newest source when querying with an unconfirmed pending token.
            if end >= self.confirmed_len {
                continue;
            }
            let limit = self.config.max_match.min(available).min(end - start);
            let mut matched = 0;
            while matched < limit
                && self.history[end - start - 1 - matched] == self.query_back(pending, matched)
            {
                matched += 1;
            }
            if matched > best_len {
                best_end = end;
                best_len = matched;
            }
        }
        if best_len < self.config.min_match {
            return None;
        }
        let count = budget.min(self.confirmed_len - best_end);
        let tokens = (0..count)
            .map(|i| self.history[best_end - start + i])
            .collect();
        Some(Candidates {
            tokens,
            matched_tokens: best_len,
            source_end: best_end,
        })
    }

    /// Report accepted draft-prefix tokens / produced draft tokens, BOTH excluding pending
    /// and any target bonus token. Zero produced is ignored. Panics if accepted > produced.
    /// With adaptive_width, rejection narrows the next probe to one token, full acceptance
    /// grows the limit by one, and partial acceptance narrows it. Repeated full rejections
    /// trigger cooldown in either mode; fixed width still obeys caller/configuration bounds.
    /// This feedback changes policy only: commit actual token IDs separately.
    pub fn record_result(&mut self, accepted: usize, produced: usize) {
        assert!(
            accepted <= produced,
            "accepted lookup drafts exceed produced drafts"
        );
        if produced == 0 {
            return;
        }
        if accepted == 0 {
            if self.config.adaptive_width {
                self.draft_limit = 1;
            }
            self.rejected = self.rejected.saturating_add(1);
            if self.rejected >= self.config.reject_threshold {
                self.cooldown = self.config.cooldown_steps;
                self.rejected = 0;
            }
        } else {
            self.rejected = 0;
            self.cooldown = 0;
            if self.config.adaptive_width {
                self.draft_limit = if accepted == produced {
                    self.draft_limit
                        .saturating_add(1)
                        .min(self.config.max_drafts)
                } else {
                    self.draft_limit.min(accepted)
                };
            }
        }
    }

    fn query_back(&self, pending: Option<u32>, offset: usize) -> u32 {
        match pending {
            Some(token) if offset == 0 => token,
            Some(_) => self.history[self.history.len() - offset],
            None => self.history[self.history.len() - 1 - offset],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasherDefault, Hasher};

    fn small_config() -> LookupConfig {
        LookupConfig {
            min_match: 3,
            max_match: 8,
            history_capacity: 64,
            ..LookupConfig::default()
        }
    }

    fn proposer(history: &[u32]) -> LookupProposer {
        let mut lookup = LookupProposer::new(small_config()).unwrap();
        lookup.sync(history);
        lookup
    }

    #[test]
    fn defaults_and_validation() {
        let config = LookupConfig::default();
        assert!(config.adaptive_width);
        assert_eq!(
            (config.min_match, config.max_match, config.max_drafts),
            (8, 64, 3)
        );
        for invalid in [
            LookupConfig {
                min_match: 2,
                ..config
            },
            LookupConfig {
                max_match: 7,
                ..config
            },
            LookupConfig {
                history_capacity: 8,
                ..config
            },
            LookupConfig {
                max_drafts: 0,
                ..config
            },
            LookupConfig {
                positions_per_key: 0,
                ..config
            },
            LookupConfig {
                reject_threshold: 0,
                ..config
            },
        ] {
            assert!(LookupProposer::new(invalid).is_err());
        }
    }

    #[test]
    fn duplicate_and_ambiguous_suffixes_choose_longest_then_newest() {
        let mut lookup = proposer(&[
            7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3,
        ]);
        let found = lookup.propose(None, 2).unwrap();
        assert_eq!(found.tokens, [100, 101]);
        assert_eq!((found.matched_tokens, found.source_end), (5, 5));

        let mut lookup = proposer(&[1, 2, 3, 10, 1, 2, 3, 20, 1, 2, 3]);
        let found = lookup.propose(None, 1).unwrap();
        assert_eq!((found.tokens, found.source_end), (vec![20], 7));
        assert_eq!(lookup.propose(None, 1).unwrap().source_end, 7);
    }

    #[test]
    fn overlap_is_allowed_without_extrapolating_past_confirmed_tokens() {
        let mut lookup = proposer(&[7; 20]);
        let found = lookup.propose(None, 3).unwrap();
        assert_eq!(found.tokens, [7]);
        assert_eq!((found.matched_tokens, found.source_end), (8, 19));
        let mut lookup = proposer(&[1, 2, 3, 1, 2, 3, 1, 2, 3]);
        let found = lookup.propose(None, 3).unwrap();
        assert_eq!(found.tokens, [1, 2, 3]);
        assert_eq!(found.matched_tokens, 6);
    }

    #[test]
    fn no_match_short_match_and_self_match_produce_nothing() {
        for history in [&[][..], &[1, 2][..], &[1, 2, 3][..], &[1, 2, 3, 4, 5][..]] {
            assert!(proposer(history).propose(None, 3).is_none());
        }
        let mut lookup = LookupProposer::default();
        lookup.sync(&[1, 2, 3, 9, 1, 2, 3]);
        assert!(lookup.propose(None, 3).is_none());
    }

    #[test]
    fn pending_is_queried_once_and_never_indexed() {
        let mut lookup = proposer(&[1, 2, 3, 4, 5, 9, 1, 2]);
        let before = (
            lookup.history.clone(),
            lookup.index.clone(),
            lookup.confirmed_len,
        );
        assert_eq!(lookup.propose(Some(3), 2).unwrap().tokens, [4, 5]);
        assert!(lookup.propose(None, 2).is_none());
        assert_eq!(lookup.propose(Some(3), 2).unwrap().tokens, [4, 5]);
        lookup.record_result(0, 2);
        assert_eq!(
            before,
            (
                lookup.history.clone(),
                lookup.index.clone(),
                lookup.confirmed_len
            )
        );
        lookup.append_confirmed(&[3]);
        assert_eq!(lookup.propose(None, 2).unwrap().tokens, [4]);
        assert!(proposer(&[1, 2]).propose(Some(3), 3).is_none());
        // A source ending at the confirmed frontier has no confirmed continuation.
        assert!(proposer(&[7, 7, 7]).propose(Some(7), 3).is_none());
    }

    #[test]
    fn only_accepted_tokens_are_committed_after_partial_verification() {
        let mut lookup = proposer(&[1, 2, 3, 4, 5, 9, 1, 2]);
        let found = lookup.propose(Some(3), 2).unwrap();
        assert_eq!(found.tokens, [4, 5]);
        lookup.record_result(1, found.tokens.len());
        lookup.append_confirmed(&[3, 4, 99]); // pending, accepted draft, target correction
        assert_eq!(
            lookup.history.iter().copied().collect::<Vec<_>>(),
            [1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 99]
        );
        assert_eq!(lookup.index[&[3, 4, 5]], VecDeque::from([5]));
    }

    #[test]
    fn draft_budget_and_configured_max_are_independent_of_match_length() {
        let mut lookup = LookupProposer::new(LookupConfig {
            max_drafts: 6,
            ..small_config()
        })
        .unwrap();
        lookup.sync(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3]);
        assert!(lookup.propose(None, 0).is_none());
        assert_eq!(lookup.propose(None, 1).unwrap().tokens, [4]);
        assert_eq!(
            lookup.propose(None, usize::MAX).unwrap().tokens,
            [4, 5, 6, 7, 8, 9]
        );
        let mut lookup = proposer(&[1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]);
        assert_eq!(lookup.propose(None, 100).unwrap().tokens, [4, 5, 6]);
    }

    #[test]
    fn match_bounds_and_full_u32_tokens() {
        let block: Vec<u32> = (100..180).collect();
        let mut history = block.clone();
        history.push(0);
        history.extend_from_slice(&block[..70]);
        let mut lookup = LookupProposer::default();
        lookup.sync(&history);
        let found = lookup.propose(None, 3).unwrap();
        assert_eq!(found.matched_tokens, 64);
        assert_eq!(found.tokens, [170, 171, 172]);
        let mut lookup = proposer(&[u32::MAX, 0, 0x10ffff, 0x80000000, 5, u32::MAX, 0, 0x10ffff]);
        assert_eq!(lookup.propose(None, 1).unwrap().tokens, [0x80000000]);
    }

    #[test]
    fn default_minimum_is_eight_exact_tokens() {
        let mut lookup = LookupProposer::default();
        lookup.sync(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(lookup.propose(None, 3).is_none());
        let found = lookup.propose(Some(8), 1).unwrap();
        assert_eq!((found.tokens, found.matched_tokens), (vec![9], 8));
    }

    #[test]
    fn sync_handles_same_length_edits_rewind_and_shorter_context() {
        let original = [1, 2, 3, 10, 11, 0, 1, 2, 3];
        let mut lookup = proposer(&original);
        assert_eq!(lookup.propose(None, 1).unwrap().tokens, [10]);
        let mut edited = original;
        edited[3] = 20;
        lookup.sync(&edited);
        assert_eq!(lookup.propose(None, 1).unwrap().tokens, [20]);
        edited[0] = 99; // same length and unchanged query suffix
        lookup.sync(&edited);
        assert!(lookup.propose(None, 3).is_none());
        lookup.sync(&original);
        lookup.sync(&original[..6]);
        assert_eq!(lookup.confirmed_len, 6);
        lookup.append_confirmed(&[1, 2, 3]);
        assert_eq!(lookup.propose(None, 1).unwrap().tokens, [10]);
        lookup.sync(&[]);
        assert!(lookup.history.is_empty() && lookup.index.is_empty());
    }

    #[test]
    fn ring_and_per_key_bounds_survive_many_appends_and_rebuilds() {
        let config = LookupConfig {
            history_capacity: 17,
            positions_per_key: 3,
            ..small_config()
        };
        let mut lookup = LookupProposer::new(config).unwrap();
        let mut history = Vec::new();
        for i in 0..500 {
            let token = if i < 250 { 7 } else { (i % 13) as u32 };
            history.push(token);
            lookup.append_confirmed(&[token]);
            let start = history.len().saturating_sub(config.history_capacity);
            assert_eq!(
                lookup.history.iter().copied().collect::<Vec<_>>(),
                history[start..]
            );
            assert!(lookup.index.len() <= config.history_capacity - 2);
            assert!(
                lookup.index.values().map(VecDeque::len).sum::<usize>()
                    <= config.history_capacity - 2
            );
            for (key, positions) in &lookup.index {
                assert!(!positions.is_empty() && positions.len() <= config.positions_per_key);
                for &end in positions {
                    assert!(end >= start + 3 && end <= history.len());
                    assert_eq!(&history[end - 3..end], key);
                }
            }
            let mut fresh = LookupProposer::new(config).unwrap();
            fresh.sync(&history);
            assert_eq!(lookup.index, fresh.index);
            for pending in [None, Some(7), Some(1)] {
                assert_eq!(lookup.propose(pending, 3), fresh.propose(pending, 3));
            }
            lookup.sync(&history); // idempotent; never duplicate positions
            assert_eq!(lookup.index, fresh.index);
        }
        lookup.sync(&history[..100]); // rewind well before the old retained window
        let mut fresh = LookupProposer::new(config).unwrap();
        fresh.sync(&history[..100]);
        assert_eq!(lookup.index, fresh.index);
        assert_eq!(lookup.propose(None, 3), fresh.propose(None, 3));
    }

    #[test]
    fn sync_extensions_preserve_feedback_and_ignore_discarded_prefix() {
        let config = LookupConfig {
            history_capacity: 16,
            ..small_config()
        };
        let mut lookup = LookupProposer::new(config).unwrap();
        let mut history = vec![1, 2, 3, 10, 11, 12, 0, 1, 2, 3];
        lookup.sync(&history);
        lookup.record_result(0, 3);
        history.push(10);
        lookup.sync(&history); // small append keeps rejection feedback
        assert_eq!((lookup.draft_limit, lookup.rejected), (1, 1));
        history.extend(100..140);
        lookup.sync(&history); // large append rebuilds only the retained window
        assert_eq!((lookup.draft_limit, lookup.rejected), (1, 1));
        assert_eq!(lookup.history.len(), config.history_capacity);
        let index = lookup.index.clone();
        history[0] = 99; // no retained token or position depends on this prefix
        lookup.sync(&history);
        assert_eq!(lookup.index, index);
        assert_eq!((lookup.draft_limit, lookup.rejected), (1, 1));
        let last = history.len() - 1;
        history[last] = 999;
        lookup.sync(&history);
        assert_eq!((lookup.draft_limit, lookup.rejected), (3, 0));
        let mut fresh = LookupProposer::new(config).unwrap();
        fresh.sync(&history);
        assert_eq!(lookup.index, fresh.index);
    }

    #[test]
    fn recent_position_cap_evicts_older_matches_deterministically() {
        let mut lookup = LookupProposer::new(LookupConfig {
            positions_per_key: 2,
            ..small_config()
        })
        .unwrap();
        lookup.sync(&[1, 2, 3, 10, 1, 2, 3, 20, 1, 2, 3]);
        assert_eq!(lookup.index[&[1, 2, 3]], VecDeque::from([7, 11]));
        assert_eq!(lookup.propose(None, 1).unwrap().tokens, [20]);
        lookup.append_confirmed(&[30, 1, 2, 3]);
        assert_eq!(lookup.index[&[1, 2, 3]], VecDeque::from([11, 15]));
        assert_eq!(lookup.propose(None, 1).unwrap().tokens, [30]);
    }

    #[test]
    fn rejection_cooldown_is_bounded_and_success_restores_width() {
        let config = LookupConfig {
            cooldown_steps: 2,
            ..small_config()
        };
        let mut lookup = LookupProposer::new(config).unwrap();
        let history = [1, 2, 3, 10, 11, 12, 0, 1, 2, 3];
        lookup.sync(&history);
        lookup.record_result(0, 0);
        assert_eq!(lookup.propose(None, 3).unwrap().tokens, [10, 11, 12]);
        for _ in 0..config.reject_threshold {
            let produced = lookup.propose(None, 3).unwrap().tokens.len();
            lookup.record_result(0, produced);
        }
        lookup.sync(&history); // unchanged history does not reset cooldown
        assert!(lookup.propose(None, 0).is_none());
        assert!(lookup.propose(None, 3).is_none());
        assert!(lookup.propose(None, 3).is_none());
        assert_eq!(lookup.propose(None, 3).unwrap().tokens, [10]);
        lookup.record_result(1, 1);
        assert_eq!(lookup.propose(None, 3).unwrap().tokens, [10, 11]);
        lookup.record_result(2, 2);
        assert_eq!(lookup.propose(None, 3).unwrap().tokens, [10, 11, 12]);
        lookup.record_result(1, 3);
        assert_eq!(lookup.propose(None, 3).unwrap().tokens, [10]);
        lookup.record_result(0, 1);
        let mut edited = history;
        edited[3] = 20;
        lookup.sync(&edited);
        assert_eq!(lookup.propose(None, 3).unwrap().tokens, [20, 11, 12]);
    }

    #[test]
    fn width_feedback_is_optional_and_never_commits_tokens() {
        let history = [1, 2, 3, 10, 11, 12, 0, 1, 2, 3];
        for adaptive_width in [false, true] {
            let mut lookup = LookupProposer::new(LookupConfig {
                adaptive_width,
                ..small_config()
            })
            .unwrap();
            lookup.sync(&history);
            let before = (
                lookup.history.clone(),
                lookup.index.clone(),
                lookup.confirmed_len,
            );
            for (accepted, expected_adaptive) in [(2, 2), (1, 1), (0, 1), (1, 2), (2, 3)] {
                let produced = lookup.propose(None, 100).unwrap().tokens.len();
                lookup.record_result(accepted, produced);
                let expected = if adaptive_width { expected_adaptive } else { 3 };
                assert_eq!(lookup.draft_limit, expected);
                assert_eq!(
                    lookup.propose(None, 100).unwrap().tokens,
                    [10, 11, 12][..expected]
                );
                assert_eq!(lookup.propose(None, 1).unwrap().tokens, [10]);
                assert_eq!(
                    (&lookup.history, &lookup.index, lookup.confirmed_len),
                    (&before.0, &before.1, before.2)
                );
            }
            lookup.sync(&[90, 91, 92, 93, 94, 95]);
            assert!(lookup.propose(None, 100).is_none());
        }
    }

    #[test]
    fn both_width_policies_keep_reject_cooldown_and_caller_configured_caps() {
        for adaptive_width in [false, true] {
            let config = LookupConfig {
                adaptive_width,
                max_drafts: 5,
                reject_threshold: 2,
                cooldown_steps: 2,
                ..small_config()
            };
            let mut lookup = LookupProposer::new(config).unwrap();
            lookup.sync(&[1, 2, 3, 10, 11, 12, 13, 14, 0, 1, 2, 3]);
            assert_eq!(lookup.propose(None, 100).unwrap().tokens.len(), 5);
            for _ in 0..config.reject_threshold {
                let produced = lookup.propose(None, 100).unwrap().tokens.len();
                lookup.record_result(0, produced);
            }
            assert_eq!(lookup.cooldown, 2);
            assert!(lookup.propose(None, 0).is_none());
            assert_eq!(lookup.cooldown, 2);
            assert!(lookup.propose(None, 100).is_none());
            assert!(lookup.propose(None, 100).is_none());
            let width = if adaptive_width { 1 } else { 5 };
            assert_eq!(lookup.propose(None, 100).unwrap().tokens.len(), width);
            assert_eq!(lookup.propose(None, 2).unwrap().tokens.len(), width.min(2));
            lookup.record_result(0, 0);
            assert_eq!(lookup.propose(None, 100).unwrap().tokens.len(), width);
            // Edits reset feedback; the configured maximum remains unchanged for either policy.
            lookup.sync(&[1, 2, 3, 20, 21, 22, 23, 24, 0, 1, 2, 3]);
            assert_eq!(lookup.propose(None, 100).unwrap().tokens, [20, 21, 22, 23, 24]);
        }
    }

    #[test]
    #[should_panic(expected = "accepted lookup drafts exceed produced drafts")]
    fn invalid_feedback_is_rejected() {
        LookupProposer::default().record_result(2, 1);
    }

    #[derive(Default)]
    struct ConstantHasher;

    impl Hasher for ConstantHasher {
        fn finish(&self) -> u64 {
            0
        }
        fn write(&mut self, _: &[u8]) {}
    }

    #[test]
    fn hash_collisions_preserve_exact_keys_and_continuations() {
        // Deliberately collide every key using the same key/value representation.
        let mut collided =
            HashMap::<[u32; 3], VecDeque<usize>, BuildHasherDefault<ConstantHasher>>::default();
        let history = [8, 2, 3, 90, 1, 2, 3, 40, 41, 0, 1, 2, 3];
        let mut lookup = proposer(&history);
        for (key, positions) in &lookup.index {
            collided.insert(*key, positions.clone());
        }
        assert_eq!(collided.len(), lookup.index.len());
        assert_eq!(collided.get(&[1, 2, 3]), lookup.index.get(&[1, 2, 3]));
        assert_ne!(collided.get(&[8, 2, 3]), collided.get(&[1, 2, 3]));
        assert_eq!(lookup.propose(None, 2).unwrap().tokens, [40, 41]);
    }

    #[test]
    fn suffix_verification_rejects_false_index_hits() {
        let mut lookup = proposer(&[8, 2, 3, 90, 1, 2, 3]);
        // Even an injected false hit sharing only the final two tokens cannot draft.
        lookup.index.insert([1, 2, 3], VecDeque::from([3]));
        assert!(lookup.propose(None, 3).is_none());
    }
}
