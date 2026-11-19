//! Greedy accepted-prefix counting shared by token-only proposers.
//!
//! Inputs are `[pending, candidates...]`; target row `j` selects the token AFTER input `j`.
//! Callers repair target ids before using this module and perform all delivery/backend work.
//! The frontier/identity model below is a test-only reference, not a runtime transaction API.

use anyhow::{ensure, Result};

pub(crate) const MAX_VERIFY_ROWS: usize = 4;

#[derive(Default)]
pub(crate) struct MtpCostPolicy {
    samples: [u8; MAX_VERIFY_ROWS],
    seconds: [f64; MAX_VERIFY_ROWS],
    committed: [f64; MAX_VERIFY_ROWS],
    choose_tick: u8,
    next_probe: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct MtpChoice {
    pub(crate) rows: usize,
    pub(crate) probe: bool,
}

impl MtpCostPolicy {
    pub(crate) fn choose(&mut self, max_rows: usize, allow_plain: bool) -> MtpChoice {
        let max_rows = max_rows.clamp(2, MAX_VERIFY_ROWS);
        self.choose_tick = (self.choose_tick + 1) % 16;

        for rows in (2..=max_rows).chain(allow_plain.then_some(1)) {
            if self.samples[rows - 1] < 2 {
                return MtpChoice { rows, probe: true };
            }
        }

        if self.choose_tick == 0 {
            let arm_count = max_rows - 1 + usize::from(allow_plain);
            let index = self.next_probe % arm_count;
            self.next_probe = (index + 1) % arm_count;
            let rows = if index < max_rows - 1 { index + 2 } else { 1 };
            return MtpChoice { rows, probe: true };
        }

        let mut rows = 2;
        for candidate in 3..=max_rows {
            if self.cost(candidate) < self.cost(rows) {
                rows = candidate;
            }
        }
        if allow_plain && self.cost(rows) > self.cost(1) * 0.97 {
            rows = 1;
        }
        MtpChoice { rows, probe: false }
    }

    pub(crate) fn record(&mut self, rows: usize, seconds: f64, committed: usize) {
        if !(1..=MAX_VERIFY_ROWS).contains(&rows)
            || !seconds.is_finite()
            || seconds <= 0.0
            || committed == 0
        {
            return;
        }
        let index = rows - 1;
        if self.samples[index] == 0 {
            self.seconds[index] = seconds;
            self.committed[index] = committed as f64;
        } else {
            self.seconds[index] = self.seconds[index] * 0.875 + seconds * 0.125;
            self.committed[index] = self.committed[index] * 0.875 + committed as f64 * 0.125;
        }
        self.samples[index] = (self.samples[index] + 1).min(2);
    }

    fn cost(&self, rows: usize) -> f64 {
        self.seconds[rows - 1] / self.committed[rows - 1]
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VerifyIdentity {
    pub request_id: u64,
    pub slot: usize,
    pub slot_generation: u64,
    pub step_id: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VerifyFrontier {
    pub identity: VerifyIdentity,
    pub materialized_len: usize,
    pub pending_id: u32,
    pub pending_emitted: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VerifyDelivery {
    /// Newly retained/counted input tokens, including EOS even without a text event.
    /// Excludes a pending token already emitted before this window.
    pub new_tokens: usize,
    /// Output budget at the start of this window, excluding already-emitted tokens.
    pub remaining: usize,
    /// Delivery stopped after this prefix (including an ACK returning false).
    pub cancelled: bool,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VerifyRestore {
    Base,
    /// Intermediate trace immediately after this zero-based input row.
    Row(usize),
    Live,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VerifyStop {
    Continue,
    OutputCap,
    Eos,
    Cancelled,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VerifyCommit {
    pub frontier: VerifyFrontier,
    pub committed_rows: usize,
    pub new_tokens: usize,
    /// `None` preserves the predecessor hidden from the initial frontier.
    pub hidden_row: Option<usize>,
    pub restore: VerifyRestore,
    pub stop: VerifyStop,
}

/// Includes the known pending input; there is no additional materialized bonus row.
pub(crate) fn accepted_rows(candidates: &[u32], verified: &[u32]) -> Result<usize> {
    ensure!(
        (2..=MAX_VERIFY_ROWS).contains(&verified.len()),
        "Verify width must be 2..={MAX_VERIFY_ROWS}, got {}",
        verified.len()
    );
    ensure!(
        candidates.len() == verified.len() - 1,
        "Verify needs one more target row than candidates"
    );
    Ok(1 + candidates
        .iter()
        .zip(verified)
        .take_while(|(draft, target)| draft == target)
        .count())
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct GreedyVerify<'a> {
    frontier: VerifyFrontier,
    candidates: &'a [u32],
    verified: &'a [u32],
    accepted: usize,
}

#[cfg(test)]
impl<'a> GreedyVerify<'a> {
    pub(crate) fn new(
        frontier: VerifyFrontier,
        result_identity: VerifyIdentity,
        candidates: &'a [u32],
        verified: &'a [u32],
    ) -> Result<Self> {
        ensure!(
            frontier.identity == result_identity,
            "stale Verify result identity"
        );
        let accepted = accepted_rows(candidates, verified)?;
        ensure!(
            frontier
                .materialized_len
                .checked_add(verified.len())
                .is_some(),
            "Verify materialized length overflows"
        );
        Ok(Self {
            frontier,
            candidates,
            verified,
            accepted,
        })
    }

    pub(crate) fn accepted_rows(&self) -> usize {
        self.accepted
    }

    pub(crate) fn width(&self) -> usize {
        self.verified.len()
    }

    fn input(&self, row: usize) -> u32 {
        if row == 0 {
            self.frontier.pending_id
        } else {
            self.candidates[row - 1]
        }
    }

    /// Empty `eos_ids` implements ignore-EOS. Only input EOS stops this cycle;
    /// an EOS selected as the next pending is deferred to the next window.
    ///
    /// A published pending is materialized even with zero new delivery. Use `abort`
    /// instead to preserve that unmaterialized frontier debt unchanged.
    pub(crate) fn commit(
        &self,
        current_identity: VerifyIdentity,
        delivery: VerifyDelivery,
        eos_ids: &[u32],
    ) -> Result<VerifyCommit> {
        ensure!(
            self.frontier.identity == current_identity,
            "stale Verify commit identity"
        );
        let already = usize::from(self.frontier.pending_emitted);
        ensure!(
            delivery.new_tokens <= delivery.remaining,
            "Verify delivery exceeds output cap"
        );
        ensure!(
            delivery.new_tokens <= self.accepted - already,
            "Verify delivery exceeds accepted prefix"
        );
        let capacity = already + (self.accepted - already).min(delivery.remaining);
        let eos_row = (0..capacity).find(|&row| eos_ids.contains(&self.input(row)));
        let limit = eos_row.map_or(capacity, |row| row + 1);
        let kept = already + delivery.new_tokens;
        ensure!(kept <= limit, "Verify delivery continues past EOS");
        ensure!(
            delivery.cancelled || kept == limit,
            "short Verify delivery needs cancellation"
        );
        let stop = if eos_row.is_some_and(|row| kept == row + 1) {
            VerifyStop::Eos
        } else if delivery.new_tokens == delivery.remaining {
            VerifyStop::OutputCap
        } else if delivery.cancelled {
            VerifyStop::Cancelled
        } else {
            VerifyStop::Continue
        };
        Ok(self.retained(kept, delivery.new_tokens, stop))
    }

    pub(crate) fn abort(&self, current_identity: VerifyIdentity) -> Result<VerifyCommit> {
        ensure!(
            self.frontier.identity == current_identity,
            "stale Verify abort identity"
        );
        Ok(self.retained(0, 0, VerifyStop::Cancelled))
    }

    fn retained(&self, kept: usize, new_tokens: usize, stop: VerifyStop) -> VerifyCommit {
        let mut frontier = self.frontier;
        let hidden_row = kept.checked_sub(1);
        if let Some(row) = hidden_row {
            frontier.materialized_len += kept; // Constructor checked the entire forward width.
            frontier.pending_id = self.verified[row];
            frontier.pending_emitted = false;
        }
        VerifyCommit {
            frontier,
            committed_rows: kept,
            new_tokens,
            hidden_row,
            restore: if kept == 0 {
                VerifyRestore::Base
            } else if kept == self.width() {
                VerifyRestore::Live
            } else {
                VerifyRestore::Row(kept - 1)
            },
            stop,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_choice_respects_the_clamped_width_cap() {
        for requested in [0, 1, 2, 3, 4, 5, usize::MAX] {
            let max_rows = requested.clamp(2, MAX_VERIFY_ROWS);
            let mut policy = MtpCostPolicy::default();
            for rows in 1..=MAX_VERIFY_ROWS {
                for _ in 0..2 {
                    policy.record(rows, (MAX_VERIFY_ROWS + 1 - rows) as f64, rows);
                }
            }
            for _ in 0..128 {
                let choice = policy.choose(requested, true);
                assert!((1..=max_rows).contains(&choice.rows));
                if !choice.probe {
                    assert_eq!(choice.rows, max_rows);
                }
            }
        }
    }

    #[test]
    fn mtp_never_selects_plain_when_disabled_even_if_it_is_fastest() {
        let mut policy = MtpCostPolicy::default();
        for _ in 0..2 {
            policy.record(1, 0.001, 1);
            for rows in 2..=MAX_VERIFY_ROWS {
                policy.record(rows, rows as f64, 1);
            }
        }
        for _ in 0..128 {
            let choice = policy.choose(MAX_VERIFY_ROWS, false);
            assert!((2..=MAX_VERIFY_ROWS).contains(&choice.rows));
            if !choice.probe {
                assert_eq!(choice.rows, 2);
            }
        }
    }

    #[test]
    fn mtp_cold_start_collects_two_valid_samples_in_arm_order() {
        for max_rows in 2..=MAX_VERIFY_ROWS {
            for allow_plain in [false, true] {
                let mut policy = MtpCostPolicy::default();
                let unrecorded = policy.choose(max_rows, allow_plain);
                assert_eq!(unrecorded.rows, 2);
                assert!(unrecorded.probe);
                for rows in (2..=max_rows).chain(allow_plain.then_some(1)) {
                    for _ in 0..2 {
                        let choice = policy.choose(max_rows, allow_plain);
                        assert_eq!(choice.rows, rows);
                        assert!(choice.probe);
                        policy.record(choice.rows, 1.0, choice.rows);
                    }
                }
                let choice = policy.choose(max_rows, allow_plain);
                assert_eq!(choice.rows, max_rows);
                assert!(!choice.probe);
            }
        }
    }

    #[test]
    fn mtp_slow_verification_falls_back_to_plain() {
        let mut policy = MtpCostPolicy::default();
        for _ in 0..MAX_VERIFY_ROWS * 2 {
            let choice = policy.choose(MAX_VERIFY_ROWS, true);
            assert!(choice.probe);
            let seconds = if choice.rows == 1 {
                1.0
            } else {
                choice.rows as f64 * 2.0
            };
            policy.record(choice.rows, seconds, choice.rows);
        }
        let choice = policy.choose(MAX_VERIFY_ROWS, true);
        assert_eq!(choice.rows, 1);
        assert!(!choice.probe);
    }

    #[test]
    fn mtp_requires_at_least_three_percent_gain_over_plain() {
        for (seconds, expected) in [(1.0, 1), (0.98, 1), (0.971, 1), (0.97, 2), (0.96, 2)] {
            let mut policy = MtpCostPolicy::default();
            for _ in 0..2 {
                policy.record(1, 1.0, 1);
                policy.record(2, seconds, 1);
            }
            let choice = policy.choose(2, true);
            assert_eq!(choice.rows, expected, "seconds={seconds}");
            assert!(!choice.probe);
        }
    }

    #[test]
    fn mtp_cost_uses_separate_emas_not_the_ema_of_per_round_ratios() {
        let mut policy = MtpCostPolicy::default();
        for _ in 0..2 {
            policy.record(1, 1.5, 1);
            policy.record(2, 3.0, 1);
            policy.record(3, 3.0, 1);
        }
        policy.record(4, 2.0, 1);
        assert_eq!(policy.seconds[3], 2.0);
        assert_eq!(policy.committed[3], 1.0);
        policy.record(4, 1.0, 4);
        assert_eq!(policy.seconds[3], 1.875);
        assert_eq!(policy.committed[3], 1.375);
        let choice = policy.choose(MAX_VERIFY_ROWS, true);
        assert_eq!(choice.rows, 4);
        assert!(!choice.probe);
    }

    #[test]
    fn mtp_periodic_probes_rotate_through_every_allowed_arm() {
        for max_rows in 2..=MAX_VERIFY_ROWS {
            for allow_plain in [false, true] {
                let mut policy = MtpCostPolicy::default();
                for _ in 0..2 {
                    policy.record(1, 1.0, 1);
                    for rows in 2..=MAX_VERIFY_ROWS {
                        policy.record(rows, 10.0, 1);
                    }
                }
                let mut expected = (2..=max_rows).chain(allow_plain.then_some(1)).cycle();
                for call in 1..=16 * MAX_VERIFY_ROWS * 2 {
                    let choice = policy.choose(max_rows, allow_plain);
                    assert_eq!(choice.probe, call % 16 == 0);
                    if choice.probe {
                        assert_eq!(choice.rows, expected.next().unwrap());
                    } else {
                        assert_eq!(choice.rows, if allow_plain { 1 } else { 2 });
                    }
                }
            }
        }
    }

    #[test]
    fn mtp_reprobe_restores_a_previously_slow_wide_arm() {
        for allow_plain in [false, true] {
            let mut policy = MtpCostPolicy::default();
            for _ in 0..2 {
                policy.record(1, 1.0, 1);
                policy.record(2, 2.0, 2);
                policy.record(3, 6.0, 3);
                policy.record(4, 8.0, 4);
            }
            let baseline = if allow_plain { 1 } else { 2 };
            let initial = policy.choose(MAX_VERIFY_ROWS, allow_plain);
            assert_eq!(initial.rows, baseline);
            assert!(!initial.probe);
            let mut wide_probes = 0;
            let mut recovered = false;
            for _ in 0..16 * MAX_VERIFY_ROWS * 16 {
                let choice = policy.choose(MAX_VERIFY_ROWS, allow_plain);
                if choice.rows == 4 {
                    if !choice.probe {
                        recovered = true;
                        break;
                    }
                    wide_probes += 1;
                    policy.record(4, 0.4, 4);
                } else {
                    if !choice.probe {
                        assert_eq!(choice.rows, baseline);
                    }
                    let seconds = match choice.rows {
                        1 => 1.0,
                        2 => 2.0,
                        3 => 6.0,
                        _ => unreachable!(),
                    };
                    policy.record(choice.rows, seconds, choice.rows);
                }
            }
            assert!(wide_probes > 0);
            assert!(recovered, "allow_plain={allow_plain}");
        }
    }

    #[test]
    fn mtp_invalid_samples_do_not_change_observations() {
        for trained in [false, true] {
            let mut policy = MtpCostPolicy::default();
            if trained {
                for rows in 1..=MAX_VERIFY_ROWS {
                    for _ in 0..2 {
                        policy.record(rows, 1.0, rows);
                    }
                }
            }
            let samples = policy.samples;
            let seconds = policy.seconds;
            let committed = policy.committed;
            for (rows, elapsed, tokens) in [
                (0, 1.0, 1),
                (MAX_VERIFY_ROWS + 1, 1.0, 1),
                (usize::MAX, 1.0, 1),
                (1, f64::NAN, 1),
                (2, f64::INFINITY, 1),
                (3, f64::NEG_INFINITY, 1),
                (4, 0.0, 1),
                (1, -0.0, 1),
                (2, -1.0, 1),
                (4, 1.0, 0),
            ] {
                policy.record(rows, elapsed, tokens);
                assert_eq!(policy.samples, samples);
                assert_eq!(policy.seconds, seconds);
                assert_eq!(policy.committed, committed);
            }
            let choice = policy.choose(MAX_VERIFY_ROWS, true);
            assert_eq!(choice.rows, if trained { 4 } else { 2 });
            assert_eq!(choice.probe, !trained);
        }
    }

    const EOS: u32 = 1_000;
    const ID: VerifyIdentity = VerifyIdentity {
        request_id: 7,
        slot: 2,
        slot_generation: 9,
        step_id: 11,
    };

    fn frontier(emitted: bool) -> VerifyFrontier {
        VerifyFrontier {
            identity: ID,
            materialized_len: 100,
            pending_id: 10,
            pending_emitted: emitted,
        }
    }

    fn target(candidates: &[u32], accepted: usize) -> Vec<u32> {
        let mut ids = vec![999; candidates.len() + 1];
        ids[..accepted - 1].copy_from_slice(&candidates[..accepted - 1]);
        ids
    }

    #[test]
    fn every_match_pattern_uses_only_the_contiguous_prefix() {
        for width in 2..=MAX_VERIFY_ROWS {
            let candidates = [11, 12, 13];
            let candidates = &candidates[..width - 1];
            for mask in 0usize..(1 << candidates.len()) {
                let mut ids = vec![99; width];
                for row in 0..candidates.len() {
                    if mask & (1 << row) != 0 {
                        ids[row] = candidates[row];
                    }
                }
                let expected = 1
                    + (0..candidates.len())
                        .take_while(|&row| mask & (1 << row) != 0)
                        .count();
                assert_eq!(accepted_rows(candidates, &ids).unwrap(), expected);
                for last in [0, EOS, u32::MAX] {
                    ids[width - 1] = last;
                    assert_eq!(accepted_rows(candidates, &ids).unwrap(), expected);
                }
            }
        }
    }

    #[test]
    fn exhaustive_accepted_delivered_caps_eos_and_cancellation() {
        for width in 2..=MAX_VERIFY_ROWS {
            for accepted in 1..=width {
                for emitted in [false, true] {
                    // Include EOS in rejected inputs, which must never stop the retained prefix.
                    for eos_at in 0..=width {
                        let mut base = frontier(emitted);
                        let mut inputs = vec![10, 11, 12, 13][..width].to_vec();
                        if eos_at < width {
                            inputs[eos_at] = EOS;
                        }
                        base.pending_id = inputs[0];
                        let ids = target(&inputs[1..], accepted);
                        let window = GreedyVerify::new(base, ID, &inputs[1..], &ids).unwrap();
                        assert_eq!(window.accepted_rows(), accepted);
                        for remaining in 0..=width {
                            // Sequential oracle: already emitted pending costs no output budget.
                            let mut allowed = 0;
                            let mut budget = remaining;
                            for (row, &token) in inputs[..accepted].iter().enumerate() {
                                if !(emitted && row == 0) {
                                    if budget == 0 {
                                        break;
                                    }
                                    budget -= 1;
                                }
                                allowed += 1;
                                if token == EOS {
                                    break;
                                }
                            }
                            for delivered in 0..=width {
                                for cancelled in [false, true] {
                                    let kept = delivered + usize::from(emitted);
                                    let valid = kept <= allowed && (cancelled || kept == allowed);
                                    let result = window.commit(
                                        ID,
                                        VerifyDelivery {
                                            new_tokens: delivered,
                                            remaining,
                                            cancelled,
                                        },
                                        &[EOS],
                                    );
                                    assert_eq!(result.is_ok(), valid,
                                        "width={width} accepted={accepted} emitted={emitted} eos={eos_at} cap={remaining} delivered={delivered} cancelled={cancelled}");
                                    let Ok(commit) = result else {
                                        continue;
                                    };
                                    assert_eq!(commit.committed_rows, kept);
                                    assert_eq!(commit.new_tokens, delivered);
                                    assert_eq!(commit.frontier.identity, ID);
                                    assert_eq!(commit.frontier.materialized_len, 100 + kept);
                                    assert_eq!(commit.hidden_row, kept.checked_sub(1));
                                    if kept == 0 {
                                        assert_eq!(commit.frontier, base);
                                        assert_eq!(commit.restore, VerifyRestore::Base);
                                    } else {
                                        assert_eq!(commit.frontier.pending_id, ids[kept - 1]);
                                        assert!(!commit.frontier.pending_emitted);
                                        assert_eq!(
                                            commit.restore,
                                            if kept == width {
                                                VerifyRestore::Live
                                            } else {
                                                VerifyRestore::Row(kept - 1)
                                            }
                                        );
                                    }
                                    let expected_stop = if kept > 0 && inputs[kept - 1] == EOS {
                                        VerifyStop::Eos
                                    } else if delivered == remaining {
                                        VerifyStop::OutputCap
                                    } else if cancelled {
                                        VerifyStop::Cancelled
                                    } else {
                                        VerifyStop::Continue
                                    };
                                    assert_eq!(commit.stop, expected_stop);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn abort_preserves_every_frontier_and_previously_emitted_debt() {
        for width in 2..=MAX_VERIFY_ROWS {
            let candidates = [11, 12, 13];
            for accepted in 1..=width {
                let ids = target(&candidates[..width - 1], accepted);
                for emitted in [false, true] {
                    let base = frontier(emitted);
                    let window =
                        GreedyVerify::new(base, ID, &candidates[..width - 1], &ids).unwrap();
                    let result = window.abort(ID).unwrap();
                    assert_eq!(result.frontier, base);
                    assert_eq!(result.committed_rows, 0);
                    assert_eq!(result.new_tokens, 0);
                    assert_eq!(result.hidden_row, None);
                    assert_eq!(result.restore, VerifyRestore::Base);
                    assert_eq!(result.stop, VerifyStop::Cancelled);
                }
            }
        }
    }

    #[test]
    fn next_pending_eos_is_deferred_and_empty_eos_set_ignores_eos() {
        for width in 2..=MAX_VERIFY_ROWS {
            let candidates = vec![EOS; width - 1];
            let ids = vec![EOS; width];
            let window = GreedyVerify::new(frontier(false), ID, &candidates, &ids).unwrap();
            let all = window
                .commit(
                    ID,
                    VerifyDelivery {
                        new_tokens: width,
                        remaining: width + 1,
                        cancelled: false,
                    },
                    &[],
                )
                .unwrap();
            assert_eq!(all.stop, VerifyStop::Continue);
            assert_eq!(all.frontier.pending_id, EOS);
            let first = window
                .commit(
                    ID,
                    VerifyDelivery {
                        new_tokens: 1,
                        remaining: width + 1,
                        cancelled: true,
                    },
                    &[EOS],
                )
                .unwrap();
            assert_eq!(first.stop, VerifyStop::Cancelled);
            assert_eq!(first.frontier.pending_id, EOS);
        }
    }

    #[test]
    fn identity_is_checked_at_result_commit_and_abort_for_every_component() {
        let candidates = [11];
        let ids = [11, 12];
        let window = GreedyVerify::new(frontier(false), ID, &candidates, &ids).unwrap();
        for stale in [
            VerifyIdentity {
                request_id: 8,
                ..ID
            },
            VerifyIdentity { slot: 3, ..ID },
            VerifyIdentity {
                slot_generation: 10,
                ..ID
            },
            VerifyIdentity { step_id: 12, ..ID },
        ] {
            assert!(GreedyVerify::new(frontier(false), stale, &candidates, &ids).is_err());
            assert!(window
                .commit(
                    stale,
                    VerifyDelivery {
                        new_tokens: 2,
                        remaining: 2,
                        cancelled: false,
                    },
                    &[]
                )
                .is_err());
            assert!(window.abort(stale).is_err());
        }
    }

    #[test]
    fn rejects_invalid_width_shape_overflow_and_delivery_without_panics() {
        for width in [0, 1, 5, 6] {
            assert!(accepted_rows(&[], &vec![0; width]).is_err());
        }
        for width in 2..=MAX_VERIFY_ROWS {
            for drafts in 0..=5 {
                assert_eq!(
                    accepted_rows(&vec![1; drafts], &vec![1; width]).is_ok(),
                    drafts == width - 1
                );
            }
        }
        let candidates = [11];
        let ids = [11, 12];
        let overflow = VerifyFrontier {
            materialized_len: usize::MAX - 1,
            ..frontier(false)
        };
        assert!(GreedyVerify::new(overflow, ID, &candidates, &ids).is_err());
        let window = GreedyVerify::new(frontier(true), ID, &candidates, &ids).unwrap();
        for delivery in [
            VerifyDelivery {
                new_tokens: usize::MAX,
                remaining: usize::MAX,
                cancelled: true,
            },
            VerifyDelivery {
                new_tokens: 1,
                remaining: 0,
                cancelled: true,
            },
            VerifyDelivery {
                new_tokens: 0,
                remaining: 1,
                cancelled: false,
            },
        ] {
            assert!(window.commit(ID, delivery, &[]).is_err());
        }
        let base = VerifyFrontier {
            materialized_len: usize::MAX - 2,
            ..frontier(false)
        };
        let boundary = GreedyVerify::new(base, ID, &candidates, &ids).unwrap();
        assert_eq!(
            boundary
                .commit(
                    ID,
                    VerifyDelivery {
                        new_tokens: 2,
                        remaining: 2,
                        cancelled: false,
                    },
                    &[]
                )
                .unwrap()
                .frontier
                .materialized_len,
            usize::MAX
        );
    }
}
