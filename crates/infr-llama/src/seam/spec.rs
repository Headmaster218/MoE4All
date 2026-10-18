//! SPECULATIVE (multi-row) decode for the seam session paths — the primitive the layer-split
//! (multi-GPU) backend never had.
//!
//! WHY THIS EXISTS: a single-row decode step costs a flat ~26 ms per forward on this engine (253
//! dispatches, each serialized behind the previous one's dependency: ~103 µs/dispatch with one row
//! in flight, vs ~30 µs when many rows are in flight — see the campaign doc's measured forward-cost
//! curve). Each ADDITIONAL row costs only ~0.76 ms. So one forward over `k` rows costs nearly what
//! one row costs, and the entire multi-GPU decode gap is "we never run more than one row per
//! forward".
//!
//! STAGE B (this file): the drafter is a PLACEHOLDER — a token-history n-gram, no model, no device
//! work (see [`draft_lookup`]). It exists to prove the plumbing and to measure rows-per-forward on
//! the real pipeline. Stage A swaps the drafter for the MTP head (`crate::mtp::qwen4`), whose ~70%
//! acceptance is what turns `k` rows into nearly `k` tokens.
//!
//! The loop mirrors `mtp::qwen4`'s verified protocol exactly, including the rollback:
//!   prime (one ordinary row) → snapshot the recurrent state → { arm the per-row trace, one `k`-row
//!   VERIFY forward, accept the matching prefix, restore the state captured after `accepted` rows,
//!   re-snapshot }.
//! Rejected K/V rows need no explicit rewind: the next cycle writes over them and attention never
//! reads past the live position. Only the RECURRENT state (Qwen3.8's hyper-connection/DeltaNet
//! mixers) must be restored, which is what the trace is for.
//!
//! Enable: `INFR_SPEC_LOOKUP=1` (width = `spec.k`, clamped to 2..=`SPEC_MAX_K`).

use anyhow::{anyhow, Result};
use infr_core::backend::Backend;
use infr_gguf::Gguf;

use super::{
    generate_dense_backend, BindWeight, MropePlan, PerLayerEmbd, SeamKv, TokenEmbd, TurnCheckpoint,
};
use crate::{Config, EngineConfig, GenStats};

/// The widest VERIFY batch this loop can run. The recurrent trace is sized separately from this
/// ceiling (see [`recurrent_trace_rows`]), because allocating the maximum trace for a smaller
/// configured `spec.k` can spill the KV cache even though those rows are never used.
pub(crate) const SPEC_MAX_K: usize = crate::mtp::DRAFT_TOKENS;

/// The VERIFY width this engine instance actually runs. `spec.k` is an upper bound; the loops
/// clamp it to the architecture's supported `[2, SPEC_MAX_K]` range.
pub(crate) fn verify_width(ec: &EngineConfig) -> usize {
    ec.spec.k.clamp(2, SPEC_MAX_K)
}

/// Number of intermediate recurrent snapshots needed for a partial-accept rollback.
///
/// A VERIFY of `k` rows keeps row zero in the clean checkpoint, so it needs exactly `k - 1`
/// trace rows. The concurrent MTP path is a special case: it runs the fixed maximum-width cohort
/// independent of `spec.k`, so it must retain the maximum trace capacity.
pub(crate) fn recurrent_trace_rows(ec: &EngineConfig) -> usize {
    if window_pipe_enabled() {
        // The native window pipeline splits a VERIFY into a front span and a back span and runs
        // them one window apart, so the guessed window's front half has already advanced the live
        // recurrent state past the batch being verified. A fully accepted batch therefore has to
        // restore from a snapshot for every row, including the last one — one more row than the
        // serial path, which leaves the accepted batch's state live.
        verify_width(ec).max(1)
    } else if ec.spec.mtp_concurrent {
        SPEC_MAX_K.saturating_sub(1)
    } else {
        verify_width(ec).saturating_sub(1)
    }
}

/// The native window pipeline's opt-in (`INFR_MTP_WINDOW_PIPE`). Splits each Qwen3.8 MTP VERIFY by
/// LAYER into a front span (device 0) and a back span (device 1) so the back half of the window
/// being verified runs on one card while the front half of the next window fills the other — see
/// `WindowPipeRequest` in `seam::runner`. Read here as well as in the runner because it decides the
/// slot's recurrent-trace capacity at slot-BUILD time, long before a verify is requested.
pub(crate) fn window_pipe_enabled() -> bool {
    std::env::var_os("INFR_MTP_WINDOW_PIPE").is_some()
}

/// The B-stage placeholder drafter's flag. Deleted in stage A (MTP head).
fn lookup_enabled() -> bool {
    std::env::var_os("INFR_SPEC_LOOKUP").is_some()
}

/// Whether the session must reserve the per-row recurrent trace (the `MtpDeltaCkpt` of
/// `SeamKv`): MTP needs it, and so does this loop's rollback. Read by the runner's slot build —
/// WITHOUT it `mtp_arm_delta_trace` has no storage and a rejected draft is unrecoverable.
pub(crate) fn needs_recurrent_trace(ec: &EngineConfig, cfg: &Config) -> bool {
    cfg.qwen4exp && (ec.spec.mtp || lookup_enabled())
}

/// The VERIFY width for this request, or 0 to keep the ordinary per-token path.
///
/// Greedy only, and never grammar-constrained: acceptance compares a draft against the model's own
/// argmax, so a stochastic/penalized/constrained request must not take this path. Also requires a
/// recurrent trace to exist (`qwen4exp` + recurrent layers) — without one the loop could not roll
/// a rejected draft back out of the slot's state, and a silently-poisoned slot is worse than a
/// slow one.
pub(crate) fn spec_width(
    ec: &EngineConfig,
    cfg: &Config,
    req: Option<&crate::sampling::RequestCtx>,
    constrained: bool,
) -> usize {
    if constrained || !lookup_enabled() {
        return 0;
    }
    if !(cfg.qwen4exp && cfg.recurrent_state_elems() > 0) {
        return 0;
    }
    let base = crate::sampling::Sampler::from_cfg(&ec.sampling);
    let effective = crate::sampling::Sampler::resolve(req, &ec.sampling);
    let greedy = |s: crate::sampling::Sampler| s.temp <= 0.0 || s.top_k == 1;
    if !greedy(base) || !greedy(effective) {
        return 0;
    }
    if req.is_some_and(|ctx| ctx.sampling().penalties_active()) {
        return 0;
    }
    verify_width(ec)
}

/// Placeholder drafter (stage B): the classic token-history n-gram — copy what followed the most
/// recent earlier occurrence of the last `CTX` tokens. Deterministic, zero weights, zero device
/// work. A wrong draft costs one rolled-back row; an EMPTY slot would cost the whole cycle's row,
/// so the tail is padded by repeating the last token.
pub(crate) fn draft_lookup_with_match(history: &[u32], n: usize) -> (Vec<u32>, usize) {
    const CTX: usize = 3;
    let mut out = Vec::with_capacity(n);
    let mut matched = 0usize;
    if history.len() > CTX {
        let tail = &history[history.len() - CTX..];
        let window = history.len() - CTX + 1;
        if let Some(i) = (0..window).rev().find(|&i| &history[i..i + CTX] == tail) {
            matched = CTX;
            let mut s = i + CTX;
            while out.len() < n && s < history.len() {
                out.push(history[s]);
                matched += 1;
                s += 1;
            }
        }
    }
    while out.len() < n {
        out.push(*history.last().unwrap_or(&0));
    }
    (out, matched)
}

fn draft_lookup(history: &[u32], n: usize) -> Vec<u32> {
    draft_lookup_with_match(history, n).0
}

/// Speculative decode over `state` (the session's own slot — the SAME weights and KV the ordinary
/// path would use; nothing extra is allocated except the trace the runner already reserved).
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_spec_backend(
    be: &dyn Backend,
    bind: &BindWeight<'_>,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    mut on_token: impl FnMut(u32),
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    turn_checkpoint: Option<TurnCheckpoint>,
    req: Option<&crate::sampling::RequestCtx>,
    mm: Option<&MropePlan>,
    k: usize,
) -> Result<(Vec<u32>, GenStats)> {
    let k = k.clamp(2, SPEC_MAX_K);
    let hit_eos = |token: u32| {
        !ec.sampling.ignore_eos && (cfg.eos_ids.contains(&token) || token == cfg.eos)
    };
    let cached_before = state.as_ref().map_or(0, SeamKv::cached_len);

    // ── PRIME: one ordinary row — the same kernel path plain decode runs for this shape, so the
    // spec stream cannot introduce a numerical asymmetry against it. Also builds the slot's KV /
    // recurrent state from cold.
    let t_prime = std::time::Instant::now();
    let (prime, _) = generate_dense_backend(
        be,
        bind,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        1,
        |_| {},
        state,
        want_ctx,
        None,
        None,
        None,
        None,
        None,
        None,
        turn_checkpoint,
        req,
        None,
        mm,
    )?;
    let prompt_secs = t_prime.elapsed().as_secs_f64();
    let Some(&first) = prime.first() else {
        return Ok((
            prime,
            GenStats {
                n_prompt: prompt.len().saturating_sub(cached_before),
                n_cached: cached_before,
                prompt_secs,
                n_gen: 0,
                decode_secs: 0.0,
            },
        ));
    };
    // The trace must be usable or a rejected draft poisons the slot: fail loudly, never silently.
    if !state.as_ref().is_some_and(|kv| kv.mtp_can_trace(k)) {
        return Err(anyhow!(
            "speculative decode: the slot has no {k}-row recurrent trace to roll back with \
             (allocate it: `seam::spec::needs_recurrent_trace` must be true at slot build)"
        ));
    }

    let mut committed = prompt.to_vec();
    let mut pending = first;
    let mut out: Vec<u32> = Vec::new();
    let mut generated = 0usize;
    let mut accepted_total = 0usize;
    let mut drafted_total = 0usize;
    let t_decode = std::time::Instant::now();
    if let Some(kv) = state.as_mut() {
        kv.mtp_snapshot_delta(be, cfg)?;
    }

    'decode: while generated < max_new {
        if req.is_some_and(crate::sampling::RequestCtx::aborted) {
            break;
        }
        let n_past = committed.len();
        let drafts = draft_lookup(&committed, k - 1);
        let mut feed = Vec::with_capacity(k);
        feed.push(pending);
        feed.extend_from_slice(&drafts);
        // The runner shortens `prompt` against the slot's cached prefix, so the WHOLE sequence is
        // passed and only these `k` rows are computed (same contract as MTP's VERIFY).
        let mut whole = Vec::with_capacity(n_past + k);
        whole.extend_from_slice(&committed);
        whole.extend_from_slice(&feed);
        if let Some(kv) = state.as_mut() {
            kv.mtp_arm_delta_trace(k)?;
        }
        let mut logits = Vec::new();
        generate_dense_backend(
            be,
            bind,
            g,
            cfg,
            ec,
            token_embd,
            None,
            &whole,
            0,
            |_| {},
            state,
            want_ctx,
            None,
            Some(&mut logits), // `verify`: the m×vocab logits out
            None,              // verify_ids: None ⇒ host argmax below
            None,
            None,
            None,
            None,
            None,
            None,
            None, // mm: the verify's rope positions come from the graph's dynamic pos (as in MTP)
        )?;
        // `verify_ids: None` above is deliberate: the fused GPU-argmax accept path
        // (`caps.argmax_rows`) is NOT enabled on the layer-split pipeline — asking for it there
        // returns ZERO ids (its `Op::Argmax` output lands on a device the readback doesn't read),
        // which silently turns every draft comparison into `draft == 0`. Downloading the rows and
        // argmaxing on the host is backend-independent, and the pipeline pays that download for
        // ordinary decode's logits anyway.
        if logits.len() != k * cfg.vocab {
            return Err(anyhow!(
                "speculative verify: expected {k} rows of {wide} logits, got {} values",
                logits.len(),
                wide = cfg.vocab
            ));
        }
        let ids: Vec<u32> = (0..k)
            .map(|j| crate::mtp::argmax_row(&logits[j * cfg.vocab..(j + 1) * cfg.vocab]))
            .collect();
        let accepted_spec = (0..k - 1).take_while(|&i| drafts[i] == ids[i]).count();
        let accepted = accepted_spec + 1;
        if ec.prof.stages || infr_core::pager_profile::active() {
            tracing::info!(
                "[spec cycle] n_past={n_past} cached_after={} pending={pending} drafts={drafts:?} \
                 ids={ids:?} accepted={accepted}",
                state.as_ref().map_or(0, SeamKv::cached_len)
            );
        }
        accepted_total += accepted_spec;
        drafted_total += k - 1;
        if accepted < k {
            if let Some(kv) = state.as_mut() {
                kv.mtp_restore_delta_row(be, accepted)?;
            }
        }
        let new_tokens: Vec<u32> = feed[..accepted].to_vec();
        committed.extend_from_slice(&new_tokens);
        pending = ids[accepted - 1];
        if let Some(kv) = state.as_mut() {
            kv.mtp_snapshot_delta(be, cfg)?;
        }
        for token in new_tokens {
            out.push(token);
            generated += 1;
            let eos = hit_eos(token);
            if !eos {
                on_token(token);
            }
            if eos || generated >= max_new {
                break 'decode;
            }
        }
    }

    if ec.prof.stages || infr_core::pager_profile::active() {
        let cycles = drafted_total as f64 / (k - 1).max(1) as f64;
        tracing::info!(
            "[spec] {generated} tokens in {cycles:.0} cycles, {accepted_total}/{drafted_total} \
             drafts accepted (alpha={:.3}), {:.1} ms/token",
            accepted_total as f64 / drafted_total.max(1) as f64,
            t_decode.elapsed().as_secs_f64() * 1e3 / generated.max(1) as f64,
        );
    }

    Ok((
        out,
        GenStats {
            n_prompt: prompt.len().saturating_sub(cached_before),
            n_cached: cached_before,
            prompt_secs,
            n_gen: generated,
            decode_secs: t_decode.elapsed().as_secs_f64(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::draft_lookup;

    #[test]
    fn lookup_copies_the_continuation_of_the_last_repeat() {
        // ...a b c | a b c -> the tail `a b c` repeats, so the draft continues from the first run.
        let h = [1u32, 2, 3, 9, 1, 2, 3];
        assert_eq!(draft_lookup(&h, 2), vec![9, 1]);
        // No repeat: pad by repeating the last token rather than returning a short batch.
        assert_eq!(draft_lookup(&[7, 8, 9], 2), vec![9, 9]);
        assert_eq!(draft_lookup(&[], 2), vec![0, 0]);
    }
}
