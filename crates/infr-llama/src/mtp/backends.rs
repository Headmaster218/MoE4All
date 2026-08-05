//! Per-backend MTP spec-decode drivers (Vulkan/Metal/CPU) — thin wrappers over the shared
//! [`super::generate_mtp_spec_core`] loop that differ only in which [`infr_core::backend::Backend`]
//! + weight-bind closure they construct, and which [`super::MtpHeadSession`] constructor they use.

use anyhow::{anyhow, Result};
use infr_core::backend::{Backend, BufferUsage};

use super::{
    generate_mtp_spec_core, BindWeightFn, MtpHeadSession, MtpHeadWeights, MtpTiming, DEFAULT_N_MAX,
};

/// The MTP self-speculative generation loop (issue #33, Phase 3 — see `docs/mtp.md`'s driver
/// section and `crate::seam::model::SeamModel::generate_metal_spec`, whose two-model draft/verify/
/// commit shape this mirrors for a SINGLE self-speculating trunk on the Vulkan production
/// backend). **Greedy only** (temp 0) — the commit/accept invariant below only holds at temp 0;
/// sampling-temperature support is a follow-up phase.
///
/// No cross-turn KV reuse: every call builds a FRESH trunk session + [`MtpHeadSession`] and
/// re-prefills the WHOLE (rendered, multi-turn-inclusive) prompt from scratch. `DenseSeamChat`'s
/// non-MTP path keeps a persistent per-conversation `DenseVulkanSession`; MTP's
/// win is DECODE throughput (the thing `docs/mtp.md`'s 2.0x oracle number measures), not prefill
/// reuse, so trading a little prefill cost for a MUCH simpler (and correct) session lifetime is
/// the pragmatic call this phase makes — a real persistent MTP session hits a self-referential-
/// struct problem ([`MtpHeadSession`] borrows the backend + embedding table it's built from) that
/// isn't worth solving until multi-turn MTP throughput is actually the bottleneck being chased.
///
/// ## Round structure (ported from `speculative.cpp`'s `common_speculative_impl_draft_mtp` driver
/// loop around the target decode, adapted to infr's own VERIFY primitive)
/// 1. **Prime** (once): a single batched VERIFY forward over the WHOLE prompt captures `h` for
///    EVERY prompt row in one shot (no chunked-prefill h gap for the prompt sizes this phase
///    validates — a genuinely huge prompt would need the lazy-priming fallback `docs/mtp.md`
///    sanctions instead: "MTP attention over a partial-history KV degrades gracefully"). `catch_up`
///    primes the head over the whole prompt — mirrors `mtp_head_forward_finite`'s `prime_head` test
///    helper exactly (`shifted_h[0]` zero, `pending_h` = the last row).
/// 2. **Cycle**: `draft` up to `n_max` tokens off `(id_last, pending_h)`; one batched VERIFY over
///    `[committed | drafted]`; `crate::seam::model::spec_accept` picks the longest accepted prefix +
///    the target's correction/bonus token — EXACTLY the macOS spec driver's rule, so the same
///    "committed stream == target's own greedy argmax stream" invariant holds here.
///
/// ## The `+1` leading row, and why cycle 1 is special
/// Every cycle's VERIFY feed is `committed ++ drafted`, and `committed` always includes ONE token
/// the trunk session hasn't been fed yet (the previous cycle's correction/bonus token) — so every
/// VERIFY call after the first returns `drafted.len() + 1` rows: a leading row that re-confirms/
/// predicts `drafted[0]`, found at `base = m - (drafted.len() + 1)`. Cycle 1 has NO such leading
/// row (the prime step already verified — and cached — the WHOLE prompt, so `committed` exactly
/// equals what's cached when cycle 1's draft is proposed): its leading prediction is instead the
/// prime step's own LAST logits/h row (the prompt's last token predicting whatever comes right
/// after it), spliced in as a virtual row so the SAME accept/catch-up code handles cycle 1
/// identically to every later cycle.
///
/// ## KV-overwrite / no-rewind semantics this relies on (see `seam.rs`)
/// The trunk session's VERIFY branch always extends `cached` by the WHOLE fed suffix, accepted or
/// not (`cached.extend_from_slice(&prompt[start..])`) — a rejected draft's rows simply get
/// OVERWRITTEN by the next cycle's differing suffix, exactly like the macOS dense/attention spec
/// driver (`SeamModel::generate_metal_spec`'s own doc: "Rollback is the session prefix-diff:
/// rejected rows just get overwritten by the next round's suffix prefill"). qwen35's gated-
/// DeltaNet layers can't rewind that way (an append-only recurrent summary, not a per-position
/// cache — `docs/qwen35.md`): the SAME `generate_dense_backend` prefix-diff logic that reuses
/// dense KV rows falls back to a FULL reprefill from position 0 whenever a cycle's committed
/// stream doesn't EXACTLY extend the trunk's cached tokens (`seam.rs`'s `start`
/// computation, the `c.qwen35` branch) — i.e. every partial/zero-accept cycle pays a full
/// re-prefill of the WHOLE conversation so far. This is an EXISTING limitation of qwen35's KV
/// reuse (predates MTP; it would bite ANY qwen35 speculative scheme, draft-model or MTP), not
/// something this phase introduces — see `INFR_PROF_STAGES=1`'s per-cycle report for its measured
/// cost, and the accompanying report for whether it's the dominant cost. The head's OWN KV has no
/// such issue: `MtpHeadSession`'s attention layer is an ordinary per-position cache (`docs/mtp.md`),
/// so `catch_up` only ever (re)writes the newly-committed rows, never the whole history.
///
/// ## Perf instrumentation (`INFR_PROF_STAGES=1`)
/// Prints one line per cycle (`drafted`/`accepted` counts, `draft`/`verify`/`catchup` wall time,
/// and the `build`/`exec` split `MtpHeadSession` accumulates internally — Phase 2's doc already
/// flags the head graph as rebuilding + recompiling from scratch every `forward` call, since
/// `Op::Attention::kv_len`/`Op::WriteKv::pos` are baked into the graph rather than bound inputs;
/// this is the number that quantifies that cost) plus a final aggregate-`alpha` summary. Phase 4
/// (issue #33, perf-bottleneck visibility) plumbs the SAME per-cycle sums out programmatically as
/// [`MtpTiming`] — see [`generate_mtp_spec_vulkan_timed`] — rather than adding a second stderr
/// scraper: this function is now a thin wrapper that drops the timing half of that return, so
/// `run`/`serve` (which only want [`crate::GenStats`]) and `bench`/`compare` (which want both)
/// share one implementation.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn generate_mtp_spec_vulkan(
    model: &crate::SeamModel,
    head: &MtpHeadWeights,
    prompt: &str,
    max_new: usize,
    on_piece: impl FnMut(&str),
) -> Result<crate::GenStats> {
    generate_mtp_spec_vulkan_timed(model, head, prompt, max_new, on_piece).map(|(stats, _)| stats)
}

/// [`generate_mtp_spec_vulkan`] plus the [`MtpTiming`] breakdown `INFR_PROF_STAGES=1`'s per-cycle
/// `eprintln!`s already compute — this is that same accounting, returned instead of only printed,
/// so `infr bench`'s `mtp` segment and `infr compare`'s MTP DECODE section can report the
/// draft/verify/catchup phase split and alpha without scraping stderr.
///
/// Constructs its OWN [`infr_vulkan::VulkanBackend`] (a fresh device every call) — this is what
/// `infr bench`'s MTP arm wants (each rep measures a cold run, `bench_mtp_tg`'s doc). A caller
/// that invokes the MTP driver repeatedly on the SAME device (e.g. `DenseSeamChat`'s `warmup()`
/// plus every later chat turn) should call [`generate_mtp_spec_vulkan_timed_on`] instead and hold
/// the backend itself — otherwise every call re-pays a full VkDevice + allocator + pipeline-cache
/// init (issue: an `INFR_MTP=1` chat run used to construct TWO full Vulkan backends — one for
/// `warmup()`, one for the first real turn — for exactly this reason).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn generate_mtp_spec_vulkan_timed(
    model: &crate::SeamModel,
    head: &MtpHeadWeights,
    prompt: &str,
    max_new: usize,
    on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, MtpTiming)> {
    let vk = infr_vulkan::VulkanBackend::new_with(model.cfg().clone())
        .map_err(|e| anyhow!("vulkan init: {e}"))?;
    generate_mtp_spec_vulkan_timed_on(&vk, model, head, prompt, max_new, on_piece)
}

/// [`generate_mtp_spec_vulkan_timed`] over a CALLER-supplied Vulkan backend, so a persistent
/// caller can share ONE device (+ allocator + pipeline cache) across every MTP `generate()` call
/// instead of constructing a fresh one per call. Each call still rebuilds a fresh trunk+head
/// SESSION (KV cache state) by design — see this module's doc on "no cross-turn KV reuse" — that
/// is an ordinary-sized allocation, not a VkDevice re-init, so sharing the backend loses nothing
/// while dropping the redundant device construction.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn generate_mtp_spec_vulkan_timed_on(
    vk: &infr_vulkan::VulkanBackend,
    model: &crate::SeamModel,
    head: &MtpHeadWeights,
    prompt: &str,
    max_new: usize,
    on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, MtpTiming)> {
    let cfg = model.config();
    let max_ctx = model.encode(prompt)?.len() + max_new + DEFAULT_N_MAX + 8;
    let bind: &BindWeightFn = &|_name, tb, dt, _n| {
        let bytes = tb.materialize();
        let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
        let buf = vk
            .alloc(padded.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        vk.upload(buf.as_ref(), &padded)
            .map_err(|e| anyhow!("{e}"))?;
        Ok((buf, dt))
    };
    let mut head_sess = MtpHeadSession::new_vulkan(
        vk,
        model.gguf(),
        cfg,
        model.engine_cfg(),
        head,
        model.token_embd()?,
        max_ctx,
    )?;
    generate_mtp_spec_core(
        vk,
        bind,
        model,
        &mut head_sess,
        max_ctx,
        prompt,
        max_new,
        on_piece,
    )
}

/// Metal twin of [`generate_mtp_spec_vulkan_timed`] — the SAME draft-verify-catchup driver over
/// the Apple-GPU trunk + head (raw native-dtype weight upload, `MtpHeadSession::new_metal`).
#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn generate_mtp_spec_metal_timed(
    model: &crate::SeamModel,
    head: &MtpHeadWeights,
    prompt: &str,
    max_new: usize,
    on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, MtpTiming)> {
    let cfg = model.config();
    let max_ctx = model.encode(prompt)?.len() + max_new + DEFAULT_N_MAX + 8;
    let mtl = infr_metal::MetalBackend::new_with(model.cfg().clone())
        .map_err(|e| anyhow!("metal init: {e}"))?;
    let bind: &BindWeightFn = &|_name, tb, dt, _n| {
        let buf = mtl
            .alloc(tb.len().max(1), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        mtl.upload(buf.as_ref(), &tb.materialize())
            .map_err(|e| anyhow!("{e}"))?;
        Ok((buf, dt))
    };
    let mut head_sess = MtpHeadSession::new_metal(
        &mtl,
        model.gguf(),
        cfg,
        model.engine_cfg(),
        head,
        model.token_embd()?,
        max_ctx,
    )?;
    generate_mtp_spec_core(
        &mtl,
        bind,
        model,
        &mut head_sess,
        max_ctx,
        prompt,
        max_new,
        on_piece,
    )
}

/// CPU MTP driver — the exact-f32 reference (no GPU). Same draft-verify-catchup loop; used to
/// establish the acceptance-rate oracle (a backend whose head numerics differ from the trunk's
/// shows up as a lower alpha here vs on a GPU backend).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn generate_mtp_spec_cpu_timed(
    model: &crate::SeamModel,
    head: &MtpHeadWeights,
    prompt: &str,
    max_new: usize,
    on_piece: impl FnMut(&str),
) -> Result<(crate::GenStats, MtpTiming)> {
    let cfg = model.config();
    let max_ctx = model.encode(prompt)?.len() + max_new + DEFAULT_N_MAX + 8;
    let cpu = infr_cpu::CpuBackend::new_with(model.engine_cfg().clone());
    let bind: &BindWeightFn = &|_name, tb, dt, _n| match tb {
        crate::seam::WBytes::Mmap(tb) => Ok((cpu.map_weight(tb), dt)),
        other => {
            let v = other.materialize();
            let buf = cpu
                .alloc(v.len().max(1), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            cpu.upload(buf.as_ref(), &v).map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        }
    };
    let mut head_sess = MtpHeadSession::new_cpu(
        &cpu,
        model.gguf(),
        cfg,
        model.engine_cfg(),
        head,
        model.token_embd()?,
        max_ctx,
    )?;
    generate_mtp_spec_core(
        &cpu,
        bind,
        model,
        &mut head_sess,
        max_ctx,
        prompt,
        max_new,
        on_piece,
    )
}

/// Non-timed Metal MTP driver (drops the [`MtpTiming`]) — the `ChatModel::generate` entry.
#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn generate_mtp_spec_metal(
    model: &crate::SeamModel,
    head: &MtpHeadWeights,
    prompt: &str,
    max_new: usize,
    on_piece: impl FnMut(&str),
) -> Result<crate::GenStats> {
    generate_mtp_spec_metal_timed(model, head, prompt, max_new, on_piece).map(|(stats, _)| stats)
}
