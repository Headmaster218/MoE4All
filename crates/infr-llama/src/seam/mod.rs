//! CPU model runner — builds and drives the agnostic decode [`Graph`] through [`CpuBackend`].
//! The backend itself lives in `infr-cpu`; this module is the model-specific "glue" that
//! assembles the layer graph, uploads weights, and steps the KV cache.
//!
//! Split into submodules (pure move, zero behavior change): [`weights`] holds the per-layer
//! weight-handle structs and the persistent seam session state; [`sc`] holds the DiffusionGemma
//! self-conditioning pieces; [`runner`] holds the giant backend-generic `generate_dense_backend`
//! and its `DecodeHandles`. This file keeps the thin per-backend entry wrappers, the `verify_*`
//! family, and the small shared helpers every submodule reaches into.
#![allow(clippy::too_many_arguments)]

use crate::{dequant_block, Config, EngineConfig, GenStats, PerLayerEmbd};
use anyhow::{anyhow, Result as AResult};
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::tensor::DType;
use infr_core::WeightSource;
use infr_cpu::CpuBackend;
use infr_gguf::{Gguf, TensorBytes};

pub mod model;
mod runner;
mod sc;
mod weights;

pub(crate) use runner::generate_dense_backend;
pub(crate) use sc::DenoiseReq;
pub use sc::{DenoiseOutcome, EbReduced};
pub(crate) use weights::SeamKv;

/// A LAZILY-dequantized host f32 token-embedding table, threaded through the seam runners in place
/// of a `&[f32]`.
///
/// `token_embd.weight` blown up to f32 is enormous — Qwen3-14B's 151936×5120 Q4_K table becomes
/// 3.1 GiB of host RAM and costs ~4s of dequant, which used to be paid EAGERLY by every
/// `SeamModel::load`, i.e. by every model load on every backend. But the Vulkan and Metal dense
/// paths upload `token_embd.weight` to the device in its NATIVE dtype and gather embeddings ON
/// DEVICE (`Op::EmbedGather` / the tied-lm_head `Op::Linear`), so they never look at the host
/// table. Only the host-gather consumers touch it: the CPU runner's embed, the DiffusionGemma SC
/// soft-embed, and the MTP heads.
///
/// Passing this handle instead of a materialized slice keeps the dequant OFF the GPU load path
/// while leaving every host consumer byte-for-byte identical — they call [`get`](Self::get), which
/// dequantizes once into the owning [`model::SeamModel`]'s cache and returns the cached table on
/// every later call.
#[derive(Clone, Copy)]
pub(crate) struct TokenEmbd<'a> {
    cell: &'a std::sync::OnceLock<Vec<f32>>,
    gguf: &'a Gguf,
}

impl<'a> TokenEmbd<'a> {
    pub(crate) fn new(cell: &'a std::sync::OnceLock<Vec<f32>>, gguf: &'a Gguf) -> Self {
        Self { cell, gguf }
    }

    /// The dequantized `[vocab, n_embd]` row-major table — dequantized on first call, cached after.
    /// `Config::from_gguf` already validated the tensor exists at load, but a truncated/corrupt GGUF
    /// can still fail the dequant here (the first host gather, lazily). FALLIBLE so that a real
    /// non-programmer input surfaces a clear error at the call site instead of aborting the process.
    pub(crate) fn get(&self) -> AResult<&'a [f32]> {
        if let Some(v) = self.cell.get() {
            return Ok(v);
        }
        let (v, _) =
            crate::quant::load_tensor_dequant(self.gguf, "token_embd.weight").map_err(|e| {
                anyhow!("token_embd.weight: dequant failed (corrupt/truncated GGUF?): {e}")
            })?;
        // Race-safe cache fill: another thread may have won the init; `set` drops ours if so, then
        // `get` returns whichever value stuck (get_or_init semantics without a fallible init).
        let _ = self.cell.set(v);
        Ok(self
            .cell
            .get()
            .expect("cell was just set or already initialized"))
    }
}

/// The CPU seam weight binder, hoisted so the CPU runner and every CPU `verify_*` entry share one
/// copy: map an mmap slice zero-copy; alloc+upload owned bytes (the combined gate+up concat — never
/// produced for CPU since `combined_gu` is false there, but stays correct if it ever is).
fn cpu_upload_bind(be: &CpuBackend) -> Box<BindWeight<'_>> {
    Box::new(move |_name, tb, dt, _n| match tb {
        WBytes::Mmap(tb) => Ok((be.map_weight(tb), dt)),
        WBytes::Owned(v) => {
            let buf = be
                .alloc(v.len().max(1), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(buf.as_ref(), &v).map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        }
    })
}

/// The Metal seam weight binder (raw native-dtype upload; the backend dequantizes lazily), hoisted
/// so the Metal session runner and `verify_dense_metal2` share one copy.
#[cfg(target_os = "macos")]
fn metal_upload_bind(be: &infr_metal::MetalBackend) -> Box<BindWeight<'_>> {
    Box::new(move |_name, tb, dt, _n| {
        let buf = be
            .alloc(tb.len().max(1), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(buf.as_ref(), &tb).map_err(|e| anyhow!("{e}"))?;
        Ok((buf, dt))
    })
}

// ─── Qwen3 dense CPU decode runner ───────────────────────────────────────────────
//
// Builds the n=1 decode Graph and drives it through `CpuBackend`, one token at a time, for BOTH
// prompt ingestion (looped) and generation — so no GEMM/flash prefill kernels are needed on CPU.
// The KV cache grows one row per step. Validates the agnostic seam end-to-end against the GPU path.

/// Greedy CPU generation for a decoder (Qwen3 / Llama / Gemma 3 / Gemma 4 dense+E2B / qwen3moe). The
/// attention block is shared; the FFN is either a dense gated FFN or a routed-expert MoE bank; gemma4
/// E2B adds per-layer input embeddings + KV-layer sharing. `prompt` is the full token prefix; returns
/// the generated continuation. Stops at EOS or `max_new`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dense_cpu(
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    req: Option<&crate::sampling::RequestCtx>,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    // S4 deletes S3's bridge: the CPU backend takes the config the SEAM was handed instead of
    // building one from the environment inside `CpuBackend::new`.
    generate_dense_cpu_mode(
        CpuBackend::new_with(ec.clone()),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        req,
        on_token,
    )
}

/// [`generate_dense_cpu`] on a caller-supplied `CpuBackend`, so a comparison can pick the
/// REFERENCE backend ([`CpuBackend::reference`]) instead of the production int8-activation
/// kernels. The distinction matters at low bit-widths: the int8 activation quant carries ~4e-3
/// relative error on every quant dtype, which is invisible at Q4_K but flips greedy tokens at
/// Q2_K — so a backend-vs-backend token comparison must be scored against the reference mode.
#[cfg_attr(infr_profile, infr_prof::instrument)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dense_cpu_mode(
    cpu_be: CpuBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    req: Option<&crate::sampling::RequestCtx>,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    // Thin CPU wrapper over the backend-generic runner: a CpuBackend + a zero-copy weight binder
    // (maps each tensor straight from the GGUF mmap — no alloc, no memcpy).
    let bind = cpu_upload_bind(&cpu_be);
    generate_dense_backend(
        &cpu_be,
        &bind,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        &mut None,
        prompt.len() + max_new + 1,
        None, // constraint
        None, // verify
        None, // verify_ids
        None, // logits_out
        None, // h_out
        None, // denoise_req
        req,
    )
}

/// GPU seam runner: the SAME dense forward as [`generate_dense_cpu`], but on the Vulkan backend
/// through the agnostic [`Graph`] adapter (weights padded + uploaded to VRAM instead of mmap-mapped).
/// This is the end-to-end GPU parity/perf path — running it and diffing the CPU oracle proves the
/// adapter, and its decode tok/s (still recompiling the graph per token) is the baseline
/// record-once replay must close. Prefill's batched attention is decode-only on the seam, so the
/// caller may pass short prompts to force the per-token path.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_vulkan(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_vulkan_session(
        vk,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        &mut None,
        prompt.len() + max_new + 1,
        None, // constraint
        None, // req: the one-shot runner is a sole sequence — config sampling, no gate
    )
}

/// [`generate_dense_vulkan`] with a caller-held [`SeamKv`]: hold `state` (+ a `want_ctx` capacity)
/// across calls and each turn prefills only the suffix that differs from the cached tokens —
/// ChatSession-style KV reuse on the agnostic seam.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_vulkan_session(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    constraint: Option<&mut crate::grammar::Constraint>,
    req: Option<&crate::sampling::RequestCtx>,
) -> AResult<(Vec<u32>, GenStats)> {
    // Placement can allocate + upload (the pager arenas, a weight re-bind), i.e. it RECORDS on the
    // Vulkan command pool — so it takes a turn on the baton like any other GPU region. Scoped: the
    // baton is released before the runner starts stepping. See `StepGate`.
    //
    // A WARM call (`state.is_some()`) has every weight already resident, and the runner never calls
    // `bind_weight` again (see the `state.is_none()` init block in `generate_dense_backend`), so
    // building the full `vulkan_moe_binder` — which re-resolves the placement tiers and allocs a
    // Box — is pure per-turn waste. Skip it: a no-op binder that errors loudly if the invariant
    // ever breaks.
    let warm_binder: Box<BindWeight<'_>> =
        Box::new(|name: &str, _tb, _dt, _n| Err(anyhow!("warm session must not re-bind {name}")));
    let bind = if state.is_some() {
        warm_binder
    } else {
        let _gp = req.and_then(|r| r.gate_pass());
        vulkan_moe_binder(vk, g, cfg, ec, true, want_ctx)?
    };
    let out = generate_dense_backend(
        vk, &*bind, g, cfg, ec, token_embd, ple, prompt, max_new, on_token, state, want_ctx,
        constraint, None, None, None, None, None, req,
    )?;
    // INFR_PAGER_STATS=1: cumulative hit/miss/eviction counters since this pager was installed
    // (persists across calls on the same session — see `MoePagerSession`). A no-op when no paged
    // model is loaded. Printed every call rather than gated to "last call only" since neither the
    // CLI's run/serve loop nor this function know which call is the process's last one.
    vk.print_moe_pager_stats();
    vk.print_dense_pager_stats();
    Ok(out)
}

/// Honest activation/scratch reservation for a DENSE model's placement decision: the transient
/// VRAM a resident session needs BEYOND weights + KV, at the largest shape it will ever run — a
/// full prefill chunk of `rows = min(ubatch, want_ctx)` rows (the runner chunks batched prefill at
/// INFR_UBATCH, default 1024; decode's single row is dwarfed by this).
///
/// Derivation — measured on gemma-4-31B UD-Q5_K_XL (n_embd 5376, n_ff 32768, n_head 32,
/// head_dim 512/256 full/SWA) on a 24 GiB 7900 XTX with a tagged allocation trace
/// (INFR_ALLOC_TRACE-style eprintln on every ≥16 MiB activation alloc) at a 1024-row prefill
/// chunk, ctx 2064:
/// - Internal graph tensors (`alloc_scratch`, ~850 MiB): fused gate_up out `[rows, 2*n_ff]` f32
///   (256 MiB) + activated intermediate `[rows, n_ff]` f32 (128 MiB) + fused qkv staging
///   (168 MiB = rows*8*n_embd*4 here) + ~a dozen `[rows, n_embd]`-class f32/f16 temps.
///   Modeled as `12*n_ff + 96*n_embd` per row (the n_embd umbrella also absorbs the
///   lin_a16/mmq activation-quant pools, which are n_embd/n_ff-wide f16/i8).
/// - Attention pools — the term a previous calibration MISATTRIBUTED as "rows*vocab*2
///   whole-chunk f16 logits" (batched prefill has run a last-row-only m=1 LM head since long
///   before that trace, and is fully headless now — no logits allocation scales with rows;
///   2*vocab merely coincided with the real per-row attention bytes on this model, where
///   2*262144 == 8*n_head*head_dim*4 at head_dim 512):
/// - `nonfa_pv`/`flash_po`: `8*rows*n_head*head_dim*4` per DISTINCT head shape — gemma4
///   alternates SWA(256)/full(512) head dims, so BOTH pools live at once (512 + 256 MiB
///   measured) → `32*n_head*(head_dim + head_dim_swa-if-distinct)` per row.
/// - `nonfa_s` (score tiles, non-flash tier only — any model with SWA layers or
///   head_dim != 128): `n_head*rows*kv_pad*2`, kv_pad = kv_len rounded up to 256. The pool
///   key includes the byte size, so as kv grows across chunks stale sizes are retained —
///   modeled as 2 live pools at the final ctx: `4*n_head*ctx_pad` per row. Uniform-hd-128
///   no-SWA models (llama/qwen3) ride the single-pass flash tier: no score tiles, only the
///   (negligible) flash_pm/pl partials — term skipped.
///
/// All times a 1.25 margin for unmeasured tails (split-path partials, per-shape pool
/// duplicates), plus a fixed 256 MiB for what shapes don't scale: gpu-allocator's block
/// granularity, retained upload staging (device-local under ReBAR), and the weight-buffer
/// u32/dedicated-alloc padding not in `weight_footprint`. Deliberately a slight over-reserve:
/// under-reserving makes the alloc-time VRAM guard error a live request mid-prefill (exactly
/// what the old formula did on this 31B at pp2048 — the second chunk's 512 MiB `nonfa_pv`
/// tripped the guard mid-run), over-reserving only streams/clamps a borderline model.
/// Always taken at an EXPLICIT chunk height: every caller (the try-resident sweep, the streaming
/// budget, the context-fit math) walks [`ubatch_candidates`] rather than assuming the default
/// 1024-row chunk — assuming it is what let the KV-format decision and the placement decision
/// disagree.
pub(crate) fn dense_act_reserve_at(cfg: &Config, want_ctx: usize, ubatch: usize) -> u64 {
    // Prefill GEMM outputs pad rows to 64 (see the Vulkan adapter's `alloc_scratch`).
    let rows = ubatch.min(want_ctx).max(1).next_multiple_of(64) as u64;
    // Attention pv accumulators: one pool per distinct (n_head, head_dim) shape.
    let hd_shapes = if cfg.swa_window > 0 && cfg.head_dim_swa != cfg.head_dim {
        cfg.head_dim + cfg.head_dim_swa
    } else {
        cfg.head_dim
    };
    let attn_pv = 32 * cfg.n_head * hd_shapes;
    // Non-flash score tiles (see the doc above): skipped for uniform-hd-128 no-SWA models.
    // When ONLY the SWA layers miss the flash tier (hd == 128, e.g. gemma3-12b: full layers are
    // Causal+hd128 = flash), the widest score tile is the SWA ring's `window + chunk` rows, not
    // the full context.
    let attn_s = if cfg.swa_window == 0 && cfg.max_head_dim() == 128 {
        0
    } else {
        let kv_span = if cfg.max_head_dim() == 128 {
            want_ctx.min(cfg.swa_window + ubatch)
        } else {
            want_ctx
        };
        4 * cfg.n_head * kv_span.next_multiple_of(256)
    };
    let per_row = (12 * cfg.n_ff + 96 * cfg.n_embd + attn_pv + attn_s) as u64;
    const FIXED: u64 = 256 * 1024 * 1024;
    let est = FIXED + rows * per_row * 5 / 4;
    est * ACT_RESERVE_MARGIN.0 / ACT_RESERVE_MARGIN.1
}

/// INTERIM safety margin on [`dense_act_reserve_at`], as `(numerator, denominator)`.
///
/// **Why this term and not the KV term.** KV bytes are computed EXACTLY — per-layer geometry and
/// each side's dtype, run through `kv_fmt_bytes`, the same sizer the runner hands `Backend::alloc`
/// — so they need no margin and get none. The activation reserve is a MODEL, and a
/// measured-wrong one: it prices the non-flash score tile as two live pools at the final context,
/// while the real pools are keyed by byte size and ACCUMULATE as `kv_len` grows across the ~150+
/// chunks of a deep prefill. `FIT_FRACTION = 0.95` used to blur the two together; the slack
/// belongs on the estimate alone.
///
/// **Why it is needed at all.** Once [`kv_fit_ctx_for`] returns the exact boundary rather than a
/// 5%-shy one, any error in this model lands directly as a mid-prefill allocation failure —
/// i.e. as a default context whose top cannot be filled. Measured on gemma-4-31B UD-Q5_K_XL
/// (24 GiB 7900 XTX): with no margin the fit advertised 19 973 tokens and a `-d 19000` prefill
/// died with `VRAM budget exceeded` on a 4 MiB activation alloc.
///
/// **Sized by measurement, at the ADVERTISED context, not by choosing a number.** Each candidate
/// was built, the default context it produces read off the `ctx clamp` line, and a prefill run at
/// ~100% of THAT context (gemma-4-31B UD-Q5_K_XL, `infr bench -p 0 -n 4 -d <ctx-8>`):
///
/// | margin | advertised ctx | prefill at ~100% of it            | peak MiB |
/// | ------ | -------------- | --------------------------------- | -------- |
/// | 1.00   | 19 973         | `VRAM budget exceeded` @ d19000   | 24 298   |
/// | 1.25   | 17 848         | `VRAM budget exceeded` @ d17840   | 24 299   |
/// | 1.50   | 15 872         | **ok**, 30.2 t/s @ d15864         | 24 142   |
///
/// 1.50 is the FIRST rung that fills the window it advertises, so that is what shipped — nothing
/// was added on top of the measurement. It also holds at d15000 / d12000 / d8000 (30.3 / 30.6 /
/// 31.2 t/s) and the d15864 peak reproduces byte-for-byte across runs. Remaining spare at the
/// advertised context is 157 MiB of the 24 299 MiB guard budget (~0.6%) — thin, which is the
/// argument for B8 rather than for a bigger constant.
///
/// Cost: gemma-4-31B's default context goes 19 973 -> 15 872 (against 1024 before this slice, so
/// the win is 15.5x rather than 19.5x) and it stays FULLY RESIDENT, which is the property worth
/// 10x on decode. gemma-3-12b is unaffected — still f16 @131072 with no auto-quant, because the
/// shared chunk ladder absorbs the wider reserve one rung down.
///
/// **DELETE THIS when the reserve becomes path-aware (backlog B8).** It stands in for that work;
/// it is not an input to it, and a path-aware estimate must replace it rather than be stacked on
/// top of it.
const ACT_RESERVE_MARGIN: (u64, u64) = (3, 2);

/// Batched-prefill micro-batch: rows per prefill chunk (`device.ubatch` / `INFR_UBATCH`, default
/// 1024 — but see [`default_ubatch_rows`] for the INTEGRATED-GPU default). ONE reader funnel — the
/// prefill loop, the activation reserve, and the SWA ring sizing below all derive from this,
/// because the ring's correctness bound is "window + one whole prefill chunk".
///
/// **`ubatch_specified` IS needed** (the S0 report's open question; §10's `INFR_UBATCH=abc` note).
/// `INFR_UBATCH` is a §6.12 two-consumer knob — the VALUE here, and the PRESENCE the placement
/// sweeps read ([`user_pinned_ubatch`]) — and the two DISAGREE about an unusable value, exactly as
/// the KV dtypes do (§11 decision 8). The old value site was
/// `.parse().ok().filter(|&v| v > 0).unwrap_or_else(pin/default)` and the old presence site was a
/// bare `is_err()` on the raw variable, so `INFR_UBATCH=0` (and `=abc`) yielded NO height
/// while still disabling the sweep. That is not a corner case: `infr … -u 0` is the DOCUMENTED
/// "stay adaptive" spelling and the CLI publishes it verbatim. Collapsing both readers onto
/// `device.ubatch.is_some()` would silently re-enable the residency sweep for it, so `DeviceCfg`
/// carries the presence flag separately and every input — valid value, `0`, garbage, unset — keeps
/// today's behaviour bit-for-bit (R1).
pub(crate) fn ubatch_rows(ec: &EngineConfig) -> usize {
    ec.device.ubatch.filter(|&v| v > 0).unwrap_or_else(|| {
        with_placement_pins(|p| p.ubatch.get().copied()).unwrap_or_else(default_ubatch_rows)
    })
}

/// Did the user PIN a prefill chunk height? The PRESENCE half of `INFR_UBATCH` (§6.12) — the dense
/// placement sweeps skip themselves when it is true, because the user's height is authoritative.
/// True even for a value this reader cannot use (`0`, garbage): see [`ubatch_rows`].
pub(crate) fn user_pinned_ubatch(ec: &EngineConfig) -> bool {
    ec.device.ubatch_specified
}

/// The SHRINK ladder the dense placement sweeps walk when the default prefill chunk's activation
/// reserve is what tips a model out of residency: 512 → 256 → 128 rows. A shorter chunk shrinks
/// both the activation reserve (whole-chunk scratch scales with rows) and the SWA ring rows
/// (`window + chunk`), and resident-at-512 decodes ~10x faster than streaming at the PCIe ceiling
/// — so trading prefill chunk height for residency is strictly the right call.
///
/// 128 is the floor: below it the per-dispatch launch overhead dominates prefill entirely.
pub(crate) const DENSE_UBATCH_LADDER: [usize; 3] = [512, 256, 128];

/// Every prefill chunk height a dense placement decision is allowed to settle on, TALLEST FIRST:
/// the current/default height ([`ubatch_rows`]) followed by the [`DENSE_UBATCH_LADDER`] rungs
/// BELOW it. A user-pinned `INFR_UBATCH` is authoritative — the sweeps skip themselves — so the
/// list is then just that one height.
///
/// **One ladder, two readers.** `vulkan_moe_binder`'s residency / auto-q8 / streaming sweeps walk
/// this to pick a height, and `SeamModel::kv_fit_ctx_fmt` walks the SAME list to decide how much
/// context fits: a context is accepted when it fits at ANY height placement could settle on.
/// Before this was shared, the fit math priced the DEFAULT 1024-row chunk's reserve while
/// placement went on to pick 512 — the KV format was chosen against an assumption the very next
/// step abandoned (gemma-3-12b: an unnecessary auto-q8 at ctx 131072, while f16 at a 512-row
/// chunk fits with room to spare). `dense_ubatch_ladder_is_the_only_one` guards the drift.
///
/// Filtering to heights `< ubatch_rows(ec)` matters on an INTEGRATED GPU, whose default chunk is
/// already below 512 ([`default_ubatch_rows`]): an unfiltered ladder would let a "shrink" sweep
/// RAISE the chunk past the watchdog-safe default and trip `VK_ERROR_DEVICE_LOST`.
pub(crate) fn ubatch_candidates(ec: &EngineConfig) -> Vec<usize> {
    let now = ubatch_rows(ec);
    let mut cands = vec![now];
    if !user_pinned_ubatch(ec) {
        cands.extend(DENSE_UBATCH_LADDER.into_iter().filter(|&c| c < now));
    }
    cands
}

/// The prefill chunk when neither INFR_UBATCH nor the placement sweep pinned one: 1024 rows, EXCEPT
/// on an integrated GPU, where a chunk that big is a single multi-second command buffer and trips
/// the ~10 s GPU watchdog (`ring gfx_0.0.0 timeout` -> `VK_ERROR_DEVICE_LOST`). See
/// [`infr_core::integrated_ubatch_rows`] for the measurements behind the smaller default.
///
/// A DISCRETE device (and a CPU/Metal run, where no Vulkan backend was constructed and
/// `device_class()` is `None`) takes the 1024 branch — byte-identical to before this existed, so
/// no tuned dGPU shape moves.
fn default_ubatch_rows() -> usize {
    match infr_vulkan::device_class() {
        Some(d) if d.integrated => infr_core::integrated_ubatch_rows(d.compute_units),
        _ => 1024,
    }
}

/// Prefill chunk (rows) for a sequence SHARING the GPU with other in-flight sequences
/// (`infr serve --parallel N`, i.e. the runner's `req` carries a `StepGate`).
///
/// A prefill chunk is unpreemptible GPU: the whole chunk holds the baton, so it is exactly how long
/// a newly-admitted request's prefill stalls every in-flight decode. The solo default (1024 rows,
/// [`ubatch_rows`]) is ~100ms+ on a 14B — a visible hitch across 3 other streams. 256 rows bounds
/// that to ~25-30ms (about the cost of ~4 decode steps) at a small prefill-throughput cost, which
/// is the right trade when N clients are streaming. Never applies to a sole request: `infr run`,
/// `bench`, the goldens, and a `-np 1` server all keep the full [`ubatch_rows`] chunk, so prefill
/// throughput there is UNCHANGED. INFR_UBATCH_PARALLEL overrides; it only ever SHRINKS the chunk
/// (the runner takes the `min` with [`ubatch_rows`]).
pub(crate) fn ubatch_rows_parallel(ec: &EngineConfig) -> usize {
    ec.device.ubatch_parallel
}

/// The two placement decisions the VRAM ladder pins for a session and then keeps STABLE for its
/// whole lifetime (set before the first KV allocation, never changed after — warm calls and rebuilt
/// graphs must agree with the buffers they were sized with):
///   - `ubatch`: the pinned prefill chunk (rows) when the dense try-resident sweep found a smaller
///     chunk is what makes a big model fully resident (residency at a 512-row chunk decodes ~10x
///     faster than streaming at the PCIe ceiling — see `vulkan_moe_binder`'s dense tier). Read by
///     [`ubatch_rows`] when INFR_UBATCH is unset, so the prefill loop, the activation reserve, and
///     the SWA ring sizing all agree on the same height.
///   - `kv_q8`: auto-q8 KV, chosen to keep a session RESIDENT (dense try-resident tier) or to avoid
///     shrinking a DEFAULT context (`SeamModel::clamp_default_ctx`) — the "q8 KV" rung of the VRAM
///     ladder. Read by the runner's per-side KV-format selection (and every KV-footprint estimate)
///     ONLY when the user set none of INFR_KV_TYPE_K / INFR_KV_TYPE_V / INFR_KV_Q8 — an explicit
///     setting always wins, both directions. Policy: BOTH sides go q8_0 (coupled Q8 keeps
///     record-once decode replay and satisfies K>=V symmetrically); never below q8_0.
///
/// These were process-global `OnceLock`s, which LEAKED across models in a multi-model process
/// (`infr multi` hosts N models, each its own `DenseVulkanSession`, and the server runs their
/// generations CONCURRENTLY): a second model's `.set()` was a silent no-op, so it inherited model
/// A's pinned chunk height / q8 decision — which may not fit B's VRAM. They now live PER SESSION
/// (owned by `DenseVulkanSession`, entered via [`PlacementScope`] around each placement + generate),
/// so each model's ladder decision is isolated.
#[derive(Default)]
pub(crate) struct PlacementPins {
    ubatch: std::sync::OnceLock<usize>,
    kv_q8: std::sync::OnceLock<()>,
}

thread_local! {
    /// The session whose placement pins the current thread's `ubatch_rows`/`kv_auto_q8` reads and
    /// writes resolve against — set by [`PlacementScope`] for the duration of a placement + runner
    /// call. `None` outside any scope (the one-shot `generate_dense_*`, CPU/Metal, tests), which
    /// falls back to [`FALLBACK_PINS`].
    static CURRENT_PINS: std::cell::RefCell<Option<std::sync::Arc<PlacementPins>>> =
        const { std::cell::RefCell::new(None) };
}

/// Process-wide fallback pins for callers with no active [`PlacementScope`] (one-shot run/bench,
/// CPU/Metal, tests). Those paths host a SINGLE model per process, so a process-global is correct
/// there and byte-identical to the old two-`OnceLock` behavior.
static FALLBACK_PINS: std::sync::OnceLock<std::sync::Arc<PlacementPins>> =
    std::sync::OnceLock::new();

/// Run `f` against the current thread's session pins (or the process fallback when no session scope
/// is active). The single reader/writer funnel for both placement pins.
fn with_placement_pins<R>(f: impl FnOnce(&PlacementPins) -> R) -> R {
    CURRENT_PINS.with(|c| match c.borrow().as_ref() {
        Some(p) => f(p),
        None => f(FALLBACK_PINS.get_or_init(|| std::sync::Arc::new(PlacementPins::default()))),
    })
}

/// RAII guard binding `pins` as the current thread's placement scope. Every seam entry that runs a
/// placement decision or a runner step for a session (`DenseVulkanSession::generate*`, the
/// default-ctx clamp) enters one around its work so the free-function readers
/// ([`ubatch_rows`]/[`kv_auto_q8`]) resolve against THAT session's pins. Restores the previous
/// scope on drop (scopes may nest — the clamp runs inside session construction).
pub(crate) struct PlacementScope {
    prev: Option<std::sync::Arc<PlacementPins>>,
}

impl PlacementScope {
    pub(crate) fn enter(pins: std::sync::Arc<PlacementPins>) -> Self {
        let prev = CURRENT_PINS.with(|c| c.borrow_mut().replace(pins));
        Self { prev }
    }
}

impl Drop for PlacementScope {
    fn drop(&mut self) {
        CURRENT_PINS.with(|c| *c.borrow_mut() = self.prev.take());
    }
}

/// Pin the placement prefill chunk (rows) for the current session scope. Idempotent (OnceLock).
fn pin_ubatch(rows: usize) {
    with_placement_pins(|p| {
        let _ = p.ubatch.set(rows);
    });
}

/// Whether the placement ladder pinned auto-q8 KV for the current session scope (see
/// [`PlacementPins`]).
pub(crate) fn kv_auto_q8() -> bool {
    with_placement_pins(|p| p.kv_q8.get().is_some())
}

/// Pin auto-q8 KV for the current session scope (the default-ctx clamp path in `model.rs`, and the
/// binder's own dense rung). Idempotent (OnceLock).
pub(crate) fn pin_kv_auto_q8() {
    with_placement_pins(|p| {
        let _ = p.kv_q8.set(());
    });
}

/// True when the user expressed NO explicit KV-format choice — the only state auto-q8 may fill.
///
/// §11 decision 8: the `*_specified` flags, NOT `type_k.is_some()`. An unrecognized format name
/// (`INFR_KV_TYPE_K=nonsense`) parses to no dtype, so the runner falls through to f16 — but it was
/// still SUPPLIED, and today's `is_err()` reads it as "the user chose", suppressing auto-q8. Both
/// halves of that asymmetry are preserved.
pub(crate) fn kv_unset(ec: &EngineConfig) -> bool {
    !ec.kv.type_k_specified && !ec.kv.type_v_specified && !ec.kv.force_q8
}

/// Layout gate for a Q8_0 KV cache — every layer's KV row must be whole 32-elem blocks (the
/// same alignment the runner's own `parse_kv_fmt` gate checks; keeping them identical means a
/// pinned auto-q8 is never silently gated back to f16 with an under-sized placement estimate).
pub(crate) fn kv_q8_layout_ok(cfg: &Config) -> bool {
    (0..cfg.n_layer).all(|l| (cfg.layer_n_kv(l) * cfg.layer_head_dim(l)).is_multiple_of(32))
}

/// [`kv_rows`] at an EXPLICIT chunk height (the try-resident sweep prices candidate heights
/// before pinning one; everyone else goes through `kv_rows`/`ubatch_rows`).
pub(crate) fn kv_rows_at(
    cfg: &Config,
    l: usize,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
) -> usize {
    if ring && cfg.is_swa_layer(l) {
        want_ctx.min((cfg.swa_window + ubatch).next_multiple_of(64))
    } else {
        want_ctx
    }
}

/// Row capacity of layer `l`'s K/V cache at context `want_ctx`. With `ring` (SWA ring sizing on
/// for this session — see [`kv_ring_wanted`] + the backend's `Capabilities::kv_swa_ring`), a
/// sliding-window layer allocates only `min(want_ctx, round64(window + ubatch))` rows and the
/// backends write/read position `p` at row `p % rows` (WriteKv/Attention ring semantics).
///
/// Correctness bound: during one forward of `B <= ubatch` rows starting at position `p0`, the
/// oldest position any query's window reaches is `p0 + 1 - window` and the newest written is
/// `p0 + B - 1` — at most `window + B - 1 <= window + ubatch` distinct live positions, so a ring
/// of `window + ubatch` rows never recycles a row the sliding-window mask hasn't ALREADY excluded
/// (that mask discards everything older than `pos - window`); attention output is therefore
/// identical to the full-context cache. Global (non-SWA) layers keep full `want_ctx` rows.
pub(crate) fn kv_rows(
    cfg: &Config,
    l: usize,
    want_ctx: usize,
    ring: bool,
    ec: &EngineConfig,
) -> usize {
    kv_rows_at(cfg, l, want_ctx, ring, ubatch_rows(ec))
}

/// KV-cache footprint ESTIMATE summed over all layers at chunk height `ubatch` and side format:
/// `q8` prices BOTH sides Q8_0 (34 B / 32-elem block, `next_multiple_of(4)`, mirroring
/// [`kv_fmt_bytes`]), else f16 (2 B/elem); +64 rows/layer slop. The ONE pricing helper the dense
/// placement sweep AND the MoE expert-placement budget share, so both reserve exactly the cache the
/// runner will allocate — including a pinned auto-q8 cache. (The MoE path used to hard-code f16
/// `*2*2` regardless, over-reserving ~2× under auto-q8 and forcing avoidable expert paging.)
pub(crate) fn kv_bytes_estimate(
    cfg: &Config,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    q8: bool,
) -> u64 {
    let side = if q8 { DType::Q8_0 } else { DType::F16 };
    kv_bytes_estimate_fmt(cfg, want_ctx, ring, ubatch, side, side)
}

/// Rows of headroom every KV footprint estimate adds PER LAYER on top of the context's own rows.
///
/// The runner allocates exactly `kv_rows(..) * n_kv * head_dim` elements per side (see
/// `generate_dense_backend`'s KV loop and `SeamKv::fork`) — this is deliberate slop, not a
/// pad the allocation actually contains, so every estimate that uses it is conservative by a
/// bounded, layer-proportional amount rather than by an unexplained percentage.
pub(crate) const KV_SLOP_ROWS: usize = 64;

/// [`kv_bytes_estimate`] with the two sides priced INDEPENDENTLY. `INFR_KV_TYPE_K` and
/// `INFR_KV_TYPE_V` are separate knobs and the runner allocates each side in its own dtype, so
/// the context-fit math — which must compare against the REAL allocation size — cannot go through
/// the single-`q8` flavor above.
pub(crate) fn kv_bytes_estimate_fmt(
    cfg: &Config,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> u64 {
    (0..cfg.n_layer)
        .map(|l| {
            let elems = (cfg.layer_n_kv(l) * cfg.layer_head_dim(l))
                * (kv_rows_at(cfg, l, want_ctx, ring, ubatch) + KV_SLOP_ROWS);
            kv_pair_bytes(elems, k_fmt, v_fmt)
        })
        .sum()
}

/// K+V byte footprint for ONE layer at `elems` per-side elements (rows × n_kv × head_dim), each
/// side in its own dtype. The pure per-layer core of [`kv_bytes_estimate_fmt`].
fn kv_pair_bytes(elems: usize, k_fmt: DType, v_fmt: DType) -> u64 {
    // Priced through the SAME sizer the runner allocates with, so "mirrors `kv_fmt_bytes`" is
    // now a call rather than a comment.
    (kv_fmt_bytes(k_fmt, elems) + kv_fmt_bytes(v_fmt, elems)) as u64
}

/// Config/env-level gate for SWA ring KV sizing, shared by the runner's allocation and the
/// KV-footprint ESTIMATES (ctx clamp, dense/MoE placement) so they price the same allocation the
/// runner will make. The runner additionally requires the backend capability
/// (`Capabilities::kv_swa_ring`) and the FINAL per-side KV formats; this checks the env-requested
/// formats (a format the runner gates back to f16 stays ring-capable, so the estimate is only
/// ever conservative). Gated OFF for:
///   - non-SWA models (no window — nothing to ring);
///   - DiffusionGemma (its canvas denoise attends a fixed bidirectional `[lo, kv_len)` range that
///     is NOT a per-query sliding window, so the ring's mask-already-excludes-it argument doesn't
///     hold there);
///   - non-f16/q8 KV formats (the low-bit block quants / bf16 / f32 / turbo read the cache
///     through a dequant-the-prefix prepass sized in positions, and their static-only writes
///     never learned the ring split — they keep full-context caches, documented scope gate);
///   - `kv.ring = false` (`INFR_NO_KV_RING=1`, A/B and escape hatch).
pub(crate) fn kv_ring_wanted(cfg: &Config, ec: &EngineConfig) -> bool {
    // Not-supplied = the f16 default = ring-capable; otherwise the requested format must PARSE to
    // f16 or q8 (a name the runner would not recognize either is not ring-capable — the
    // `specified && dtype.is_none()` case, §11 decision 8). The dtype comes from the ONE shared
    // spelling table (`budget::parse_kv_dtype`, now applied in the config's env layer), so adding
    // an alias cannot make this gate and the runner disagree.
    let fmt_ok = |specified: bool, dt: Option<DType>| {
        !specified || matches!(dt, Some(DType::F16 | DType::Q8_0))
    };
    cfg.swa_window > 0
        && !cfg.diffusion_gemma
        && ec.kv.ring
        && fmt_ok(ec.kv.type_k_specified, ec.kv.type_k)
        && fmt_ok(ec.kv.type_v_specified, ec.kv.type_v)
}

/// The smallest context a session is worth opening with. A window below this is useless for any
/// real prompt, so it is the line between "clamp the default context" and "this model cannot be
/// served on this device at all" (`SeamModel::clamp_default_ctx`'s refuse rung), and the floor
/// every derived per-slot / fractional window is held to.
pub(crate) const MIN_SESSION_CTX: usize = 1024;

/// Hard ceiling on [`kv_fit_ctx_for`]'s search. Reached only by a model whose KV bytes AND
/// activation reserve both PLATEAU with context — every attention layer sliding-window (so the
/// ring caps its rows) and head_dim 128 with no score tile. There the fit is bounded by nothing
/// physical, and a number is still needed; 4 Mi tokens is past any trained window in existence.
const CTX_FIT_SEARCH_CAP: usize = 1 << 22;

/// The EXACT largest context whose KV cache + activation reserve fit `available` bytes alongside
/// `weights` — the arithmetic half of [`SeamModel::kv_fit_ctx_fmt`], split out so it is testable
/// without a GPU (`kv_fit_*` in `seam_helper_tests`).
///
/// Two things make it exact where the old estimator was not:
///
///  - **KV bytes are the real allocation**, not a bytes-per-token rate divided into a budget:
///    [`kv_bytes_estimate_fmt`] prices each layer's K and V buffers through the same
///    `kv_fmt_bytes` sizer the runner hands `Backend::alloc`, including the SWA ring's row cap
///    and each side's own dtype. A `0.95` fudge factor used to stand in for the block-quant
///    rounding and the ring split that this now computes directly.
///  - **The reserve is priced at the chunk height placement will ACTUALLY settle on.** A context
///    is accepted when it fits at ANY height in [`ubatch_candidates`], because the dense
///    placement sweep walks that same ladder and will shrink the prefill chunk to keep the
///    session resident. Pricing only the default 1024-row chunk here made the KV-format decision
///    against an assumption placement then abandoned — which is exactly how gemma-3-12b talked
///    itself into an unnecessary q8 cache at ctx 131072.
///
/// Returns the RAW fit, which may be below [`MIN_SESSION_CTX`] (or `0`) — that is the signal the
/// refuse rung reads. `None` for a pure recurrent-state arch: no per-token KV to size.
///
/// MoE models keep the plain `total/12` heuristic: their placement budgets pager arenas
/// separately from this and never walks the dense chunk ladder.
pub(crate) fn kv_fit_ctx_for(
    cfg: &Config,
    ec: &EngineConfig,
    weights: u64,
    available: u64,
    vram_total: u64,
    k_fmt: DType,
    v_fmt: DType,
) -> Option<usize> {
    if (0..cfg.n_layer).all(|l| cfg.layer_n_kv(l) * cfg.layer_head_dim(l) == 0) {
        return None;
    }
    // Same gate the runner applies (`generate_dense_backend`'s `kv_ring`): the config/env gate
    // AND f16/q8 on both sides. A low-bit side keeps full-context caches, so pricing it as a ring
    // would hand out a context the allocation cannot honor.
    let ring = kv_ring_wanted(cfg, ec)
        && matches!(k_fmt, DType::F16 | DType::Q8_0)
        && matches!(v_fmt, DType::F16 | DType::Q8_0);
    let cands = ubatch_candidates(ec);
    let reserve = |ctx: usize, ubatch: usize| -> u64 {
        if cfg.moe.is_some() {
            (vram_total / 12).max(1024 * 1024 * 1024)
        } else {
            dense_act_reserve_at(cfg, ctx, ubatch)
        }
    };
    let fits = |ctx: usize| -> bool {
        cands.iter().any(|&ubatch| {
            weights
                .saturating_add(kv_bytes_estimate_fmt(cfg, ctx, ring, ubatch, k_fmt, v_fmt))
                .saturating_add(reserve(ctx, ubatch))
                <= available
        })
    };
    // Monotone in ctx (both terms grow with it), so double-then-bisect finds the exact boundary.
    if !fits(0) {
        return Some(0);
    }
    let (mut lo, mut hi) = (0usize, 1usize);
    while hi < CTX_FIT_SEARCH_CAP && fits(hi) {
        lo = hi;
        hi = (hi * 2).min(CTX_FIT_SEARCH_CAP);
    }
    if fits(hi) {
        return Some(hi); // plateaued: the cap IS the answer.
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// One resident-BDA / streamed-arena addressing unit's ELEMENT-count cap (the invariant's element
/// half; see `infr_vulkan`'s `BdaWeightArena` doc and `BDA_ADDRESSING_UNIT_MAX`). In-kernel element
/// indices are u32, so an addressing unit — one dense tensor, or one per-expert slice of a stacked
/// bank — must stay under 2^32 ELEMENTS. This is the binding limit for sub-byte / low-bpw quants
/// (Q2_K etc.), where 4 Gi elements is only ~1.3 GiB of bytes and so trips well before the 4 GiB
/// BYTE cap `bda_weight_alloc` enforces. Load-time guard: reject LOUDLY here (shape+dtype are known)
/// instead of letting a u32 index wrap into coherent-but-wrong reads in-shader. Enforced today by
/// model reality (no single tensor / expert slice is anywhere near 4 Gi elements); a model that
/// crossed it would need a wider in-kernel addressing scheme, not just a bigger allocation.
const BDA_ELEMENT_UNIT_MAX: usize = 1 << 32;

fn check_bda_element_cap(name: &str, unit: &str, elems: usize) -> AResult<()> {
    if elems as u64 >= BDA_ELEMENT_UNIT_MAX as u64 {
        return Err(anyhow!(
            "weight tensor {name}: {unit} has {elems} elements, at/above the u32 addressing unit \
             cap (2^32) — in-kernel element indices are u32 and would wrap; this needs a wider \
             addressing scheme, not a bigger allocation"
        ));
    }
    Ok(())
}

/// The dense weights whose ONLY GPU consumers are the chunk-covered dispatches (issue #77): the
/// output projection's decode GEMV and `Op::EmbedGather`, both of which split a `>= 2^32`-element
/// tensor into output-row chunks at dispatch time (`infr_vulkan`'s `dispatch_gemv_chunked` /
/// `embed_gather`, u64 per-row base). These may exceed the u32 ELEMENT cap; every OTHER dense
/// tensor keeps the loud whole-tensor `check_bda_element_cap`, and the 4 GiB BYTE cap in
/// `bda_weight_alloc` still bounds the single contiguous allocation for ALL of them (so an
/// over-cap table must be low-bpw enough to fit — a quantized frontier 256k-vocab lm_head).
/// `output.weight` = the untied lm_head; `token_embd.weight` = the input embedding (and the TIED
/// lm_head, read by both `Op::EmbedGather` and the lm_head `Op::Linear`);
/// `per_layer_token_embd.weight` = gemma4-E2B's per-layer gather table.
fn chunk_covered_dense_tensor(name: &str) -> bool {
    matches!(
        name,
        "output.weight" | "token_embd.weight" | "per_layer_token_embd.weight"
    )
}

/// Decide this model's MoE expert placement, install the pager session when the decision pages
/// (FIRST load only), and return the Vulkan weight binder that implements it. Shared by every
/// Vulkan weight-uploading session — [`generate_dense_vulkan_session`] and the DiffusionGemma
/// session (`model.rs`), which drives `generate_dense_backend` directly and would otherwise
/// silently skip placement (observed: `INFR_CACHE` was a no-op on DG).
///
/// For a non-MoE model (or a warm call — `first_load == false`) this degrades to the plain
/// pad-and-upload resident binder: placement is decided ONCE per weight upload, both because
/// only the first load ever calls the binder, and because the tier-3 budget math is only
/// consistent BEFORE the upload (see the double-count note inside).
pub(crate) fn vulkan_moe_binder<'a>(
    vk: &'a infr_vulkan::VulkanBackend,
    g: &'a Gguf,
    cfg: &'a Config,
    ec: &EngineConfig,
    first_load: bool,
    want_ctx: usize,
) -> AResult<Box<BindWeight<'a>>> {
    // ── MoE expert placement ─────────────────────────────────────────────────────────────────
    // The pager (`infr_vulkan::pager`) is the ONLY MoE offload mechanism — the legacy
    // host-visible (HostWeights/GTT) split and its INFR_NCMOE knob are gone. Tiers, in
    // precedence order:
    //   1. `INFR_CACHE=<size>` EXPLICIT override — force EVERY expert layer through the pager
    //      with that byte budget, regardless of whether the banks would fit resident. Lets a
    //      caller (or a test) force the paged path deterministically instead of depending on
    //      this box's free VRAM — see the `gpu_seam_paged_moe_matches_*` tests. The value is the
    //      shared size grammar (`infr_core::parse_size`): plain bytes, `k/m/g/t` 1024-suffixes
    //      (`INFR_CACHE=19g`), or a percentage of the device's AVAILABLE VRAM at first load
    //      (`INFR_CACHE=80%` — device-appropriate base: the cache lives in VRAM).
    //   2. Auto (unset): fully resident (the fast path, zero change) when the banks fit VRAM;
    //      otherwise the pager with budget = remaining VRAM after dense+KV+headroom.
    // Paging rides the adapter's paged executor split (`infr_vulkan::adapter::execute_static`'s
    // paged branch). FUSED gate_up banks (gemma-4 MoE / DiffusionGemma) page under `Role::Gate`
    // with a double-width slot, and MIXED-dtype roles (unsloth-dynamic quants bumping a subset
    // of layers' banks to a wider format) split into per-(role, slot_bytes) arena pools — see
    // `infr_vulkan::pager`'s MoE-session doc.
    let cache_override = ec.paging.cache;

    let mut n_paged = 0usize; // paged layer-count (0 = fully resident, or all = cfg.n_layer)
    let mut pager_budget_bytes = 0u64;
    // Placement is decided ONCE, on the session's FIRST load — the only call where `bind_weight`
    // runs (see the `state.is_none()` init block in `generate_dense_backend`) and the only moment
    // the tier-3 budget math is consistent: `vram.available` is LIVE (heapBudget − heapUsage), so
    // once this model's weights are resident a recompute would subtract `fp.dense` from an
    // `available` that ALREADY excludes it — double-counting the model against itself and
    // collapsing the budget (observed: a fully-resident 16.4 GB model "re-placed" as 5/30
    // resident on the warm second call of a bench). Warm calls leave `n_paged` at 0; nothing
    // consumes it (no binding, and the pager init below is first_load-gated anyway). A first
    // load racing ANOTHER resident model (swap mid-drain) still reads reduced `available` —
    // that's real pressure, deliberately not compensated; the alloc-time VRAM budget guard is
    // the backstop against over-commit.
    if first_load && cfg.moe.is_some() {
        // NB: the load-time expert-bank dtype gate that used to live here (field report:
        // MXFP4_MOE expert banks `expect`-panicked mid-inference before it existed) is GONE —
        // the id-indexed GEMV floor now covers every dtype a GGUF expert bank can hold (the
        // full dense native set plus F16/F32 for float banks), so
        // `infr_vulkan::linear::moe_expert_dtype_ok` is true for all of them; the invariant is
        // pinned by `moe_expert_floor_covers_dense_set` in infr-vulkan's linear.rs tests.
        match cache_override {
            Some(spec) => {
                n_paged = cfg.n_layer;
                // Percent resolves against AVAILABLE VRAM at this (first-load, pre-upload)
                // moment — the same consistent snapshot the auto tier uses below.
                pager_budget_bytes = spec.resolve(vk.vram().available);
            }
            None => {
                let fp = crate::weights::weight_footprint(g);
                let vram = vk.vram();
                // Per-layer rows: SWA layers ring at window+ubatch rows (see `kv_rows`), so a
                // mostly-SWA model's KV prices far below n_layer * ctx. +64 rows/layer slop.
                let ring = kv_ring_wanted(cfg, ec);
                // Route through the shared q8-aware helper (the dense placement path already does),
                // so a pinned auto-q8 KV cache is priced at ~half the bytes instead of the old
                // hard-coded f16 `*2*2` — which over-reserved ~2× and forced avoidable expert paging.
                let kv_bytes: u64 =
                    kv_bytes_estimate(cfg, want_ctx, ring, ubatch_rows(ec), kv_auto_q8());
                // 2 GiB: covers activation scratch (pooled, but per-tag sizes scale with n_embd/
                // n_ff and this budget calc's `fp`/`kv_bytes` are estimates, not the exact bytes
                // `alloc`/gpu-allocator's 256 MiB block granularity ends up committing) plus the
                // pager's own arena+staging+LUT allocations, which aren't counted in `fp` at all
                // (found empirically sizing Scout's 48-layer, 37 GB Q2_K pager placement — 512 MiB
                // undershot by a few hundred MiB and the guard rightly refused to over-commit).
                const ACT_HEADROOM: u64 = 2 * 1024 * 1024 * 1024;
                // Mixed oversize (an MoE model whose DENSE part alone doesn't fit) is out of
                // scope for dense layer streaming — fail with a clear message instead of letting
                // a degenerate expert budget stumble into the alloc-time VRAM guard's generic
                // over-commit error.
                if vram.available < fp.dense + kv_bytes {
                    return Err(anyhow!(
                        "this MoE model's dense weights ({:.2} GB) + KV cache ({:.2} GB) exceed \
                         available VRAM ({:.2} GB) — dense layer streaming does not cover MoE \
                         models' dense parts; reduce ctx or run on the CPU backend (INFR_DEV=cpu)",
                        fp.dense as f64 / 1e9,
                        kv_bytes as f64 / 1e9,
                        vram.available as f64 / 1e9,
                    ));
                }
                let budget = vram
                    .available
                    .saturating_sub(fp.dense + kv_bytes + ACT_HEADROOM);
                if budget < fp.expert {
                    // Page EVERY expert layer with the WHOLE budget (tier-2 semantics), NOT
                    // "keep the first gpu_layers banks resident and page the overflow with
                    // the leftover": the leftover degenerates to a few slots and the paged
                    // layers thrash at a 0% hit rate (measured, Scout Q2_K on 24GB:
                    // resident-26-layers + 3-slot pager decoded at 3.3 t/s; all-paged with
                    // the same total VRAM = 307 slots/role, 66% hits, 6.2 t/s). A shared LRU
                    // arena keeps the hot experts of EVERY layer resident — strictly more
                    // flexible than pinning whole layer banks.
                    n_paged = cfg.n_layer;
                    pager_budget_bytes = budget;
                }
            }
        }
    }
    // The layer index of a `blk.{l}.…_exps…` tensor name.
    let exps_layer = |name: &str| -> Option<usize> {
        if !name.contains("_exps") {
            return None;
        }
        name.strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|l| l.parse::<usize>().ok())
    };
    // A FUSED gate_up bank pages under `Role::Gate` (one double-width slot per expert; the model
    // then has no `Role::Up` sources at all) — see `infr_vulkan::pager`'s MoE-session doc.
    let moe_role_of = |name: &str| -> Option<infr_vulkan::pager::Role> {
        use infr_vulkan::pager::Role;
        if name.ends_with("ffn_gate_exps.weight") || name.ends_with("ffn_gate_up_exps.weight") {
            Some(Role::Gate)
        } else if name.ends_with("ffn_up_exps.weight") {
            Some(Role::Up)
        } else if name.ends_with("ffn_down_exps.weight") {
            Some(Role::Down)
        } else {
            None
        }
    };
    // The session must exist (and answer `Backend::moe_paged` truthy) BEFORE `generate_dense_backend`
    // below uploads a single weight: the FIRST paged tensor the `bind_weight` closure sees still
    // has to bind a placeholder the adapter recognizes as paged the very first time a graph
    // executes — sizing (and installing) it AFTER the call, once every weight was already bound
    // to a 4-byte placeholder nobody registered, would leave `execute_static` reading that
    // placeholder as if it were the full bank (see `pager::MoePagerLayout`'s doc). Only on the
    // FIRST load of a session (`bind_weight` isn't called again once `state` already holds
    // uploaded weights, so a second `init_moe_pager` would wipe an already-warm cache for nothing;
    // `n_paged > 0` already implies `first_load` — the placement calc above is first_load-gated —
    // but keep the guard explicit).
    if first_load && n_paged > 0 {
        use infr_vulkan::pager::Role;
        let moe = cfg.moe.as_ref().expect("n_paged > 0 implies MoE");
        let n_expert = moe.n_expert.max(1);
        // Enumerate every paged `_exps` weight bank's (role, per-expert bytes): one arena POOL
        // per distinct pair. A uniform split-bank model (Scout) yields the classic three pools;
        // a fused-bank model (gemma-4 MoE / DiffusionGemma) yields a double-width Gate pool and
        // no Up pool; a mixed-dtype role (UD quants) yields one pool per byte size, each holding
        // exactly the layers whose banks match it (`MoePagerSession::register` re-derives the
        // same key from the tensor bytes). `(slot_bytes, blocks-in-pool)` per pool.
        let mut pool_blocks: Vec<(Role, usize, usize)> = Vec::new();
        for t in g.tensors() {
            if exps_layer(&t.name).is_none_or(|l| l >= n_paged) {
                continue; // not a paged layer's `_exps` tensor
            }
            let Some(role) = moe_role_of(&t.name) else {
                continue; // `_exps` but not a weight bank (e.g. the per-expert `.scale` vector)
            };
            let sb = (t.nbytes / n_expert).max(4);
            match pool_blocks
                .iter_mut()
                .find(|(r, s, _)| *r == role && *s == sb)
            {
                Some((_, _, n)) => *n += n_expert,
                None => pool_blocks.push((role, sb, n_expert)),
            }
        }
        if pool_blocks.is_empty() {
            // Defensive: an MoE config with NO pageable `_exps` weight banks (no arch this crate
            // loads ships that). Nothing to page — stay fully resident and let the alloc-time
            // VRAM budget guard produce its clear error if that overflows. `n_paged = 0` also
            // turns the binder's paged divert below into a no-op (it re-checks `n_paged`).
            eprintln!(
                "MoE pager: no pageable `_exps` weight banks found — keeping every expert \
                 resident (the VRAM budget guard is the backstop)"
            );
            n_paged = 0;
        }
        if n_paged > 0 {
            let n_blocks = n_paged * n_expert;
            // The session's pinned upload ring (two fence-rotated halves — see
            // `MoePagerSession`'s `ring` doc) lives in the same VRAM the arenas do: subtract it
            // from the budget BEFORE splitting arena shares so the paged footprint stays within
            // what the caller granted (INFR_CACHE) / what the auto tier measured as free.
            let ring_bytes = infr_core::pager::ring_bytes(pager_budget_bytes, vk.cfg().paging.ring);
            pager_budget_bytes = pager_budget_bytes.saturating_sub(ring_bytes as u64);
            let total_bytes: u64 = pool_blocks
                .iter()
                .map(|&(_, sb, nb)| (sb * nb) as u64)
                .sum::<u64>()
                .max(1);
            // No per-pool arena ceiling: each MoE pool is a `bufferDeviceAddress` buffer read by
            // pointer, so it is NOT capped by one SSBO binding's maxStorageBufferRange (~4 GiB on
            // RADV) the way it was when the arena was a bound SSBO. A pool now holds as many experts
            // as its budget share allows — the whole point of the u64 addressing lift. The only
            // backstop is the alloc-time VRAM budget guard (`GpuPager::new` -> `alloc_arena_bda`);
            // the proportional split below never over-subscribes VRAM because it partitions
            // `pager_budget_bytes`, which the caller derived from the remaining VRAM.
            let pools: Vec<infr_vulkan::pager::MoePoolSpec> = pool_blocks
                .iter()
                .map(|&(role, sb, nb)| {
                    // Budget split PROPORTIONALLY to each pool's total bank bytes — the byte share is
                    // also the access share under uniform routing (every (layer, expert) read touches
                    // gate+up+down alike), so proportional slots equalize expected hit rates across
                    // pools; any fancier split would need routing statistics that don't exist at
                    // load time.
                    let share = (pager_budget_bytes as u128 * (sb * nb) as u128
                        / total_bytes as u128) as u64;
                    // Floor at `min(n_expert, nb)`: a chunked batched-prefill `Op::MoeFfn` (rows>1)
                    // runs ALL of a layer's routed buckets in ONE dispatch
                    // (`matmul_mmq_experts_paged`), touching up to `n_expert` DISTINCT experts of
                    // that layer that must be simultaneously resident (the within-batch safety
                    // invariant — see `infr_core::pager::Pager::new`'s doc). Decode's rows=1 needs
                    // only `n_used`, but the batched bound subsumes it and `n_expert` slots is tiny
                    // next to any real budget (Scout: 16 x ~18 MB per role). Capped at `nb` (no
                    // point holding more slots than the pool has distinct experts).
                    let floor = n_expert.min(nb).max(1);
                    let budget_slots = ((share / sb as u64) as usize).clamp(floor, nb);
                    infr_vulkan::pager::MoePoolSpec {
                        role,
                        slot_bytes: sb,
                        n_slots: budget_slots,
                    }
                })
                .collect();
            let cached: usize = pools.iter().map(|p| p.n_slots).sum();
            let pool_desc: Vec<String> = pool_blocks
                .iter()
                .zip(&pools)
                .map(|(&(role, sb, nb), p)| {
                    format!("{role:?}[{:.1}MB] {}/{}", sb as f64 / 1e6, p.n_slots, nb)
                })
                .collect();
            eprintln!(
                "MoE pager: {n_paged}/{} expert layers PAGED ({cached} expert blocks cached — {}; \
             {:.2} GB budget; ctx={want_ctx})",
                cfg.n_layer,
                pool_desc.join(", "),
                pager_budget_bytes as f64 / 1e9,
            );
            vk.init_moe_pager(infr_vulkan::pager::MoePagerLayout {
                n_blocks,
                pools,
                ring_bytes,
            })
            .map_err(|e| anyhow!("{e}"))?;
        }
    }

    // ── Dense layer streaming placement ──────────────────────────────────────────────────────
    // The DENSE twin of the MoE tiers above (`infr_vulkan::pager::DensePagerSession`): when a
    // dense model's per-layer weights (minus what must stay resident) exceed the budget, stream
    // them through per-(dtype, stride) arena pools driven by the exact cyclic-sweep policy
    // (`infr_core::pager::Pager::schedule`). One block = one weight GROUP exactly as `wload`
    // uploads it (fused qkv / gate_up concats are one block — the shared `fuse_*_decision`
    // helpers keep this enumeration and the runner's upload order from drifting). Embeddings,
    // lm_head, norms and biases stay resident: norms/biases are consumed by ops without weight
    // offsets and are tiny; token_embd/lm_head are read at every token edge, so streaming them
    // adds their full bytes to every token's PCIe bill with zero locality to exploit.
    //
    //   1. `INFR_CACHE=<size>` on a DENSE model — force EVERY streamable block through the
    //      streamer with that byte budget (deterministic test hook, same grammar as the MoE tier).
    //   2. Auto (unset): TRY RESIDENT FIRST — fully resident (the fast path, zero change) when
    //      weights + KV + the honest dense activation reserve (`dense_act_reserve_at`) fit live
    //      VRAM; otherwise stream with budget = remaining VRAM after resident-weights+KV+reserve.
    //      An explicit oversized INFR_CTX whose KV can't sit beside resident weights falls back
    //      to streaming the same way (never clamped here — the ctx the caller asked for is kept).
    // Streamable = the per-layer Linear projection groups whose dtype has offset-capable native
    // kernels (`native_dense_supported`, F16/F32 excluded — `matmul_proj`/`linear_f32` take no
    // weight offset) and whose bytes upload unmodified from the mmap (the qwen2 NEOX q/k row
    // permute rewrites bytes at load, so those tensors stay resident).
    let mut dense_plan: std::collections::HashMap<String, (usize, u32, Vec<String>)> =
        std::collections::HashMap::new();
    if first_load && cfg.moe.is_none() {
        let fuse_gu = runner::fuse_gu_decision(vk.capabilities().combined_gu, g, cfg);
        let fuse_qkv = runner::fuse_qkv_decision(vk.capabilities().combined_gu, g, cfg, ec);
        // Candidate groups in LAYER ORDER — the cyclic-sweep schedule key. Key = names[0] (what
        // `bind_weight` receives for the group).
        let mut groups: Vec<Vec<String>> = Vec::new();
        for l in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            if !cfg.permute_qk_neox {
                if fuse_qkv {
                    groups.push(vec![
                        p("attn_q.weight"),
                        p("attn_k.weight"),
                        p("attn_v.weight"),
                    ]);
                } else {
                    groups.push(vec![p("attn_q.weight")]);
                    groups.push(vec![p("attn_k.weight")]);
                    groups.push(vec![p("attn_v.weight")]);
                }
            } else if !fuse_qkv {
                // Permuted q/k stay resident (their upload bytes are load-time rewrites of the
                // mmap); v uploads raw and can still stream.
                groups.push(vec![p("attn_v.weight")]);
            }
            groups.push(vec![p("attn_output.weight")]);
            if fuse_gu {
                groups.push(vec![p("ffn_gate.weight"), p("ffn_up.weight")]);
            } else {
                groups.push(vec![p("ffn_gate.weight")]);
                groups.push(vec![p("ffn_up.weight")]);
            }
            groups.push(vec![p("ffn_down.weight")]);
        }
        let tinfo = |n: &str| g.tensors().iter().find(|t| t.name == n);
        // Eligible groups with their (dtype, raw bytes, numel) — a group whose tensors are
        // missing (DeltaNet layers, gemma4 V-less layers) or whose dtype lacks offset-capable
        // kernels simply stays resident.
        let eligible: Vec<(Vec<String>, infr_core::DType, usize, usize)> = groups
            .into_iter()
            .filter_map(|comps| {
                let infos: Vec<_> = comps.iter().map(|n| tinfo(n)).collect::<Option<_>>()?;
                let dt = infos[0].dtype;
                if !infos.iter().all(|t| t.dtype == dt)
                    || !infr_vulkan::linear::native_dense_supported(dt)
                {
                    return None;
                }
                let raw: usize = infos.iter().map(|t| t.nbytes).sum();
                let numel: usize = infos
                    .iter()
                    .map(|t| t.shape.iter().product::<usize>())
                    .sum();
                Some((comps, dt, raw, numel))
            })
            .collect();
        let streamable_resident: u64 = eligible
            .iter()
            .map(|(_, dt, raw, numel)| crate::weights::tensor_resident_bytes(*dt, *numel, *raw))
            .sum();
        let fp = crate::weights::weight_footprint(g);
        let vram = vk.vram();
        // Per-layer rows: SWA layers ring at window+ubatch rows (see `kv_rows`) — this is what
        // lets a mostly-SWA model (gemma-4-31B: 50/60 layers SWA) price its KV small enough to
        // take the try-resident tier at real contexts instead of streaming. +64 rows/layer slop.
        let ring = kv_ring_wanted(cfg, ec);
        // KV bytes at an EXPLICIT chunk height and side format — the ONE pricing helper every
        // decision below shares (try-resident, the chunk sweeps, the auto-q8 rung, the streaming
        // budget), so they all price exactly the allocation the runner will make. `q8` prices
        // BOTH sides Q8_0 (34 bytes / 32 elems, mirroring `kv_fmt_bytes`); false = f16 (2 B/elem).
        let kv_total_at =
            |ubatch: usize, q8: bool| -> u64 { kv_bytes_estimate(cfg, want_ctx, ring, ubatch, q8) };
        // Does weights + KV + the honest activation reserve fit live VRAM at this (chunk, fmt)?
        let fits = |ubatch: usize, q8: bool| {
            fp.total() + kv_total_at(ubatch, q8) + dense_act_reserve_at(cfg, want_ctx, ubatch)
                <= vram.available
        };
        // Try-resident-first: a dense model goes FULLY RESIDENT (the exact pre-streaming fast
        // path) whenever weights + this session's KV + an HONEST dense activation estimate fit
        // the live free VRAM; only a genuine miss streams. The MoE tier's 2 GiB ACT_HEADROOM is
        // sized for pager arenas/staging that a dense-resident session doesn't have — reusing it
        // here streamed gemma-4-31B (21.9 GB weights on a 24 GB card, decode 33 t/s resident vs
        // ~3 t/s streamed at the PCIe ceiling). If residency is chosen but a later activation
        // alloc still misses (fragmentation, another process grabbing VRAM), the alloc-time VRAM
        // guard fails that request cleanly — INFR_CACHE=<size> is the escape hatch that forces
        // streaming. `kv_auto_q8()` may already be pinned by the default-ctx clamp path (see
        // `SeamModel::clamp_default_ctx`) — then every check here prices the q8 cache the runner
        // will actually allocate.
        //
        // Residency sweep: when the FULL-chunk reserve is what tips a big model into streaming,
        // try smaller prefill chunks — a smaller chunk shrinks BOTH the activation reserve
        // (whole-chunk logits/gate_up scratch scale with rows) and the SWA ring rows
        // (window + chunk). Resident-with-a-512-row-chunk decodes ~10x faster than streaming at
        // the PCIe ceiling (gemma-4-31B @ d4096: 27.6 vs 2.9 t/s), so trading prefill chunk
        // height for residency is strictly the right call. Pinned per-session (`PlacementPins`)
        // so the prefill loop and the runner's ring sizing use exactly the priced height; an
        // explicit INFR_UBATCH disables the sweep (the user's height is authoritative). Runs
        // BEFORE the auto-q8 rung below: a shorter prefill chunk costs only some prefill
        // throughput, while q8 KV costs ~10-16% GQA decode — prefer the cheaper concession.
        let mut resident = fits(ubatch_rows(ec), kv_auto_q8());
        if !resident && !user_pinned_ubatch(ec) && cache_override.is_none() {
            // Shared ladder (`ubatch_candidates`) minus its first rung, which is the height
            // `resident` above already priced.
            for cand in ubatch_candidates(ec).into_iter().skip(1) {
                if fits(cand, kv_auto_q8()) {
                    pin_ubatch(cand);
                    // Re-read through the pin (a racing earlier set wins — use whatever stuck).
                    if fits(ubatch_rows(ec), kv_auto_q8()) {
                        eprintln!(
                            "dense placement: resident with a {}-row prefill chunk (the default \
                             1024-row chunk's activation reserve wouldn't fit); set INFR_UBATCH \
                             to override",
                            ubatch_rows(ec).min(want_ctx),
                        );
                        resident = true;
                    }
                    break;
                }
            }
        }
        // ── auto-q8 KV rung (the placement-ladder step between the SWA ring and streaming):
        // f16 KV missed residency at every chunk height, but a Q8_0 cache (roughly HALF the KV
        // bytes) might fit — placing RESIDENT with q8 KV beats the remaining rungs by an order
        // of magnitude (streaming decodes at the PCIe ceiling; the explicit-ctx path never
        // clamps). Gates: the user set NO KV format (see `PlacementPins`'s policy doc — explicit
        // settings always win, both sides go q8_0, never below q8), no INFR_CACHE override
        // (that's the deterministic force-streaming hook), and the runner's own q8 layout gate
        // (32-elem block alignment; this binder is the Vulkan path, a native q8-KV backend —
        // decode reads Q8 natively and coupled K==V==Q8 keeps record-once replay; batched
        // prefill reads it through the dequant prepass, which the SWA ring kept for q8).
        // Tries the pinned/current chunk height first, then the same smaller-chunk ladder as
        // the f16 sweep (floor 128).
        if !resident
            && cache_override.is_none()
            && !kv_auto_q8()
            && kv_unset(ec)
            && kv_q8_layout_ok(cfg)
        {
            for cand in ubatch_candidates(ec) {
                if fits(cand, true) {
                    pin_kv_auto_q8();
                    if cand != ubatch_rows(ec) {
                        pin_ubatch(cand);
                    }
                    // Re-read through the pins (racing earlier sets win — use whatever stuck).
                    if kv_auto_q8() && fits(ubatch_rows(ec), true) {
                        tracing::warn!(
                            requested_ctx = want_ctx,
                            prefill_chunk = ubatch_rows(ec),
                            kv_dtype = "q8_0",
                            "kv auto-quant: q8_0 KV cache — an f16 cache would not fit resident \
                             at ctx={want_ctx} at any prefill chunk height; set \
                             INFR_KV_TYPE_K/V=f16 to force f16 (decode is ~10-16% slower on q8)"
                        );
                        resident = true;
                    }
                    break;
                }
            }
        }
        // Slot stride: the group's raw bytes padded to a whole number of quant blocks AND
        // u32 words, so every slot base is block-aligned (the kernels' element-offset weight
        // addressing needs `slot_byte_base = whole blocks`) and the arena binds as
        // `array<u32>`. Hoisted above the budget decision so the streaming-chunk sweep below
        // can price the full-arena need with the exact strides the pools use.
        let lcm = |a: usize, b: usize| {
            let gcd = {
                let (mut x, mut y) = (a, b);
                while y != 0 {
                    (x, y) = (y, x % y);
                }
                x
            };
            a / gcd * b
        };
        let stride_of = |dt: infr_core::DType, raw: usize| {
            raw.next_multiple_of(lcm(infr_gguf::block_layout(dt).1, 4))
        };
        let budget = match cache_override {
            Some(spec) => Some(spec.resolve(vram.available)),
            None if resident => None,
            None => {
                // Streaming is inevitable. Edge-aware chunk sweep — the STREAMING twin of the
                // residency sweep above: a smaller prefill chunk shrinks the activation reserve
                // and the SWA ring rows, and every byte freed is a byte of streaming budget →
                // more resident slots, fewer PCIe refetches per weight sweep. But a taller
                // chunk prefills faster (fewer whole-model weight sweeps per prompt), so don't
                // shrink past the point of gain: pick the TALLEST chunk whose budget already
                // holds EVERY streamable block resident (extra budget past that buys nothing);
                // if no chunk reaches that, take the floor — 128 rows, the maximum-budget
                // choice. An explicit INFR_UBATCH is authoritative and skips the sweep; the
                // INFR_CACHE tier above is untouched (its budget is the caller's, not derived
                // from the reserve). Pinned via `PlacementPins` like the residency sweep, so the
                // prefill loop, the runner's ring sizing, and this budget all agree.
                let q8 = kv_auto_q8();
                let base = fp.total() - streamable_resident;
                let budget_at = |ub: usize| {
                    vram.available.saturating_sub(
                        base + kv_total_at(ub, q8) + dense_act_reserve_at(cfg, want_ctx, ub),
                    )
                };
                if !user_pinned_ubatch(ec) && !eligible.is_empty() {
                    let need: u64 = eligible
                        .iter()
                        .map(|(_, dt, raw, _)| stride_of(*dt, *raw) as u64)
                        .sum();
                    // "Covers": the budget minus its own upload-ring share holds every block.
                    let covers = |b: u64| {
                        b.saturating_sub(
                            infr_core::pager::ring_bytes(b, vk.cfg().paging.ring) as u64
                        ) >= need
                    };
                    let ub_now = ubatch_rows(ec);
                    let cands = ubatch_candidates(ec);
                    let pick = cands
                        .iter()
                        .copied()
                        .find(|&c| covers(budget_at(c)))
                        .unwrap_or(*cands.last().expect("cands is never empty"));
                    if pick != ub_now {
                        pin_ubatch(pick);
                    }
                }
                Some(budget_at(ubatch_rows(ec)))
            }
        };
        if let (Some(mut budget), false) = (budget, eligible.is_empty()) {
            // Pools keyed by (dtype, stride); blocks assigned ids in layer order per pool.
            let mut pools: Vec<(infr_core::DType, usize, usize)> = Vec::new(); // (dt, stride, n_blocks)
            let mut planned: Vec<(Vec<String>, usize, u32)> = Vec::new(); // (comps, pool, block_id)
            for (comps, dt, raw, _numel) in &eligible {
                let stride = stride_of(*dt, *raw);
                let pool = match pools.iter().position(|&(d, s, _)| d == *dt && s == stride) {
                    Some(i) => i,
                    None => {
                        pools.push((*dt, stride, 0));
                        pools.len() - 1
                    }
                };
                let block_id = pools[pool].2 as u32;
                pools[pool].2 += 1;
                planned.push((comps.clone(), pool, block_id));
            }
            // The pinned upload ring lives in the same VRAM the arenas do — subtract it first.
            let ring_bytes = infr_core::pager::ring_bytes(budget, vk.cfg().paging.ring);
            budget = budget.saturating_sub(ring_bytes as u64);
            let total_bytes: u64 = pools
                .iter()
                .map(|&(_, s, nb)| (s * nb) as u64)
                .sum::<u64>()
                .max(1);
            let specs: Vec<infr_vulkan::pager::DensePoolSpec> = pools
                .iter()
                .map(|&(_, stride, nb)| {
                    // Proportional budget split (byte share == access share: every block is read
                    // exactly once per sweep). Floor 2 slots so the next block's upload can
                    // overlap the previous block's dispatch instead of serializing on one slot.
                    // `n_slots` is bounded ONLY by the VRAM budget share and this floor: the pool
                    // arena is a `bufferDeviceAddress` buffer read purely by 64-bit pointer (see
                    // `DensePagerSession`), NEVER bound as a descriptor, so BOTH pre-BDA caps are
                    // gone — no `maxStorageBufferRange` binding ceiling and no u32 element-reach
                    // limit. A single pool may span well past 4 GiB (matching the paged-MoE arena,
                    // e.g. Scout's 6.57 GB role pools).
                    let share =
                        (budget as u128 * (stride * nb) as u128 / total_bytes as u128) as u64;
                    let floor = 2.min(nb).max(1);
                    let budget_slots = ((share / stride as u64) as usize).clamp(floor, nb);
                    infr_vulkan::pager::DensePoolSpec {
                        slot_bytes: stride,
                        n_slots: budget_slots,
                        n_blocks: nb,
                    }
                })
                .collect();
            let alloc: u64 = specs
                .iter()
                .map(|s| (s.n_slots * s.slot_bytes) as u64)
                .sum();
            if alloc > budget.max(1) && cache_override.is_none() {
                // Auto tier only: the floors overran what's actually free — streaming can't help.
                return Err(anyhow!(
                    "dense weights exceed VRAM and the leftover budget ({:.2} GB) can't hold \
                     even the streaming floor ({:.2} GB) — reduce ctx or run on the CPU backend \
                     (INFR_DEV=cpu)",
                    budget as f64 / 1e9,
                    alloc as f64 / 1e9,
                ));
            }
            let cached: usize = specs.iter().map(|s| s.n_slots).sum();
            let n_blocks: usize = specs.iter().map(|s| s.n_blocks).sum();
            eprintln!(
                "dense streaming: {n_blocks} weight blocks across {} pools, {cached} slots \
                 cached ({:.2} GB arena + {:.2} GB ring; budget {:.2} GB; ctx={want_ctx}; \
                 chunk={})",
                specs.len(),
                alloc as f64 / 1e9,
                ring_bytes as f64 / 1e9,
                (budget + ring_bytes as u64) as f64 / 1e9,
                ubatch_rows(ec).min(want_ctx),
            );
            vk.init_dense_pager(infr_vulkan::pager::DensePagerLayout {
                pools: specs,
                ring_bytes,
            })
            .map_err(|e| anyhow!("{e}"))?;
            for (comps, pool, block_id) in planned {
                dense_plan.insert(comps[0].clone(), (pool, block_id, comps));
            }
        }
    }

    Ok(Box::new(move |name, tb, dt, numel| {
        // Raw upload for EVERY dtype — the file's bytes go straight to VRAM (u32-padded) and the
        // kernel reads/dequants the native dtype in-shader. F16 → f16 coopmat GEMM / f16 GEMV;
        // F32 stays native (rmsnorm/qk_norm_rope read f32); bf16 → in-shader expand (bf16 is the
        // top 16 bits of an f32, EXACT; the warp GEMM narrows to f16 for the matrix cores like
        // every other format); quant weights → raw blocks. No host dtype conversion on any path.
        //
        // Paged: register this layer's mmap bytes with the pager and bind a tiny
        // placeholder instead of uploading the full bank — the Vulkan adapter recognizes the
        // placeholder's identity (see `infr_vulkan::pager`'s module doc) and diverts to the
        // paged executor split at execute time. `down_scale`/router/every other tensor of a
        // paged layer is unaffected — only the `_exps` weight banks divert here.
        // Dense layer streaming: bind a tiny placeholder and register the group's ZERO-COPY mmap
        // segments with the dense session instead of uploading (the adapter recognizes the
        // placeholder's identity at execute time and dispatches against the pool arena — see
        // `infr_vulkan::pager`'s dense-session doc). The `tb` byte-length check is the drift
        // guard between this plan's group enumeration and the runner's actual upload grouping
        // (`fuse_*_decision` keeps them aligned; a mismatch here is a bug, caught loudly).
        if let Some((pool, block_id, comps)) = dense_plan.get(name) {
            let segments: Vec<std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>> = comps
                .iter()
                .map(|c| {
                    Ok(
                        std::sync::Arc::new(g.tensor_bytes_arc(c).map_err(|e| anyhow!("{e}"))?)
                            as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>,
                    )
                })
                .collect::<AResult<_>>()?;
            let seg_total: usize = segments.iter().map(|s| s.as_ref().as_ref().len()).sum();
            if seg_total != tb.len() {
                return Err(anyhow!(
                    "dense streaming plan out of sync with the upload order for {name}: plan \
                     bytes {seg_total} != uploaded bytes {}",
                    tb.len()
                ));
            }
            let placeholder = vk
                .alloc_uninit(4, BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            let buf_id = infr_vulkan::pager::buffer_identity(placeholder.as_ref());
            vk.register_dense_stream(
                *pool,
                buf_id,
                infr_vulkan::pager::DenseSource {
                    segments,
                    block_id: *block_id,
                },
            )
            .map_err(|e| anyhow!("{e}"))?;
            return Ok((placeholder, dt));
        }
        if let Some(l) = exps_layer(name).filter(|&l| l < n_paged) {
            if let (WBytes::Mmap(bytes), Some(role)) = (&tb, moe_role_of(name)) {
                let n_expert = cfg
                    .moe
                    .as_ref()
                    .expect("a paged tensor implies an MoE config")
                    .n_expert
                    .max(1);
                // The ADDRESSING UNIT of a stacked bank is ONE per-expert slice, not the whole
                // bank (the arena kernels index `arena + expert * stride`), so the whole-bank
                // element count may legitimately exceed 2^32 while each slice must not.
                check_bda_element_cap(name, "per-expert slice", numel / n_expert)?;
                let stride_bytes = bytes.len() / n_expert;
                let placeholder = vk
                    .alloc_uninit(4, BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                let buf_id = infr_vulkan::pager::buffer_identity(placeholder.as_ref());
                let source = infr_vulkan::pager::ExpertSource {
                    bytes: std::sync::Arc::new(bytes.clone())
                        as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>,
                    stride_bytes,
                    layer_base: (l * n_expert) as u32,
                };
                vk.register_paged_expert(role, buf_id, source)
                    .map_err(|e| anyhow!("{e}"))?;
                return Ok((placeholder, dt));
            }
        }
        // Ordinary (non-paged, non-streamed) weight. A RESIDENT stacked expert bank addresses ONE
        // per-expert slice at a time (the arena kernels index `arena + expert * stride`), exactly
        // like the paged path above — so its whole-bank element count may legitimately exceed 2^32
        // while each per-expert slice must not. This used to take the conservative WHOLE-tensor cap
        // because the flag-off resident id kernels did whole-bank u32 element math; those kernels
        // are gone (weights are u64 BDA-addressed only), so the slice cap is now correct. Every
        // non-expert weight is still one whole-tensor addressing unit.
        if moe_role_of(name).is_some() {
            let n_expert = cfg
                .moe
                .as_ref()
                .expect("an expert-bank tensor implies an MoE config")
                .n_expert
                .max(1);
            check_bda_element_cap(name, "per-expert slice", numel / n_expert)?;
        } else if chunk_covered_dense_tensor(name) {
            // lm_head / embedding tables (issue #77): read ONLY by chunk-covered dispatches (the
            // output projection's decode GEMV + Op::EmbedGather), which split a >= 2^32-element
            // tensor into output-row chunks at DISPATCH, so the whole-tensor element cap no longer
            // applies. `bda_weight_alloc`'s 4 GiB BYTE cap still bounds the single contiguous
            // allocation (this over-cap table must fit — a quantized 256k-vocab lm_head), and a
            // multi-row lm_head GEMM (MTP verify / all-position logits) is still caught loudly at
            // dispatch (the adapter's tiled-GEMM breach guard). Every OTHER dense tensor below
            // keeps the loud whole-tensor element cap.
        } else {
            check_bda_element_cap(name, "tensor", numel)?;
        }
        let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
        // alloc_uninit: the `upload` right below writes the buffer's FULL extent (it is sized to
        // exactly `padded.len()`), so the calloc contract's zero-fill is dead work — and an
        // expensive kind: on the device-local path it costs a `vkCmdFillBuffer` over the whole
        // model plus a submit + `queue_wait_idle` PER TENSOR, doubling the load's stall count.
        let buf = vk
            .alloc_uninit(padded.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        vk.upload(buf.as_ref(), &padded)
            .map_err(|e| anyhow!("{e}"))?;
        Ok((buf, dt))
    }))
}

/// Open one Vulkan backend per physical device index (the shared front of every multi-GPU
/// `generate_*` wrapper). Errors name the failing `VulkanN`.
///
/// Every backend gets the SAME `Arc<Config>` — `docs/config-plan.md` §5.1: per-device configs are
/// explicitly out of scope for the config campaign, and today's multi-GPU paths all read one
/// process environment, so one shared handle reproduces that exactly.
fn open_vulkan_devices(
    devices: &[usize],
    ec: &EngineConfig,
) -> AResult<Vec<infr_vulkan::VulkanBackend>> {
    let cfg = std::sync::Arc::new(ec.clone());
    devices
        .iter()
        .map(|&idx| {
            infr_vulkan::VulkanBackend::new_on_with(idx, cfg.clone())
                .map_err(|e| anyhow!("vulkan init (Vulkan{idx}): {e}"))
        })
        .collect()
}

/// The arch guards the DENSE multi-GPU paths (pipeline, tensor-parallel) share: they cover
/// dense-attention Qwen3/Llama/Gemma only, rejecting MoE / qwen35 DeltaNet / gemma-E2B /
/// diffusion-gemma with a clear per-flag message (`label` = the env var). Expert parallelism has
/// its own guard (it REQUIRES MoE).
fn dense_multi_gpu_guard(cfg: &Config, label: &str) -> AResult<()> {
    if cfg.moe.is_some() {
        return Err(anyhow!(
            "{label} supports dense models only; this is an MoE model — use INFR_EXPERT_PARALLEL"
        ));
    }
    if cfg.qwen35 {
        return Err(anyhow!(
            "{label} does not support qwen35 (DeltaNet recurrent-state placement is a separate slice)"
        ));
    }
    if cfg.n_embd_per_layer > 0 {
        return Err(anyhow!(
            "{label} does not support gemma E2B per-layer embeddings"
        ));
    }
    if cfg.diffusion_gemma {
        return Err(anyhow!("{label} does not support diffusion-gemma"));
    }
    Ok(())
}

/// Drive `generate_dense_backend` as a ONE-SHOT (single conversation, no slot pool): fresh `None`
/// state, `want_ctx = prompt + max_new + 1`, and the eight trailing hooks (constraint / verify /
/// verify_ids / logits_out / h_out / denoise_req / req) all unused. The shared tail of the
/// multi-GPU `generate_*` wrappers.
fn run_dense_oneshot(
    be: &dyn Backend,
    bind: &BindWeight<'_>,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_backend(
        be,
        bind,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        &mut None,
        prompt.len() + max_new + 1,
        None, // constraint
        None, // verify
        None, // verify_ids
        None, // logits_out
        None, // h_out
        None, // denoise_req
        None, // req
    )
}

/// The `multi.pipeline` device list (`INFR_PIPELINE=Vulkan0,Vulkan1,…`), or `None` for
/// single-device. Needs >=2 devices for a layer split; garbage or too-few errors LOUDLY — but now
/// at `Config::load`, not here (see `docs/config-plan.md`'s S1 note), since the grammar and the
/// minimum both moved into `infr_core::config::parse_device_spec`. The `parse_device_spec` /
/// `parse_device_list` pair this crate used to own is DELETED — it was a second copy of that
/// grammar (§6.11).
pub fn pipeline_devices(ec: &EngineConfig) -> Option<&[usize]> {
    ec.multi.pipeline.as_deref()
}

/// A device-aware [`BindWeight`] for the multi-GPU pipeline: each weight is placed on the device
/// [`PipelineBackend::device_for_weight`] chooses (by tensor name — `blk.{l}.*` → layer `l`'s
/// device, `output*`/`token_embd` → last device) and wrapped in a single-device
/// [`infr_vulkan::PipelineBuffer`] so the executor can pin its readers to that device. Dense
/// resident weights only (pipeline v1 is dense-attention; the MoE-paged / dense-streamed placement
/// tiers of [`vulkan_moe_binder`] are out of scope).
fn pipeline_binder<'a>(pb: &'a infr_vulkan::PipelineBackend) -> Box<BindWeight<'a>> {
    Box::new(move |name: &str, tb: WBytes, dt: DType, numel: usize| {
        // lm_head / embedding tables (issue #77) are read only by dispatch-chunked ops, so they may
        // legitimately exceed the u32 element cap — mirror the resident/EP binders and exempt them
        // (a large/quantized-vocab lm_head must not hard-reject a model that runs single-device/EP).
        if !chunk_covered_dense_tensor(name) {
            check_bda_element_cap(name, "tensor", numel)?;
        }
        let d = pb.device_for_weight(name);
        let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
        let buf = pb
            .backend(d)
            .alloc_uninit(padded.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        pb.backend(d)
            .upload(buf.as_ref(), &padded)
            .map_err(|e| anyhow!("{e}"))?;
        Ok((infr_vulkan::PipelineBuffer::single(d, buf), dt))
    })
}

/// Multi-GPU PIPELINE (layer-split) dense generation — the layers of ONE model are split across
/// the `devices` (a physical index list, e.g. `[0, 1]`), each layer's weights + KV resident on its
/// device, the residual hidden state handed across at the split (P2P dma-buf when available, else
/// host-bounce). The forward is BIT-IDENTICAL to the same model run single-device (identical ops +
/// per-device kernels; only the boundary residual crosses via a value-preserving copy).
///
/// Dense attention models only (Qwen3/Llama/Gemma-dense); MoE / qwen35 DeltaNet / gemma E2B / the
/// paged + streamed weight tiers are rejected with a clear message (they need per-layer placement
/// work beyond this slice). Runs one-shot (a single conversation, no slot pool).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_vulkan_pipeline(
    devices: &[usize],
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    dense_multi_gpu_guard(cfg, "INFR_PIPELINE")?;
    let backends = open_vulkan_devices(devices, ec)?;
    let layer_map = infr_vulkan::PipelineBackend::balanced_layer_map(cfg.n_layer, backends.len());
    // Placement report: how many layers landed on each physical device.
    let names = backends
        .iter()
        .map(|b| {
            use infr_core::backend::Backend;
            b.capabilities().name
        })
        .collect::<Vec<_>>();
    // `multi.pipeline_p2p` (`INFR_PIPELINE_HOST` inverted): its ABSENCE selects P2P.
    let use_p2p = ec.multi.pipeline_p2p;
    let pb = infr_vulkan::PipelineBackend::new(backends, layer_map.clone(), use_p2p)
        .map_err(|e| anyhow!("{e}"))?;
    eprintln!(
        "pipeline: {}-way layer split of {} layers:",
        devices.len(),
        cfg.n_layer
    );
    for (di, &idx) in devices.iter().enumerate() {
        let lo = layer_map.iter().position(|&d| d == di);
        let hi = layer_map.iter().rposition(|&d| d == di);
        let count = layer_map.iter().filter(|&&d| d == di).count();
        match (lo, hi) {
            (Some(lo), Some(hi)) => eprintln!(
                "  Vulkan{idx} ({}): layers [{lo}..={hi}] ({count} layers){}",
                names[di],
                if di + 1 < devices.len() {
                    ""
                } else {
                    " + final norm + lm_head"
                }
            ),
            _ => eprintln!("  Vulkan{idx} ({}): (no layers)", names[di]),
        }
    }
    let bind = pipeline_binder(&pb);
    run_dense_oneshot(
        &pb, &*bind, g, cfg, ec, token_embd, ple, prompt, max_new, on_token,
    )
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Tensor parallelism (dense) — Megatron-style intra-op weight sharding. See `infr_vulkan::tp`.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The `multi.tensor_parallel` device list (`INFR_TENSOR_PARALLEL=Vulkan0,Vulkan1,…`), or `None`
/// when unset. Sibling of [`pipeline_devices`]; needs >=2 devices for a real split.
pub fn tensor_parallel_devices(ec: &EngineConfig) -> Option<&[usize]> {
    ec.multi.tensor_parallel.as_deref()
}

/// The tensor-parallel device role of a weight (by GGUF tensor name), plus its INNER dim `in_f` (for
/// the row-parallel byte stride). `None` = replicated (norms, biases, embeddings, the LM head).
///
/// Column-parallel (sliced by output rows): q/k/v/gate/up. Row-parallel (sliced by input columns):
/// attn_output / ffn_down.
fn tp_weight_role(name: &str, cfg: &Config) -> Option<(infr_vulkan::TpRole, usize)> {
    let layer = name
        .strip_prefix("blk.")
        .and_then(|r| r.split('.').next())
        .and_then(|s| s.parse::<usize>().ok());
    let l = layer?;
    let after = name.rsplit('.').nth(1)?; // e.g. "attn_q" from "blk.3.attn_q.weight"
    let ne = cfg.n_embd;
    let qrow = cfg.n_head * cfg.layer_head_dim(l);
    let nff = cfg.layer_n_ff(l);
    match after {
        "attn_q" | "attn_k" | "attn_v" => Some((infr_vulkan::TpRole::Column, ne)),
        "ffn_gate" | "ffn_up" => Some((infr_vulkan::TpRole::Column, ne)),
        "attn_output" => Some((infr_vulkan::TpRole::Row, qrow)),
        "ffn_down" => Some((infr_vulkan::TpRole::Row, nff)),
        _ => None, // attn_norm/ffn_norm/q_norm/k_norm/biases → replicated
    }
}

/// Column slice: rank `r` of `world` takes the contiguous output-row band
/// `[r·out_f/W, (r+1)·out_f/W)` of a row-major `[out_f, in_f]` tensor. Quant-block-safe: blocks tile
/// along `in_f` (within a row), so a whole-row band never cuts a block.
fn tp_slice_column(
    bytes: &[u8],
    dt: DType,
    in_f: usize,
    r: usize,
    world: usize,
) -> AResult<Vec<u8>> {
    let (be, bb) = infr_gguf::block_layout(dt);
    if !in_f.is_multiple_of(be) {
        return Err(anyhow!(
            "tp column slice: in_f={in_f} not a multiple of block {be}"
        ));
    }
    let row_bytes = (in_f / be) * bb;
    if !bytes.len().is_multiple_of(row_bytes) {
        return Err(anyhow!(
            "tp column slice: {} bytes not a multiple of row {row_bytes}",
            bytes.len()
        ));
    }
    let out_f = bytes.len() / row_bytes;
    if !out_f.is_multiple_of(world) {
        return Err(anyhow!(
            "tp column slice: out_f={out_f} not divisible by world {world}"
        ));
    }
    let rows = out_f / world;
    let start = r * rows * row_bytes;
    Ok(bytes[start..start + rows * row_bytes].to_vec())
}

/// Row slice: rank `r` takes the input-column band `[r·in_f/W, (r+1)·in_f/W)` of every one of the
/// `out_f` rows and re-packs them contiguously into a `[out_f, in_f/W]` tensor. Needs `in_f/W`
/// block-aligned so each per-row band is a whole number of quant blocks.
fn tp_slice_row(bytes: &[u8], dt: DType, in_f: usize, r: usize, world: usize) -> AResult<Vec<u8>> {
    let (be, bb) = infr_gguf::block_layout(dt);
    if !in_f.is_multiple_of(be * world) {
        return Err(anyhow!(
            "tp row slice: in_f={in_f} not divisible by world·block ({world}·{be}) — the input-column \
             split must land on quant-block boundaries"
        ));
    }
    let row_bytes = (in_f / be) * bb;
    if !bytes.len().is_multiple_of(row_bytes) {
        return Err(anyhow!(
            "tp row slice: {} bytes not a multiple of row {row_bytes}",
            bytes.len()
        ));
    }
    let out_f = bytes.len() / row_bytes;
    let band_bytes = ((in_f / world) / be) * bb;
    let col_off = r * band_bytes;
    let mut out = Vec::with_capacity(out_f * band_bytes);
    for row in 0..out_f {
        let s = row * row_bytes + col_off;
        out.extend_from_slice(&bytes[s..s + band_bytes]);
    }
    Ok(out)
}

/// A device-aware [`BindWeight`] for tensor parallelism: q/k/v/gate/up are COLUMN-sliced (output
/// rows), attn_output/ffn_down are ROW-sliced (input columns) and each rank uploads only its slice;
/// norms/biases/embeddings/lm_head are REPLICATED to every rank. Each slice is padded to u32
/// alignment and uploaded to its rank's device, returned as an `infr_vulkan::TpBuffer` carrying the
/// device role the TP lowering reads.
fn tensor_parallel_binder<'a>(
    tp: &'a infr_vulkan::TensorParallelBackend,
    cfg: &'a Config,
) -> Box<BindWeight<'a>> {
    let world = tp.world();
    Box::new(move |name: &str, tb: WBytes, dt: DType, numel: usize| {
        // lm_head / embedding tables (issue #77) are read only by dispatch-chunked ops, so they may
        // legitimately exceed the u32 element cap — mirror the resident/EP binders and exempt them
        // (they are replicated below, never sharded, so the whole-tensor cap would wrongly reject a
        // large/quantized-vocab lm_head that runs fine single-device/EP).
        if !chunk_covered_dense_tensor(name) {
            check_bda_element_cap(name, "tensor", numel)?;
        }
        match tp_weight_role(name, cfg) {
            Some((role, in_f)) => {
                let mut bufs = Vec::with_capacity(world);
                for r in 0..world {
                    let slice = match role {
                        infr_vulkan::TpRole::Column => tp_slice_column(&tb, dt, in_f, r, world)?,
                        infr_vulkan::TpRole::Row => tp_slice_row(&tb, dt, in_f, r, world)?,
                        infr_vulkan::TpRole::Replicated => tb.to_vec(),
                    };
                    let padded = infr_vulkan::linear::pad_to_u32_align(&slice);
                    let buf = tp
                        .rank(r)
                        .alloc_uninit(padded.len(), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    tp.rank(r)
                        .upload(buf.as_ref(), &padded)
                        .map_err(|e| anyhow!("{e}"))?;
                    bufs.push(buf);
                }
                Ok((infr_vulkan::TpBuffer::weight(role, bufs), dt))
            }
            None => {
                // Replicated: the full padded tensor on every rank.
                let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
                let mut bufs = Vec::with_capacity(world);
                for r in 0..world {
                    let buf = tp
                        .rank(r)
                        .alloc_uninit(padded.len(), BufferUsage::Weights)
                        .map_err(|e| anyhow!("{e}"))?;
                    tp.rank(r)
                        .upload(buf.as_ref(), &padded)
                        .map_err(|e| anyhow!("{e}"))?;
                    bufs.push(buf);
                }
                Ok((infr_vulkan::TpBuffer::replica(bufs), dt))
            }
        }
    })
}

/// Multi-GPU TENSOR-PARALLEL (dense) generation — each transformer layer's weight matrices are
/// SHARDED across the `devices` (column-parallel q/k/v/gate/up, row-parallel o/down), each device
/// computes its shard and the partials are all-reduced (P2P dma-buf) per attention + per FFN. The
/// output equals the single-device forward to reduction-order tolerance. Dense attention models only.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_vulkan_tp(
    devices: &[usize],
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    dense_multi_gpu_guard(cfg, "INFR_TENSOR_PARALLEL")?;
    let backends = open_vulkan_devices(devices, ec)?;
    let names = backends
        .iter()
        .map(|b| {
            use infr_core::backend::Backend;
            b.capabilities().name
        })
        .collect::<Vec<_>>();
    // `multi.tp_p2p` (`INFR_TP_HOST` inverted): its ABSENCE selects P2P.
    let use_p2p = ec.multi.tp_p2p;
    let tp =
        infr_vulkan::TensorParallelBackend::new(backends, cfg.n_head, cfg.n_kv, cfg.n_ff, use_p2p)
            .map_err(|e| anyhow!("{e}"))?;
    eprintln!(
        "tensor-parallel: {}-way weight split (n_head={}, n_kv={}, n_ff={}):",
        devices.len(),
        cfg.n_head,
        cfg.n_kv,
        cfg.n_ff
    );
    for (di, &idx) in devices.iter().enumerate() {
        eprintln!(
            "  Vulkan{idx} ({}): rank {di} — {}/{} heads, {}/{} kv-heads, {}/{} ffn per matrix",
            names[di],
            cfg.n_head / devices.len(),
            cfg.n_head,
            cfg.n_kv / devices.len(),
            cfg.n_kv,
            cfg.n_ff / devices.len(),
            cfg.n_ff,
        );
    }
    let bind = tensor_parallel_binder(&tp, cfg);
    run_dense_oneshot(
        &tp, &*bind, g, cfg, ec, token_embd, ple, prompt, max_new, on_token,
    )
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// Expert parallelism (MoE) — shard the experts across devices. See `infr_vulkan::ep`.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The `multi.expert_parallel` device list (`INFR_EXPERT_PARALLEL=Vulkan0,Vulkan1,…`), or `None`
/// when unset. Sibling of [`tensor_parallel_devices`], but its minimum is **1**, not 2 (a single
/// device is the identity, used only as the correctness reference) — the three minimums differ and
/// are preserved by the env layer (§6.11).
pub fn expert_parallel_devices(ec: &EngineConfig) -> Option<&[usize]> {
    ec.multi.expert_parallel.as_deref()
}

/// Whether `name` is a stacked expert-bank weight (`ffn_{gate,up,down}_exps` or fused
/// `ffn_gate_up_exps`) — the ONLY tensors Expert Parallelism shards; everything else is replicated.
fn is_expert_bank(name: &str) -> bool {
    name.ends_with("ffn_gate_exps.weight")
        || name.ends_with("ffn_up_exps.weight")
        || name.ends_with("ffn_down_exps.weight")
        || name.ends_with("ffn_gate_up_exps.weight")
}

/// A device-aware [`BindWeight`] for Expert Parallelism: the stacked expert banks
/// (`ffn_{gate,up,down}_exps`) are split by EXPERT — rank `r` of `world` gets the contiguous band
/// `[r·E/W, (r+1)·E/W)` (each per-expert slice is `nbytes/n_expert`, block-aligned, so a whole-band
/// cut never splits a quant block) and uploads ONLY that band; every other weight (router,
/// attention, norms, embeddings, the LM head) is REPLICATED to every rank. Each slice/replica is
/// padded to u32 alignment and uploaded to its rank's device, returned as an
/// `infr_vulkan::EpBuffer`.
fn expert_parallel_binder<'a>(
    ep: &'a infr_vulkan::ExpertParallelBackend,
    cfg: &'a Config,
) -> Box<BindWeight<'a>> {
    let world = ep.world();
    let n_expert = cfg.moe.as_ref().map(|m| m.n_expert).unwrap_or(0).max(1);
    Box::new(move |name: &str, tb: WBytes, dt: DType, numel: usize| {
        if is_expert_bank(name) {
            // Split the bank by expert. The stacked bank is `n_expert` contiguous per-expert slices
            // (the arena kernels index `arena + expert·stride`), so band `[r·nl, (r+1)·nl)` is a
            // contiguous byte range and its per-expert element cap is `numel/n_expert`.
            let total = tb.len();
            if !total.is_multiple_of(n_expert) {
                return Err(anyhow!(
                    "ep: expert bank '{name}' {total} bytes not divisible by n_expert={n_expert}"
                ));
            }
            let stride_bytes = total / n_expert;
            if !n_expert.is_multiple_of(world) {
                return Err(anyhow!(
                    "ep: n_expert={n_expert} not divisible by world {world}"
                ));
            }
            let nl = n_expert / world;
            check_bda_element_cap(name, "per-expert slice", numel / n_expert)?;
            let mut bufs = Vec::with_capacity(world);
            for r in 0..world {
                let start = r * nl * stride_bytes;
                let slice = &tb[start..start + nl * stride_bytes];
                let padded = infr_vulkan::linear::pad_to_u32_align(slice);
                let buf = ep
                    .rank(r)
                    .alloc_uninit(padded.len(), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                ep.rank(r)
                    .upload(buf.as_ref(), &padded)
                    .map_err(|e| anyhow!("{e}"))?;
                bufs.push(buf);
            }
            Ok((infr_vulkan::EpBuffer::wrap(bufs), dt))
        } else {
            // Replicated: the full padded tensor on every rank. Mirror the resident binder's
            // per-expert-slice / chunk-covered / whole-tensor element-cap policy for non-bank
            // weights (a replicated router is a whole tensor; lm_head/token_embd are chunk-covered).
            if chunk_covered_dense_tensor(name) {
                // no whole-tensor cap (dispatch-chunked reads)
            } else {
                check_bda_element_cap(name, "tensor", numel)?;
            }
            let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
            let mut bufs = Vec::with_capacity(world);
            for r in 0..world {
                let buf = ep
                    .rank(r)
                    .alloc_uninit(padded.len(), BufferUsage::Weights)
                    .map_err(|e| anyhow!("{e}"))?;
                ep.rank(r)
                    .upload(buf.as_ref(), &padded)
                    .map_err(|e| anyhow!("{e}"))?;
                bufs.push(buf);
            }
            Ok((infr_vulkan::EpBuffer::wrap(bufs), dt))
        }
    })
}

/// Multi-GPU EXPERT-PARALLEL (MoE) generation — the model's experts are split across the `devices`
/// (rank `r` owns experts `[r·E/W, (r+1)·E/W)`), the router + attention + norms run replicated on
/// every rank, each rank computes only its band's experts, and one P2P all-reduce per MoE layer
/// combines the partial expert outputs. The output equals the single-device MoE to reduction-order
/// tolerance (token-identical greedy). qwen3moe (split gate/up, softmax, no shared expert) in v1.
#[cfg_attr(infr_profile, infr_prof::instrument)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_moe_vulkan_ep(
    devices: &[usize],
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    let Some(moe) = cfg.moe.as_ref() else {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL needs a routed-expert (MoE) model — this is a dense model (use \
             INFR_TENSOR_PARALLEL or INFR_PIPELINE)"
        ));
    };
    if cfg.qwen35 {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL does not yet support qwen35 (DeltaNet recurrent-state replication \
             is a separate slice)"
        ));
    }
    if cfg.shexp_ff > 0 {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL v1 does not yet place a SHARED expert (qwen35moe / llama4 \
             MoeSharedExpertAdd) — only routed-only MoE (qwen3moe). Shared-expert placement is a \
             separate slice"
        ));
    }
    if cfg.n_embd_per_layer > 0 {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL does not support gemma E2B per-layer embeddings"
        ));
    }
    if cfg.diffusion_gemma {
        return Err(anyhow!(
            "INFR_EXPERT_PARALLEL does not support diffusion-gemma"
        ));
    }
    let backends = open_vulkan_devices(devices, ec)?;
    let names = backends
        .iter()
        .map(|b| {
            use infr_core::backend::Backend;
            b.capabilities().name
        })
        .collect::<Vec<_>>();
    // `multi.ep_p2p` (`INFR_EP_HOST` inverted): its ABSENCE selects P2P.
    let use_p2p = ec.multi.ep_p2p;
    let ep = infr_vulkan::ExpertParallelBackend::new(backends, moe.n_expert, use_p2p)
        .map_err(|e| anyhow!("{e}"))?;
    let nl = ep.experts_per_device();
    eprintln!(
        "expert-parallel: {}-way expert split ({} experts, {} used/token → {nl} experts/device):",
        devices.len(),
        moe.n_expert,
        moe.n_used,
    );
    for (di, &idx) in devices.iter().enumerate() {
        eprintln!(
            "  Vulkan{idx} ({}): rank {di} — experts [{}..{}) ({nl} of {})",
            names[di],
            di * nl,
            (di + 1) * nl,
            moe.n_expert,
        );
    }
    let bind = expert_parallel_binder(&ep, cfg);
    run_dense_oneshot(
        &ep, &*bind, g, cfg, ec, token_embd, ple, prompt, max_new, on_token,
    )
}

/// Metal seam runner: the SAME dense forward as [`generate_dense_cpu`], on the reference Metal
/// backend through the agnostic [`Graph`]. Weights are uploaded to Metal buffers in their NATIVE
/// GGUF dtype (the backend dequantizes lazily in its own `bytes_to_f32`, exactly like the CPU
/// interpreter — so a quant weight occupies ~quant size, not 8× f32).
#[cfg(target_os = "macos")]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_metal(
    mtl: &infr_metal::MetalBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    req: Option<&crate::sampling::RequestCtx>,
    on_token: impl FnMut(u32),
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_metal_session(
        mtl,
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        &mut None,
        prompt.len() + max_new + 1,
        None,
        req,
    )
}

/// Persistent-session Metal seam runner — the Metal twin of [`generate_dense_vulkan_session`]:
/// weights upload once into `state`, the KV cache is sized to `want_ctx`, and each call prefills
/// only the suffix that differs from the tokens already materialized in the cache.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn generate_dense_metal_session(
    mtl: &infr_metal::MetalBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    prompt: &[u32],
    max_new: usize,
    on_token: impl FnMut(u32),
    state: &mut Option<SeamKv>,
    want_ctx: usize,
    constraint: Option<&mut crate::grammar::Constraint>,
    req: Option<&crate::sampling::RequestCtx>,
) -> AResult<(Vec<u32>, GenStats)> {
    generate_dense_backend(
        mtl,
        &metal_upload_bind(mtl),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        prompt,
        max_new,
        on_token,
        state,
        want_ctx,
        constraint,
        None,
        None,
        None,
        None,
        None,
        req,
    )
}

/// Speculative VERIFY on the Metal seam: one batched forward of `tokens`' un-cached suffix with
/// the LM head on every suffix row. Returns the [m, vocab] logits plus the graph-execute
/// seconds, and leaves the session's KV + `cached` covering all of `tokens` — the caller
/// commits the accepted prefix and the next call's prefix diff overwrites whatever was
/// speculatively written past it.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_metal2(
    mtl: &infr_metal::MetalBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
    state: &mut Option<SeamKv>,
    want_ctx: usize,
) -> AResult<(Vec<f32>, f64)> {
    let mut logits = Vec::new();
    let (_, stats) = generate_dense_backend(
        mtl,
        &metal_upload_bind(mtl),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        0,
        |_| {},
        state,
        want_ctx,
        None,
        Some(&mut logits),
        None,
        None,
        None,
        None,
        None,
    )?;
    Ok((logits, stats.prompt_secs))
}

/// DiffusionGemma Phase-1 validation: a causal prefill of `tokens` (a fresh one-shot forward, no
/// session) through the CPU reference backend, returning the LAST token's raw (pre-softmax, post-
/// softcap) logits. Rides the ordinary per-token decode loop (`max_new = 1`, the one generated
/// token discarded) — MoE-compatible, unlike the batched `verify` path.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_cpu(
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<Vec<f32>> {
    let cpu_be = CpuBackend::new_with(ec.clone());
    let mut logits = Vec::new();
    let mut state = None;
    generate_dense_backend(
        &cpu_be,
        &cpu_upload_bind(&cpu_be),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        1,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,
        None,
        None,
        Some(&mut logits),
        None,
        None,
        None,
    )?;
    Ok(logits)
}

/// [`verify_dense_cpu`]'s MTP Phase 1 twin (issue #33, docs/mtp.md): ALSO captures the LM-head
/// input rows (`h_out` — `DecodeHandles::h_out`'s doc) alongside the logits, for the
/// `lm_head(h_row) == logits_row` consistency check `docs/mtp.md`'s Phase 1 validation calls for.
/// Returns `(logits, h)`, both `[vocab]`/`[n_embd]` for the last prompt token.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_cpu_with_h(
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<(Vec<f32>, Vec<f32>)> {
    let cpu_be = CpuBackend::new_with(ec.clone());
    let mut logits = Vec::new();
    let mut h = Vec::new();
    let mut state = None;
    generate_dense_backend(
        &cpu_be,
        &cpu_upload_bind(&cpu_be),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        1,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,
        None,
        None,
        Some(&mut logits),
        Some(&mut h),
        None,
        None,
    )?;
    Ok((logits, h))
}

/// [`verify_dense_cpu_with_h`]'s ALL-ROWS twin (MTP Phase 2, issue #33): rides the speculative-
/// VERIFY batched forward (the `verify` param, not `logits_out`) so `h`/`logits` cover EVERY one of
/// `tokens`, not just the last — the shape `crate::mtp::catch_up` needs to prime the head's KV over
/// a whole prompt in one call (`docs/mtp.md`'s `process()` runs after every target ubatch, not just
/// the sampled row). Dense non-MoE models only (mirrors the VERIFY branch's own guard). Returns
/// `(logits [tokens.len()*vocab], h [tokens.len()*n_embd])`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_rows_cpu_with_h(
    g: &Gguf,
    cfg: &Config,
    ec: &std::sync::Arc<EngineConfig>,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<(Vec<f32>, Vec<f32>)> {
    let cpu_be = CpuBackend::new_with(ec.clone());
    let mut logits = Vec::new();
    let mut h = Vec::new();
    let mut state = None;
    generate_dense_backend(
        &cpu_be,
        &cpu_upload_bind(&cpu_be),
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        0,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,
        Some(&mut logits),
        None,
        None,
        Some(&mut h),
        None,
        None,
    )?;
    Ok((logits, h))
}

/// [`verify_dense_cpu`]'s Vulkan twin — the same one-shot causal prefill through the production
/// Vulkan seam, for the CPU/Vulkan cross-backend parity check.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn verify_dense_vulkan(
    vk: &infr_vulkan::VulkanBackend,
    g: &Gguf,
    cfg: &Config,
    ec: &EngineConfig,
    token_embd: TokenEmbd<'_>,
    ple: Option<&PerLayerEmbd>,
    tokens: &[u32],
) -> AResult<Vec<f32>> {
    let mut logits = Vec::new();
    let mut state = None;
    generate_dense_backend(
        vk,
        &|_name, tb, dt, _n| {
            let padded = infr_vulkan::linear::pad_to_u32_align(&tb);
            let buf = vk
                .alloc(padded.len(), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            vk.upload(buf.as_ref(), &padded)
                .map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        },
        g,
        cfg,
        ec,
        token_embd,
        ple,
        tokens,
        1,
        |_| {},
        &mut state,
        tokens.len() + 2,
        None,
        None,
        None,
        Some(&mut logits),
        None,
        None,
        None,
    )?;
    Ok(logits)
}

/// Backend-generic dense decode runner. Builds the agnostic decode [`Graph`] per token and runs it
/// on `be` (CPU reference or Vulkan). `bind_weight` turns each native-dtype GGUF tensor into a
/// backend buffer: the CPU maps it zero-copy from the mmap; the GPU pads + uploads it to VRAM. This
/// is the single forward both backends share — running it on Vulkan and diffing the CPU oracle is
/// the end-to-end dense parity check.
/// Weight bytes handed to a binder: a zero-copy mmap slice (the normal case), or an owned
/// concatenation (the combined gate+up upload — only produced when `Capabilities::combined_gu`).
pub(crate) enum WBytes {
    Mmap(TensorBytes),
    Owned(Vec<u8>),
}

impl std::ops::Deref for WBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            WBytes::Mmap(tb) => tb,
            WBytes::Owned(v) => v,
        }
    }
}

/// Turns a native-dtype GGUF tensor into a backend buffer + the EFFECTIVE dtype it now holds (the
/// GPU binder may convert float weights to f16), so the graph declares the handle to match. The
/// tensor NAME lets a binder place specific tensors differently (the Vulkan binder puts
/// auto-fit-offloaded MoE expert banks in host-visible memory instead of VRAM).
type BindWeight<'a> = dyn Fn(&str, WBytes, DType, usize) -> AResult<(Box<dyn Buffer>, DType)> + 'a;

/// Persistent per-session seam state: the uploaded weights, the KV cache (sized to `max_ctx`
/// once), the per-step IO buffers, and the token ids currently MATERIALIZED in the cache. A caller
/// holding one across `generate_dense_backend` calls gets ChatSession-style KV reuse — each turn
/// prefills only the token suffix that differs from `cached` (the common-prefix diff), so a
/// growing conversation stops re-prefilling its whole history. Pass a fresh `None` for the old
/// one-shot behavior.
/// Byte size of `elems` KV-cache elements stored as `dt`. Q8_0 = 34 bytes / 32-elem block
/// (a 2-byte f16 scale + 32 int8), F16 = 2 bytes, else raw f32. K and V pick their dtype
/// independently, so this is called per-side. Q8_0 is rounded up to a u32 multiple so the Vulkan
/// backend can bind the buffer as a `uint` array (its planar Q8 layout reads codes/scales as words).
/// A quantized KV cache dtype that forces per-execute static decode on the GPU (record-once replay
/// is disabled for it). Must match the adapter's `decode_eligible` rejection — with one pair-wise
/// exception the caller handles: COUPLED Q8_0 (K==V==Q8) replays (store_q8_dyn + the planar-Q8 dyn
/// attention read), so `runner`'s gate checks the pair before consulting this per-side predicate.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn kv_forces_static(dt: DType) -> bool {
    matches!(
        dt,
        DType::Q8_0
            | DType::Q4_0
            | DType::Q4_1
            | DType::Q5_0
            | DType::Q5_1
            | DType::Iq4Nl
            | DType::Turbo2
            | DType::Turbo3
            | DType::Turbo4
            // Dense f32/bf16 caches also un-fuse the K write on the GPU → force static decode.
            | DType::F32
            | DType::Bf16
    )
}

/// Exact KV buffer size for a format — pure format arithmetic, so it lives in the shared seam
/// ([`infr_core::budget::kv_fmt_bytes`], which owns the doc and the pinning tests) next to the
/// per-element rate the placement estimates use. Re-exported under the old crate-private name
/// because every call site here is a `Backend::alloc` argument.
pub(crate) use infr_core::budget::kv_fmt_bytes;

/// gemma4 E2B: gather + dequant this chunk's per-layer TOKEN embedding rows on the host — the ONLY
/// part llama.cpp keeps host-side ("very little benefit to offloading the input layer"); the
/// model_proj GEMV + RMSNorm + combine now run as GPU graph ops (see the E2B prologue in `build`).
/// Returns `pl_tok_scaled[r][l*npl+j] = per_layer_tok_embd[tok_r][l*npl+j] * √npl`, `[rows,
/// n_layer*npl]` row-major — uploaded to `ipl_buf` and bound to the graph Input `pl_tok_in`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn e2b_ipl_rows(g: &Gguf, ple: &PerLayerEmbd, tokens: &[u32]) -> AResult<Vec<f32>> {
    use rayon::prelude::*;
    let (npl, nl) = (ple.npl, ple.n_layer);
    let sqrt_npl = (npl as f32).sqrt();
    let te_bytes = g
        .tensor_bytes("per_layer_token_embd.weight")
        .map_err(|e| anyhow!("{e}"))?;
    let mut out = vec![0f32; tokens.len() * nl * npl];
    out.par_chunks_mut(nl * npl)
        .zip(tokens.par_iter())
        .try_for_each(|(dst, &tok)| -> AResult<()> {
            let tok = tok as usize;
            let r0 = tok * ple.tok_embd_row_bytes;
            let pl_tok = dequant_block(
                ple.tok_embd_dtype,
                &te_bytes[r0..r0 + ple.tok_embd_row_bytes],
            )
            .map_err(|e| anyhow!("{e}"))?;
            for (d, s) in dst.iter_mut().zip(pl_tok.iter()) {
                *d = s * sqrt_npl;
            }
            Ok(())
        })?;
    Ok(out)
}

/// Longest shared prefix of the cached tokens and the new prompt (the KV rows that stay valid).
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod bda_cap_tests {
    use super::{check_bda_element_cap, chunk_covered_dense_tensor, BDA_ELEMENT_UNIT_MAX};

    #[test]
    fn chunk_covered_names_are_exactly_lm_head_and_embed() {
        // Only the output projection / embedding tables (chunk-covered, issue #77) may exceed the
        // element cap — the binder skips check_bda_element_cap for exactly these.
        for n in [
            "output.weight",
            "token_embd.weight",
            "per_layer_token_embd.weight",
        ] {
            assert!(chunk_covered_dense_tensor(n), "{n} must be chunk-covered");
        }
        // A per-layer projection / norm / bias is NOT — it keeps the loud whole-tensor element cap
        // (and never breaches anyway: its element count is orders of magnitude under 2^32).
        for n in [
            "blk.0.attn_q.weight",
            "blk.10.ffn_down.weight",
            "output_norm.weight",
            "blk.0.ffn_gate_exps.weight",
        ] {
            assert!(!chunk_covered_dense_tensor(n), "{n} must stay capped");
        }
    }

    #[test]
    fn element_cap_accepts_realistic_and_boundary_below() {
        // A big-but-real per-expert slice (2112 * 5120 ≈ 10.8M elems) and the last legal count
        // both pass — model reality stays well under the cap.
        check_bda_element_cap(
            "blk.0.ffn_gate_exps.weight",
            "per-expert slice",
            2112 * 5120,
        )
        .expect("realistic expert slice must pass");
        check_bda_element_cap("t", "tensor", BDA_ELEMENT_UNIT_MAX - 1)
            .expect("2^32 - 1 elements is the last legal count");
    }

    #[test]
    fn element_cap_rejects_at_and_above_2p32() {
        // At the cap and above it fail LOUDLY (u32 index would wrap) rather than corrupt output.
        assert!(check_bda_element_cap("t", "tensor", BDA_ELEMENT_UNIT_MAX).is_err());
        assert!(check_bda_element_cap("t", "per-expert slice", BDA_ELEMENT_UNIT_MAX + 7).is_err());
    }

    #[test]
    fn element_cap_resident_expert_bank_uses_per_slice() {
        // A RESIDENT stacked expert bank whose WHOLE-bank element count clears 2^32 but whose
        // per-expert slice stays legal: the binder's `moe_role_of(name)` branch checks the
        // per-expert slice (arena kernels index `arena + expert * stride`), NOT the whole bank —
        // the same relaxation the paged path already made. This is the invariant that branch relies
        // on, and the flag-off resident id kernels that once forced the conservative whole-bank
        // count are gone (weights are u64 BDA-addressed only).
        let n_expert = 512usize;
        let slice = 9_000_000usize; // < 2^32
        let whole = n_expert * slice; // 4.608e9 > 2^32
        assert!(whole > BDA_ELEMENT_UNIT_MAX && slice < BDA_ELEMENT_UNIT_MAX);
        // The old WHOLE-bank cap would (wrongly) reject this legal resident bank...
        assert!(check_bda_element_cap("blk.0.ffn_up_exps.weight", "tensor", whole).is_err());
        // ...but the per-expert-slice cap the resident-bank path now applies accepts it.
        check_bda_element_cap(
            "blk.0.ffn_up_exps.weight",
            "per-expert slice",
            whole / n_expert,
        )
        .expect("resident expert bank per-slice must pass");
    }

    #[test]
    fn tp_pipeline_cap_exempts_chunk_covered_enforces_others() {
        // The TP and pipeline binders now gate the whole-tensor cap on `!chunk_covered_dense_tensor`
        // (mirroring EP/resident) — model that exact decision: a >= 2^32-element lm_head / embedding
        // table is EXEMPT (dispatch-chunked reads), a normal huge non-chunk-covered tensor is caught.
        let over_cap = BDA_ELEMENT_UNIT_MAX + 1;
        for chunked in [
            "output.weight",
            "token_embd.weight",
            "per_layer_token_embd.weight",
        ] {
            assert!(chunk_covered_dense_tensor(chunked));
            // The binder's gate: `if !chunk_covered_dense_tensor(name) { check_bda_element_cap(..) }`
            // — chunk-covered means the cap is SKIPPED, so an over-cap table binds fine.
            if !chunk_covered_dense_tensor(chunked) {
                check_bda_element_cap(chunked, "tensor", over_cap).expect("unreachable");
            }
        }
        // A normal (non-chunk-covered) tensor over the cap is still rejected loudly under both.
        assert!(!chunk_covered_dense_tensor("blk.0.attn_q.weight"));
        assert!(check_bda_element_cap("blk.0.attn_q.weight", "tensor", over_cap).is_err());
    }
}

#[cfg(test)]
mod seam_helper_tests {
    use super::{kv_pair_bytes, Config, DType, EngineConfig, PlacementPins, PlacementScope};

    // NB: `parse_device_spec`'s own cases moved to `infr_core::config::tests` with the function
    // (S4 deleted this crate's duplicate of that grammar — §6.11).

    /// `device.ubatch` is TWO readers, and an unusable value must split them (§6.12, and the
    /// `ubatch_specified` decision recorded on [`super::ubatch_rows`]): `-u 0` / a typo yields no
    /// chunk height yet still counts as "the user pinned one", which is what disables the dense
    /// placement sweeps. Collapsing them onto `Option::is_some` would silently re-enable the sweep.
    #[test]
    fn ubatch_value_and_presence_are_separate_readers() {
        // Own scope: the fallback pins are process-global, and this test asserts the UNPINNED
        // default. (That this is even possible per-scope is the point of `PlacementPins`.)
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let unset = EngineConfig::default();
        assert!(!super::user_pinned_ubatch(&unset));
        assert_eq!(
            super::ubatch_rows(&unset),
            1024,
            "no pin, no iGPU: the 1024 default"
        );

        let pinned = EngineConfig {
            device: infr_core::config::DeviceCfg {
                ubatch: Some(512),
                ubatch_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(super::user_pinned_ubatch(&pinned));
        assert_eq!(super::ubatch_rows(&pinned), 512);

        // `-u 0` / `INFR_UBATCH=abc`: specified, but no usable height.
        let adaptive = EngineConfig {
            device: infr_core::config::DeviceCfg {
                ubatch: Some(0),
                ubatch_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            super::user_pinned_ubatch(&adaptive),
            "an unusable value is still a pin — the sweep must stay off"
        );
        assert_eq!(
            super::ubatch_rows(&adaptive),
            1024,
            "…and the height falls back"
        );
    }

    /// The `*_specified` rule (§11 decision 8): an UNRECOGNIZED KV format name still suppresses
    /// auto-q8 (it was supplied) while yielding no dtype, and it is not ring-capable either.
    #[test]
    fn kv_specified_beats_a_parsed_dtype() {
        let unset = EngineConfig::default();
        assert!(
            super::kv_unset(&unset),
            "nothing supplied ⇒ auto-q8 may fill"
        );

        // `INFR_KV_TYPE_K=nonsense`: specified, no dtype.
        let nonsense = EngineConfig {
            kv: infr_core::config::KvCfg {
                type_k: None,
                type_k_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            !super::kv_unset(&nonsense),
            "an unparseable name still counts as an explicit choice"
        );

        // The legacy both-sides alias counts too.
        let q8 = EngineConfig {
            kv: infr_core::config::KvCfg {
                force_q8: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!super::kv_unset(&q8));
    }

    #[test]
    fn kv_pair_bytes_prices_each_side_in_its_own_dtype() {
        // q8 prices K+V at ~half the f16 bytes (34 B / 32-elem block vs 2 B/elem, ×2 sides).
        let elems = 32_000usize;
        let f16 = kv_pair_bytes(elems, DType::F16, DType::F16);
        let q8 = kv_pair_bytes(elems, DType::Q8_0, DType::Q8_0);
        assert_eq!(f16, 2 * 2 * elems as u64); // 128_000
        assert_eq!(q8, 2 * (elems as u64 / 32 * 34)); // 68_000, already u32-aligned
        assert!(
            q8 < f16 && q8 * 2 > f16,
            "q8 must be ~half of f16, not equal"
        );
        // The MIXED case the single-flag helper could not express: `INFR_KV_TYPE_K=q8_0` with V
        // left at f16 is exactly one of each, not two of either.
        let mixed = kv_pair_bytes(elems, DType::Q8_0, DType::F16);
        assert_eq!(mixed, q8 / 2 + f16 / 2);
        assert!(mixed > q8 && mixed < f16);
    }

    // ── context-fit math (`kv_fit_ctx_for`) ──────────────────────────────────────────────────
    //
    // GPU-free: the function takes the weight footprint and the device's byte figures as plain
    // arguments, so every case below is exact arithmetic. Model geometry and weight footprints are
    // read off the real GGUFs (`Config::from_gguf` + `weights::weight_footprint`); the byte figures
    // are this box's 7900 XTX (`/sys/class/drm/card1/device/mem_info_vram_total` = 25 753 026 560,
    // live free ~23.94 GiB on an idle desktop).

    /// RX 7900 XTX, 24 GB.
    const XTX_TOTAL: u64 = 25_753_026_560;
    /// What `VulkanBackend::alloc_room()` reports on an idle XTX: live free (~23.94 GiB) minus the
    /// allocator guard's own 256 MiB headroom. Anything the fit math plans past this the VRAM
    /// guard will refuse, so this — not the raw free figure — is the budget.
    const XTX_ROOM: u64 = 25_701_257_216 - 256 * 1024 * 1024;

    /// gemma-3-12b-it-Q4_K_M — the reported case. Geometry from the GGUF metadata.
    fn gemma3_12b() -> Config {
        Config {
            n_layer: 48,
            n_head: 16,
            n_kv: 8,
            n_kv_swa: 8,
            head_dim: 256,
            head_dim_swa: 256,
            n_embd: 3840,
            n_ff: 15360,
            swa_window: 1024,
            swa_pattern: 6,
            n_ctx_train: 131072,
            ..Default::default()
        }
    }
    /// `weights::weight_footprint` of that GGUF.
    const GEMMA3_12B_WEIGHTS: u64 = 7_292_694_912;

    /// A plain dense model with NO sliding window and head_dim 128 — the flash tier, so
    /// `dense_act_reserve_at`'s score-tile term is zero and the KV grows strictly linearly.
    /// Qwen3-14B's geometry.
    fn qwen3_14b() -> Config {
        Config {
            n_layer: 40,
            n_head: 40,
            n_kv: 8,
            head_dim: 128,
            n_embd: 5120,
            n_ff: 17408,
            n_ctx_train: 40960,
            ..Default::default()
        }
    }

    /// Re-derive the accept predicate from the shared primitives and assert `fit` is EXACTLY its
    /// boundary: it fits, and one more token does not. Returns nothing — it panics with detail.
    fn assert_exact_boundary(
        cfg: &Config,
        ec: &EngineConfig,
        weights: u64,
        room: u64,
        k: DType,
        v: DType,
        fit: usize,
    ) {
        let ring = super::kv_ring_wanted(cfg, ec)
            && matches!(k, DType::F16 | DType::Q8_0)
            && matches!(v, DType::F16 | DType::Q8_0);
        let need = |ctx: usize, ub: usize| {
            weights
                + super::kv_bytes_estimate_fmt(cfg, ctx, ring, ub, k, v)
                + super::dense_act_reserve_at(cfg, ctx, ub)
        };
        let cands = super::ubatch_candidates(ec);
        let best = |ctx: usize| cands.iter().map(|&ub| need(ctx, ub)).min().expect("ladder");
        assert!(
            best(fit) <= room,
            "fit {fit} must FIT: cheapest need {} > room {room}",
            best(fit)
        );
        assert!(
            best(fit + 1) > room,
            "fit {fit} must be the LARGEST: {} tokens also fits (need {} <= room {room})",
            fit + 1,
            best(fit + 1)
        );
    }

    /// The linear (no-ring) branch, both element sizes: the returned context is the exact largest
    /// that fits, not an approximation of it. A `0.95` haircut or a `saturating_sub(64)` on the
    /// RESULT — what this replaced — fails the second half of the boundary check.
    #[test]
    fn kv_fit_linear_is_the_exact_largest_context() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = qwen3_14b();
        let ec = EngineConfig::default();
        let weights = 9 * (1u64 << 30);
        for (k, v) in [(DType::F16, DType::F16), (DType::Q8_0, DType::Q8_0)] {
            let fit = super::kv_fit_ctx_for(&cfg, &ec, weights, XTX_ROOM, XTX_TOTAL, k, v)
                .expect("has KV");
            assert_exact_boundary(&cfg, &ec, weights, XTX_ROOM, k, v, fit);
        }
        // q8 is ~half the bytes per token, so it must buy materially more context than f16.
        let f16 = super::kv_fit_ctx_for(
            &cfg,
            &ec,
            weights,
            XTX_ROOM,
            XTX_TOTAL,
            DType::F16,
            DType::F16,
        )
        .expect("has KV");
        let q8 = super::kv_fit_ctx_for(
            &cfg,
            &ec,
            weights,
            XTX_ROOM,
            XTX_TOTAL,
            DType::Q8_0,
            DType::Q8_0,
        )
        .expect("has KV");
        assert!(q8 > f16, "q8 {q8} must beat f16 {f16}");
        assert!(q8 < 2 * f16, "q8 {q8} is ~2x f16 {f16}, not more");
    }

    /// The SWA-ring branch, both element sizes. A window layer's KV stops growing with context
    /// once the ring caps its rows, so the fit is enormously larger than the same model priced
    /// without the ring — and it is still the EXACT boundary.
    #[test]
    fn kv_fit_swa_ring_is_exact_and_much_larger_than_linear() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let no_ring = EngineConfig {
            kv: infr_core::config::KvCfg {
                ring: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let ring = EngineConfig::default();
        assert!(super::kv_ring_wanted(&cfg, &ring) && !super::kv_ring_wanted(&cfg, &no_ring));
        for (k, v) in [(DType::F16, DType::F16), (DType::Q8_0, DType::Q8_0)] {
            let a =
                super::kv_fit_ctx_for(&cfg, &ring, GEMMA3_12B_WEIGHTS, XTX_ROOM, XTX_TOTAL, k, v)
                    .expect("has KV");
            let b = super::kv_fit_ctx_for(
                &cfg,
                &no_ring,
                GEMMA3_12B_WEIGHTS,
                XTX_ROOM,
                XTX_TOTAL,
                k,
                v,
            )
            .expect("has KV");
            assert_exact_boundary(&cfg, &ring, GEMMA3_12B_WEIGHTS, XTX_ROOM, k, v, a);
            assert_exact_boundary(&cfg, &no_ring, GEMMA3_12B_WEIGHTS, XTX_ROOM, k, v, b);
            assert!(
                a > b * 4,
                "40 of 48 layers ring: {a} should dwarf the full-cache fit {b}"
            );
        }
    }

    /// **Regression, the reported case.** gemma-3-12b at its trained 131072 window on a 24 GiB
    /// XTX fits at f16 — it never needed the q8 cache the clamp used to pin. What used to make it
    /// miss is priced here explicitly: the DEFAULT 1024-row prefill chunk's activation reserve
    /// does not fit, a SHORTER rung of the same ladder the dense placement sweep walks does, and
    /// the sweep would have shrunk to it anyway. Pricing only the default chunk decided the KV
    /// format against an assumption the very next step abandoned.
    ///
    /// Deliberately does not name the winning rung. Which one it is moves with
    /// `ACT_RESERVE_MARGIN` (512 unmargined, 256 at the 1.5 shipped today) and that is not what
    /// this is guarding — the invariant is "the default chunk misses, a lower rung on the SHARED
    /// ladder saves it, and f16 therefore reaches the trained window".
    ///
    /// (Whole-run confirmation is `infr bench -d 120000` on the real model, which peaks at
    /// 17.5 GiB of 24.0 GiB — GPU + 8 GiB of weights, so not something a unit test can host.)
    #[test]
    fn kv_fit_walks_the_placement_chunk_ladder_gemma3_12b() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        let want = cfg.n_ctx_train;
        let need = |ub: usize| {
            GEMMA3_12B_WEIGHTS
                + super::kv_bytes_estimate_fmt(&cfg, want, true, ub, DType::F16, DType::F16)
                + super::dense_act_reserve_at(&cfg, want, ub)
        };
        let cands = super::ubatch_candidates(&ec);
        assert_eq!(cands[0], 1024, "the default chunk leads the ladder");
        assert!(
            need(1024) > XTX_ROOM,
            "the default chunk really does not fit — otherwise this test proves nothing"
        );
        let rescued_by = cands.iter().copied().find(|&ub| need(ub) <= XTX_ROOM);
        assert!(
            rescued_by.is_some_and(|ub| ub < 1024),
            "…but a SHORTER rung of the shared ladder must: {:?}",
            cands.iter().map(|&ub| (ub, need(ub))).collect::<Vec<_>>()
        );

        let fit = super::kv_fit_ctx_for(
            &cfg,
            &ec,
            GEMMA3_12B_WEIGHTS,
            XTX_ROOM,
            XTX_TOTAL,
            DType::F16,
            DType::F16,
        )
        .expect("has KV");
        assert!(
            fit >= want,
            "f16 must reach the trained window ({want}); got {fit} — the auto-q8 rung would fire"
        );
    }

    /// The interim activation-reserve margin (`ACT_RESERVE_MARGIN`) is REALLY in the number, and
    /// the formula around it is the one documented on `dense_act_reserve_at`. Spelled out term by
    /// term for gemma-4-31B's shape at a 128-row chunk and ctx 16384, so deleting the margin — or
    /// quietly changing a coefficient — fails here rather than showing up as a `VRAM budget
    /// exceeded` on someone's deep prefill.
    ///
    /// When backlog B8 lands (a path-aware reserve), this test goes with the margin. It is not a
    /// floor to build on.
    #[test]
    fn act_reserve_carries_the_interim_margin() {
        let cfg = Config {
            n_head: 32,
            head_dim: 512,
            head_dim_swa: 256,
            swa_window: 1024,
            swa_pattern: 6,
            n_embd: 5376,
            n_ff: 21504,
            ..Default::default()
        };
        let (ctx, ubatch) = (16384usize, 128usize);
        let rows: u64 = 128; // already a multiple of 64
        let attn_pv: u64 = 32 * 32 * 768; // n_head x (head_dim + head_dim_swa), both shapes live
        let attn_s: u64 = 4 * 32 * 16384; // non-flash score tiles: hd 512 misses the flash tier
        let per_row = 12 * 21504 + 96 * 5376 + attn_pv + attn_s;
        let unmargined = 256 * 1024 * 1024 + rows * per_row * 5 / 4;
        assert_eq!(unmargined, 853_671_936);

        let got = super::dense_act_reserve_at(&cfg, ctx, ubatch);
        assert_eq!(
            got,
            unmargined * super::ACT_RESERVE_MARGIN.0 / super::ACT_RESERVE_MARGIN.1,
            "reserve must be the model TIMES the interim margin"
        );
        assert!(
            got > unmargined,
            "the margin must actually widen the reserve — {got} vs {unmargined}"
        );
        assert_eq!(
            got, 1_280_507_904,
            "1.5x, the measured value that lets \
             gemma-4-31B fill the context it advertises"
        );
    }

    /// The ladder is ONE list. Both readers — `vulkan_moe_binder`'s residency / auto-q8 /
    /// streaming sweeps and `kv_fit_ctx_for` — call [`super::ubatch_candidates`], so a rung added
    /// or removed cannot reach one and miss the other. This pins its shape, including the two
    /// rules that are easy to lose in a refactor: the current height leads, and a user-pinned
    /// `INFR_UBATCH` collapses the list to that one height.
    #[test]
    fn dense_ubatch_ladder_is_the_only_one() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        assert_eq!(super::DENSE_UBATCH_LADDER, [512, 256, 128]);
        let unset = EngineConfig::default();
        assert_eq!(super::ubatch_rows(&unset), 1024);
        assert_eq!(super::ubatch_candidates(&unset), vec![1024, 512, 256, 128]);

        // Rungs at or above the current height are filtered out — a SHRINK ladder must never
        // raise an integrated GPU past its watchdog-safe default.
        let pinned = |rows: usize| EngineConfig {
            device: infr_core::config::DeviceCfg {
                ubatch: Some(rows),
                ubatch_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(super::ubatch_candidates(&pinned(256)), vec![256]);
        assert_eq!(super::ubatch_candidates(&pinned(2048)), vec![2048]);
    }

    /// The refuse rung's input: when the weights leave no usable room, the fit reports the honest
    /// small number (possibly `0`) rather than a floored 1024 that reads as "a session fits".
    /// `clamp_default_ctx` turns that into an error naming the numbers; that half needs a live
    /// backend, so it is only exercised by the GPU tests.
    #[test]
    fn kv_fit_reports_below_the_floor_instead_of_pretending() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        // Weights fill the card: room left is under the 256 MiB fixed activation reserve alone.
        let fit = super::kv_fit_ctx_for(
            &cfg,
            &ec,
            XTX_ROOM - 64 * 1024 * 1024,
            XTX_ROOM,
            XTX_TOTAL,
            DType::F16,
            DType::F16,
        )
        .expect("has KV");
        assert!(
            fit < super::MIN_SESSION_CTX,
            "must report the real (unusable) fit, got {fit}"
        );
    }

    /// A pure recurrent-state arch has no per-token KV to size — `None`, not `0`, so callers keep
    /// the trained window instead of clamping to nothing.
    #[test]
    fn kv_fit_is_none_without_a_per_token_cache() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = Config {
            n_layer: 24,
            n_ctx_train: 32768,
            ..Default::default() // n_kv == head_dim == 0
        };
        assert!(super::kv_fit_ctx_for(
            &cfg,
            &EngineConfig::default(),
            1 << 30,
            XTX_ROOM,
            XTX_TOTAL,
            DType::F16,
            DType::F16,
        )
        .is_none());
    }

    #[test]
    fn placement_pins_are_per_scope_not_process_global() {
        // The multi-model fix: two independent sessions' pins are isolated. Session A pins a chunk
        // and q8; inside A's scope the readers see them, and OUTSIDE any scope (or in a fresh
        // session B's scope) they are unset — the old process-global `OnceLock` leaked A's decision
        // into B (a silent no-op `.set()`).
        let a = std::sync::Arc::new(PlacementPins::default());
        let b = std::sync::Arc::new(PlacementPins::default());
        {
            let _sa = PlacementScope::enter(a.clone());
            super::pin_ubatch(512);
            super::pin_kv_auto_q8();
            assert!(super::kv_auto_q8(), "A's scope sees its own q8 pin");
            assert_eq!(a.ubatch.get().copied(), Some(512));
        }
        {
            let _sb = PlacementScope::enter(b.clone());
            assert!(!super::kv_auto_q8(), "B must NOT inherit A's q8 pin");
            assert_eq!(
                b.ubatch.get().copied(),
                None,
                "B must NOT inherit A's chunk"
            );
        }
    }
}
