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
use infr_core::backend::{Backend, Buffer, BufferUsage, Capabilities};
use infr_core::tensor::DType;
use infr_core::WeightSource;
use infr_cpu::CpuBackend;
use infr_gguf::{Gguf, TensorBytes};

pub mod model;
mod ple;
mod runner;
mod sc;
mod segmented_kv;
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

fn host_ram_request(ec: &EngineConfig) -> infr_core::hostmem::RamRequest {
    let total_host_ram = infr_core::hostmem::total_bytes();
    let total_process_budget = ec.device.ram_budget.and_then(|size| match size {
        infr_core::SizeSpec::Bytes(bytes) => Some(bytes),
        infr_core::SizeSpec::Percent(_) => match total_host_ram {
            Some(total) => Some(size.resolve(total)),
            None => {
                tracing::warn!(
                    "device.ram_budget is a percentage, but total physical RAM could not be \
                     detected; falling back to automatic host-memory sizing"
                );
                None
            }
        },
    });
    let legacy_cache_budget = ec.paging.dram.map(|size| size.resolve(0));
    if total_process_budget.is_some() && legacy_cache_budget.is_some() {
        tracing::warn!(
            "both device.ram_budget and legacy paging.dram are set; using device.ram_budget"
        );
    }
    infr_core::hostmem::RamRequest::from_config(
        total_process_budget,
        legacy_cache_budget,
        ec.paging.dram_bypass,
    )
}

fn log_host_ram_request(
    what: &str,
    request: infr_core::hostmem::RamRequest,
    process_resident: Option<u64>,
    cache_bytes: u64,
) {
    match (request, process_resident) {
        (infr_core::hostmem::RamRequest::TotalProcessBudget(total), Some(resident)) => {
            tracing::info!(
                total_process_ram_budget_bytes = total,
                observed_process_resident_bytes = resident,
                host_cache_budget_bytes = cache_bytes,
                "{what} host tier: resolved total-process RAM budget"
            )
        }
        (infr_core::hostmem::RamRequest::TotalProcessBudget(total), None) => tracing::warn!(
            total_process_ram_budget_bytes = total,
            host_cache_budget_bytes = cache_bytes,
            "{what} host tier: this platform could not measure process resident RAM; treating \
             device.ram_budget as a best-effort host-cache ceiling"
        ),
        (infr_core::hostmem::RamRequest::LegacyCacheBudget(bytes), _) => tracing::warn!(
            legacy_host_cache_bytes = bytes,
            "{what} host tier: paging.dram / INFR_DRAM_CACHE is a deprecated raw cache override; \
             use device.ram_budget / INFR_RAM_BUDGET for a total-process RAM budget"
        ),
        _ => {}
    }
}

/// The CPU seam weight binder, hoisted so the CPU runner and every CPU `verify_*` entry share one
/// copy: map an mmap slice zero-copy; alloc+upload owned bytes (the combined gate+up concat — never
/// produced for CPU since `combined_gu` is false there, but stays correct if it ever is).
/// Build the CPU backend's host weight cache, or `None` to keep the zero-copy mmap path.
///
/// # When this turns itself on
/// **Only when the weights do not fit.** The CPU backend has no VRAM ladder to tell it that, the
/// way the Vulkan paths do (see [`vulkan_host_tier`], whose callers are already past that
/// decision), so it asks directly: do the pageable weights fit the host memory this machine can
/// actually spare? If they do, the mmap path is right and stays — it is zero-copy, the page cache
/// holds the model, and an arena would only add copies. If they do not, the page cache is about to
/// thrash on the cyclic sweep a forward pass performs (`docs/perf/results.md` measured the
/// collapse: decode 23-33x slower once the weights stop fitting), and the arena's own policy is
/// what fixes it.
///
/// An explicit `device.ram_budget` always wins in BOTH directions — it forces the arena on a model
/// that would have fit and sets the process's total resident-RAM target. The current working set
/// and a small allowance for process objects created after planning are subtracted before this
/// arena is sized. Legacy `paging.dram` remains an exact-cache diagnostic override only.
///
/// Returns `None` — not an error — when nothing seats, so every degraded case falls back to
/// today's behaviour rather than failing the load.
fn cpu_paged_store(
    ec: &EngineConfig,
    g: &Gguf,
) -> AResult<Option<std::sync::Arc<infr_cpu::paged::PagedWeights>>> {
    // `0` is the explicit OFF switch, not "unset" — preserve it through request resolution.
    let ram_request = host_ram_request(ec);
    // Only the weights this backend would actually page count toward "does it fit": the sub-floor
    // tensors stay mapped either way, so counting them could page a model that fits.
    let pageable: u64 = g
        .tensors()
        .iter()
        .filter(|t| t.nbytes >= infr_cpu::paged::MIN_PAGED_BYTES)
        .map(|t| t.nbytes as u64)
        .sum();
    let available = infr_core::hostmem::available_bytes();
    let process_resident = infr_core::hostmem::process_resident_bytes();
    let arena_plan =
        infr_core::hostmem::cpu_arena_plan(ram_request, available, process_resident, pageable);
    let cache_bytes = match arena_plan {
        infr_core::hostmem::ArenaPlan::Take(bytes) => bytes,
        _ => 0,
    };
    log_host_ram_request("CPU", ram_request, process_resident, cache_bytes);
    let budget = match arena_plan {
        // Only the GPU tiers can want a reader with no cache under them (their arena IS the cache
        // on unified memory). For the CPU backend this arena is the only tier there is, so a
        // read-through with nothing behind it would be a pure regression on the mapping.
        infr_core::hostmem::ArenaPlan::StreamOnly => {
            debug_assert!(false, "cpu_arena_plan never answers StreamOnly");
            return Ok(None);
        }
        infr_core::hostmem::ArenaPlan::Take(n) => {
            if ram_request == infr_core::hostmem::RamRequest::Auto {
                tracing::info!(
                    "host paging: {:.2} GB of weights exceed the {:.2} GB of host memory \
                     available, so they stream from disk through a {:.2} GB arena instead of the \
                     OS page cache (set INFR_RAM_BUDGET to override)",
                    pageable as f64 / 1e9,
                    available.unwrap_or(0) as f64 / 1e9,
                    n as f64 / 1e9,
                );
            }
            n as usize
        }
        infr_core::hostmem::ArenaPlan::Skip(why) => {
            use infr_core::hostmem::Skip;
            // `Fits` and `NoProbe` are the ordinary paths — the weights are mapped, exactly as
            // before this tier existed — so neither says anything. `TooLittle` is the one case
            // where the run is about to be slow for a reason the user can act on.
            if why == Skip::TooLittle {
                tracing::warn!(
                    "host paging: {:.2} GB of weights do not fit the {:.2} GB of host memory \
                     available, but too little is free to seat a useful arena — falling back to \
                     the OS page cache, which thrashes on a forward pass's cyclic sweep. Free \
                     memory, or set INFR_RAM_BUDGET explicitly",
                    pageable as f64 / 1e9,
                    available.unwrap_or(0) as f64 / 1e9,
                );
            }
            return Ok(None);
        }
    };
    let plans = infr_cpu::paged::plan_pools(budget, g.tensors());
    if plans.is_empty() {
        tracing::warn!(
            "host paging: a {:.2} GB budget seats no weight class of this model — keeping the \
             mmap path (raise INFR_RAM_BUDGET enough to leave room for one tensor)",
            budget as f64 / 1e9,
        );
        return Ok(None);
    }
    let io = std::sync::Arc::new(
        infr_core::blockio::FileBlockIo::open_shards(&g.shards()).map_err(|e| anyhow!("{e}"))?,
    );
    let store = infr_cpu::paged::PagedWeights::new(&plans, io).map_err(|e| anyhow!("{e}"))?;
    tracing::info!(
        "host paging: {} weight class(es), {:.2} GB arena of a {:.2} GB budget",
        plans.len(),
        store.arena_bytes() as f64 / 1e9,
        budget as f64 / 1e9,
    );
    Ok(Some(std::sync::Arc::new(store)))
}

/// Build the host DRAM tier under a set of Vulkan arena pools — one
/// [`infr_core::hostpager::HostPager`] per pool, since a pool is already exactly a block-size class,
/// which is the uniform-slot shape the host tier needs.
///
/// `classes` is `(slot stride, block count)` per pool, in pool order; the result matches that
/// order, `None` where the budget seated nothing. `what` names the caller in the log line — the
/// dense streaming pools and the MoE expert pools both come through here, and a run reporting one
/// arena should say which. A budget too small to seat a pool leaves that pool on the mmap path
/// rather than failing the load.
///
/// # Where the budget comes from
/// **Both call sites are already past the point where the model did not fit VRAM** — dense
/// streaming and the paged MoE cache are each only reached when residency was rejected — so
/// reaching here IS the signal that this run has to stream. An explicit `device.ram_budget` still
/// wins; otherwise the budget is sized from what the host can actually spare
/// ([`infr_core::hostmem`]), because a tier that is off by default helps nobody on the one run
/// that needs it, and the alternative is a user guessing a number that is worth 1.6x
/// (`docs/perf/results.md`).
///
/// # Forcing this path on hardware that would not take it
/// Auto-sizing never REPLACES the two knobs, because a machine big enough to hold the model
/// resident is exactly the machine you want to test streaming on. `INFR_CACHE` (`paging.cache`)
/// caps the VRAM paging budget, which is what forces residency to be rejected and this function to
/// be reached at all; `INFR_RAM_BUDGET` (`device.ram_budget`) then pins total process RAM instead
/// of letting it be measured. Both together can force this path on a machine that would otherwise
/// keep the model resident. Legacy `paging.dram` pins only the raw host-cache size.
///
/// # Unified memory has only ONE host tier, and it is the arena above this one
/// `unified` is `DeviceCaps::unified_memory` — an iGPU or APU whose "VRAM" IS system RAM. There the
/// arena above already sits in host memory, so a host tier beneath it would hold extra blocks in
/// the same RAM that the GPU CANNOT read directly: every hit would still be copied through the
/// staging ring, while the identical bytes spent on the arena above are read in place. It is not
/// merely double-counted, it is strictly worse than making that arena bigger. So auto-sizing
/// declines on unified memory and says why; an explicit `device.ram_budget` is still honoured,
/// because a user asking for it by name may be working around something this does not model.
fn vulkan_host_tier(
    ec: &EngineConfig,
    g: &Gguf,
    what: &str,
    classes: &[(usize, usize)],
    unified: bool,
) -> AResult<Vec<Option<std::sync::Arc<infr_core::hostpager::HostPager>>>> {
    let unpaged = || vec![None; classes.len()];
    // `0` is the explicit OFF switch, not "unset" — preserve it through request resolution.
    let ram_request = host_ram_request(ec);
    let available = infr_core::hostmem::available_bytes();
    let process_resident = infr_core::hostmem::process_resident_bytes();
    let pageable: u64 = classes.iter().map(|&(s, n)| (s * n) as u64).sum();
    let arena_plan = infr_core::hostmem::streaming_arena_plan(
        ram_request,
        available,
        process_resident,
        unified,
        pageable,
    );
    let cache_bytes = match arena_plan {
        infr_core::hostmem::ArenaPlan::Take(bytes) => bytes,
        _ => 0,
    };
    log_host_ram_request(what, ram_request, process_resident, cache_bytes);
    let budget = match arena_plan {
        // Unified memory: the arena above is already GPU-accessible RAM, so there is nothing
        // to cache down here — but its misses still come from BLOCK reads instead of the
        // mapping, which is what lets a big model run on these machines at all.
        infr_core::hostmem::ArenaPlan::StreamOnly => {
            let io = std::sync::Arc::new(
                infr_core::blockio::FileBlockIo::open_shards(&g.shards())
                    .map_err(|e| anyhow!("{e}"))?,
            );
            tracing::info!(
                "{what} host tier: streaming DISK -> GPU-accessible RAM with no host cache — \
                     this device's memory IS host memory, so the {} pool(s) above are already the \
                     only useful cache and a second one would hold bytes the GPU cannot read in \
                     place. Raise INFR_CACHE to cache more; INFR_RAM_BUDGET forces a host arena",
                classes.len(),
            );
            let mut out = Vec::with_capacity(classes.len());
            for &(slot_bytes, _) in classes {
                let p = infr_core::hostpager::HostPager::stream_only(slot_bytes, io.clone())
                    .map_err(|e| anyhow!("{e}"))?;
                out.push(Some(std::sync::Arc::new(p)));
            }
            return Ok(out);
        }
        infr_core::hostmem::ArenaPlan::Take(n) => {
            if ram_request == infr_core::hostmem::RamRequest::Auto {
                tracing::info!(
                    "{what} host tier: sized automatically to {:.2} GB of {:.2} GB available host \
                     memory (set INFR_RAM_BUDGET to override)",
                    n as f64 / 1e9,
                    available.unwrap_or(0) as f64 / 1e9,
                );
            }
            n as usize
        }
        infr_core::hostmem::ArenaPlan::Skip(why) => {
            use infr_core::hostmem::Skip;
            match why {
                Skip::NoProbe => tracing::warn!(
                    "{what} host tier: this model must stream, but there is no host-memory probe \
                     on this platform, so the arena cannot be sized automatically — set \
                     INFR_RAM_BUDGET to page weights through DRAM instead of leaving them to the \
                     OS page cache"
                ),
                Skip::TooLittle => tracing::warn!(
                    "{what} host tier: this model must stream, but only {:.2} GB of host memory \
                     is available — too little to seat a useful arena, so the weights stay on the \
                     OS page cache. Free memory, or set INFR_RAM_BUDGET explicitly",
                    available.unwrap_or(0) as f64 / 1e9,
                ),
                // Off by name: say so once, because a user who set it and then wonders why
                // streaming is slow should find the reason in the log.
                Skip::Disabled => tracing::info!(
                    "{what} host tier: disabled by an explicit zero RAM setting — weights stream \
                     from the OS page cache instead of an arena"
                ),
                // Unreachable here by construction: `streaming_arena_plan` is for callers already
                // past the fit decision, so it never answers `Fits`.
                Skip::Fits => tracing::warn!("{what} host tier: not built (weights fit)"),
            }
            return Ok(unpaged());
        }
    };
    let slots = infr_core::hostpager::plan_slots(budget, classes);
    if slots.iter().all(|&n| n == 0) {
        tracing::warn!(
            "{what} host tier: a {:.2} GB budget seats no block class of this model — keeping \
             the mmap path (raise INFR_RAM_BUDGET enough to leave room for one block)",
            budget as f64 / 1e9,
        );
        return Ok(unpaged());
    }
    let io = std::sync::Arc::new(
        infr_core::blockio::FileBlockIo::open_shards(&g.shards()).map_err(|e| anyhow!("{e}"))?,
    );
    let mut arena = 0usize;
    let mut out = Vec::with_capacity(classes.len());
    for (&n_slots, &(slot_bytes, _)) in slots.iter().zip(classes) {
        if n_slots == 0 {
            out.push(None);
            continue;
        }
        let p = infr_core::hostpager::HostPager::new(n_slots, slot_bytes, io.clone())
            .map_err(|e| anyhow!("{e}"))?;
        arena += p.arena_bytes();
        out.push(Some(std::sync::Arc::new(p)));
    }
    tracing::info!(
        "{what} host tier: {} of {} pool(s) paged, {:.2} GB arena of a {:.2} GB budget",
        slots.iter().filter(|&&n| n > 0).count(),
        classes.len(),
        arena as f64 / 1e9,
        budget as f64 / 1e9,
    );
    Ok(out)
}

fn cpu_upload_bind(be: &CpuBackend) -> Box<BindWeight<'_>> {
    cpu_bind_with(be, None)
}

/// The CPU binder. With a `store`, a weight whose bytes come straight off disk (no load-time
/// rewrite) and whose size class has a pool is registered and read on demand; everything else takes
/// the same paths as before.
fn cpu_bind_with<'a>(
    be: &'a CpuBackend,
    store: Option<std::sync::Arc<infr_cpu::paged::PagedWeights>>,
) -> Box<BindWeight<'a>> {
    Box::new(move |_name, tb, dt, _n| {
        if let Some(store) = &store {
            // `file_ranges` is `None` exactly for bytes the loader REWROTE (the qwen2 q/k permute,
            // the BitNet dequant), which correspond to nothing on disk and so cannot be re-read.
            if let Some(ranges) = tb.file_ranges() {
                if let Some(id) = store.register(&ranges).map_err(|e| anyhow!("{e}"))? {
                    let nbytes = ranges.iter().map(|(_, l)| l).sum();
                    return Ok((be.paged_weight(store.clone(), id, nbytes), dt));
                }
            }
        }
        cpu_bind_resident(be, tb, dt)
    })
}

/// The non-paged half of the CPU binder: map an mmap slice zero-copy, or materialize owned/fused
/// bytes into one host buffer.
fn cpu_bind_resident(be: &CpuBackend, tb: WBytes, dt: DType) -> AResult<(Box<dyn Buffer>, DType)> {
    match tb {
        WBytes::Mmap(tb) => Ok((be.map_weight(tb), dt)),
        // Owned bytes (a load-time rewrite) and a fused group both become one host buffer here.
        // The concat is materialized only on this arm — a binder that pages the group never
        // reaches it.
        other => {
            let v = other.materialize();
            let buf = be
                .alloc(v.len().max(1), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(buf.as_ref(), &v).map_err(|e| anyhow!("{e}"))?;
            Ok((buf, dt))
        }
    }
}

/// The Metal seam weight binder (raw native-dtype upload; the backend dequantizes lazily), hoisted
/// so the Metal session runner and `verify_dense_metal2` share one copy.
#[cfg(target_os = "macos")]
fn metal_upload_bind(be: &infr_metal::MetalBackend) -> Box<BindWeight<'_>> {
    Box::new(move |_name, tb, dt, _n| {
        let buf = be
            .alloc(tb.len().max(1), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(buf.as_ref(), &tb.materialize())
            .map_err(|e| anyhow!("{e}"))?;
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
    // Thin CPU wrapper over the backend-generic runner: a CpuBackend + a weight binder that maps
    // each tensor straight from the GGUF mmap (no alloc, no memcpy) — or, when the host RAM plan asks
    // for a host weight cache, registers the big ones to be read from the file on demand.
    let store = cpu_paged_store(ec, g)?;
    let bind = cpu_bind_with(&cpu_be, store.clone());
    let out = generate_dense_backend(
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
        None, // turn checkpoint boundary
        req,
        None, // mm
    );
    if let Some(store) = store.filter(|_| ec.paging.stats) {
        report_host_paging(&store);
    }
    out
}

/// `paging.stats` (`INFR_PAGER_STATS=1`): what the host tier actually did, per size class.
///
/// The hit rate alone cannot distinguish a tier that is working from one that was never asked, so
/// the read count and the bytes moved are reported beside it: those are what a run's throughput is
/// bounded by, and what the mmap baseline in `docs/perf/results.md` is compared against.
fn report_host_paging(store: &infr_cpu::paged::PagedWeights) {
    for (slot_bytes, n_slots, s) in store.pool_stats() {
        tracing::info!(
            "[host pager] {:.1}MB x {n_slots} slots: {} hits, {} misses ({:.1}% hit), {} evictions, \
             {} reads, {:.2} GB from disk",
            slot_bytes as f64 / 1e6,
            s.pager.hits,
            s.pager.misses,
            s.pager.hit_rate() * 100.0,
            s.pager.evictions,
            s.reads,
            s.bytes_read as f64 / 1e9,
        );
    }
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
        None, // turn checkpoint boundary
        None, // constraint
        None, // req: the one-shot runner is a sole sequence — config sampling, no gate
        None, // mm
    )
}

/// [`generate_dense_vulkan`] with a caller-held [`SeamKv`]: hold `state` (+ a `want_ctx` capacity)
/// across calls and each turn prefills only the suffix that differs from the cached tokens —
/// ChatSession-style KV reuse on the agnostic seam.
#[derive(Clone, Copy)]
pub(crate) enum TurnCheckpoint {
    /// Allocate the rolling recurrent snapshot before session allocation finalization, without
    /// taking a snapshot during this warmup call.
    Enable,
    /// Allocate if needed and capture state after this many prompt tokens.
    Boundary(usize),
}

/// One image span's ViT embeddings as the seam consumes them (stage V4b): the rows overwrite the
/// token-embedding-table rows of the span's `<|image_pad|>` tokens during host prefill embedding,
/// and the span's tokens rope on a (T,H,W) sub-grid instead of the linear position.
pub struct ImageSpanEmbeds {
    /// Token index (in the EXPANDED prompt token array) of the span's first token.
    pub start: usize,
    /// Number of tokens the span occupies (nx*ny merged tokens).
    pub n_tokens: usize,
    /// ViT output rows, [n_tokens * n_embd] f32 row-major.
    pub embeds: std::sync::Arc<Vec<f32>>,
}

/// The vision mrope plan a caller hands [`generate_dense_vulkan_session`] (and, transitively,
/// `generate_dense_backend`) for a turn whose rendered prompt contains `<|image_pad|>` tokens and
/// whose image embeddings come from the ViT (`infr-vision`). `None` on every text-only turn —
/// byte-for-byte the pre-V4b graph and uploads.
pub struct MropePlan {
    /// Per-token (T,H,W,E) for every PROMPT token, [plen*4] i32 row-major.
    pub prompt_pos4: Vec<i32>,
    pub spans: Vec<ImageSpanEmbeds>,
    /// Rope position of generated token i (i = 0,1,2... after the prompt): decode_base + i.
    pub decode_base: i32,
}

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
    turn_checkpoint: Option<TurnCheckpoint>,
    constraint: Option<&mut crate::grammar::Constraint>,
    req: Option<&crate::sampling::RequestCtx>,
    // Vision mrope plan (stage V4b): `None` on every text-only caller.
    mm: Option<&MropePlan>,
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
        match vulkan_moe_binder(vk, g, cfg, ec, true, want_ctx) {
            Ok(bind) => bind,
            Err(error) => {
                vk.release_moe_load_reservation();
                return Err(error);
            }
        }
    };
    let out = generate_dense_backend(
        vk,
        &*bind,
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
        turn_checkpoint,
        req,
        mm,
    );
    if out.is_err() {
        vk.release_moe_load_reservation();
    }
    let out = out?;
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
/// Every term is per ROW of the prefill chunk, and the whole estimate is now checked against the
/// backend's own high-water mark of live activation bytes at the end of every generation (see
/// `Backend::activation_peak` and the runner's `activation reserve too low` warning), so the
/// numbers below are a fit to measurements rather than an argument:
/// - Internal graph tensors (`alloc_scratch`): fused gate_up out `[rows, 2*n_ff]` f32 + activated
///   intermediate `[rows, n_ff]` f32 + fused qkv staging + ~a dozen `[rows, n_embd]`-class f32/f16
///   temps. Modeled as `12*n_ff + 96*n_embd` per row (the n_embd umbrella also absorbs the
///   lin_a16/mmq activation-quant pools, which are n_embd/n_ff-wide f16/i8).
/// - `nonfa_pv`/`flash_po`: `8*rows*n_head*head_dim*4` per DISTINCT head shape — gemma4 alternates
///   SWA(256)/full(512) head dims, so BOTH pools live at once →
///   `32*n_head*(head_dim + head_dim_swa-if-distinct)` per row.
/// - `nonfa_s` (score tiles, non-flash tier only): full-context layers whose head dim has no
///   FlashAttention implementation reserve the full context; when all full-context layers use
///   flash and only SWA layers miss it, the score span is just `window + chunk`. The established
///   hd128 path is always scoreless. The hd256 path is scoreless only on a device with the exact
///   f16 coopmat/shared-memory capabilities its Vulkan shader requires; other devices retain the
///   conservative non-flash reserve.
///   `n_head*rows*kv_pad*2`, kv_pad = kv_len rounded up to 256, i.e. `2*n_head*ctx_pad` per row at
///   the final context. A phase's first mixed-geometry recording may need one buffer per distinct
///   size because a smaller buffer already referenced by that command stream cannot be released
///   when a later layer asks for more capacity. Once that execute drains, the adapter retains only
///   the high-water tile. Uniform-hd-128 models, plus capability-qualified hd256 models, ride the
///   single-pass flash tier: no score tiles, only the (negligible) flash_pm/pl partials — term
///   skipped when no SWA layer remains.
///
/// Times [`ACT_RESERVE_PAD`]. What is deliberately NOT here any more: a fixed 256 MiB that stood
/// in for gpu-allocator block granularity, retained upload staging and weight-buffer padding.
/// Those are not activations at all — they are exactly what the runner's post-load re-clamp
/// ([`reclamp_ctx_to_live_room`]) prices by ASKING the device, so carrying an estimate of them
/// here as well is double-counting, and the estimate was 3.5x wrong at a 128-row chunk.
///
/// Always taken at an EXPLICIT chunk height: every caller (the try-resident sweep, the streaming
/// budget, the context-fit math) walks [`ubatch_candidates`] rather than assuming the default
/// 1024-row chunk — assuming it is what let the KV-format decision and the placement decision
/// disagree.
pub(crate) fn dense_act_reserve_at(
    cfg: &Config,
    caps: &Capabilities,
    want_ctx: usize,
    ubatch: usize,
) -> u64 {
    // Prefill GEMM outputs pad rows to 64 (see the Vulkan adapter's `alloc_scratch`). Qwen3.8 now
    // builds the same ubatch-height graph as the other supported architectures, including its
    // four-stream residual, PLE and row-aware QSA scratch, so it must reserve the real row count.
    let rows = ubatch.min(want_ctx).max(1).next_multiple_of(64) as u64;
    // Only Op::Attention uses the pooled score/PV scratch below. Op::Mla scans compressed KV and
    // accumulates softmax/value inside its dedicated kernel, while recurrent mixers have no
    // context attention at all. Keep n_layer == 0 conservative for geometry-only configs/tests.
    let ordinary_attention = |l: usize| !cfg.is_mla_layer(l) && !cfg.is_recurrent_layer(l);
    let has_ordinary_attention = cfg.n_layer == 0 || (0..cfg.n_layer).any(ordinary_attention);
    // Attention pv accumulators: one pool per distinct (n_head, head_dim) shape.
    let hd_shapes = if cfg.swa_window > 0 && cfg.head_dim_swa != cfg.head_dim {
        cfg.head_dim + cfg.head_dim_swa
    } else {
        cfg.head_dim
    };
    let attn_pv = if has_ordinary_attention {
        32 * cfg.n_head * hd_shapes
    } else {
        0
    };
    // Non-flash score tiles (see the doc above). Keep hd128's established reservation behavior;
    // hd256 is scoreless only when the device can actually take M2's dedicated shader. This must
    // mirror the adapter's `flash_hd` gate rather than infer support from model geometry alone:
    // NVIDIA/Intel/older drivers with a smaller shared-memory or coopmat tier still run non-FA.
    let flash_scoreless = |hd: usize| {
        hd == 128
            || (hd == 256
                && caps.f16_coopmat()
                && caps.max_shared_memory_bytes >= infr_vulkan::FLASH_HD256_BM16_SHARED)
    };
    // `n_layer == 0` appears in geometry-only tests/config fragments; preserve the historical
    // conservative assumption that such a shape represents at least one full-attention layer.
    let has_full =
        cfg.n_layer == 0 || (0..cfg.n_layer).any(|l| ordinary_attention(l) && !cfg.is_swa_layer(l));
    let has_swa = (0..cfg.n_layer).any(|l| ordinary_attention(l) && cfg.is_swa_layer(l));
    let full_needs_scores = has_full && !flash_scoreless(cfg.head_dim);
    let kv_span = if full_needs_scores {
        want_ctx
    } else if has_swa {
        // FlashAttention is causal-only; SWA layers still use non-FA, but their cache is bounded.
        want_ctx.min(cfg.swa_window.saturating_add(ubatch))
    } else {
        0
    };
    let attn_s = 2 * cfg.n_head * kv_span.next_multiple_of(256);
    // MoE expert scratch (`moe_*` / `moe_pgb_*` in the Vulkan adapter). Its pools are sized by
    // (row, expert) PAIRS, not rows — the batched path buckets every row's `n_used` picks and runs
    // them through the expert FFN together — so the per-row cost carries an `n_used` multiplier.
    // Per pair: the gate+up output `2*n_ff_exp` f32, the activated intermediate `n_ff_exp` f32,
    // the down-projection's f32 result `n_embd`, and the int8 activation-quant pools (one byte per
    // element plus two f16 scales per 32-element block, on both the `n_embd` and `n_ff_exp` sides).
    let moe = cfg.moe.as_ref().map_or(0, |m| {
        let per_pair = 3 * m.n_ff_exp * 4 + cfg.n_embd * 4 + m.n_ff_exp + cfg.n_embd;
        // The paged/batched MoE executor also holds routing, bucket/scatter and quantized-
        // activation pools beside the expert pair buffers. Their exact set varies with expert
        // dtype, so use the backend high-water measurement's stable n_embd envelope. Qwen3.6
        // A3B at 512 rows measured 540 MiB live against 476 MiB without this term; 48*n_embd
        // raises the reserve to 548 MiB while leaving dense models unchanged.
        m.n_used * per_pair + 48 * cfg.n_embd
    });
    // qwen35's gated-DeltaNet mixer scratch, which the `n_embd` umbrella above does not cover: its
    // buffers are keyed on the SSM dims, not on n_embd, and a hybrid model held 1.53x the umbrella
    // (Qwen3.5-9B, `activation reserve too low` at a 1024-row chunk). Named one for one after the
    // `dn_*` internals the runner's graph declares, all f32 and all `batch`-wide, so the two lists
    // can be read side by side: qkv + conv out (conv channels each), z (d_inner), q + k (key dim
    // each), v + out (value dim each), beta + alpha (one per v-head).
    // Plus its attention out-gate pair (`qg` + `gate_a`), which every arch declares but only this
    // one makes big: qwen35 interleaves q and gate in one projection, so `qg` is DOUBLE the q
    // width and the umbrella's n_embd term no longer covers the three of them.
    let deltanet = if cfg.qwen35 || cfg.qwen4exp {
        4 * (2 * cfg.q35_conv_channels()
            + cfg.ssm_d_inner
            + 2 * cfg.q35_num_k_heads() * cfg.q35_head_k_dim()
            + 2 * cfg.q35_num_v_heads() * cfg.q35_head_v_dim()
            + 2 * cfg.q35_num_v_heads())
            + 12 * cfg.n_head * cfg.max_head_dim()
    } else {
        0
    };
    // Qwen3.8's caller-owned wide residual; qwen_alt/normed/gate scratch; and PLE
    // key/query/gated/conv rows are eight f32 `[rows, hc*n_embd]` buffers in total. The low-rank
    // projection and per-stream injection are f32 too. The generic n_embd umbrella cannot absorb
    // these hc-wide tensors.
    let qwen4_hc = if cfg.qwen4exp {
        32 * cfg.hc_mult * cfg.n_embd + 4 * cfg.hc_low_rank + 4 * cfg.hc_mult
    } else {
        0
    };
    let per_row =
        (12 * cfg.n_ff + 96 * cfg.n_embd + attn_pv + attn_s + moe + deltanet + qwen4_hc) as u64;
    let row_reserve = rows * per_row * ACT_RESERVE_PAD.0 / ACT_RESERVE_PAD.1;
    // Qwen3.8 keeps its scalar decode graph live while a separately compiled batched-prefill
    // graph owns the row-scaled pools above. Two real runs at d4096 measured the fixed intercept
    // at 54.6 MiB (64 rows: 143.6 MiB peak; 128 rows: 232.7 MiB), so round it up to 64 MiB.
    // This is a fixed plan-overlap cost, not another per-row multiplier.
    let reserve = row_reserve.saturating_add(if cfg.qwen4exp {
        QWEN4_PLAN_OVERLAP_RESERVE
    } else {
        0
    });
    // Fixed slack over the algebraic reserve. Measured on Ornith-1.5-35B (qwen35moe, MTP verify
    // re-prefill, chunk 128): the true peak ran 174 KB over the reserve — the follow-up unified
    // arena growth request then had to carve a fresh window beside the pager's permanent expert
    // slots and failed outright ("cannot create a contiguous window ... expert minimum working
    // set"). 8 MiB absorbs the model-specific residue the per-row terms miss; when the reserve is
    // already generous this is plan headroom, never waste (see the unified arena's sizing).
    reserve.saturating_add(DENSE_ACT_RESERVE_SLACK)
}

/// F16 expansion buffers held by Vulkan while a batched attention op reads a Q8_0 KV cache.
///
/// The adapter uses separate `kvdeq_k` and `kvdeq_v` logical tags. A phase's first recording may
/// hold one buffer per distinct byte size when a later layer grows a tag already referenced by the
/// command stream; after that execute drains, only its high-water capacity remains. Mirror the
/// cold-execute peak here instead of charging once per layer or once per historical depth.
fn q8_prefill_scratch_bytes(
    cfg: &Config,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> u64 {
    if ubatch.min(want_ctx) <= 1 {
        return 0;
    }

    let mut k_sizes = std::collections::BTreeSet::new();
    let mut v_sizes = std::collections::BTreeSet::new();
    for l in 0..cfg.n_layer {
        if cfg.is_mla_layer(l) || cfg.is_recurrent_layer(l) {
            continue;
        }
        let rows = kv_rows_at(cfg, l, want_ctx, ring, ubatch) as u64;
        let (k_row, v_row) = kv_row_elems(cfg, l);
        if k_fmt == DType::Q8_0 {
            k_sizes.insert(rows.saturating_mul(k_row as u64).saturating_mul(2));
        }
        if v_fmt == DType::Q8_0 {
            v_sizes.insert(rows.saturating_mul(v_row as u64).saturating_mul(2));
        }
    }

    k_sizes
        .into_iter()
        .chain(v_sizes)
        .fold(0u64, u64::saturating_add)
}

/// Runtime workspace priced by placement and checked against Vulkan's measured activation peak.
/// This includes the graph's ordinary activation pools plus format-specific KV read scratch.
pub(crate) fn runtime_reserve_at(
    cfg: &Config,
    caps: &Capabilities,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> u64 {
    dense_act_reserve_at(cfg, caps, want_ctx, ubatch).saturating_add(q8_prefill_scratch_bytes(
        cfg, want_ctx, ring, ubatch, k_fmt, v_fmt,
    ))
}

/// Number of ubatches in one Qwen3.8 layer-major prefill group. Eight leaves enough same-layer
/// work to amortize/pipeline expert staging without retaining a long prompt's four residual
/// streams all at once.
pub(crate) const QWEN4_PREFILL_GROUP_CHUNKS: usize = 8;
// RX 7900 XTX pp8192 measured the combined group + graph high-water mark 18 MiB above the
// algebraic reserve. Keep a small fixed cushion so boundary placements do not depend on that
// model-specific scratch residue.
const QWEN4_PREFILL_GROUP_PAD: u64 = 32 * 1024 * 1024;

/// Activation bytes LAYER-MAJOR prefill holds ON TOP of [`dense_act_reserve_at`]. Ordinary
/// streaming models retain the whole prompt's residual rows. Qwen3.8 retains one bounded group,
/// including its caller-owned four-stream residual and PLE rows.
///
/// Priced at the full context because that is the longest prompt the session can be handed, and
/// these buffers are allocated mid-prefill out of whatever the arenas left: an under-reserve
/// surfaces as a VRAM-guard refusal on a long prompt, not as a smaller cache. A short prompt
/// really does allocate less, so on a session that never fills its window this is held back and
/// unused — the price of sizing an arena before the prompts arrive.
///
/// Zero when the session prefills chunk-major ([`layer_major_prefill`]).
pub(crate) fn layer_major_act_bytes(cfg: &Config, want_ctx: usize, ubatch: usize) -> u64 {
    let ctx = want_ctx.max(1);
    let ub = ubatch.clamp(1, ctx);
    if cfg.qwen4exp {
        let rows = ctx.min(ub.saturating_mul(QWEN4_PREFILL_GROUP_CHUNKS)) as u64;
        let residual = cfg.n_embd.saturating_mul(1 + cfg.hc_mult);
        let ple = cfg
            .ple_ngram_size
            .saturating_sub(1)
            .saturating_mul(cfg.ple_heads_per_ngram)
            .saturating_mul(cfg.ple_head_dim);
        rows.saturating_mul(residual.saturating_add(ple).saturating_mul(4) as u64)
            .saturating_add(QWEN4_PREFILL_GROUP_PAD)
    } else {
        ctx.div_ceil(ub) as u64 * ub as u64 * cfg.n_embd as u64 * 4
    }
}

/// Does this session prefill LAYER-MAJOR — every chunk through layer L before any chunk reaches
/// L+1 — instead of running the whole model per chunk?
///
/// The two orders compute the same thing; they differ in what they re-read. Chunk-major sweeps the
/// entire weight set once PER CHUNK. Layer-major can reduce those reads, at the cost of retaining
/// residual streams and introducing one execute boundary per layer/chunk pair
/// ([`layer_major_act_bytes`]). That trade helped one streamed dense model, but regressed paged
/// Qwen3.8 prefill by more than an order of magnitude, so chunk-major is the default everywhere.
///
/// `paging.layer_major = true` is the only way onto this diagnostic path. It needs a backend that
/// carries a bound `Input` from one execute to the next, which is what threads the residual stream
/// between two layers' dispatches, and an architecture whose layer stack can be entered past layer
/// 0 at all (`spannable`).
pub(crate) fn layer_major_prefill(
    ec: &EngineConfig,
    caps: &infr_core::backend::Capabilities,
    spannable: bool,
) -> bool {
    let want = ec.paging.layer_major == Some(true);
    if want && !spannable {
        // gemma4-E2B: its layer stack reads `per_layer_inp`, which the graph PROLOGUE builds, so a
        // span starting past layer 0 would read an unbound tensor. This is the gate for it — the
        // matching `assert!` in `build` is an internal invariant, and a streamed E2B model reached
        // it as a PANIC on an ordinary `bench`/`run` until this arm existed.
        tracing::warn!(
            "layer-major prefill cannot split this architecture's layer stack (per-layer inputs \
             are built by the graph prologue) — prefilling chunk-major"
        );
        return false;
    }
    if want && !caps.graph_input_inplace {
        // Only reachable through the explicit override on a wrapper backend (TP/EP/pipeline) —
        // the auto rule needs a dense pager, which those do not host. Say so rather than silently
        // running the other order.
        tracing::warn!(
            backend = caps.name,
            "layer-major prefill needs a backend that carries a bound graph Input across \
             executes; this one does not — prefilling chunk-major"
        );
        return false;
    }
    want
}

/// Safety pad on [`dense_act_reserve_at`]'s per-row terms, as `(numerator, denominator)`.
///
/// **What it covers.** The terms above name the pools a forward allocates, but WHICH pools it
/// allocates is a tier decision the Vulkan adapter takes per layer, per op, on row count / head
/// dim / mask / KV dtype / coopmat capability — and the per-arch algebra is fit to the graphs that
/// have been measured, not to the ones that have not. This pad is that distance.
///
/// **Sized by measurement, against the backend's own high-water mark** (`Backend::activation_peak`
/// — every row below is reproducible with `RUST_LOG=infr_llama=debug` on the run named, and the
/// runner warns when a peak lands above the reserve). Unpadded model against measured peak, all on
/// a 24 GiB 7900 XTX:
///
/// | model                | chunk | ctx     | modeled MiB | measured MiB | measured / modeled |
/// | -------------------- | ----- | ------- | ----------- | ------------ | ------------------ |
/// | Qwen3.5-4B-MTP Q4_K_M| 1024  |   2 064 |         724 |        1 027 |              1.42x |
/// | Qwen3.5-9B Q4_K_M    | 1024  |   2 064 |         904 |        1 112 |              1.23x |
/// | Llama-3.2-1B Q4_K_M  | 1024  |   2 064 |         406 |          429 |              1.06x |
/// | Qwen3-30B-A3B Q4_K_M | 1024  |     528 |         309 |          291 |              0.94x |
/// | gemma-3-12b Q4_K_M   | 1024  | 131 072 |       3 811 |        4 735 |              1.24x |
/// | gemma-4-31B UD-Q5_K_XL| 128  |  15 440 |         261 |          334 |              1.28x |
///
/// 1.5 is the first rung above the worst of them (the MTP 4B at 1.42x). The hybrid archs are what
/// set it: their DeltaNet mixer and double-width q projection are named terms now, and the residue
/// is still the largest — which is the argument for deriving these bytes from the graph the runner
/// already builds rather than re-deriving them here (backlog B8).
const ACT_RESERVE_PAD: (u64, u64) = (3, 2);
const QWEN4_PLAN_OVERLAP_RESERVE: u64 = 64 * 1024 * 1024;
/// Fixed slack added on top of the algebraic [`dense_act_reserve_at`] model — see the call site
/// for the measurement that motivated it.
const DENSE_ACT_RESERVE_SLACK: u64 = 8 * 1024 * 1024;

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
    let configured = ec.device.ubatch.filter(|&v| v > 0);
    let (placed, moe_cap) = with_placement_pins(|p| {
        (
            match p.ubatch.load(std::sync::atomic::Ordering::Relaxed) {
                0 => None, // nothing pinned
                rows => Some(rows),
            },
            match p.moe_ubatch_cap.load(std::sync::atomic::Ordering::Relaxed) {
                0 => None,
                rows => Some(rows),
            },
        )
    });
    let selected = configured.or(placed).unwrap_or_else(default_ubatch_rows);
    match moe_cap {
        Some(cap) => selected.min(cap),
        None => selected,
    }
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

/// Emergency MoE-only shrink ladder. Unlike [`ubatch_candidates`], this remains available after
/// an explicit `INFR_UBATCH`: the requested height is priced first and is lowered only when its
/// activation reserve leaves less than one complete whole-layer Prefill lane. This is a viability
/// fallback, not a throughput/residency policy sweep.
fn moe_ubatch_fallback_candidates(ec: &EngineConfig) -> Vec<usize> {
    const FALLBACKS: [usize; 4] = [1024, 512, 256, 128];
    let now = ubatch_rows(ec);
    let mut candidates = vec![now];
    candidates.extend(FALLBACKS.into_iter().filter(|&rows| rows < now));
    candidates
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
/// The chunk height has ONE exception to "set once, then stable": the runner's post-load
/// re-clamp ([`reclamp_ctx_to_live_room`]) may LOWER it, because the sweep that pinned it decided
/// against an estimate of the weight bytes and the re-clamp knows the real ones. It still runs
/// before the first KV allocation, so everything that reads the height — the prefill loop, the
/// activation reserve, the SWA ring sizing — still sees one value for the session's whole life.
/// `0` = unpinned (see [`repin_ubatch`]).
#[derive(Default)]
pub(crate) struct PlacementPins {
    ubatch: std::sync::atomic::AtomicUsize,
    /// Emergency upper bound used only when the requested/current chunk leaves no complete MoE
    /// Prefill lane. Kept separate so ordinary placement/re-clamp pins cannot override an explicit
    /// `INFR_UBATCH`.
    moe_ubatch_cap: std::sync::atomic::AtomicUsize,
    kv_q8: std::sync::OnceLock<()>,
    /// Has this session already reported an activation peak above what it reserved (the runner's
    /// `activation reserve too low` warning)? The condition persists for the session's whole life —
    /// the peak is a high-water mark and the reserve is fixed — so without a latch a server would
    /// repeat the same line on every request for as long as it runs.
    act_over_reserve_reported: std::sync::atomic::AtomicBool,
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

/// Pin the placement prefill chunk (rows) for the current session scope, if nothing pinned one
/// yet — the placement sweeps' spelling, which keeps the first decision (a racing second sweep
/// must not move a height an earlier one already priced buffers against).
fn pin_ubatch(rows: usize) {
    with_placement_pins(|p| {
        let _ = p.ubatch.compare_exchange(
            0,
            rows,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        );
    });
}

/// LOWER the pinned prefill chunk, overriding whatever the placement sweeps pinned — the runner's
/// post-load re-clamp only. A shorter chunk shrinks both the activation reserve and the SWA ring,
/// so it buys context; the sweeps chose their height against an ESTIMATE of the weight bytes,
/// and by here the real ones are known. Refuses to RAISE a height (that could outgrow buffers a
/// pre-load decision already sized, and on an integrated GPU it is the watchdog bound).
fn repin_ubatch_lower(rows: usize) {
    with_placement_pins(|p| {
        let cur = p.ubatch.load(std::sync::atomic::Ordering::Relaxed);
        if cur == 0 || rows < cur {
            p.ubatch.store(rows, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

fn cap_moe_ubatch(rows: usize) {
    with_placement_pins(|p| {
        let cur = p.moe_ubatch_cap.load(std::sync::atomic::Ordering::Relaxed);
        if cur == 0 || rows < cur {
            p.moe_ubatch_cap
                .store(rows, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

/// Claim the one-shot "activation reserve too low" report for the current session scope: `true`
/// exactly once, for the first caller that finds the peak above the reserve.
pub(crate) fn claim_act_over_reserve_report() -> bool {
    with_placement_pins(|p| {
        !p.act_over_reserve_reported
            .swap(true, std::sync::atomic::Ordering::Relaxed)
    })
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

/// The automatic Vulkan KV format when the user did not choose one explicitly. Keep this out of
/// [`EngineConfig`]'s generic defaults: CPU/Metal sessions retain their backend-specific behavior,
/// while every Vulkan placement estimate and allocation can ask the same model-layout gate.
pub(crate) fn kv_default_q8(cfg: &Config, ec: &EngineConfig) -> bool {
    kv_unset(ec) && kv_q8_layout_ok(cfg)
}

/// Per-token (K, V) ELEMENT counts for layer `l` — the one answer to "how wide is a row of this
/// layer's KV cache", shared by the runner's allocation, the graph declaration, the fork/seed
/// copies and every footprint estimate. Multiply by a row count for a whole cache side.
///
/// DeepSeek2 MLA caches ONE compressed row per token (`kv_lora_rank + qk_rope_dim` — 576 on
/// V2-Lite) and has **no V cache at all**: V is an aliased prefix view of that same row, so the V
/// side is 0 elements and its buffer/tensor is a placeholder ([`kv_side_elems`]). Every other arch
/// stores `n_kv * head_dim` per side. Reading it off `layer_n_kv * layer_head_dim` for an MLA model
/// yields `1 * 192` — a third of what the kernels index — which is exactly what `SeamKv::fork`,
/// `SeamKv::seed_from` and the VRAM estimate each did while the allocation carried its own copy of
/// the branch (docs/backlog.md B41).
///
/// **deepseek32 (V3.2) puts its lightning indexer's SECOND cache on that free V side**: one
/// `indexer_head_size`-wide row per token per layer, on top of the compressed MLA row. It is a
/// genuinely independent per-token cache written by its own `Op::WriteKv` and read by
/// `Op::LightningIndexer`, and it is carried here — rather than as a third per-layer buffer — so
/// that the SIX sites this helper feeds (allocation, graph declaration, `SeamKv::fork`,
/// `SeamKv::seed_from`, and both VRAM estimates) size and copy it without any of them growing a
/// private branch. Under-reserving it is exactly the failure B41 records. The precedent is
/// `MixerW::DeltaNet`, which puts a qwen35 layer's recurrent `S` state in the same `v_cache[l]`
/// slot; the difference is that DeltaNet's state is fixed-size and this one is per-token, which is
/// what makes it fit the K side's own geometry.
///
/// The indexer cache **must never ring**: `Op::LightningIndexer` masks causally only, so position 0
/// stays eligible for every query row and every backend refuses a cache holding fewer rows than
/// `kv_len`. That is why it is sized off [`kv_rows`] like any full-context side — V3.2 has no
/// sliding-window layer, so `kv_rows` returns the whole context — and why the allocation asserts it
/// (see `generate_dense_backend`'s KV loop).
///
/// Recurrent-state layers (qwen35 DeltaNet) have no per-token cache to size;
/// [`layer_state_bytes`] branches to their fixed conv/S-state allocation before reaching here.
/// **deepseek4 (V4) caches ONE `head_dim`-wide MQA row per side per token**, and the V side is a
/// DUPLICATE of the K side rather than the MLA-style 0-width placeholder (docs/backlog.md B53).
///
/// V4's raw attention is `build_attn_mha(q, k_all, k_all, …)` — K and V really are the same rows —
/// so `(head_dim, 0)` with `Op::Attention` pointed at one buffer for both sides is the shape the
/// arithmetic wants. It is not the shape this codebase can execute: the CPU backend's
/// `Op::Attention` arm takes `cpu_buf(kbuf).read()` and `cpu_buf(vbuf).read()` as two
/// SIMULTANEOUSLY-LIVE guards (`crates/infr-cpu/src/lib.rs`), and a KV buffer is
/// `CpuStore::Owned(Mutex<Vec<u8>>)` — a non-reentrant `std::sync::Mutex`, so one id bound to both
/// sides self-deadlocks the moment the first V4 attention op runs. (Vulkan is fine with it: both
/// bindings are `readonly` and the hazard tracker de-dupes.) Aliasing therefore costs a one-line
/// change in a crate the V4 wiring slice does not own, in exchange for halving a cache that, at
/// V4's MQA width, is `head_dim` floats per token per layer. The emit writes the same normed+roped
/// row to BOTH caches with two `Op::WriteKv`s instead — see `docs/backlog.md` § B53.
///
/// The compressed caches (CSA/HCA/LID) and the three compressor states a ratio-4 / ratio-128 layer
/// owns are NOT modelled here. They cannot be: they are per-layer (a ratio-0 layer has none) and
/// the states are fixed-size recurrent buffers, not per-token rows. `generate_dense_backend`
/// refuses any non-zero ratio before a graph is built, so nothing reads a geometry that does not
/// exist yet.
pub(crate) fn kv_row_elems(cfg: &Config, l: usize) -> (usize, usize) {
    if cfg.is_mla_layer(l) {
        return (cfg.kv_lora_rank + cfg.qk_rope_dim, cfg.indexer_head_size);
    }
    if cfg.is_recurrent_layer(l) {
        return (0, 0);
    }
    if cfg.deepseek4 {
        // Read straight off `head_dim` rather than through `layer_head_dim`/`layer_n_kv`: those
        // route via `is_swa_layer`, which is TRUE for every V4 layer, and so would answer with
        // gemma4's `head_dim_swa`/`n_kv_swa` fields. They happen to equal `head_dim`/`n_kv` for
        // every non-gemma4 model, which is an accident this arch should not depend on.
        let row = cfg.head_dim;
        return (row, row);
    }
    let row = cfg.layer_n_kv(l) * cfg.layer_head_dim(l);
    (row, row)
}

/// Elements a KV side that the arch does NOT cache still declares and allocates: MLA's V, whose
/// [`kv_row_elems`] count is 0. No backend ever indexes it (the MLA kernels read V as a prefix of
/// the K row), but every backend still binds a buffer and a graph Input for the side, and a
/// zero-size allocation is not portable — so the placeholder is a handful of elements, sized
/// identically at the allocation and the declaration.
pub(crate) const KV_PLACEHOLDER_ELEMS: usize = 4;

/// A KV side's element count as DECLARED in the graph: `elems`, or the placeholder when the arch
/// does not cache that side at all (see [`KV_PLACEHOLDER_ELEMS`]).
pub(crate) fn kv_side_elems(elems: usize) -> usize {
    if elems == 0 {
        KV_PLACEHOLDER_ELEMS
    } else {
        elems
    }
}

/// Smallest KV-side allocation any backend will accept.
///
/// The floor has to be in BYTES rather than elements, because a block-quant format prices
/// [`KV_PLACEHOLDER_ELEMS`] at zero bytes (`kv_fmt_bytes(Q8_0, 4)` is `4 / 32 * 34`) and a
/// zero-size allocation is refused outright by the Vulkan allocator.
const KV_MIN_SIDE_BYTES: usize = 4;

/// Bytes to ALLOCATE for one KV side holding [`kv_side_elems`]`(elems)` values of `fmt`, floored at
/// [`KV_MIN_SIDE_BYTES`]. A side the arch really caches is far past that floor at any usable
/// context; only the placeholder ever meets it.
pub(crate) fn kv_side_bytes(fmt: DType, elems: usize) -> usize {
    kv_fmt_bytes(fmt, kv_side_elems(elems)).max(KV_MIN_SIDE_BYTES)
}

/// Layout half of the Q8_0 / block-quant KV gate — every layer's KV row must be whole 32-elem
/// blocks. Pure geometry: it says the format *fits* the rows, not that this model may use it (see
/// [`kv_q8_layout_ok`], which adds the MLA exclusion).
pub(crate) fn kv_row_align_ok(cfg: &Config) -> bool {
    (0..cfg.n_layer).all(|l| {
        let (k, v) = kv_row_elems(cfg, l);
        k.is_multiple_of(32) && v.is_multiple_of(32)
    })
}

/// Whether a Q8_0 KV cache may be CHOSEN for this model: the rows are 32-block aligned
/// ([`kv_row_align_ok`]) **and** the model is not MLA.
///
/// The MLA exclusion is not about alignment (576 is 18 whole blocks) — it is that the Vulkan and
/// Metal MLA kernels read the cache as f16 unconditionally, so `generate_dense_backend` forces f16
/// there ([`mla_kv_fmt`]). This gate is what the Vulkan auto-q8 placement PIN consults before
/// pricing the VRAM estimate at q8 (`model.rs`'s `clamp_default_ctx`), so excluding MLA here is
/// what keeps the estimate and the allocation in agreement: a format the runner will refuse to
/// build can never be pinned and priced in the first place.
pub(crate) fn kv_q8_layout_ok(cfg: &Config) -> bool {
    !cfg.deepseek2 && !cfg.deepseek4 && !cfg.bailingmoe3 && kv_row_align_ok(cfg)
}

/// Resolve the KV dtype the Vulkan runner will allocate for one side, for placement accounting.
/// This mirrors `runner.rs`'s capability gates closely enough that an explicit Q8/F16 choice is
/// priced exactly instead of the MoE planner treating every non-auto choice as f16.
fn vulkan_kv_fmt_for_budget(cfg: &Config, ec: &EngineConfig, requested: Option<DType>) -> DType {
    if cfg.deepseek2 || cfg.deepseek4 || cfg.bailingmoe3 {
        return DType::F16;
    }
    let block_aligned = kv_row_align_ok(cfg);
    // Qwen3.8's QSA gather/attention reads the ordinary K/V cache in F16 or planar Q8_0. Its
    // separate raw index-key cache remains F16 and is priced by `qsa_cache_bytes` below.
    if cfg.qwen4exp {
        return match requested {
            Some(DType::Q8_0) if block_aligned => DType::Q8_0,
            Some(DType::F16) => DType::F16,
            _ if (ec.kv.force_q8 || kv_auto_q8() || kv_default_q8(cfg, ec)) && block_aligned => {
                DType::Q8_0
            }
            _ => DType::F16,
        };
    }
    let turbo_aligned = (0..cfg.n_layer).all(|l| cfg.layer_head_dim(l).is_multiple_of(128));
    match requested {
        Some(dt @ (DType::Turbo2 | DType::Turbo3 | DType::Turbo4)) if turbo_aligned => dt,
        Some(DType::Q8_0) if block_aligned => DType::Q8_0,
        Some(dt @ (DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Iq4Nl))
            if block_aligned =>
        {
            dt
        }
        Some(dt @ (DType::F16 | DType::Bf16 | DType::F32)) => dt,
        _ if (ec.kv.force_q8 || kv_auto_q8() || kv_default_q8(cfg, ec)) && block_aligned => {
            DType::Q8_0
        }
        _ => DType::F16,
    }
}

/// The KV formats an MLA (deepseek2) session may actually run with on `backend`, given the pair
/// the dtype ladder resolved.
///
/// The GPU MLA kernels type the cache f16 UNCONDITIONALLY — `mla.comp`'s `kread` is
/// `unpackHalf2x16`, Metal's `mla_f16kv_one` takes `device const half*` — so any other dtype is
/// REINTERPRETED rather than converted: silent wrong output, no error. (The CPU `Op::Mla` arm
/// dequantizes through its own closure and is dtype-correct, which is why a CPU-vs-GPU parity run
/// would show a correct oracle beside a wrong GPU and read as a kernel bug — docs/backlog.md B42.)
///
/// So on every backend but `cpu`, a deepseek2 session is f16 on both sides. A format the USER named
/// is REFUSED rather than downgraded behind their back; a format nothing asked for (the ladder's
/// own default, or a placement pin — which [`kv_q8_layout_ok`] already prevents for MLA) is forced.
pub(crate) fn mla_kv_fmt(
    cfg: &Config,
    backend: &str,
    ec: &EngineConfig,
    k_fmt: DType,
    v_fmt: DType,
) -> AResult<(DType, DType)> {
    if (!cfg.deepseek2 && !cfg.bailingmoe3) || backend == "cpu" {
        return Ok((k_fmt, v_fmt));
    }
    let named = |dt: Option<DType>| matches!(dt, Some(dt) if dt != DType::F16);
    if named(ec.kv.type_k) || named(ec.kv.type_v) || ec.kv.force_q8 {
        return Err(anyhow!(
            "MLA KV cache is f16-only on the {backend} backend: its attention kernel \
             reads the compressed KV row as f16 and would reinterpret any other dtype. Requested \
             k={:?} (INFR_KV_TYPE_K), v={:?} (INFR_KV_TYPE_V), force_q8={} (INFR_KV_Q8) — unset \
             them, ask for f16, or run this model on the CPU backend, which dequantizes every KV \
             dtype.",
            ec.kv.type_k,
            ec.kv.type_v,
            ec.kv.force_q8
        ));
    }
    Ok((DType::F16, DType::F16))
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

/// DeepSeek V4's official mixed-FP8 cache is paged in groups of 64 rows. Each page stores all
/// 576 data bytes per row first (448 E4M3 bytes + 64 BF16 values), followed by eight UE8M0 scale
/// bytes per row. The page is therefore exactly `64 * 584` bytes.
pub(crate) const DSV4_FP8_PAGE_ROWS: usize = 64;
pub(crate) const DSV4_FP8_DATA_BYTES: usize = 576;
pub(crate) const DSV4_FP8_SCALE_BYTES: usize = 8;
pub(crate) const DSV4_FP8_PAGE_BYTES: usize =
    DSV4_FP8_PAGE_ROWS * (DSV4_FP8_DATA_BYTES + DSV4_FP8_SCALE_BYTES);
pub(crate) const DSV4_MXFP4_ROW_BYTES: usize = 68;

pub(crate) fn dsv4_fp8_cache_bytes(rows: usize) -> usize {
    rows.max(1).div_ceil(DSV4_FP8_PAGE_ROWS) * DSV4_FP8_PAGE_BYTES
}

/// Byte layout of a DeepSeek V4 layer's two persistent buffers. `kbuf` is only the mixed-FP8 raw
/// SWA ring. `vbuf` owns the compressed cache and its recurrent compressor state; ratio-4 adds the
/// MXFP4 indexer cache and its second overlapping state. Packing these into the existing K/V pair
/// keeps allocation, binding, fork geometry and unified-budget accounting on one established path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Dsv4LayerLayout {
    pub raw_rows: usize,
    pub raw_bytes: usize,
    pub comp_rows: usize,
    pub comp_off: usize,
    pub comp_bytes: usize,
    pub state_values_off: usize,
    pub state_scores_off: usize,
    pub lid_off: usize,
    pub lid_bytes: usize,
    pub lid_state_values_off: usize,
    pub lid_state_scores_off: usize,
    pub state_bytes: usize,
}

pub(crate) fn dsv4_layer_layout(cfg: &Config, l: usize, want_ctx: usize) -> Dsv4LayerLayout {
    debug_assert!(cfg.deepseek4);
    let raw_rows = want_ctx.min(cfg.swa_window.max(1)).max(1);
    let raw_bytes = dsv4_fp8_cache_bytes(raw_rows);
    let ratio = cfg.layer_compress_ratio(l);
    if ratio == 0 {
        return Dsv4LayerLayout {
            raw_rows,
            raw_bytes,
            state_bytes: KV_MIN_SIDE_BYTES,
            ..Dsv4LayerLayout::default()
        };
    }

    let comp_rows = want_ctx.div_ceil(ratio).max(1);
    let comp_off = 0usize;
    let comp_bytes = dsv4_fp8_cache_bytes(comp_rows);
    let overlap = ratio == 4;
    let state_rows = if overlap { 2 * ratio } else { ratio };
    let state_width = if overlap {
        2 * cfg.head_dim
    } else {
        cfg.head_dim
    };
    let state_one = state_rows * state_width * 4;
    let state_values_off = comp_off + comp_bytes;
    let state_scores_off = state_values_off + state_one;
    let mut end = state_scores_off + state_one;

    let (lid_off, lid_bytes, lid_state_values_off, lid_state_scores_off) = if overlap {
        let lid_off = end;
        let lid_bytes = comp_rows * DSV4_MXFP4_ROW_BYTES;
        let lid_width = 2 * cfg.indexer_head_size;
        let lid_state_one = state_rows * lid_width * 4;
        let lid_state_values_off = lid_off + lid_bytes;
        let lid_state_scores_off = lid_state_values_off + lid_state_one;
        end = lid_state_scores_off + lid_state_one;
        (
            lid_off,
            lid_bytes,
            lid_state_values_off,
            lid_state_scores_off,
        )
    } else {
        (0, 0, 0, 0)
    };

    Dsv4LayerLayout {
        raw_rows,
        raw_bytes,
        comp_rows,
        comp_off,
        comp_bytes,
        state_values_off,
        state_scores_off,
        lid_off,
        lid_bytes,
        lid_state_values_off,
        lid_state_scores_off,
        state_bytes: end.max(KV_MIN_SIDE_BYTES),
    }
}

/// Exact bytes allocated for the two persistent state buffers owned by one layer.
///
/// Most layers store context-scaled K/V rows. Qwen3.5/3.6 DeltaNet layers instead reuse the same
/// two buffer slots for fixed-size f32 convolution history and recurrent S state. Keeping that
/// architecture decision here lets allocation and budget accounting consume one answer without
/// teaching either caller a second copy of the model-specific branch.
pub(crate) fn layer_state_bytes(
    cfg: &Config,
    l: usize,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> (usize, usize) {
    if cfg.deepseek4 {
        let d = dsv4_layer_layout(cfg, l, want_ctx);
        return (d.raw_bytes, d.state_bytes);
    }
    if cfg.is_recurrent_layer(l) {
        let conv_elems = (cfg.ssm_d_conv - 1) * cfg.recurrent_conv_channels();
        let state_elems = cfg.recurrent_state_elems();
        return (conv_elems * 4, state_elems * 4);
    }

    let rows = kv_rows_at(cfg, l, want_ctx, ring, ubatch);
    let (k_row, v_row) = kv_row_elems(cfg, l);
    (
        kv_side_bytes(k_fmt, rows * k_row),
        kv_side_bytes(v_fmt, rows * v_row),
    )
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

/// Persistent KV/recurrent-state footprint summed over all layers at chunk height `ubatch` and
/// side format, including one rolling copy of append-only recurrent state. Qwen3.5/3.6 DeltaNet
/// layers contribute their fixed f32 conv/S state instead of a fictional context-scaled KV cache.
/// The ONE pricing helper the dense placement sweep and MoE expert-placement budget share, so both
/// price exactly what stateful chat may allocate.
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
    let primary: u64 = (0..cfg.n_layer)
        .map(|l| {
            let (k_bytes, v_bytes) =
                layer_state_bytes(cfg, l, want_ctx, ring, ubatch, k_fmt, v_fmt);
            (k_bytes + v_bytes) as u64
        })
        .sum();
    let qwen4_extra = if cfg.qwen4exp {
        let hc_dim = cfg.hc_mult.saturating_mul(cfg.n_embd);
        let ple_hist = (cfg.ple_conv_kernel.saturating_sub(1))
            .saturating_mul(cfg.ple_ngram_size)
            .saturating_mul(hc_dim);
        let ple_in = cfg
            .ple_head_dim
            .saturating_mul(cfg.ple_ngram_size.saturating_sub(1))
            .saturating_mul(cfg.ple_heads_per_ngram);
        let qsa: u64 = (0..cfg.n_layer)
            .map(|l| qsa_cache_bytes(cfg, l, want_ctx) as u64)
            .sum();
        (hc_dim
            .saturating_add(ple_hist)
            .saturating_add(ple_in)
            .saturating_mul(4) as u64)
            .saturating_add(qsa)
    } else {
        0
    };
    primary
        .saturating_add(recurrent_checkpoint_bytes(cfg))
        .saturating_add(qwen4_extra)
}

/// Raw Qwen3.8 QSA index keys: one unnormalised, unroped F16 row per token.
pub(crate) fn qsa_raw_cache_bytes(cfg: &Config, layer: usize, ctx: usize) -> usize {
    if cfg.qwen4exp && cfg.is_qwen_hybrid_attn_layer(layer) {
        ctx.saturating_mul(cfg.indexer_head_size).saturating_mul(2)
    } else {
        0
    }
}

/// Persistent final QSA keys: one F32 RMS-normalised and roped row per complete compressed block.
pub(crate) fn qsa_block_cache_bytes(cfg: &Config, layer: usize, ctx: usize) -> usize {
    if cfg.qwen4exp && cfg.is_qwen_hybrid_attn_layer(layer) {
        let ratio = cfg.layer_compress_ratio(layer).max(1);
        (ctx / ratio)
            .max(1)
            .saturating_mul(cfg.indexer_head_size)
            .saturating_mul(4)
    } else {
        0
    }
}

/// Total per-layer QSA state charged to context placement.
pub(crate) fn qsa_cache_bytes(cfg: &Config, layer: usize, ctx: usize) -> usize {
    qsa_raw_cache_bytes(cfg, layer, ctx).saturating_add(qsa_block_cache_bytes(cfg, layer, ctx))
}

/// One rolling copy of every append-only recurrent layer's fixed f32 state. Stateful Vulkan chat
/// allocates this lazily at the first stable conversation boundary, but placement must reserve it
/// up front so the allocation cannot unexpectedly consume the last expert/activation bytes.
pub(crate) fn recurrent_checkpoint_bytes(cfg: &Config) -> u64 {
    (0..cfg.n_layer)
        .filter(|&l| cfg.is_recurrent_layer(l))
        .map(|l| {
            let (k_bytes, v_bytes) = layer_state_bytes(cfg, l, 1, false, 1, DType::F16, DType::F16);
            (k_bytes + v_bytes) as u64
        })
        .sum()
}

/// Read-only KV footprint estimate for control planes and launch planners. This is the same
/// arithmetic placement uses before allocating anything; exposing it avoids a GUI maintaining a
/// second approximation that silently drifts when an architecture's KV layout changes.
pub fn estimate_kv_bytes(
    cfg: &Config,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> u64 {
    kv_bytes_estimate_fmt(cfg, want_ctx, ring, ubatch, k_fmt, v_fmt)
}

/// K+V byte footprint for ONE layer at `k_elems`/`v_elems` per-side elements, each side in its own
/// dtype. The pure per-layer core of [`kv_bytes_estimate_fmt`].
///
/// The two sides are counted SEPARATELY because they are not always the same width: an MLA layer
/// caches a wide K row and no V at all, so a single `elems` argument (what this took while three of
/// the five geometry sites had drifted) cannot express it.
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
    // Not supplied is ring-capable under either automatic Vulkan Q8 or another backend's F16
    // default; otherwise the requested format must PARSE to f16 or q8 (a name the runner would
    // not recognize either is not ring-capable — the
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

// ── one budget, two families of callers ───────────────────────────────────────────────────────
//
// The context-fit math ([`kv_fit_ctx_for`], via `SeamModel::kv_fit_ctx_fmt`) and the placement
// sweeps (`vulkan_moe_binder`'s residency / auto-q8 / streaming / MoE-expert budgets) are two
// readers of ONE question: what still fits this device? Every helper below takes the raw
// [`infr_vulkan::VramInfo`] snapshot and derives its ceiling from [`VramInfo::alloc_room`] —
// the allocator's own limit — so neither family can plan bytes `check_vram_budget` will refuse.
// `budgets_agree_with_the_allocator_ceiling` and `fit_math_and_placement_pick_the_same_rung`
// guard the drift.

/// Placement-time view of the unified device-memory budget. Placement runs before this model has
/// committed its weights, so `tracked_used=0`; the allocator applies the same helper later with its
/// live per-backend allocation tally. The physical side starts at `VramInfo::alloc_room()` (already
/// net of Vulkan's mandatory guard), then applies the caller's additional reserve.
fn planned_vram_room(vram: &infr_vulkan::VramInfo, ec: &EngineConfig) -> u64 {
    infr_core::budget::unified_vram_room(
        vram.total,
        vram.alloc_room(),
        0,
        ec.device.vram_budget,
        ec.device.vram_reserve,
    )
}

/// Will the KV cache a placement/fit decision is pricing actually RING (SWA rows capped at
/// `window + chunk`)? The config/env gate AND f16/q8 on BOTH sides — the same pair of conditions
/// the runner applies. A low-bit side keeps full-context caches, so pricing it as a ring would
/// hand out a context the allocation cannot honor.
pub(crate) fn placement_ring(cfg: &Config, ec: &EngineConfig, k_fmt: DType, v_fmt: DType) -> bool {
    kv_ring_wanted(cfg, ec)
        && matches!(k_fmt, DType::F16 | DType::Q8_0)
        && matches!(v_fmt, DType::F16 | DType::Q8_0)
}

/// Whether this session requests the segmented Qwen KV layout. The implementation currently
/// relies on Q8_0's compact fixed-size segments; explicit F16 or a mixed pair keeps the established
/// flat allocation path until segmented F16 has its own performance and allocation contract.
pub(crate) fn segmented_kv_wanted(
    cfg: &Config,
    ec: &EngineConfig,
    ring: bool,
    k_fmt: DType,
    v_fmt: DType,
) -> bool {
    ec.kv.dynamic
        && (cfg.qwen35 || cfg.qwen4exp)
        && !ring
        && !ec.kv.overflow
        && k_fmt == DType::Q8_0
        && v_fmt == DType::Q8_0
}

/// Bytes a FULLY-RESIDENT dense session needs at one EXPLICIT prefill chunk height and KV format
/// pair: weights + the exact KV allocation ([`kv_bytes_estimate_fmt`]) + the activation reserve
/// ([`dense_act_reserve_at`]). The arithmetic both the fit math and the placement sweep compare
/// against the ceiling, in one place so the two cannot price a session differently.
pub(crate) fn dense_resident_need(
    cfg: &Config,
    caps: &Capabilities,
    weights: u64,
    want_ctx: usize,
    ring: bool,
    ubatch: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> u64 {
    weights
        .saturating_add(kv_bytes_estimate_fmt(
            cfg, want_ctx, ring, ubatch, k_fmt, v_fmt,
        ))
        .saturating_add(runtime_reserve_at(
            cfg, caps, want_ctx, ring, ubatch, k_fmt, v_fmt,
        ))
}

/// Does a fully-resident dense session at this chunk height fit the ALLOCATOR's ceiling?
///
/// `vram.alloc_room()`, NOT `vram.available`: the VRAM guard reserves a fixed headroom below the
/// free figure and refuses anything that reaches into it, so a residency decision taken against
/// the raw figure can declare a model resident while planning 256 MiB the allocator will never
/// hand out — and then fail on an activation alloc mid-prefill.
pub(crate) fn dense_placement_fits(
    cfg: &Config,
    caps: &Capabilities,
    ec: &EngineConfig,
    weights: u64,
    vram: &infr_vulkan::VramInfo,
    want_ctx: usize,
    ubatch: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> bool {
    let ring = placement_ring(cfg, ec, k_fmt, v_fmt);
    // Every session prefills chunk-major unless the user forces the other order, so the
    // whole-prompt residual stream is priced only for the explicit `Some(true)` case.
    let lm = if ec.paging.layer_major == Some(true) {
        layer_major_act_bytes(cfg, want_ctx, ubatch)
    } else {
        0
    };
    dense_resident_need(cfg, caps, weights, want_ctx, ring, ubatch, k_fmt, v_fmt).saturating_add(lm)
        <= planned_vram_room(vram, ec)
}

/// The TALLEST [`ubatch_candidates`] rung at which this dense session fits resident, or `None` when
/// none of them does (the caller streams). The rung the placement sweep settles on — and the same
/// walk [`kv_fit_ctx_for`] makes when it decides a context fits.
pub(crate) fn dense_resident_rung(
    cfg: &Config,
    caps: &Capabilities,
    ec: &EngineConfig,
    weights: u64,
    vram: &infr_vulkan::VramInfo,
    want_ctx: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> Option<usize> {
    ubatch_candidates(ec)
        .into_iter()
        .find(|&ub| dense_placement_fits(cfg, caps, ec, weights, vram, want_ctx, ub, k_fmt, v_fmt))
}

/// Streaming budget: what is left of the allocator's ceiling for the dense weight-streaming arenas
/// once the always-resident weights, the KV cache and the activation reserve are paid for. Same
/// ceiling as [`dense_placement_fits`] — every byte this over-states is a slot the arena allocates
/// and the guard then refuses.
pub(crate) fn dense_stream_budget_at(
    cfg: &Config,
    caps: &Capabilities,
    ec: &EngineConfig,
    resident_weights: u64,
    vram: &infr_vulkan::VramInfo,
    want_ctx: usize,
    ubatch: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> u64 {
    let ring = placement_ring(cfg, ec, k_fmt, v_fmt);
    // Chunk-major is the default for streamed sessions too. Only an explicit layer-major request
    // holds residual streams across layers, so only that mode reserves the extra bytes.
    let lm = if ec.paging.layer_major == Some(true) {
        layer_major_act_bytes(cfg, want_ctx, ubatch)
    } else {
        0
    };
    planned_vram_room(vram, ec)
        .saturating_sub(dense_resident_need(
            cfg,
            caps,
            resident_weights,
            want_ctx,
            ring,
            ubatch,
            k_fmt,
            v_fmt,
        ))
        .saturating_sub(lm)
}

/// One lightweight account of the VRAM ceiling shared by placement, the loader and control planes.
/// All inputs are already-resolved logical bytes; this type only prevents consumers from applying
/// the subtraction in different orders or forgetting one category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelMemoryPlan {
    pub total_room_bytes: u64,
    pub fixed_weight_bytes: u64,
    pub persistent_state_bytes: u64,
    pub runtime_reserve_bytes: u64,
    /// Maximum lazily committed per-token state. It shares the unified arena with experts and
    /// runtime scratch, but remains separately named so diagnostics never call KV "runtime".
    pub dynamic_state_reserve_bytes: u64,
    pub weight_packing_margin_bytes: u64,
    pub load_driver_reserve_bytes: u64,
    pub post_load_reserve_bytes: u64,
    pub expert_cache_bytes: u64,
}

impl ModelMemoryPlan {
    pub fn new(
        total_room_bytes: u64,
        fixed_weight_bytes: u64,
        persistent_state_bytes: u64,
        runtime_reserve_bytes: u64,
    ) -> Option<Self> {
        Self::new_with_packing_margin(
            total_room_bytes,
            fixed_weight_bytes,
            persistent_state_bytes,
            runtime_reserve_bytes,
            0,
        )
    }

    pub fn new_with_packing_margin(
        total_room_bytes: u64,
        fixed_weight_bytes: u64,
        persistent_state_bytes: u64,
        runtime_reserve_bytes: u64,
        weight_packing_margin_bytes: u64,
    ) -> Option<Self> {
        Self::new_with_reserves(
            total_room_bytes,
            fixed_weight_bytes,
            persistent_state_bytes,
            runtime_reserve_bytes,
            weight_packing_margin_bytes,
            0,
            0,
        )
    }

    pub fn new_with_reserves(
        total_room_bytes: u64,
        fixed_weight_bytes: u64,
        persistent_state_bytes: u64,
        runtime_reserve_bytes: u64,
        weight_packing_margin_bytes: u64,
        load_driver_reserve_bytes: u64,
        post_load_reserve_bytes: u64,
    ) -> Option<Self> {
        Self::new_with_dynamic_reserve(
            total_room_bytes,
            fixed_weight_bytes,
            persistent_state_bytes,
            runtime_reserve_bytes,
            0,
            weight_packing_margin_bytes,
            load_driver_reserve_bytes,
            post_load_reserve_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_dynamic_reserve(
        total_room_bytes: u64,
        fixed_weight_bytes: u64,
        persistent_state_bytes: u64,
        runtime_reserve_bytes: u64,
        dynamic_state_reserve_bytes: u64,
        weight_packing_margin_bytes: u64,
        load_driver_reserve_bytes: u64,
        post_load_reserve_bytes: u64,
    ) -> Option<Self> {
        let persistent = fixed_weight_bytes.saturating_add(persistent_state_bytes);
        (persistent <= total_room_bytes).then(|| Self {
            total_room_bytes,
            fixed_weight_bytes,
            persistent_state_bytes,
            runtime_reserve_bytes,
            dynamic_state_reserve_bytes,
            weight_packing_margin_bytes,
            load_driver_reserve_bytes,
            post_load_reserve_bytes,
            expert_cache_bytes: total_room_bytes
                .saturating_sub(persistent)
                .saturating_sub(runtime_reserve_bytes)
                .saturating_sub(dynamic_state_reserve_bytes)
                .saturating_sub(weight_packing_margin_bytes)
                .saturating_sub(load_driver_reserve_bytes)
                .saturating_sub(post_load_reserve_bytes),
        })
    }

    pub fn minimum_required_bytes(self) -> u64 {
        self.fixed_weight_bytes
            .saturating_add(self.persistent_state_bytes)
            .saturating_add(self.runtime_reserve_bytes)
            .saturating_add(self.dynamic_state_reserve_bytes)
            .saturating_add(self.weight_packing_margin_bytes)
            .saturating_add(self.load_driver_reserve_bytes)
            .saturating_add(self.post_load_reserve_bytes)
    }

    /// Physical size of the elastic Expert/runtime arena. Runtime is deducted while calculating
    /// the target Expert occupancy, then added back here because it borrows the same bytes only
    /// while a graph is active instead of living in a separate permanently idle reservation.
    pub fn elastic_pool_bytes(self, expert_cache_target_bytes: u64) -> u64 {
        expert_cache_target_bytes.saturating_add(self.elastic_reserve_bytes())
    }

    pub fn elastic_reserve_bytes(self) -> u64 {
        self.runtime_reserve_bytes
            .saturating_add(self.dynamic_state_reserve_bytes)
    }
}

/// Build the same conservative pre-load memory account used by the Vulkan MoE loader.
///
/// Control planes use this instead of duplicating the resident-weight packing margin and
/// architecture-specific driver reserves. `persistent_state_bytes` is the already-selected KV /
/// recurrent-state layout, and `runtime_reserve_bytes` is the peak elastic activation estimate.
pub fn estimate_model_memory_plan(
    cfg: &Config,
    dense_weight_bytes: u64,
    total_room_bytes: u64,
    persistent_state_bytes: u64,
    runtime_reserve_bytes: u64,
) -> Option<ModelMemoryPlan> {
    ModelMemoryPlan::new_with_reserves(
        total_room_bytes,
        dense_weight_bytes,
        persistent_state_bytes,
        runtime_reserve_bytes,
        resident_weight_packing_margin(dense_weight_bytes),
        load_driver_reserve(cfg),
        POST_KV_DEVICE_RESERVE,
    )
}

/// Resident BDA weights are packed into adaptive 64/128/256 MiB blocks, so summing logical tensor
/// bytes understates the committed allocation by the unused block tails. Measurements across
/// supported model families put that delta at 1.16%-2.43%; 3%, rounded to the initial block unit
/// and floored at 256 MiB, protects new architectures without carrying a second copy of allocator
/// state into the planner.
fn resident_weight_packing_margin(dense_weight_bytes: u64) -> u64 {
    const BLOCK: u64 = 64 * 1024 * 1024;
    const MIN: u64 = 256 * 1024 * 1024;
    dense_weight_bytes
        .saturating_mul(3)
        .div_ceil(100)
        .max(MIN)
        .next_multiple_of(BLOCK)
}

/// Extra room withheld from the expert arena while a DeepSeek V4 model is still loading. V4's
/// unusually large resident dense set and tensor/pipeline count leave substantially more live
/// device/driver memory than the logical tensor footprint describes. On the 24 GiB Windows AMD
/// target, 13.5 GiB of expert cache loads reliably while 14 GiB fails near the final 414 MiB weight
/// allocation; 1.5 GiB brings the automatic plan to the measured safe side of that boundary.
///
/// This is deliberately separate from [`POST_KV_DEVICE_RESERVE`]. Once weights are resident,
/// [`reclamp_ctx_to_live_room`] observes this load-time overhead in the driver's live heap usage;
/// subtracting it there again would double-charge it and can collapse the selected context.
const DEEPSEEK4_LOAD_DRIVER_RESERVE: u64 = 1536 * 1024 * 1024;

/// WDDM charges large mapped ReBAR arenas more aggressively than the logical Vulkan allocation
/// tally while they are being committed. On the Windows 7900 XTX target, Qwen35 and Ling sessions
/// gain about 2 GiB of untracked heap usage between committing a large mapped arena and allocating
/// their fixed weights/state. Keep that load-only movement out of the expert arena; Linux uses the
/// live heap budget without this WDDM allowance and remains byte-for-byte unchanged.
const WINDOWS_LARGE_REBAR_LOAD_DRIVER_RESERVE: u64 = 2 * 1024 * 1024 * 1024;

/// Cold WDDM startup has a small amount of run-to-run heap-budget movement beyond the measured
/// large-ReBAR load reserve above. Automatic placement should favor a reliable first launch over the
/// last few Expert slots. An explicit total VRAM budget/reserve remains authoritative and opts out
/// of this extra policy margin.
const WINDOWS_LARGE_REBAR_AUTO_STARTUP_RESERVE: u64 = 512 * 1024 * 1024;

fn load_driver_reserve(cfg: &Config) -> u64 {
    if cfg.deepseek4 {
        DEEPSEEK4_LOAD_DRIVER_RESERVE
    } else if cfg!(windows) && (cfg.qwen35 || cfg.qwen4exp || cfg.bailingmoe3) {
        WINDOWS_LARGE_REBAR_LOAD_DRIVER_RESERVE
    } else {
        0
    }
}

fn session_load_driver_reserve(cfg: &Config, ec: &EngineConfig) -> u64 {
    let automatic = if cfg!(windows)
        && (cfg.qwen35 || cfg.qwen4exp || cfg.bailingmoe3)
        && ec.device.vram_budget.is_none()
        && ec.device.vram_reserve.is_none()
    {
        WINDOWS_LARGE_REBAR_AUTO_STARTUP_RESERVE
    } else {
        0
    };
    load_driver_reserve(cfg).saturating_add(automatic)
}

/// Conservative load-time runtime reserve for control planes that do not own a live backend yet.
/// Placement calls the same formula with the selected device's real capabilities. These helpers
/// describe the automatic Vulkan format: Q8_0 on a compatible layout, otherwise F16.
pub fn estimate_runtime_reserve_bytes(cfg: &Config, want_ctx: usize, ubatch: usize) -> u64 {
    let fmt = if kv_q8_layout_ok(cfg) {
        DType::Q8_0
    } else {
        DType::F16
    };
    runtime_reserve_at(
        cfg,
        &Capabilities::default(),
        want_ctx,
        false,
        ubatch,
        fmt,
        fmt,
    )
}

/// Device-aware form of [`estimate_runtime_reserve_bytes`] for control planes that have probed
/// whether the selected Vulkan device can run the dedicated hd256 FlashAttention kernel.
pub fn estimate_runtime_reserve_bytes_for_device(
    cfg: &Config,
    want_ctx: usize,
    ubatch: usize,
    flash_attention_hd256: bool,
) -> u64 {
    let mut caps = Capabilities::default();
    if flash_attention_hd256 {
        caps.f16 = true;
        caps.coopmat_f16 = Some(infr_core::COOPMAT_TILE_16);
        caps.max_shared_memory_bytes = infr_vulkan::FLASH_HD256_BM16_SHARED;
    }
    let fmt = if kv_q8_layout_ok(cfg) {
        DType::Q8_0
    } else {
        DType::F16
    };
    runtime_reserve_at(cfg, &caps, want_ctx, false, ubatch, fmt, fmt)
}

fn moe_expert_layer(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")
        .and_then(|r| r.split('.').next())
        .and_then(|l| l.parse::<usize>().ok())
}

fn moe_role_index(name: &str) -> Option<usize> {
    if name.ends_with("ffn_gate_exps.weight") || name.ends_with("ffn_gate_up_exps.weight") {
        Some(0)
    } else if name.ends_with("ffn_up_exps.weight") {
        Some(1)
    } else if name.ends_with("ffn_down_exps.weight") {
        Some(2)
    } else {
        None
    }
}

/// The pager's physical size classes, derived from the same expert banks its binder diverts.
/// Keeping this outside the binder lets the default-context preflight reserve the exact one-layer
/// floor before any Vulkan allocation exists.
fn moe_logical_pools(g: &Gguf, cfg: &Config, n_paged: usize) -> Vec<(usize, usize, [usize; 3])> {
    let Some(moe) = cfg.moe.as_ref() else {
        return Vec::new();
    };
    let n_expert = moe.n_expert.max(1);
    let mut by_size = std::collections::BTreeMap::<usize, (usize, [usize; 3])>::new();
    for t in g.tensors() {
        let Some(_layer) = moe_expert_layer(&t.name).filter(|&l| l < n_paged) else {
            continue;
        };
        let Some(role) = moe_role_index(&t.name) else {
            continue;
        };
        let slot_bytes = (t.nbytes / n_expert).max(4);
        let entry = by_size.entry(slot_bytes).or_insert((0, [0; 3]));
        entry.0 += n_expert;
        entry.1[role] += n_expert;
    }
    by_size
        .into_iter()
        .map(|(slot_bytes, (blocks, role_blocks))| (slot_bytes, blocks, role_blocks))
        .collect()
}

fn moe_pool_batch_slot_floors(pools: &[(usize, usize, [usize; 3])], n_expert: usize) -> Vec<usize> {
    pools
        .iter()
        .map(|&(_, n_blocks, role_blocks)| {
            let roles_per_layer = role_blocks.iter().filter(|&&n| n != 0).count().max(1);
            n_expert
                .saturating_mul(roles_per_layer)
                .min(n_blocks)
                .max(1)
        })
        .collect()
}

/// Physical pool floors include one rotating exchange slot whenever the pool can page. Bounded
/// RAM promotion keeps that slot disabled while the dispatch-visible cache retains the full batch
/// floor; full-RAM pools simply gain one harmless extra cache entry.
fn moe_pool_slot_floors(pools: &[(usize, usize, [usize; 3])], n_expert: usize) -> Vec<usize> {
    moe_pool_batch_slot_floors(pools, n_expert)
        .into_iter()
        .map(|batch_floor| batch_floor.saturating_add(1))
        .collect()
}

fn moe_pool_floor_bytes(pools: &[(usize, usize, [usize; 3])], n_expert: usize) -> Option<u64> {
    pools
        .iter()
        .zip(moe_pool_batch_slot_floors(pools, n_expert))
        .try_fold(0u64, |sum, (&(slot_bytes, ..), slots)| {
            sum.checked_add((slot_bytes as u64).checked_mul(slots as u64)?)
        })
}

fn moe_pool_physical_floor_bytes(
    pools: &[(usize, usize, [usize; 3])],
    n_expert: usize,
) -> Option<u64> {
    pools
        .iter()
        .zip(moe_pool_slot_floors(pools, n_expert))
        .try_fold(0u64, |sum, (&(slot_bytes, ..), slots)| {
            sum.checked_add((slot_bytes as u64).checked_mul(slots as u64)?)
        })
}

pub(crate) fn moe_prefill_floor_bytes(g: &Gguf, cfg: &Config) -> u64 {
    let Some(moe) = cfg.moe.as_ref() else {
        return 0;
    };
    let pools = moe_logical_pools(g, cfg, cfg.n_layer);
    moe_pool_floor_bytes(&pools, moe.n_expert.max(1)).unwrap_or(u64::MAX)
}

/// Whole-layer Prefill ring depth from the model's actual mixer topology. A recurrent run gives
/// the uploader that many fast layers in which to prepare the next slow Attention/MLA layer; the
/// extra lane is the layer currently being consumed. Models without recurrent mixers keep the
/// established all-layer target and let physical cache capacity cap it.
fn moe_prefill_target_lanes(cfg: &Config, n_paged: usize) -> usize {
    let mut current_run = 0usize;
    let mut longest_run = 0usize;
    for layer in 0..cfg.n_layer {
        if cfg.is_recurrent_layer(layer) {
            current_run = current_run.saturating_add(1);
            longest_run = longest_run.max(current_run);
        } else {
            current_run = 0;
        }
    }
    if longest_run == 0 {
        n_paged.max(1)
    } else {
        longest_run.saturating_add(1).min(n_paged.max(1))
    }
}

/// Split the MoE arena budget across slot-size pools without ever exceeding it. Each pool first
/// receives enough slots for one worst-case Prefill layer; the remaining bytes are then assigned
/// in weighted-fair order. Reserving the floors up front avoids the old `clamp(floor, nb)` corner
/// case where several small proportional shares silently summed to more than the caller's budget.
fn moe_pool_slot_counts(
    pools: &[(usize, usize, [usize; 3])],
    budget: u64,
    n_expert: usize,
    size_bias: f64,
) -> Option<Vec<usize>> {
    let floors = moe_pool_slot_floors(pools, n_expert);
    let minimum = moe_pool_physical_floor_bytes(pools, n_expert)?;
    if minimum > budget {
        return None;
    }

    let weights: Vec<f64> = pools
        .iter()
        .map(|&(slot_bytes, n_blocks, _)| {
            slot_bytes as f64 * n_blocks as f64 * (slot_bytes as f64).powf(size_bias)
        })
        .collect();
    let mut slots = floors;
    let mut remaining = budget - minimum;
    loop {
        let next = pools
            .iter()
            .enumerate()
            .filter(|(i, &(slot_bytes, n_blocks, _))| {
                slots[*i] < n_blocks.saturating_add(1) && slot_bytes as u64 <= remaining
            })
            .min_by(|(a, &(a_bytes, _, _)), (b, &(b_bytes, _, _))| {
                let a_fill = (slots[*a] * a_bytes) as f64 / weights[*a];
                let b_fill = (slots[*b] * b_bytes) as f64 / weights[*b];
                a_fill.total_cmp(&b_fill).then_with(|| a.cmp(b))
            })
            .map(|(i, _)| i);
        let Some(i) = next else { break };
        slots[i] += 1;
        remaining -= pools[i].0 as u64;
    }
    Some(slots)
}

const AUTO_MOE_ARENA_SHRINK_MIN: u64 = 256 * 1024 * 1024;
const AUTO_MOE_ARENA_MAX_ATTEMPTS: usize = 16;

/// Next automatic mapped-arena probe. An allocation failure has no trustworthy byte shortfall,
/// so retire 5% (at least 256 MiB); a successful allocation whose live budget is short can name
/// the exact deficit and skips directly past it. Explicit cache budgets never call this helper.
fn next_auto_moe_arena_budget(current: u64, minimum: u64, shortfall: u64) -> Option<u64> {
    const STEP_ALIGN: u64 = 64 * 1024 * 1024;
    let measured = shortfall.div_ceil(STEP_ALIGN).saturating_mul(STEP_ALIGN);
    let step = (current / 20).max(AUTO_MOE_ARENA_SHRINK_MIN).max(measured);
    let next = current.saturating_sub(step);
    (next >= minimum && next < current).then_some(next)
}

fn moe_pool_capacity_bytes(pools: &[(usize, usize, [usize; 3])], slots: &[usize]) -> u64 {
    pools
        .iter()
        .zip(slots)
        .fold(0u64, |total, (&(slot_bytes, ..), &n_slots)| {
            total.saturating_add((slot_bytes as u64).saturating_mul(n_slots as u64))
        })
}

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
pub(crate) fn kv_fit_ctx_for(
    cfg: &Config,
    caps: &Capabilities,
    ec: &EngineConfig,
    weights: u64,
    vram: &infr_vulkan::VramInfo,
    k_fmt: DType,
    v_fmt: DType,
) -> Option<usize> {
    // The ALLOCATOR's ceiling, derived by the same function the placement sweeps use, so the two
    // decide against one budget (see the `budgets_agree_with_the_allocator_ceiling` drift test).
    kv_fit_ctx_in_budget(
        cfg,
        caps,
        ec,
        planned_vram_room(vram, ec).saturating_sub(weights),
        &ubatch_candidates(ec),
        k_fmt,
        v_fmt,
    )
}

/// MoE counterpart of [`kv_fit_ctx_for`]. `fixed_bytes` excludes pageable expert banks but
/// includes their load-time fixed reserves. The remaining physical arena is shared by Expert
/// slots and transient runtime allocations, so it must hold the larger of the one-layer Expert
/// floor and the activation peak, not both at once.
pub(crate) fn kv_fit_ctx_for_moe(
    cfg: &Config,
    caps: &Capabilities,
    ec: &EngineConfig,
    fixed_bytes: u64,
    minimum_elastic_bytes: u64,
    vram: &infr_vulkan::VramInfo,
    k_fmt: DType,
    v_fmt: DType,
) -> Option<usize> {
    kv_fit_ctx_in_budgets(
        cfg,
        caps,
        ec,
        planned_vram_room(vram, ec).saturating_sub(fixed_bytes),
        None,
        minimum_elastic_bytes,
        &ubatch_candidates(ec),
        k_fmt,
        v_fmt,
    )
}

/// The search half of [`kv_fit_ctx_for`], against a budget that is ALREADY net of the weights:
/// the largest context whose KV cache plus its activation reserve fit `budget` bytes.
///
/// Split out because the two callers know the weight bytes with very different confidence. Before
/// the load, [`kv_fit_ctx_for`] can only subtract an ESTIMATE of them (`weight_footprint`, which
/// prices tensor bytes and not the arena block tails they land in). After the load, the runner
/// asks the device what is left (`Backend::device_alloc_room`) and passes that here — a budget with
/// the tails, the retained staging and the driver's own memory already netted out, because they
/// have been allocated. Same arithmetic either way; only the confidence in the input differs.
///
pub(crate) fn kv_fit_ctx_in_budget(
    cfg: &Config,
    caps: &Capabilities,
    ec: &EngineConfig,
    budget: u64,
    cands: &[usize],
    k_fmt: DType,
    v_fmt: DType,
) -> Option<usize> {
    kv_fit_ctx_in_budgets(cfg, caps, ec, budget, None, 0, cands, k_fmt, v_fmt)
}

/// Context fit with separate placement domains. `persistent_budget` is ordinary device room and
/// must contain KV/recurrent state. `elastic_activation_budget`, when present, is an already-
/// committed arena that only activation scratch may borrow; keeping the two tests separate avoids
/// pretending that persistent KV can consume Expert slots. `minimum_elastic_bytes` reserves a
/// physical arena floor that can itself be reused by activation scratch.
fn kv_fit_ctx_in_budgets(
    cfg: &Config,
    caps: &Capabilities,
    ec: &EngineConfig,
    persistent_budget: u64,
    elastic_activation_budget: Option<u64>,
    minimum_elastic_bytes: u64,
    cands: &[usize],
    k_fmt: DType,
    v_fmt: DType,
) -> Option<usize> {
    if (0..cfg.n_layer).all(|l| kv_row_elems(cfg, l) == (0, 0)) {
        return None;
    }
    // Same gate the runner applies (`generate_dense_backend`'s `kv_ring`) — see `placement_ring`.
    let ring = placement_ring(cfg, ec, k_fmt, v_fmt);
    let fits = |ctx: usize| -> bool {
        cands.iter().any(|&ubatch| {
            let kv = kv_bytes_estimate_fmt(cfg, ctx, ring, ubatch, k_fmt, v_fmt);
            let reserve = runtime_reserve_at(cfg, caps, ctx, ring, ubatch, k_fmt, v_fmt);
            elastic_activation_budget.map_or_else(
                || kv.saturating_add(reserve.max(minimum_elastic_bytes)) <= persistent_budget,
                |activation_budget| {
                    kv <= persistent_budget
                        && reserve.max(minimum_elastic_bytes) <= activation_budget
                },
            )
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

/// Device memory a live session commits AFTER its KV cache is allocated, which therefore has to be
/// held back from [`reclamp_ctx_to_live_room`]'s budget: the compute pipelines, descriptor pools
/// and command buffers the driver builds while recording the first forwards. It is the driver's
/// own memory, so no `Backend` allocation accounts for it and only the live budget query sees it —
/// which is why it is a term here rather than a footprint somewhere.
///
/// Sized by measurement (7900 XTX, RADV): the gap between the driver's reported used bytes and
/// this backend's own tally grows from 187 MiB right after the weight load to 368 MiB at the peak
/// of a deep gemma-4-31B prefill — 181 MiB of pipeline/descriptor memory built during the run,
/// with the idle desktop's 43 MiB present in both figures and cancelling out. Rounded up to 256
/// MiB: the term protects the LAST thing to allocate, and under-reserving it lands as the
/// mid-prefill failure this whole path exists to prevent.
///
/// Deliberately NOT the same money as `infr_vulkan`'s `GUARD_HEADROOM`. That headroom is the
/// allocator's own cushion below the free figure (alignment, block rounding); spending it on
/// pipelines is what left the guard nothing to cushion with and turned a 2 MiB overshoot into a
/// failed request.
pub(crate) const POST_KV_DEVICE_RESERVE: u64 = 256 * 1024 * 1024;

/// Re-decide the session's context against what the device says is LEFT, now that the weights are
/// resident — the runner's cold init calls this between the weight upload and the KV allocation.
///
/// **Why this exists at all.** Every pre-load estimate of "will this fit?" has to model the weight
/// bytes, and the model is wrong in a direction that hurts: `weight_footprint` sums tensor bytes,
/// while the resident-BDA arena commits those tensors into ≥64 MiB blocks whose tails nobody
/// counts (measured: **+2.20%** on gemma-4-31B UD-Q5_K_XL = 481 MiB, +2.43% on gemma-3-12b,
/// +1.16% on Qwen3-14B Q4_K_M), and neither the retained upload staging nor the driver's own
/// memory appears in any footprint. At THIS point none of that has to be modelled: the weights,
/// the tails, the staging and the driver's memory so far are all allocated, so the live budget
/// query prices them exactly. What remains predicted is the activation reserve
/// ([`dense_act_reserve_at`]) and [`POST_KV_DEVICE_RESERVE`].
///
/// **Only ever shrinks**, and only a context the SESSION chose. A user-pinned `--ctx`/`INFR_CTX`
/// is documented as taken verbatim (never clamped), so it is warned about and honored — the
/// alloc-time VRAM guard stays its backstop. Returns `want_ctx` unchanged on a backend with no
/// budget to report (`Backend::device_alloc_room` → `None`: CPU, Metal today).
///
/// May also LOWER the pinned prefill chunk ([`repin_ubatch_lower`]) when a shorter one serves more
/// context — the same trade the pre-load sweep makes, retaken with the weight bytes known.
pub(crate) fn reclamp_ctx_to_live_room(
    be: &dyn Backend,
    cfg: &Config,
    ec: &EngineConfig,
    want_ctx: usize,
    k_fmt: DType,
    v_fmt: DType,
) -> usize {
    let Some(room) = be.device_alloc_room() else {
        return want_ctx;
    };
    let caps = be.capabilities();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let budget = room.saturating_sub(POST_KV_DEVICE_RESERVE);
    let elastic_activation = be.device_elastic_activation_room();
    // Walk the chunk ladder HERE too, and price each rung on its own: a shorter chunk shrinks both
    // the activation reserve and the SWA ring, so it buys context, and the rung the pre-load sweep
    // pinned was chosen against weight bytes that turned out to be ~2% light. Pricing only the
    // pinned rung leaves that context on the floor (measured on gemma-4-31B: 10 440 tokens at the
    // pinned 256-row chunk against 15 440 at 128). Tallest-first, so a rung is only lowered when
    // the taller one genuinely cannot serve the window.
    let cands = ubatch_candidates(ec);
    // MoE: the pager's arenas are already allocated by the binder at this point, so the live room
    // has them netted out and the flat `total/12` stand-in would double-count. The dense reserve
    // is what an MoE forward's activations actually need beside them.
    let at = |ub: usize| {
        kv_fit_ctx_in_budgets(
            cfg,
            &caps,
            ec,
            budget,
            elastic_activation,
            0,
            &[ub],
            k_fmt,
            v_fmt,
        )
    };
    let Some(_) = at(cands[0]) else {
        return want_ctx; // pure recurrent-state arch: no per-token KV to size.
    };
    // The tallest rung that serves the whole window, else the rung that serves the most of it.
    let (chunk, fit) = cands
        .iter()
        .map(|&ub| (ub, at(ub).unwrap_or(0)))
        .find(|&(_, fit)| fit >= want_ctx)
        .unwrap_or_else(|| {
            cands
                .iter()
                .map(|&ub| (ub, at(ub).unwrap_or(0)))
                .max_by_key(|&(ub, fit)| (fit, ub))
                .expect("ubatch_candidates is never empty")
        });
    // Lower the pinned height BEFORE the KV buffers below are sized: the ring is `window + chunk`
    // rows, and the reserve this fit was priced with is that height's.
    if chunk < ubatch_rows(ec) {
        repin_ubatch_lower(chunk);
    }
    if fit >= want_ctx {
        return want_ctx;
    }
    if ec.device.ctx.is_some() {
        tracing::warn!(
            requested_ctx = want_ctx,
            fits_ctx = fit,
            "ctx: only {fit} tokens fit the {:.2} GiB the device reports free after the weights, \
             but the context was set explicitly — honoring it. The allocation guard will fail \
             this session if it really does not fit; lower INFR_CTX/--ctx to pick a window that \
             does.",
            gib(room),
        );
        return want_ctx;
    }
    let ubatch = ubatch_rows(ec);
    let ring = placement_ring(cfg, ec, k_fmt, v_fmt);
    tracing::warn!(
        requested_ctx = want_ctx,
        fits_ctx = fit,
        "ctx clamp (measured): {want_ctx} -> {fit} tokens against the {:.2} GiB the device reports \
         free after the weights are resident (KV {:.2} GiB + activation reserve {:.2} GiB at a \
         {ubatch}-row chunk, minus {:.2} GiB held for the driver's own later allocations) — the \
         pre-load estimate does not price the weight arena's block tails or the driver's own \
         memory; set INFR_CTX to override",
        gib(room),
        gib(kv_bytes_estimate_fmt(cfg, fit, ring, ubatch, k_fmt, v_fmt)),
        gib(runtime_reserve_at(
            cfg, &caps, fit, ring, ubatch, k_fmt, v_fmt,
        )),
        gib(POST_KV_DEVICE_RESERVE),
    );
    fit
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

/// Select the backing store for routed experts after their exact load-time layout is known.
///
/// `payload_bytes` includes the alignment of the layer-contiguous DMA layout, so `Full` means the
/// configured/automatic RAM budget can hold every byte Decode or Prefill can request. In that mode
/// the Vulkan pager never constructs a `FileBlockIo`: SSD is a
/// load-time source only, and runtime misses stop at RAM. A smaller budget selects the bounded
/// inclusive RAM/SSD tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoeHostBacking {
    Full,
    Bounded { bytes: usize },
}

fn moe_host_backing(
    ram_request: infr_core::hostmem::RamRequest,
    available: Option<u64>,
    process_resident: Option<u64>,
    payload_bytes: usize,
) -> MoeHostBacking {
    let budget = match ram_request {
        infr_core::hostmem::RamRequest::TotalProcessBudget(total) => {
            infr_core::hostmem::cache_bytes_for_total_budget(
                total,
                process_resident,
                payload_bytes as u64,
            )
        }
        infr_core::hostmem::RamRequest::LegacyCacheBudget(bytes) => bytes.min(payload_bytes as u64),
        infr_core::hostmem::RamRequest::Bypass => 0,
        infr_core::hostmem::RamRequest::Auto => available
            .map(|available| {
                infr_core::hostmem::auto_cache_bytes(available, 0, payload_bytes as u64)
            })
            .unwrap_or(0),
    } as usize;
    if budget >= payload_bytes {
        MoeHostBacking::Full
    } else {
        MoeHostBacking::Bounded { bytes: budget }
    }
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
    // with a double-width slot. Physical cache pools are keyed only by expert slot size, so roles
    // share capacity while mixed-dtype layouts still receive the distinct geometries they need.
    let cache_override = ec.paging.cache;
    let caps = vk.capabilities();

    let mut n_paged = 0usize; // paged layer-count (0 = fully resident, or all = cfg.n_layer)
    let mut expert_cache_target_bytes = 0u64;
    let mut pager_budget_bytes = 0u64;
    let mut pager_memory_plan = None;
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
    // Carried into the arena search's `required_after_arena` when MTP is opted in (see the
    // computation inside the first-load block below).
    let mut mtp_head_reserve: u64 = 0;
    if first_load && cfg.moe.is_some() {
        // NB: the load-time expert-bank dtype gate that used to live here (field report:
        // MXFP4_MOE expert banks `expect`-panicked mid-inference before it existed) is GONE —
        // the id-indexed GEMV floor now covers every dtype a GGUF expert bank can hold (the
        // full dense native set plus F16/F32 for float banks), so
        // `infr_vulkan::linear::moe_expert_dtype_ok` is true for all of them; the invariant is
        // pinned by `moe_expert_floor_covers_dense_set` in infr-vulkan's linear.rs tests.
        let fp = crate::weights::weight_footprint(g);
        let vram = vk.vram();
        let room = planned_vram_room(&vram, ec);
        // MTP headroom (qwen35/qwen35moe speculative decode): the MTP head session is built AFTER
        // placement by the chat layer and allocates its own weights/KV/draft-chain buffers through
        // the SAME unified guard. The head's need is therefore carried into the arena search's
        // `required_after_arena` (see below) instead of being subtracted from `room` up front:
        // inside the requirement, a shortfall degrades the physical runtime margin and then
        // shrinks the EXPERT arena — trading expert-cache capacity for head space — instead of
        // erroring when a static subtraction guessed wrong (a fixed scratch cannot cover both
        // 8K and 131K: measured shortfalls on the RX 7700 XT ranged from -289 MiB at 8K to
        // +287 MiB at 131K). Computed from the GGUF: the head layer's own tensor bytes (native
        // dtype, the same upload the binder does) + the draft-chain vocab embed table
        // (token_embd F16 upload, vocab*ne*2 B) + head KV + a scratch margin for the BDA block
        // geometry rounding (`bda_weight_alloc` floors each resident block at 64/128/256 MiB).
        let mtp_head_reserve = if crate::mtp::should_use_mtp(cfg, ec) {
            let head_prefix = format!("blk.{}.", cfg.n_layer);
            let head_weights: usize = g
                .tensors()
                .iter()
                .filter(|t| t.name.starts_with(&head_prefix))
                .map(|t| {
                    let numel: usize = t.shape.iter().product();
                    infr_gguf::nbytes(t.dtype, numel)
                })
                .sum();
            // The trunk's own token_embd upload during the MTP verify binds is F16 (dequantized
            // from its GGUF dtype — Q8_0 here), i.e. TWICE the bytes minimum_required prices it
            // at. This term covers that delta (and the head's device embed table when
            // `device_embed_enabled`, which shares the same magnitude).
            let embed_table = cfg.vocab * cfg.n_embd * 2;
            let head_kv = want_ctx * cfg.n_kv * cfg.head_dim * 2 * 2; // K+V f16, 1 layer
                                                                      // Scratch covers the BDA geometry rounding AND the head's later blocks beyond the
                                                                      // first `resident-bda` request: with the reserve inside required_after_arena, an
                                                                      // undersized scratch surfaces as a mid-upload guard refusal (measured: 272.0 MiB
                                                                      // block refused with 180.7 MiB left at 131K kv-q8 on the RX 7700 XT). 896 MiB
                                                                      // closes it; the shortfall-driven search pays for it out of expert slots.
                                                                      // Scratch covers the BDA geometry rounding (272/397 MiB blocks vs raw tensor bytes)
                                                                      // plus the dynamic KV expansion segments that land after placement — with the reserve
                                                                      // enforced inside the arena search's `required_after_arena`, an undersized scratch
                                                                      // surfaces as a mid-upload guard refusal; the loop pays for the bigger scratch out of
                                                                      // expert slots instead.
            let scratch = 1280 << 20;
            let mtp_reserve = head_weights + embed_table + head_kv + scratch;
            tracing::info!(
                "[mtp] headroom reserve: head_weights={:.0}MB embed_table={:.0}MB head_kv={:.0}MB \
                 scratch={:.0}MB total={:.2}GB",
                head_weights as f64 / 1e6,
                embed_table as f64 / 1e6,
                head_kv as f64 / 1e6,
                scratch as f64 / 1e6,
                mtp_reserve as f64 / 1e9,
            );
            mtp_reserve as u64
        } else {
            0
        };
        // Per-layer rows: SWA layers ring at window+ubatch rows (see `kv_rows`), so a mostly-SWA
        // model's KV prices far below n_layer * ctx. Price the actual per-side Vulkan formats:
        // explicit Q8 and F16 choices must change the expert remainder just like the allocation.
        let ring = kv_ring_wanted(cfg, ec);
        let k_fmt = vulkan_kv_fmt_for_budget(cfg, ec, ec.kv.type_k);
        let v_fmt = vulkan_kv_fmt_for_budget(cfg, ec, ec.kv.type_v);
        let dynamic_layout = segmented_kv_wanted(cfg, ec, ring, k_fmt, v_fmt).then(|| {
            segmented_kv::SegmentedKvLayout::for_qwen(cfg, want_ctx, k_fmt, v_fmt)
                .expect("Qwen hybrid models have segmented KV geometry")
        });
        let dynamic_kv_reserve = dynamic_layout
            .as_ref()
            .map(|layout| layout.committed_bytes(want_ctx))
            .unwrap_or(0);
        let kv_bytes_at = |ubatch| match (k_fmt, v_fmt) {
            (DType::Q8_0, DType::Q8_0) => kv_bytes_estimate(cfg, want_ctx, ring, ubatch, true),
            (DType::F16, DType::F16) => kv_bytes_estimate(cfg, want_ctx, ring, ubatch, false),
            _ => kv_bytes_estimate_fmt(cfg, want_ctx, ring, ubatch, k_fmt, v_fmt),
        };
        let initial_ubatch = ubatch_rows(ec);
        let mut selected_ubatch = initial_ubatch;
        let kv_bytes = kv_bytes_at(selected_ubatch);
        let persistent_state = kv_bytes.saturating_sub(dynamic_kv_reserve);
        // Reserve the workspace for the chunk this session will actually execute. A user selecting
        // 4096 rows still gets the full 4K reserve; the default 1024-row session no longer strands
        // the difference behind a permanent 2 GiB/4K assumption. Prefill's layer ring already
        // borrows only cold Decode arena ranges and returns them on `enter_decode`.
        let runtime_reserve =
            runtime_reserve_at(cfg, &caps, want_ctx, ring, selected_ubatch, k_fmt, v_fmt);
        let packing_margin = resident_weight_packing_margin(fp.dense);
        let load_driver_reserve = session_load_driver_reserve(cfg, ec);
        let Some(mut plan) = ModelMemoryPlan::new_with_dynamic_reserve(
            room,
            fp.dense,
            persistent_state,
            runtime_reserve,
            dynamic_kv_reserve,
            packing_margin,
            load_driver_reserve,
            POST_KV_DEVICE_RESERVE,
        ) else {
            return Err(anyhow!(
                "this MoE model's dense weights ({:.2} GB) + KV cache ({:.2} GB) exceed the unified \
                 VRAM room ({:.2} GB after guard/reserve/configured cap) — dense layer streaming \
                 does not cover MoE models' dense parts; reduce ctx or run on the CPU backend \
                 (INFR_DEV=cpu)",
                fp.dense as f64 / 1e9,
                kv_bytes as f64 / 1e9,
                room as f64 / 1e9,
            ));
        };
        let requested_cache = cache_override.map(|spec| spec.resolve(vram.available));
        let paged_target_at = |candidate: ModelMemoryPlan| match requested_cache {
            Some(requested) => Some(requested.min(candidate.expert_cache_bytes)),
            None if candidate.expert_cache_bytes < fp.expert => Some(candidate.expert_cache_bytes),
            None => None,
        };
        let prefill_floor = moe_prefill_floor_bytes(g, cfg);
        let mut paged_target = paged_target_at(plan);
        if paged_target.is_some_and(|bytes| bytes < prefill_floor) {
            for candidate in moe_ubatch_fallback_candidates(ec).into_iter().skip(1) {
                let candidate_kv = kv_bytes_at(candidate);
                let candidate_persistent = candidate_kv.saturating_sub(dynamic_kv_reserve);
                let candidate_runtime =
                    runtime_reserve_at(cfg, &caps, want_ctx, ring, candidate, k_fmt, v_fmt);
                let Some(candidate_plan) = ModelMemoryPlan::new_with_dynamic_reserve(
                    room,
                    fp.dense,
                    candidate_persistent,
                    candidate_runtime,
                    dynamic_kv_reserve,
                    packing_margin,
                    load_driver_reserve,
                    POST_KV_DEVICE_RESERVE,
                ) else {
                    continue;
                };
                let candidate_target = paged_target_at(candidate_plan);
                if candidate_target.is_none_or(|bytes| bytes >= prefill_floor) {
                    selected_ubatch = candidate;
                    plan = candidate_plan;
                    paged_target = candidate_target;
                    break;
                }
            }
        }
        if paged_target.is_some_and(|bytes| bytes < prefill_floor) {
            return Err(anyhow!(
                "MoE expert cache leaves {:.2} MiB, but one complete Prefill layer needs {:.2} MiB; \
                 increase INFR_VRAM_BUDGET/INFR_CACHE or reduce context/runtime memory",
                paged_target.unwrap_or(0) as f64 / 2f64.powi(20),
                prefill_floor as f64 / 2f64.powi(20),
            ));
        }
        if selected_ubatch != initial_ubatch {
            cap_moe_ubatch(selected_ubatch);
            tracing::warn!(
                "MoE placement: lowered the Prefill chunk from {initial_ubatch} to \
                 {selected_ubatch} rows because the larger chunk's runtime reserve left less than \
                 one complete Expert streaming lane"
            );
        }

        let auto_budget = plan.expert_cache_bytes;
        match cache_override {
            Some(spec) => {
                n_paged = cfg.n_layer;
                // Legacy expert-only override remains supported, but cannot punch through the new
                // total-process budget or the physical allocator ceiling. Percent keeps its old
                // AVAILABLE-VRAM base for compatibility.
                let requested = spec.resolve(vram.available);
                expert_cache_target_bytes = requested.min(auto_budget);
                if requested > auto_budget {
                    tracing::warn!(
                        "INFR_CACHE requested {:.2} GB of expert arena but the unified VRAM plan \
                         leaves {:.2} GB; clamping the arena to the safe remainder",
                        requested as f64 / 1e9,
                        auto_budget as f64 / 1e9,
                    );
                }
            }
            None if auto_budget < fp.expert => {
                // Page EVERY expert layer with the WHOLE remainder. A shared LRU arena keeps hot
                // experts from every layer and is strictly more flexible than pinning a prefix of
                // complete layers.
                n_paged = cfg.n_layer;
                expert_cache_target_bytes = auto_budget;
            }
            None => {}
        }
        if n_paged > 0 {
            // Runtime scratch and Experts now share one physical elastic arena. The planner still
            // subtracts the worst-case runtime peak to preserve the hard total-VRAM ceiling, but
            // those bytes are no longer stranded outside the cache while Decode uses only a tiny
            // workspace: they begin life as ordinary expert slots and graph execution borrows
            // exactly the ranges it actually allocates. Initializing the arena first also serves
            // the old load-time escrow purpose, so a second dedicated reservation would both
            // waste VRAM and double-charge the same bytes.
            pager_budget_bytes = plan.elastic_pool_bytes(expert_cache_target_bytes);
            pager_memory_plan = Some(plan);
        }
        let cache_layout = if cfg.deepseek4 {
            "fp8-kv+mxfp4-index".to_string()
        } else if dynamic_kv_reserve > 0 {
            format!("dynamic-32k k={k_fmt:?}, v={v_fmt:?}")
        } else {
            format!("k={k_fmt:?}, v={v_fmt:?}")
        };
        tracing::info!(
            "VRAM plan: total_room={:.2} GB fixed={:.2} GB state={:.2} GB runtime_elastic={:.2} GB \
             dynamic_state_elastic={:.2} GB \
             packing_margin={:.2} GB load_driver={:.2} GB post_load={:.2} GB \
             expert_cache_target={:.2} GB elastic_pool={:.2} GB ({cache_layout}, ctx={want_ctx})",
            room as f64 / 1e9,
            plan.fixed_weight_bytes as f64 / 1e9,
            plan.persistent_state_bytes as f64 / 1e9,
            plan.runtime_reserve_bytes as f64 / 1e9,
            plan.dynamic_state_reserve_bytes as f64 / 1e9,
            plan.weight_packing_margin_bytes as f64 / 1e9,
            plan.load_driver_reserve_bytes as f64 / 1e9,
            plan.post_load_reserve_bytes as f64 / 1e9,
            (if n_paged > 0 {
                expert_cache_target_bytes
            } else {
                fp.expert
            }) as f64
                / 1e9,
            pager_budget_bytes as f64 / 1e9,
        );
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
    // The MoE twin, keyed by the `(role, per-expert bytes)` pair that identifies an expert pool —
    // the binder receives a tensor, not a pool index, and re-derives that key the same way
    // `MoePagerSession::register` does.
    let mut moe_host_offsets =
        std::collections::HashMap::<(usize, infr_vulkan::pager::Role), usize>::new();
    let mut moe_host_by_size = std::collections::BTreeMap::<
        usize,
        std::sync::Arc<infr_core::hostpager::InclusiveHostCache>,
    >::new();
    if first_load && n_paged > 0 {
        use infr_vulkan::pager::Role;
        let moe = cfg.moe.as_ref().expect("n_paged > 0 implies MoE");
        let n_expert = moe.n_expert.max(1);
        // MoE runtime weights DMA directly from one permanent HOST_CACHED transfer source; no
        // upload ring or host pager is allocated for this path.
        // Enumerate every paged `_exps` bank's role and per-expert byte size. The role remains
        // source/dispatch metadata, while all banks with the same physical size share one logical
        // cache pool. Mixed-dtype layouts naturally create additional size pools.
        let mut pool_blocks: Vec<(Role, usize, usize)> = Vec::new();
        let mut host_banks: Vec<(usize, Role, usize)> = Vec::new();
        for t in g.tensors() {
            let Some(layer) = exps_layer(&t.name).filter(|&l| l < n_paged) else {
                continue; // not a paged layer's `_exps` tensor
            };
            let Some(role) = moe_role_of(&t.name) else {
                continue; // `_exps` but not a weight bank (e.g. the per-expert `.scale` vector)
            };
            let sb = (t.nbytes / n_expert).max(4);
            host_banks.push((layer, role, t.nbytes));
            match pool_blocks
                .iter_mut()
                .find(|(r, s, _)| *r == role && *s == sb)
            {
                Some((_, _, n)) => *n += n_expert,
                None => pool_blocks.push((role, sb, n_expert)),
            }
        }
        host_banks.sort_unstable_by_key(|&(layer, role, _)| {
            let role_order = match role {
                Role::Gate => 0usize,
                Role::Up => 1,
                Role::Down => 2,
            };
            (layer, role_order)
        });
        let mut host_bytes = 0usize;
        let mut host_layers = Vec::<(usize, usize)>::new();
        let mut current_layer = None;
        let mut layer_start = 0usize;
        for (layer, role, bytes) in host_banks {
            if current_layer != Some(layer) {
                if current_layer.is_some() {
                    host_bytes = host_bytes.next_multiple_of(256);
                    host_layers.push((layer_start, host_bytes));
                }
                host_bytes = host_bytes.next_multiple_of(256);
                layer_start = host_bytes;
                current_layer = Some(layer);
            }
            host_bytes = host_bytes.next_multiple_of(256);
            moe_host_offsets.insert((layer, role), host_bytes);
            host_bytes = host_bytes
                .checked_add(bytes)
                .ok_or_else(|| anyhow!("MoE permanent host-store size overflow"))?;
        }
        host_bytes = host_bytes.next_multiple_of(256);
        if current_layer.is_some() {
            host_layers.push((layer_start, host_bytes));
        }
        let mut host_chunks = Vec::<infr_vulkan::pager::MoeHostChunkSpec>::new();
        if pool_blocks.is_empty() {
            // Defensive: an MoE config with NO pageable `_exps` weight banks (no arch this crate
            // loads ships that). Nothing to page — stay fully resident and let the alloc-time
            // VRAM budget guard produce its clear error if that overflows. `n_paged = 0` also
            // turns the binder's paged divert below into a no-op (it re-checks `n_paged`).
            tracing::warn!(
                "MoE pager: no pageable `_exps` weight banks found — keeping every expert \
                 resident (the VRAM budget guard is the backstop)"
            );
            n_paged = 0;
        }
        if n_paged > 0 {
            let n_blocks = n_paged * n_expert;
            // The permanent HostWeights store is system RAM, so the entire configured MoE cache
            // budget remains available to the shared GPU arena.
            // Pool identity is ONLY the physical bytes per expert. Gate/Up/Down banks with the
            // same slot size are interchangeable cache occupants: the source metadata retains
            // the role and the dispatch retains the layer's dtype/shape. Mixed quant layouts
            // naturally form more size pools without reviving role-sharded cache ownership.
            let mut by_size = std::collections::BTreeMap::<usize, (usize, [usize; 3])>::new();
            for &(role, slot_bytes, blocks) in &pool_blocks {
                let entry = by_size.entry(slot_bytes).or_insert((0, [0; 3]));
                entry.0 += blocks;
                let idx = match role {
                    Role::Gate => 0,
                    Role::Up => 1,
                    Role::Down => 2,
                };
                entry.1[idx] += blocks;
            }
            let logical_pools: Vec<(usize, usize, [usize; 3])> = by_size
                .into_iter()
                .map(|(slot_bytes, (blocks, role_blocks))| (slot_bytes, blocks, role_blocks))
                .collect();
            // A miss has both a fixed paging/submit component and a size-proportional transfer
            // component. On the validated balanced two-size geometry, assigning resident
            // fraction proportional to size^2 reduced transferred bytes and paging windows. Do
            // not generalize it to arbitrary mixed quant layouts: IQ4_NL_XL and Q4_K_M regress.
            let auto_size_bias_layout = logical_pools.len() == 2
                && logical_pools
                    .iter()
                    .all(|&(_, _, role_blocks)| role_blocks.into_iter().all(|n| n != 0))
                && (1.1..=1.5).contains(&(logical_pools[1].0 as f64 / logical_pools[0].0 as f64));
            let (size_cache_bias, size_cache_bias_source) = match ec.paging.moe_size_cache_bias {
                Some(value) => (value.clamp(-8.0, 8.0) as f64, "explicit"),
                None if auto_size_bias_layout => (2.0, "auto"),
                None => (0.0, "off"),
            };
            // Commit the mapped arena before allocating host caches or loading weights, then ask
            // the driver how much room is ACTUALLY left. `VK_EXT_memory_budget` accounting for a
            // large ReBAR mapping is card/driver dependent: WDDM can charge more than the logical
            // VkDeviceMemory size, so a plan that fits arithmetically on one GPU can leave too
            // little room for the same fixed weights on another. Automatic placement shrinks and
            // retries while the arena is still empty; an explicit `paging.cache` remains exact.
            let plan = pager_memory_plan.expect("n_paged > 0 carries its selected memory plan");
            let physical_pool_floor = moe_pool_physical_floor_bytes(&logical_pools, n_expert)
                .ok_or_else(|| anyhow!("MoE physical pool floor size overflow"))?;
            let planned_pager_budget = pager_budget_bytes;
            let elastic_reserve = plan.elastic_reserve_bytes();
            let minimum_pager_budget = elastic_reserve.saturating_add(physical_pool_floor);
            // The MTP head session allocates through the same guard right after this loop, so
            // when the user opted in its reserve is part of what must stay live beside the
            // arena (a shortfall then degrades the margin and shrinks expert slots — head
            // space is traded against expert-cache capacity, never against starting at all).
            let required_after_arena = plan
                .minimum_required_bytes()
                .saturating_sub(elastic_reserve)
                .saturating_add(mtp_head_reserve);
            let adaptive_arena = cache_override.is_none();
            let mut candidate_budget = pager_budget_bytes;
            // The runtime margin became PHYSICAL trailing arena shards (see
            // `MoePagerLayout::runtime_margin_bytes`), which raises the mapped-arena size by the
            // full activation reserve. At large KV sizes that can overflow a small card even
            // though the same plan fit when the reserve borrowed expert slots at runtime. The
            // margin is a placement-QUALITY knob (runtime windows coalesce instead of borrowing),
            // not a correctness requirement, so on a fit shortfall degrade it: first by the
            // shortfall itself, then to zero (full slot-borrowing, the proven pre-margin
            // behaviour) — and only error when even the margin-free layout does not fit.
            let mut effective_margin = plan.runtime_reserve_bytes;
            let mut attempts = 0usize;
            let slot_counts = loop {
                attempts += 1;
                let Some(candidate_slots) = moe_pool_slot_counts(
                    &logical_pools,
                    candidate_budget,
                    n_expert,
                    size_cache_bias,
                ) else {
                    return Err(anyhow!(
                        "MoE expert arena budget ({:.2} MiB) cannot hold one complete Prefill \
                         layer plus runtime workspace; increase the VRAM budget or reduce context",
                        candidate_budget as f64 / 2f64.powi(20),
                    ));
                };
                let physical_bytes = moe_pool_capacity_bytes(&logical_pools, &candidate_slots);
                if physical_bytes.saturating_sub(elastic_reserve) < physical_pool_floor {
                    return Err(anyhow!(
                        "MoE mapped arena cannot retain one complete Prefill layer after its \
                         runtime reserve plus the tiered-cache exchange slots (arena {:.2} MiB, \
                         runtime {:.2} MiB, physical floor {:.2} MiB)",
                        physical_bytes as f64 / 2f64.powi(20),
                        elastic_reserve as f64 / 2f64.powi(20),
                        physical_pool_floor as f64 / 2f64.powi(20),
                    ));
                }
                let specs: Vec<(usize, usize)> = logical_pools
                    .iter()
                    .zip(&candidate_slots)
                    .map(|(&(slot_bytes, ..), &n_slots)| (slot_bytes, n_slots))
                    .collect();

                let failure = match vk.prepare_moe_unified_vram(&specs, effective_margin as usize) {
                    Ok(committed) => {
                        debug_assert_eq!(committed as u64, physical_bytes + effective_margin);
                        let live_room = vk.alloc_room();
                        if live_room >= required_after_arena {
                            pager_budget_bytes = physical_bytes;
                            expert_cache_target_bytes = expert_cache_target_bytes
                                .min(physical_bytes.saturating_sub(elastic_reserve));
                            if attempts > 1 {
                                tracing::warn!(
                                    planned_bytes = planned_pager_budget,
                                    actual_bytes = pager_budget_bytes,
                                    attempts,
                                    "automatic MoE arena reduced after live Vulkan allocation \
                                     feedback; fixed weights, KV and runtime reserves remain intact"
                                );
                            }
                            break candidate_slots;
                        }
                        let shortfall = required_after_arena - live_room;
                        vk.discard_empty_moe_unified_vram()
                            .map_err(|e| anyhow!("discarding MoE allocation probe: {e}"))?;
                        // Degrade the runtime margin before giving up: unmapping margin bytes
                        // frees them one-for-one, so shrinking by `shortfall` closes the gap
                        // directly (windows borrow expert slots for the degraded remainder,
                        // exactly the pre-margin placement behaviour).
                        if effective_margin > 0 {
                            let shrunk = effective_margin.saturating_sub(shortfall);
                            tracing::warn!(
                                from_margin_bytes = effective_margin,
                                to_margin_bytes = shrunk,
                                shortfall_bytes = shortfall,
                                "runtime margin does not fit at this KV size; degrading the \
                                 pager to expert-slot borrowing for the difference"
                            );
                            effective_margin = shrunk;
                            attempts -= 1;
                            continue;
                        }
                        (
                            shortfall,
                            format!(
                                "the mapped arena left {:.2} MiB live, {:.2} MiB short of fixed \
                                 weights/KV/reserves",
                                live_room as f64 / 2f64.powi(20),
                                shortfall as f64 / 2f64.powi(20),
                            ),
                        )
                    }
                    Err(error) => (0, error.to_string()),
                };

                if !adaptive_arena {
                    return Err(anyhow!(
                        "explicit MoE expert arena {:.2} MiB did not fit this device: {}",
                        physical_bytes as f64 / 2f64.powi(20),
                        failure.1,
                    ));
                }
                let next = (attempts < AUTO_MOE_ARENA_MAX_ATTEMPTS)
                    .then(|| {
                        next_auto_moe_arena_budget(
                            candidate_budget,
                            minimum_pager_budget,
                            failure.0,
                        )
                    })
                    .flatten()
                    .ok_or_else(|| {
                        anyhow!(
                            "automatic MoE arena could not find a device-safe size after \
                             {attempts} attempt(s): {}",
                            failure.1,
                        )
                    })?;
                tracing::warn!(
                    attempt = attempts,
                    old_bytes = candidate_budget,
                    new_bytes = next,
                    reason = %failure.1,
                    "automatic MoE arena allocation retry"
                );
                candidate_budget = next;
            };
            // The Host tier has two honest modes. If the configured/automatic RAM budget covers
            // the whole expert payload, retain the existing layer-contiguous store (fastest
            // Prefill and Decode source). Otherwise allocate bounded per-size-class victim
            // caches and leave the remaining Experts on SSD. GPU-resident Experts retain a pinned
            // RAM shadow when capacity permits, making GPU eviction metadata-only.
            let ram_request = host_ram_request(ec);
            let host_available = infr_core::hostmem::available_bytes();
            let process_resident = infr_core::hostmem::process_resident_bytes();
            let host_backing =
                moe_host_backing(ram_request, host_available, process_resident, host_bytes);
            let (host_kind, host_resident_bytes) = match host_backing {
                MoeHostBacking::Full => ("full-RAM", host_bytes),
                MoeHostBacking::Bounded { bytes } => ("inclusive-RAM/SSD", bytes),
            };
            log_host_ram_request(
                "MoE",
                ram_request,
                process_resident,
                host_resident_bytes as u64,
            );
            if let MoeHostBacking::Bounded {
                bytes: host_cache_budget,
            } = host_backing
            {
                let classes: Vec<(usize, usize)> = logical_pools
                    .iter()
                    .map(|&(slot_bytes, blocks, _)| (slot_bytes, blocks))
                    .collect();
                let ram_slots = infr_core::hostpager::plan_slots(host_cache_budget, &classes);
                let io = std::sync::Arc::new(
                    infr_core::blockio::FileBlockIo::open_shards(&g.shards())
                        .map_err(|e| anyhow!("{e}"))?,
                );
                for (&(slot_bytes, _, _), &slots) in logical_pools.iter().zip(&ram_slots) {
                    let cache = infr_core::hostpager::InclusiveHostCache::new(
                        slots,
                        slot_bytes,
                        io.clone(),
                    )
                    .map_err(|e| anyhow!("{e}"))?;
                    moe_host_by_size.insert(slot_bytes, std::sync::Arc::new(cache));
                }
                tracing::info!(
                    "MoE host plan: bounded inclusive RAM cache {:.2} GB / {:.2} GB expert payload; GPU shadows share this budget and remaining Experts stream from SSD",
                    host_cache_budget as f64 / 1e9,
                    host_bytes as f64 / 1e9,
                );
            } else {
                // Keep the complete payload in one logical layer-major store, split only BETWEEN
                // layers so each layer remains a contiguous Prefill source. Bounded-RAM mode does
                // not allocate this store and therefore must not inherit its per-chunk limit.
                const HOST_CHUNK_MAX: usize = 2 * 1024 * 1024 * 1024;
                let mut chunk_ranges = Vec::<(usize, usize)>::new();
                for &(start, end) in &host_layers {
                    if end - start > HOST_CHUNK_MAX {
                        return Err(anyhow!(
                            "MoE expert layer requires {:.2} GiB, above the {:.2} GiB permanent \
                             host-store chunk limit",
                            (end - start) as f64 / 2f64.powi(30),
                            HOST_CHUNK_MAX as f64 / 2f64.powi(30),
                        ));
                    }
                    match chunk_ranges.last_mut() {
                        Some((chunk_start, chunk_end)) if end - *chunk_start <= HOST_CHUNK_MAX => {
                            *chunk_end = end;
                        }
                        _ => chunk_ranges.push((start, end)),
                    }
                }
                host_chunks = chunk_ranges
                    .into_iter()
                    .map(|(base_offset, end)| infr_vulkan::pager::MoeHostChunkSpec {
                        base_offset,
                        bytes: end - base_offset,
                    })
                    .collect();
                tracing::info!(
                    "MoE host plan: full layer-contiguous RAM store {:.2} GB; RAM budget covers \
                     every routed expert, runtime SSD tier disabled",
                    host_bytes as f64 / 1e9,
                );
            }
            // No per-pool arena ceiling: each MoE pool is a `bufferDeviceAddress` buffer read by
            // pointer, so it is NOT capped by one SSBO binding's maxStorageBufferRange (~4 GiB on
            // RADV) the way it was when the arena was a bound SSBO. A pool now holds as many experts
            // as its budget share allows — the whole point of the u64 addressing lift. The only
            // backstop is the alloc-time VRAM budget guard (`GpuPager::new_mapped` -> per-pool
            // ReBAR BDA allocation);
            // the proportional split below never over-subscribes VRAM because it partitions
            // `pager_budget_bytes`, which the caller derived from the remaining VRAM.
            let pool_floors = moe_pool_batch_slot_floors(&logical_pools, n_expert);
            let pools: Vec<infr_vulkan::pager::MoePoolSpec> = logical_pools
                .iter()
                .zip(slot_counts)
                .zip(pool_floors)
                .map(
                    |((&(sb, _nb, _role_blocks), budget_slots), min_enabled_slots)| {
                        // Budget split PROPORTIONALLY to each pool's total bank bytes — the byte share is
                        // also the access share under uniform routing (every (layer, expert) read touches
                        // gate+up+down alike), so proportional slots equalize expected hit rates across
                        // pools; any fancier split would need routing statistics that don't exist at
                        // load time.
                        // Floor at `min(n_expert, nb)`: a chunked batched-prefill `Op::MoeFfn` (rows>1)
                        // runs ALL of a layer's routed buckets in ONE dispatch
                        // (`matmul_mmq_experts_paged`), touching up to `n_expert` DISTINCT experts of
                        // that layer that must be simultaneously resident (the within-batch safety
                        // invariant — see `infr_core::pager::Pager::new`'s doc). Decode's rows=1 needs
                        // only `n_used`, but the batched bound subsumes it and `n_expert` slots is tiny
                        // next to any real budget (Scout: 16 x ~18 MB per role). Capped at `nb` (no
                        // point holding more slots than the pool has distinct experts).
                        // The dynamic Prefill ring can legally degrade to one lane when the user
                        // budget cannot hold its topology target. The old A/B implementation
                        // required two full layers here and could silently exceed that budget.
                        infr_vulkan::pager::MoePoolSpec {
                            slot_bytes: sb,
                            n_slots: budget_slots,
                            min_enabled_slots,
                            host: moe_host_by_size.get(&sb).cloned(),
                        }
                    },
                )
                .collect();
            let cached: usize = pools.iter().map(|p| p.n_slots).sum();
            let host_chunk_count = host_chunks.len();
            let pool_desc: Vec<String> = logical_pools
                .iter()
                .zip(&pools)
                .map(|(&(sb, nb, _), p)| {
                    format!("shared[{:.1}MB] {}/{}", sb as f64 / 1e6, p.n_slots, nb)
                })
                .collect();
            tracing::info!(
                "MoE pager: {n_paged}/{} expert layers PAGED ({cached} expert blocks cached — {}; \
             {:.2} GB mapped ReBAR pool budget; Decode size bias {size_cache_bias:+.2} \
             ({size_cache_bias_source}); host={} {:.2} GB in {host_chunk_count} chunks; \
             ctx={want_ctx})",
                cfg.n_layer,
                pool_desc.join(", "),
                pager_budget_bytes as f64 / 1e9,
                host_kind,
                host_resident_bytes as f64 / 1e9,
            );
            // Recurrent hybrids use current + the longest consecutive recurrent run. The pager
            // may lower that target to fit the Expert-cache share, but never spends the runtime
            // reserve priced for the selected Prefill chunk. Pure Attention models retain their
            // established all-layer target.
            let prefill_target_lanes = moe_prefill_target_lanes(cfg, n_paged);
            vk.init_moe_pager(infr_vulkan::pager::MoePagerLayout {
                // The runtime reserve is part of `pager_budget_bytes` and therefore physically
                // escrowed by the unified arena itself. No second load-only allocation.
                load_reserve_bytes: 0,
                n_blocks,
                pools,
                host_chunks,
                prefill_target_lanes,
                prefill_cache_bytes: expert_cache_target_bytes,
                // The activation reserve becomes a PHYSICAL trailing shard (see
                // `MoePagerLayout::runtime_margin_bytes`): runtime windows live there instead of
                // borrowing expert slots, which is what wedged long serve sessions before.
                // The committed runtime margin: `effective_margin` starts at the full plan
                // reserve and degrades when the mapped arena would otherwise overflow the
                // device (windows borrow expert slots for the degraded part).
                runtime_margin_bytes: effective_margin,
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
    // The tier BELOW VRAM, one entry per dense pool (empty = none: every miss reads the mmap).
    let mut dense_host: Vec<Option<std::sync::Arc<infr_core::hostpager::HostPager>>> = Vec::new();
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
        // The per-side KV formats a chunk/format candidate prices: `q8` = BOTH sides Q8_0 (34
        // bytes / 32 elems), false = f16 (2 B/elem). Per-layer rows ring at window+ubatch rows
        // (see `kv_rows`) — what lets a mostly-SWA model (gemma-4-31B: 50/60 layers SWA) price its
        // KV small enough to take the try-resident tier at real contexts instead of streaming.
        let configured_k = vulkan_kv_fmt_for_budget(cfg, ec, ec.kv.type_k);
        let configured_v = vulkan_kv_fmt_for_budget(cfg, ec, ec.kv.type_v);
        let kv_fmts = |try_auto_q8: bool| {
            if try_auto_q8 {
                (DType::Q8_0, DType::Q8_0)
            } else {
                (configured_k, configured_v)
            }
        };
        // Does weights + KV + the honest activation reserve fit at this (chunk, fmt)? Through the
        // SHARED predicate, so this decision and `kv_fit_ctx_for`'s price the same session against
        // the same ceiling — the ALLOCATOR's (`VramInfo::alloc_room`), not the raw free figure.
        // Against `vram.available` this could declare a model resident while planning 256 MiB the
        // VRAM guard will refuse, and the two ladders picked different rungs (gemma-3-12b @131072:
        // the fit math validated the context at 256 rows while this went resident at 512).
        let fits = |ubatch: usize, q8: bool| {
            let (k, v) = kv_fmts(q8);
            dense_placement_fits(cfg, &caps, ec, fp.total(), &vram, want_ctx, ubatch, k, v)
        };
        // Try-resident-first: a dense model goes FULLY RESIDENT (the exact pre-streaming fast
        // path) whenever weights + this session's KV + an HONEST dense activation estimate fit
        // the allocatable VRAM; only a genuine miss streams. The MoE tier's 2 GiB ACT_HEADROOM is
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
            // `dense_resident_rung` walks the shared ladder — the SAME walk `kv_fit_ctx_for` makes
            // to decide a context fits, so the rung it lands on here is the rung that math priced.
            // Its first entry is the height `resident` above already priced and rejected.
            let (k, v) = kv_fmts(kv_auto_q8());
            if let Some(cand) =
                dense_resident_rung(cfg, &caps, ec, fp.total(), &vram, want_ctx, k, v)
            {
                pin_ubatch(cand);
                // Re-read through the pin (a racing earlier set wins — use whatever stuck).
                if fits(ubatch_rows(ec), kv_auto_q8()) {
                    tracing::warn!(
                        "dense placement: resident with a {}-row prefill chunk (the default \
                         1024-row chunk's activation reserve wouldn't fit); set INFR_UBATCH \
                         to override",
                        ubatch_rows(ec).min(want_ctx),
                    );
                    resident = true;
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
            && !kv_default_q8(cfg, ec)
            && kv_unset(ec)
            && kv_q8_layout_ok(cfg)
        {
            let (k, v) = kv_fmts(true);
            if let Some(cand) =
                dense_resident_rung(cfg, &caps, ec, fp.total(), &vram, want_ctx, k, v)
            {
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
        let streamed_fixed = fp.total().saturating_sub(streamable_resident);
        let budget = match cache_override {
            Some(spec) => {
                let requested = spec.resolve(vram.available);
                let (k, v) = kv_fmts(kv_auto_q8());
                let safe = dense_stream_budget_at(
                    cfg,
                    &caps,
                    ec,
                    streamed_fixed,
                    &vram,
                    want_ctx,
                    ubatch_rows(ec),
                    k,
                    v,
                );
                if requested > safe {
                    tracing::warn!(
                        "INFR_CACHE requested {:.2} GB of dense streaming arena but the unified \
                         VRAM plan leaves {:.2} GB; clamping the arena to the safe remainder",
                        requested as f64 / 1e9,
                        safe as f64 / 1e9,
                    );
                }
                Some(requested.min(safe))
            }
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
                // Same ceiling as `fits` above (`VramInfo::alloc_room`): every byte this
                // over-states is an arena slot the allocator then refuses.
                let budget_at = |ub: usize| {
                    let (k, v) = kv_fmts(q8);
                    dense_stream_budget_at(
                        cfg,
                        &caps,
                        ec,
                        streamed_fixed,
                        &vram,
                        want_ctx,
                        ub,
                        k,
                        v,
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
            // The tier below VRAM, sized before the pools so a `Host` source can name it. Pool
            // order is preserved, so pool `i`'s host tier is `dense_host[i]`.
            let classes: Vec<(usize, usize)> = pools.iter().map(|&(_, s, nb)| (s, nb)).collect();
            dense_host =
                vulkan_host_tier(ec, g, "dense", &classes, vk.capabilities().unified_memory)?;
            let specs: Vec<infr_vulkan::pager::DensePoolSpec> = pools
                .iter()
                .enumerate()
                .map(|(i, &(_, stride, nb))| {
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
                        host: dense_host[i].clone(),
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
            tracing::info!(
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
            // With a host tier under this pool, the group's bytes are read from the model file on
            // demand instead of faulted in through the mmap. `file_ranges` is `None` exactly for
            // bytes the loader REWROTE, which correspond to nothing on disk — those keep the mmap
            // source even when a tier exists (the eligibility filter already excludes them, so
            // this is a fallback, not a normal path).
            let host = dense_host.get(*pool).and_then(|h| h.as_ref());
            let (bytes, plan_bytes) = match (host, tb.file_ranges()) {
                (Some(h), Some(ranges)) => {
                    let total: usize = ranges.iter().map(|(_, l)| l).sum();
                    h.register(infr_core::blockio::BlockDesc {
                        id: *block_id,
                        extents: ranges
                            .iter()
                            .map(|&(offset, len)| infr_core::blockio::BlockExtent { offset, len })
                            .collect(),
                    })
                    .map_err(|e| anyhow!("{e}"))?;
                    (infr_vulkan::pager::DenseBytes::Host, total)
                }
                _ => {
                    let segments: Vec<std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>> = comps
                        .iter()
                        .map(|c| {
                            Ok(std::sync::Arc::new(
                                g.tensor_bytes_arc(c).map_err(|e| anyhow!("{e}"))?,
                            )
                                as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>)
                        })
                        .collect::<AResult<_>>()?;
                    let total = segments.iter().map(|s| s.as_ref().as_ref().len()).sum();
                    (infr_vulkan::pager::DenseBytes::Mmap(segments), total)
                }
            };
            if plan_bytes != tb.len() {
                return Err(anyhow!(
                    "dense streaming plan out of sync with the upload order for {name}: plan \
                     bytes {plan_bytes} != uploaded bytes {}",
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
                    bytes,
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
                let layer_base = (l * n_expert) as u32;
                let bank = std::sync::Arc::new(bytes.clone())
                    as std::sync::Arc<dyn AsRef<[u8]> + Send + Sync>;
                let host_offset = *moe_host_offsets.get(&(l, role)).ok_or_else(|| {
                    anyhow!("MoE permanent host-store plan has no offset for {name}")
                })?;
                let file = if let Some(host) = moe_host_by_size.get(&stride_bytes) {
                    let (base, len) = bytes.file_range();
                    if len != stride_bytes * n_expert {
                        return Err(anyhow!(
                            "MoE bounded Host tier: {name}'s file range is {len} bytes, expected \
                             {n_expert} x {stride_bytes}"
                        ));
                    }
                    let role_idx = match role {
                        infr_vulkan::pager::Role::Gate => 0usize,
                        infr_vulkan::pager::Role::Up => 1,
                        infr_vulkan::pager::Role::Down => 2,
                    };
                    let block_base = (role_idx * n_paged * n_expert + l * n_expert) as u32;
                    for expert in 0..n_expert {
                        host.register(infr_core::blockio::BlockDesc {
                            id: block_base + expert as u32,
                            extents: vec![infr_core::blockio::BlockExtent {
                                offset: base + (expert * stride_bytes) as u64,
                                len: stride_bytes,
                            }],
                        })
                        .map_err(|e| anyhow!("{e}"))?;
                    }
                    Some(infr_core::blockio::BlockDesc {
                        id: block_base,
                        extents: vec![infr_core::blockio::BlockExtent { offset: base, len }],
                    })
                } else {
                    None
                };
                let source = infr_vulkan::pager::ExpertSource {
                    bank,
                    stride_bytes,
                    layer_base,
                    host_offset,
                    file,
                };
                vk.register_paged_expert(role, buf_id, source, n_expert)
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
        let bytes = tb.materialize();
        let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
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
        None, // turn checkpoint boundary
        None, // req
        None, // mm
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
        let bytes = tb.materialize();
        let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
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
    tracing::info!(
        "pipeline: {}-way layer split of {} layers:",
        devices.len(),
        cfg.n_layer
    );
    for (di, &idx) in devices.iter().enumerate() {
        let lo = layer_map.iter().position(|&d| d == di);
        let hi = layer_map.iter().rposition(|&d| d == di);
        let count = layer_map.iter().filter(|&&d| d == di).count();
        match (lo, hi) {
            (Some(lo), Some(hi)) => tracing::info!(
                "  Vulkan{idx} ({}): layers [{lo}..={hi}] ({count} layers){}",
                names[di],
                if di + 1 < devices.len() {
                    ""
                } else {
                    " + final norm + lm_head"
                }
            ),
            _ => tracing::info!("  Vulkan{idx} ({}): (no layers)", names[di]),
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
                // Every rank's slice is cut from the whole tensor, so this binder needs the bytes
                // themselves — materialize once for all `world` cuts, not per rank.
                let tb = tb.materialize();
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
                let bytes = tb.materialize();
                let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
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
    tracing::info!(
        "tensor-parallel: {}-way weight split (n_head={}, n_kv={}, n_ff={}):",
        devices.len(),
        cfg.n_head,
        cfg.n_kv,
        cfg.n_ff
    );
    for (di, &idx) in devices.iter().enumerate() {
        tracing::info!(
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
            // Each rank uploads its own band of the bank, so the bytes are needed here.
            let tb = tb.materialize();
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
            let bytes = tb.materialize();
            let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
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
    tracing::info!(
        "expert-parallel: {}-way expert split ({} experts, {} used/token → {nl} experts/device):",
        devices.len(),
        moe.n_expert,
        moe.n_used,
    );
    for (di, &idx) in devices.iter().enumerate() {
        tracing::info!(
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
        None,
        req,
        None, // mm
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
        None,
        None,
        None, // mm
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
        None,
        None, // mm
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
        None,
        None, // mm
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
        None,
        None, // mm
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
            let bytes = tb.materialize();
            let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
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
        None,
        None, // mm
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
    /// Several mapped tensors to be laid down back to back — the fused qkv / gate+up groups.
    ///
    /// Kept as its COMPONENTS rather than a concatenated buffer, because a binder that pages or
    /// streams the group never wants the bytes at all: it registers the components' file ranges and
    /// has them read straight into a slot later. Materializing first meant building a multi-MB
    /// concat per fused group at load and immediately dropping it (and touching every one of those
    /// pages, which for a model that does not fit memory is the cost this whole tier exists to
    /// avoid).
    Concat(Vec<TensorBytes>),
}

impl WBytes {
    /// Byte length, without materializing anything.
    pub(crate) fn len(&self) -> usize {
        match self {
            WBytes::Mmap(tb) => tb.len(),
            WBytes::Owned(v) => v.len(),
            WBytes::Concat(parts) => parts.iter().map(|p| p.len()).sum(),
        }
    }

    /// The components' `(offset, len)` ranges in the model file, in layout order — what a binder
    /// that reads weights itself registers instead of taking the bytes.
    ///
    /// `None` for `Owned` bytes: those exist only because the loader REWROTE them (the qwen2 q/k
    /// row permute, the BitNet dequant), so they correspond to nothing on disk and re-reading the
    /// file would produce the pre-rewrite bytes.
    pub(crate) fn file_ranges(&self) -> Option<Vec<(u64, usize)>> {
        match self {
            WBytes::Mmap(tb) => Some(vec![tb.file_range()]),
            WBytes::Concat(parts) => Some(parts.iter().map(|p| p.file_range()).collect()),
            WBytes::Owned(_) => None,
        }
    }

    /// The bytes themselves. Borrowed for a single mapped tensor or an owned buffer; a `Concat` is
    /// joined HERE, so the cost lands on the binders that genuinely need the bytes and on no one
    /// else.
    pub(crate) fn materialize(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            WBytes::Mmap(tb) => std::borrow::Cow::Borrowed(tb),
            WBytes::Owned(v) => std::borrow::Cow::Borrowed(v),
            WBytes::Concat(parts) => {
                let mut out = Vec::with_capacity(self.len());
                for p in parts {
                    out.extend_from_slice(p);
                }
                std::borrow::Cow::Owned(out)
            }
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
    use super::{Config, DType, EngineConfig, PlacementPins, PlacementScope};
    use infr_core::backend::{Capabilities, COOPMAT_TILE_16};

    const GIB: usize = 1 << 30;

    #[test]
    fn host_ram_request_preserves_total_budget_and_legacy_cache_semantics() {
        use infr_core::{hostmem::RamRequest, SizeSpec};

        let mut config = EngineConfig::default();
        config.device.ram_budget = Some(SizeSpec::Bytes((50 * GIB) as u64));
        config.paging.dram = Some(SizeSpec::Bytes((7 * GIB) as u64));
        assert_eq!(
            super::host_ram_request(&config),
            RamRequest::TotalProcessBudget((50 * GIB) as u64),
            "the canonical process-wide budget must win over the compatibility cache override"
        );

        config.device.ram_budget = None;
        assert_eq!(
            super::host_ram_request(&config),
            RamRequest::LegacyCacheBudget((7 * GIB) as u64),
            "legacy paging.dram must retain its exact cache-only meaning"
        );

        config.device.ram_budget = Some(SizeSpec::Bytes(0));
        assert_eq!(
            super::host_ram_request(&config),
            RamRequest::TotalProcessBudget(0),
            "zero must retain its canonical source so diagnostics cannot mislabel it"
        );
    }

    #[test]
    fn moe_host_backing_disables_ssd_when_routed_payload_fits() {
        use infr_core::hostmem::RamRequest;

        let payload = 24 * GIB;
        assert_eq!(
            super::moe_host_backing(
                RamRequest::LegacyCacheBudget(payload as u64),
                None,
                Some(0),
                payload,
            ),
            super::MoeHostBacking::Full,
            "an exact explicit fit must disable the runtime SSD tier"
        );
        assert_eq!(
            super::moe_host_backing(
                RamRequest::LegacyCacheBudget((40 * GIB) as u64),
                None,
                Some(0),
                payload,
            ),
            super::MoeHostBacking::Full,
            "budget above the routed payload must not create a bounded SSD cache"
        );
        assert_eq!(
            super::moe_host_backing(RamRequest::Auto, Some((64 * GIB) as u64), Some(0), payload,),
            super::MoeHostBacking::Full,
            "automatic sizing must select the full store when its post-headroom budget fits"
        );
    }

    #[test]
    fn moe_host_backing_keeps_ssd_only_below_routed_payload() {
        use infr_core::hostmem::RamRequest;

        let payload = 24 * GIB;
        assert_eq!(
            super::moe_host_backing(
                RamRequest::LegacyCacheBudget((23 * GIB) as u64),
                None,
                Some(0),
                payload,
            ),
            super::MoeHostBacking::Bounded { bytes: 23 * GIB }
        );
        assert!(matches!(
            super::moe_host_backing(
                RamRequest::Auto,
                Some((25 * GIB) as u64),
                Some(0),
                payload,
            ),
            super::MoeHostBacking::Bounded { bytes } if bytes < payload
        ));
        assert_eq!(
            super::moe_host_backing(
                RamRequest::TotalProcessBudget(0),
                Some((64 * GIB) as u64),
                Some(0),
                payload,
            ),
            super::MoeHostBacking::Bounded { bytes: 0 }
        );
        assert_eq!(
            super::moe_host_backing(
                RamRequest::Bypass,
                Some((64 * GIB) as u64),
                Some(0),
                payload,
            ),
            super::MoeHostBacking::Bounded { bytes: 0 }
        );
    }

    #[test]
    fn moe_host_backing_resolves_total_ram_and_auto_requires_a_probe() {
        use infr_core::hostmem::RamRequest;

        let payload = 80 * GIB;
        assert_eq!(
            super::moe_host_backing(
                RamRequest::TotalProcessBudget((50 * GIB) as u64),
                Some((48 * GIB) as u64),
                Some((2 * GIB) as u64),
                payload,
            ),
            super::MoeHostBacking::Bounded {
                bytes: 48 * GIB - (512 << 20),
            },
            "the total target also covers persistent process objects created after planning"
        );
        assert_eq!(
            super::moe_host_backing(RamRequest::Auto, None, Some(0), payload),
            super::MoeHostBacking::Bounded { bytes: 0 },
            "auto sizing without a probe must not assume the whole payload fits RAM"
        );
    }

    /// Arithmetic tests that predate capability-aware hd256 flash use the conservative device:
    /// it preserves their old non-FA reserve exactly. Dedicated M4 cases opt into the XTX tier.
    fn conservative_caps() -> Capabilities {
        Capabilities::default()
    }

    fn hd256_flash_caps() -> Capabilities {
        Capabilities {
            coopmat_f16: Some(COOPMAT_TILE_16),
            max_shared_memory_bytes: infr_vulkan::FLASH_HD256_BM16_SHARED,
            ..Default::default()
        }
    }

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

    #[test]
    fn moe_viability_pin_may_lower_but_never_raise_explicit_ubatch() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let explicit = EngineConfig {
            device: infr_core::config::DeviceCfg {
                ubatch: Some(2048),
                ubatch_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(super::ubatch_rows(&explicit), 2048);
        assert_eq!(
            super::moe_ubatch_fallback_candidates(&explicit),
            vec![2048, 1024, 512, 256, 128]
        );

        super::repin_ubatch_lower(512);
        assert_eq!(
            super::ubatch_rows(&explicit),
            2048,
            "ordinary placement pins must not override an explicit height"
        );
        super::cap_moe_ubatch(1024);
        assert_eq!(super::ubatch_rows(&explicit), 1024);
        super::cap_moe_ubatch(4096);
        assert_eq!(super::ubatch_rows(&explicit), 1024);
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
    fn vulkan_default_kv_is_q8_only_when_unset_and_compatible() {
        let cfg = qwen3_14b();
        let unset = EngineConfig::default();
        assert!(super::kv_default_q8(&cfg, &unset));
        assert_eq!(
            super::vulkan_kv_fmt_for_budget(&cfg, &unset, None),
            DType::Q8_0
        );

        let explicit_f16 = EngineConfig {
            kv: infr_core::config::KvCfg {
                type_k: Some(DType::F16),
                type_k_specified: true,
                type_v: Some(DType::F16),
                type_v_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!super::kv_default_q8(&cfg, &explicit_f16));
        assert_eq!(
            super::vulkan_kv_fmt_for_budget(&cfg, &explicit_f16, Some(DType::F16)),
            DType::F16
        );

        let mla = deepseek_v2_lite_kv();
        assert!(!super::kv_default_q8(&mla, &unset));
        assert_eq!(
            super::vulkan_kv_fmt_for_budget(&mla, &unset, None),
            DType::F16
        );
    }

    #[test]
    fn segmented_kv_requires_q8_on_both_sides() {
        let cfg = Config {
            qwen35: true,
            n_layer: 1,
            n_kv: 2,
            head_dim: 128,
            full_attn_interval: 1,
            ..Default::default()
        };
        let mut ec = EngineConfig::default();
        ec.kv.dynamic = true;
        assert!(super::segmented_kv_wanted(
            &cfg,
            &ec,
            false,
            DType::Q8_0,
            DType::Q8_0
        ));
        assert!(!super::segmented_kv_wanted(
            &cfg,
            &ec,
            false,
            DType::F16,
            DType::F16
        ));
        assert!(!super::segmented_kv_wanted(
            &cfg,
            &ec,
            false,
            DType::Q8_0,
            DType::F16
        ));

        ec.kv.dynamic = false;
        assert!(!super::segmented_kv_wanted(
            &cfg,
            &ec,
            false,
            DType::Q8_0,
            DType::Q8_0
        ));
        ec.kv.dynamic = true;
        ec.kv.overflow = true;
        assert!(!super::segmented_kv_wanted(
            &cfg,
            &ec,
            false,
            DType::Q8_0,
            DType::Q8_0
        ));
        ec.kv.overflow = false;
        assert!(!super::segmented_kv_wanted(
            &cfg,
            &ec,
            true,
            DType::Q8_0,
            DType::Q8_0
        ));
    }

    #[test]
    fn q8_runtime_reserve_includes_pooled_f16_kv_expansion() {
        let cfg = qwen3_14b();
        let (ctx, ubatch) = (250_000usize, 1024usize);
        let caps = conservative_caps();
        let f16 =
            super::runtime_reserve_at(&cfg, &caps, ctx, false, ubatch, DType::F16, DType::F16);
        let q8 =
            super::runtime_reserve_at(&cfg, &caps, ctx, false, ubatch, DType::Q8_0, DType::Q8_0);
        let one_side = ctx as u64 * (cfg.n_kv * cfg.head_dim) as u64 * 2;
        assert_eq!(q8 - f16, 2 * one_side);
        assert!(
            super::runtime_reserve_at(
                &cfg,
                &caps,
                ctx / 2,
                false,
                ubatch,
                DType::Q8_0,
                DType::Q8_0,
            ) < q8
        );
    }

    #[test]
    fn kv_side_bytes_prices_each_side_in_its_own_dtype() {
        // q8 prices K+V at ~half the f16 bytes (34 B / 32-elem block vs 2 B/elem, ×2 sides).
        let elems = 32_000usize;
        let f16 = 2 * super::kv_side_bytes(DType::F16, elems) as u64;
        let q8 = 2 * super::kv_side_bytes(DType::Q8_0, elems) as u64;
        assert_eq!(f16, 2 * 2 * elems as u64); // 128_000
        assert_eq!(q8, 2 * (elems as u64 / 32 * 34)); // 68_000, already u32-aligned
        assert!(
            q8 < f16 && q8 * 2 > f16,
            "q8 must be ~half of f16, not equal"
        );
        // The MIXED case the single-flag helper could not express: `INFR_KV_TYPE_K=q8_0` with V
        // left at f16 is exactly one of each, not two of either.
        let mixed = (super::kv_side_bytes(DType::Q8_0, elems)
            + super::kv_side_bytes(DType::F16, elems)) as u64;
        assert_eq!(mixed, q8 / 2 + f16 / 2);
        assert!(mixed > q8 && mixed < f16);
        // A side the arch does not cache at all (MLA's V) gets only the bindable placeholder the
        // runner really allocates, not the K side's width.
        assert_eq!(super::kv_side_bytes(DType::F16, 0), 8);
    }

    // ── MLA (deepseek2) KV geometry + dtype gate — docs/backlog.md B41/B42 ───────────────────

    /// The KV-geometry fields of DeepSeek-V2-Lite-Chat, as `cpu_deepseek2_config` prints them off
    /// the real GGUF (`n_layer=27 n_head=16 n_embd=2048 kv_lora_rank=512 qk_rope_dim=64
    /// head_k_mla=128 v_head_dim=128`, and `n_kv == 1` which that test asserts). `head_dim` is
    /// MLA's `key_length_mla` = `head_k_mla + qk_rope_dim`. Every other field stays at its default:
    /// nothing below reads them.
    ///
    /// `n_kv * head_dim` is 192 here and the real cached row is 576, which is exactly why the
    /// open-coded product looked plausible at five call sites while being a third of the truth.
    fn deepseek_v2_lite_kv() -> Config {
        Config {
            deepseek2: true,
            n_layer: 27,
            n_head: 16,
            n_kv: 1,
            head_dim: 192,
            n_embd: 2048,
            kv_lora_rank: 512,
            qk_rope_dim: 64,
            head_k_mla: 128,
            v_head_dim: 128,
            ..Default::default()
        }
    }

    /// The B41 defect in one assertion: an MLA layer's cached row is `kv_lora_rank + qk_rope_dim`
    /// with NO V side, not `n_kv * head_dim` on both sides.
    #[test]
    fn kv_row_elems_mla_is_the_compressed_row_with_no_v_side() {
        let cfg = deepseek_v2_lite_kv();
        let (k, v) = super::kv_row_elems(&cfg, 0);
        assert_eq!(k, cfg.kv_lora_rank + cfg.qk_rope_dim, "MLA K row");
        assert_eq!(k, 576);
        assert_eq!(v, 0, "MLA has no V cache — V is a prefix view of the K row");
        assert_ne!(
            k,
            cfg.layer_n_kv(0) * cfg.layer_head_dim(0),
            "the open-coded product (1 x 192) is what three of the five sites used"
        );
        // Every non-MLA arch keeps the plain per-side product on both sides.
        let dense = qwen3_14b();
        assert_eq!(
            super::kv_row_elems(&dense, 0),
            (
                dense.layer_n_kv(0) * dense.layer_head_dim(0),
                dense.layer_n_kv(0) * dense.layer_head_dim(0)
            )
        );
        // The side an arch does not cache still gets a bindable placeholder, not a zero-size
        // allocation — and a real side is never rewritten.
        assert_eq!(super::kv_side_elems(v), super::KV_PLACEHOLDER_ELEMS);
        assert_eq!(super::kv_side_elems(k), k);
    }

    fn qwen35_hybrid_state() -> Config {
        Config {
            qwen35: true,
            n_layer: 40,
            full_attn_interval: 4,
            n_kv: 4,
            head_dim: 128,
            ssm_d_conv: 4,
            ssm_d_state: 128,
            ssm_d_inner: 1024,
            ssm_n_group: 8,
            ssm_dt_rank: 16,
            ..Default::default()
        }
    }

    #[test]
    fn moe_prefill_lanes_use_current_plus_longest_recurrent_run() {
        let qwen35 = qwen35_hybrid_state();
        assert_eq!(super::moe_prefill_target_lanes(&qwen35, 40), 4);

        let qwen38 = Config {
            qwen4exp: true,
            n_layer: 48,
            recurrent_layers: (0usize..48)
                .map(|layer| !(layer + 1).is_multiple_of(4))
                .collect(),
            ..Default::default()
        };
        assert_eq!(super::moe_prefill_target_lanes(&qwen38, 48), 4);

        let mut mla_layers = vec![false; 42];
        for layer in [5, 11, 17, 23, 29, 35, 41] {
            mla_layers[layer] = true;
        }
        let ling = Config {
            bailingmoe3: true,
            n_layer: 42,
            bailing_mla_layers: mla_layers,
            ..Default::default()
        };
        assert_eq!(super::moe_prefill_target_lanes(&ling, 42), 6);

        let attention_only = Config {
            n_layer: 24,
            ..Default::default()
        };
        assert_eq!(super::moe_prefill_target_lanes(&attention_only, 24), 24);
    }

    #[test]
    fn qwen38_qsa_cache_prices_raw_rows_and_final_block_keys() {
        let compress_ratios: Vec<usize> = (0usize..48)
            .map(|l| if (l + 1).is_multiple_of(4) { 4 } else { 0 })
            .collect();
        let cfg = Config {
            qwen4exp: true,
            n_layer: 48,
            full_attn_interval: 4,
            indexer_head_size: 128,
            compress_ratios,
            ..Default::default()
        };
        let ctx = 262_144usize;
        assert_eq!(super::qsa_cache_bytes(&cfg, 0, ctx), 0);
        assert_eq!(
            super::qsa_block_cache_bytes(&cfg, 3, 1),
            128 * 4,
            "a sub-block context still needs a bindable cache placeholder"
        );
        assert_eq!(super::qsa_raw_cache_bytes(&cfg, 3, ctx), ctx * 128 * 2);
        assert_eq!(
            super::qsa_block_cache_bytes(&cfg, 3, ctx),
            (ctx / 4) * 128 * 4
        );
        assert_eq!(super::qsa_cache_bytes(&cfg, 3, ctx), 96 * 1024 * 1024);
        let total: usize = (0..cfg.n_layer)
            .map(|l| super::qsa_cache_bytes(&cfg, l, ctx))
            .sum();
        assert_eq!(total, 1152 * 1024 * 1024);
    }

    #[test]
    fn qwen38_q8_prices_only_the_main_kv_cache_as_q8() {
        let recurrent_layers: Vec<bool> =
            (0usize..48).map(|l| !(l + 1).is_multiple_of(4)).collect();
        let cfg = Config {
            qwen4exp: true,
            n_layer: 48,
            full_attn_interval: 4,
            recurrent_layers,
            compress_ratios: (0usize..48)
                .map(|l| if (l + 1).is_multiple_of(4) { 4 } else { 0 })
                .collect(),
            n_kv: 2,
            head_dim: 256,
            indexer_head_size: 128,
            ssm_d_conv: 4,
            ssm_d_state: 128,
            ssm_d_inner: 2048,
            ssm_n_group: 8,
            ssm_dt_rank: 16,
            ..Default::default()
        };
        let ec = EngineConfig {
            kv: infr_core::config::KvCfg {
                type_k: Some(DType::Q8_0),
                type_k_specified: true,
                type_v: Some(DType::Q8_0),
                type_v_specified: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(super::kv_row_align_ok(&cfg));
        assert!(super::kv_q8_layout_ok(&cfg));
        assert_eq!(
            super::vulkan_kv_fmt_for_budget(&cfg, &ec, ec.kv.type_k),
            DType::Q8_0
        );

        let ctx = 262_144usize;
        let f16 = super::kv_bytes_estimate_fmt(&cfg, ctx, false, 1024, DType::F16, DType::F16);
        let q8 = super::kv_bytes_estimate_fmt(&cfg, ctx, false, 1024, DType::Q8_0, DType::Q8_0);
        let elems_per_side = ctx * cfg.n_kv * cfg.head_dim;
        let saved_per_full_layer = 2
            * (super::kv_side_bytes(DType::F16, elems_per_side)
                - super::kv_side_bytes(DType::Q8_0, elems_per_side));
        assert_eq!(f16 - q8, (12 * saved_per_full_layer) as u64);
        assert_eq!(
            (0..cfg.n_layer)
                .map(|l| super::qsa_cache_bytes(&cfg, l, ctx))
                .sum::<usize>(),
            1152 * 1024 * 1024,
            "QSA raw rows and persistent final block keys are independent of main KV dtype"
        );
    }

    #[test]
    fn qwen38_activation_reserve_tracks_the_real_prefill_batch() {
        let cfg = Config {
            qwen4exp: true,
            n_layer: 48,
            n_head: 24,
            n_embd: 2560,
            n_ff: 640,
            head_dim: 256,
            hc_mult: 4,
            hc_low_rank: 320,
            ssm_d_inner: 6144,
            ssm_n_group: 16,
            ssm_dt_rank: 48,
            ssm_d_state: 128,
            moe: Some(crate::MoeConfig {
                n_expert: 512,
                n_used: 10,
                n_ff_exp: 640,
                scale: 1.0,
                gating: infr_core::graph::MoeGating::Sigmoid,
                norm_w: true,
                weight_before: false,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
            }),
            ..Default::default()
        };
        let half = super::dense_act_reserve_at(&cfg, &conservative_caps(), 4096, 512);
        let full = super::dense_act_reserve_at(&cfg, &conservative_caps(), 4096, 1024);
        assert_eq!(
            full - super::QWEN4_PLAN_OVERLAP_RESERVE - super::DENSE_ACT_RESERVE_SLACK,
            2 * (half - super::QWEN4_PLAN_OVERLAP_RESERVE - super::DENSE_ACT_RESERVE_SLACK),
            "only the row-scaled part doubles; the retained decode plan and the fixed slack are constant"
        );
    }

    #[test]
    fn qwen38_layer_major_reserve_is_bounded_to_one_prefill_group() {
        let cfg = Config {
            qwen4exp: true,
            n_embd: 2560,
            hc_mult: 4,
            ple_ngram_size: 4,
            ple_heads_per_ngram: 2,
            ple_head_dim: 320,
            ..Default::default()
        };
        let ubatch = 1024usize;
        let rows = ubatch * super::QWEN4_PREFILL_GROUP_CHUNKS;
        let row_elems = cfg.n_embd * (1 + cfg.hc_mult)
            + (cfg.ple_ngram_size - 1) * cfg.ple_heads_per_ngram * cfg.ple_head_dim;
        let expected = rows as u64 * row_elems as u64 * 4 + super::QWEN4_PREFILL_GROUP_PAD;
        assert_eq!(
            super::layer_major_act_bytes(&cfg, 262_144, ubatch),
            expected
        );
        assert_eq!(
            super::layer_major_act_bytes(&cfg, rows / 2, ubatch),
            (expected - super::QWEN4_PREFILL_GROUP_PAD) / 2 + super::QWEN4_PREFILL_GROUP_PAD,
            "short prompts reserve only their live rows"
        );
    }

    #[test]
    fn layer_major_prefill_is_explicit_only() {
        let caps = Capabilities {
            graph_input_inplace: true,
            ..Default::default()
        };
        let default_cfg = EngineConfig::default();
        assert!(
            !super::layer_major_prefill(&default_cfg, &caps, true),
            "an unset setting must keep the chunk-major production order"
        );

        let enabled = EngineConfig {
            paging: infr_core::config::PagingCfg {
                layer_major: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(super::layer_major_prefill(&enabled, &caps, true));
        assert!(
            !super::layer_major_prefill(&enabled, &caps, false),
            "an architecture that cannot split its stack stays chunk-major"
        );

        let disabled = EngineConfig {
            paging: infr_core::config::PagingCfg {
                layer_major: Some(false),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!super::layer_major_prefill(&disabled, &caps, true));
    }

    fn deepseek4_cache_config() -> Config {
        Config {
            deepseek4: true,
            n_layer: 3,
            head_dim: 512,
            rope_dim: 64,
            swa_window: 128,
            indexer_head_size: 128,
            compress_ratios: vec![0, 4, 128],
            ..Default::default()
        }
    }

    #[test]
    fn deepseek4_128k_cache_layout_prices_fp8_and_mxfp4_exactly() {
        let cfg = deepseek4_cache_config();
        let ctx = 131_072;
        let raw = 2 * super::DSV4_FP8_PAGE_BYTES;

        let r0 = super::dsv4_layer_layout(&cfg, 0, ctx);
        assert_eq!(r0.raw_rows, 128);
        assert_eq!(r0.raw_bytes, raw);
        assert_eq!(r0.state_bytes, super::KV_MIN_SIDE_BYTES);

        let r4 = super::dsv4_layer_layout(&cfg, 1, ctx);
        assert_eq!(r4.comp_rows, 32_768);
        assert_eq!(r4.comp_bytes, 512 * super::DSV4_FP8_PAGE_BYTES);
        assert_eq!(r4.lid_bytes, 32_768 * super::DSV4_MXFP4_ROW_BYTES);
        assert_eq!(r4.state_bytes, 21_446_656);
        assert_eq!(
            super::layer_state_bytes(&cfg, 1, ctx, false, 1024, DType::F16, DType::F16),
            (raw, 21_446_656)
        );

        let r128 = super::dsv4_layer_layout(&cfg, 2, ctx);
        assert_eq!(r128.comp_rows, 1024);
        assert_eq!(r128.comp_bytes, 16 * super::DSV4_FP8_PAGE_BYTES);
        assert_eq!(r128.lid_bytes, 0);
        assert_eq!(r128.state_bytes, 1_122_304);
        assert!(!super::kv_q8_layout_ok(&cfg));
    }

    #[test]
    fn layer_state_bytes_distinguishes_qwen35_attention_and_deltanet() {
        let cfg = qwen35_hybrid_state();
        let ctx = 200_000;
        let ubatch = 4096;

        // Layer 0 is DeltaNet: its two persistent buffers are fixed f32 state and do not depend on
        // context length or the session KV dtype.
        let delta_q8 =
            super::layer_state_bytes(&cfg, 0, ctx, false, ubatch, DType::Q8_0, DType::Q8_0);
        let delta_f16 = super::layer_state_bytes(&cfg, 0, 1, false, ubatch, DType::F16, DType::F16);
        let conv_elems = (cfg.ssm_d_conv - 1) * cfg.q35_conv_channels();
        let state_elems = cfg.q35_num_v_heads() * cfg.q35_head_k_dim() * cfg.q35_head_v_dim();
        assert_eq!(delta_q8, (conv_elems * 4, state_elems * 4));
        assert_eq!(delta_q8, delta_f16);

        // Layer 3 is full attention and therefore scales with the requested context in the
        // selected KV format.
        let attention =
            super::layer_state_bytes(&cfg, 3, ctx, false, ubatch, DType::Q8_0, DType::Q8_0);
        let row = cfg.n_kv * cfg.head_dim;
        assert_eq!(
            attention,
            (
                super::kv_side_bytes(DType::Q8_0, ctx * row),
                super::kv_side_bytes(DType::Q8_0, ctx * row),
            )
        );
    }

    #[test]
    fn qwen35_persistent_state_estimate_counts_only_full_attention_layers_as_kv() {
        let cfg = qwen35_hybrid_state();
        let ctx = 200_000;
        let ubatch = 4096;
        let fmt = DType::Q8_0;
        let estimate = super::kv_bytes_estimate_fmt(&cfg, ctx, false, ubatch, fmt, fmt);
        let attention = super::layer_state_bytes(&cfg, 3, ctx, false, ubatch, fmt, fmt);
        let delta = super::layer_state_bytes(&cfg, 0, ctx, false, ubatch, fmt, fmt);
        let attention_bytes = attention.0 as u64 + attention.1 as u64;
        let delta_bytes = delta.0 as u64 + delta.1 as u64;
        let checkpoint = super::recurrent_checkpoint_bytes(&cfg);

        assert_eq!(checkpoint, 30 * delta_bytes);
        assert_eq!(
            estimate,
            10 * attention_bytes + 30 * delta_bytes + checkpoint
        );
        assert!(
            estimate < 40 * attention_bytes,
            "DeltaNet layers must not be priced as full-context KV"
        );
    }

    /// The VRAM estimate must price the row the runner ALLOCATES. Before B41 it priced
    /// `2 x (n_kv x head_dim)` = 384 elements/token/layer against a reality of 576 K-only —
    /// under-reserving by exactly 1.5x, with the context clamp and the resident-fit sweep computing
    /// off that.
    #[test]
    fn kv_bytes_estimate_prices_the_mla_row_it_allocates() {
        let cfg = deepseek_v2_lite_kv();
        let ec = EngineConfig::default();
        let (ctx, ubatch) = (4096usize, super::ubatch_rows(&ec));
        let est = super::kv_bytes_estimate_fmt(&cfg, ctx, false, ubatch, DType::F16, DType::F16);
        // 27 layers × exact K allocation plus the bindable V placeholder.
        let per_layer =
            super::kv_side_bytes(DType::F16, ctx * (cfg.kv_lora_rank + cfg.qk_rope_dim))
                + super::kv_side_bytes(DType::F16, 0);
        let want = (cfg.n_layer * per_layer) as u64;
        assert_eq!(est, want);
        // The pre-fix pricing, spelled out: K+V at the head-dim product.
        let old = (cfg.n_layer * ctx * 2 * (cfg.n_kv * cfg.head_dim) * 2) as u64;
        assert_eq!(
            est - (cfg.n_layer * super::kv_side_bytes(DType::F16, 0)) as u64,
            old * 3 / 2,
            "the estimate was 1.5x short before the placeholder"
        );
    }

    /// `kv_q8_layout_ok` is the "may this model use a q8 cache" gate the Vulkan auto-q8 placement
    /// PIN consults, so it must reject MLA — the pin is priced into the VRAM estimate, and
    /// `generate_dense_backend` will build an f16 cache for MLA no matter what was pinned. The
    /// LAYOUT question underneath it is separate and answers yes (576 is 18 whole 32-blocks),
    /// which is what keeps the CPU backend — whose `Op::Mla` dequantizes every KV dtype — able to
    /// honor an explicit q8 request.
    #[test]
    fn q8_gate_rejects_mla_even_though_its_rows_are_block_aligned() {
        let mla = deepseek_v2_lite_kv();
        assert!(super::kv_row_align_ok(&mla), "576 % 32 == 0");
        assert!(!super::kv_q8_layout_ok(&mla), "MLA may not be pinned to q8");
        let dense = qwen3_14b();
        assert!(super::kv_row_align_ok(&dense) && super::kv_q8_layout_ok(&dense));
    }

    /// B42: the GPU MLA kernels read the cache as f16 unconditionally, so a deepseek2 session on a
    /// GPU backend is f16 on both sides — a NAMED non-f16 format is refused rather than silently
    /// downgraded, and the CPU backend (dtype-correct `Op::Mla`) is untouched.
    #[test]
    fn mla_kv_fmt_forces_f16_on_gpu_and_refuses_a_named_format() {
        let mla = deepseek_v2_lite_kv();
        let unset = EngineConfig::default();
        let q8 = |k: Option<DType>, v: Option<DType>, force: bool| EngineConfig {
            kv: infr_core::config::KvCfg {
                type_k: k,
                type_k_specified: k.is_some(),
                type_v: v,
                type_v_specified: v.is_some(),
                force_q8: force,
                ..Default::default()
            },
            ..Default::default()
        };

        // Nothing named: whatever the ladder resolved is forced to f16 (this is the shape an
        // auto-q8 placement pin would arrive in, which `kv_q8_layout_ok` also prevents upstream).
        for be in ["vulkan", "metal"] {
            assert_eq!(
                super::mla_kv_fmt(&mla, be, &unset, DType::Q8_0, DType::Q8_0).expect("forced"),
                (DType::F16, DType::F16)
            );
        }
        // Named non-f16 — per side, and through the legacy both-sides alias — is refused.
        for ec in [
            q8(Some(DType::Q8_0), None, false),
            q8(None, Some(DType::Turbo3), false),
            q8(None, None, true),
        ] {
            let err = super::mla_kv_fmt(&mla, "vulkan", &ec, DType::F16, DType::F16)
                .expect_err("must refuse");
            assert!(
                err.to_string().contains("f16-only"),
                "unhelpful refusal: {err}"
            );
        }
        // f16 asked for explicitly is not a refusal.
        assert_eq!(
            super::mla_kv_fmt(
                &mla,
                "vulkan",
                &q8(Some(DType::F16), Some(DType::F16), false),
                DType::F16,
                DType::F16
            )
            .expect("f16 is what MLA runs"),
            (DType::F16, DType::F16)
        );
        // CPU keeps its dtype freedom (its MLA arm dequantizes), and a non-MLA model is untouched
        // on every backend.
        assert_eq!(
            super::mla_kv_fmt(
                &mla,
                "cpu",
                &q8(Some(DType::Q8_0), Some(DType::Q8_0), false),
                DType::Q8_0,
                DType::Q8_0
            )
            .expect("cpu passthrough"),
            (DType::Q8_0, DType::Q8_0)
        );
        assert_eq!(
            super::mla_kv_fmt(&qwen3_14b(), "vulkan", &unset, DType::Q8_0, DType::Q8_0)
                .expect("non-MLA passthrough"),
            (DType::Q8_0, DType::Q8_0)
        );
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
    /// Live FREE bytes on an idle XTX (~23.94 GiB) — the raw `VramInfo::available` figure, which is
    /// NOT the budget: see [`XTX_ROOM`].
    const XTX_FREE: u64 = 25_701_257_216;
    /// What `VramInfo::alloc_room()` yields for that snapshot: live free minus the allocator
    /// guard's own 256 MiB headroom. Anything a fit or placement decision plans past this the VRAM
    /// guard will refuse, so this — not the raw free figure — is the budget.
    const XTX_ROOM: u64 = XTX_FREE - 256 * 1024 * 1024;

    /// That box's VRAM snapshot at an arbitrary free figure, as the backend would report it.
    fn xtx(available: u64) -> infr_vulkan::VramInfo {
        infr_vulkan::VramInfo {
            total: XTX_TOTAL,
            available,
            live: true,
            uma: false,
        }
    }

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
                + super::runtime_reserve_at(cfg, &conservative_caps(), ctx, ring, ub, k, v)
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
            let fit = super::kv_fit_ctx_for(
                &cfg,
                &conservative_caps(),
                &ec,
                weights,
                &xtx(XTX_FREE),
                k,
                v,
            )
            .expect("has KV");
            assert_exact_boundary(&cfg, &ec, weights, XTX_ROOM, k, v, fit);
        }
        // q8 is ~half the bytes per token, so it must buy materially more context than f16.
        let f16 = super::kv_fit_ctx_for(
            &cfg,
            &conservative_caps(),
            &ec,
            weights,
            &xtx(XTX_FREE),
            DType::F16,
            DType::F16,
        )
        .expect("has KV");
        let q8 = super::kv_fit_ctx_for(
            &cfg,
            &conservative_caps(),
            &ec,
            weights,
            &xtx(XTX_FREE),
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
            let a = super::kv_fit_ctx_for(
                &cfg,
                &conservative_caps(),
                &ring,
                GEMMA3_12B_WEIGHTS,
                &xtx(XTX_FREE),
                k,
                v,
            )
            .expect("has KV");
            let b = super::kv_fit_ctx_for(
                &cfg,
                &conservative_caps(),
                &no_ring,
                GEMMA3_12B_WEIGHTS,
                &xtx(XTX_FREE),
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
    /// Deliberately does not name the winning rung. Which one it is moves with the reserve's own
    /// coefficients and that is not what this is guarding — the invariant is "the default chunk
    /// misses, a lower rung on the SHARED ladder saves it, and f16 therefore reaches the trained
    /// window".
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
                + super::dense_act_reserve_at(&cfg, &conservative_caps(), want, ub)
        };
        let cands = super::ubatch_candidates(&ec);
        assert_eq!(cands[0], 1024, "the default chunk leads the ladder");
        // With the measured reserve this model now fits at the DEFAULT chunk — it no longer needs
        // a shorter rung to reach its trained window, which is a strictly better outcome than the
        // one this test was written for (and matches the device: `infr bench -p 131056` runs at
        // ubatch 1024, 780 t/s, peaking 4735 MiB of activations against a 7146 MiB reserve).
        // What still has to hold is that SOME rung of the shared ladder serves it.
        assert!(
            cands.iter().any(|&ub| need(ub) <= XTX_ROOM),
            "a rung of the shared ladder must serve the trained window: {:?}",
            cands.iter().map(|&ub| (ub, need(ub))).collect::<Vec<_>>()
        );

        let fit = super::kv_fit_ctx_for(
            &cfg,
            &conservative_caps(),
            &ec,
            GEMMA3_12B_WEIGHTS,
            &xtx(XTX_FREE),
            DType::F16,
            DType::F16,
        )
        .expect("has KV");
        assert!(
            fit >= want,
            "f16 must reach the trained window ({want}); got {fit} — the auto-q8 rung would fire"
        );
    }

    /// The reserve is the model documented on `dense_act_reserve_at`, term by term, at
    /// gemma-4-31B's shape and a 128-row chunk — so changing a coefficient, or dropping the pad,
    /// fails HERE rather than as a `VRAM budget exceeded` on someone's deep prefill.
    ///
    /// The two numbers this pins that a reader is most likely to "simplify": the score tile is ONE
    /// live pool (`2 * n_head * ctx_pad`), not two, and there is no fixed byte term — the
    /// non-activation slop a 256 MiB constant used to stand in for is measured by
    /// `reclamp_ctx_to_live_room` instead of estimated twice.
    #[test]
    fn act_reserve_is_the_measured_model() {
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
        let attn_s: u64 = 2 * 32 * 16384; // ONE non-flash score tile: hd 512 misses the flash tier
        let per_row = 12 * 21504 + 96 * 5376 + attn_pv + attn_s;
        let unpadded = rows * per_row;
        let got = super::dense_act_reserve_at(&cfg, &conservative_caps(), ctx, ubatch);
        assert_eq!(
            got,
            unpadded * super::ACT_RESERVE_PAD.0 / super::ACT_RESERVE_PAD.1
                + super::DENSE_ACT_RESERVE_SLACK,
            "reserve must be the per-row model times the pad plus the fixed slack"
        );
        assert_eq!(got, 500_957_184 + super::DENSE_ACT_RESERVE_SLACK);
        // The row-scaled part still halves with the rows (the fixed slack cancels out), which a
        // constant reserve would floor.
        let half = super::dense_act_reserve_at(&cfg, &conservative_caps(), ctx, 64);
        assert_eq!(
            (half - super::DENSE_ACT_RESERVE_SLACK) * 2,
            got - super::DENSE_ACT_RESERVE_SLACK,
            "the reserve is per-row plus one fixed slack term"
        );
    }

    /// Ling's KDA/MLA hybrid never emits Op::Attention. Its dedicated MLA kernel scans the
    /// compressed cache and accumulates softmax/value internally, so charging the ordinary
    /// nonfa_s/nonfa_pv pools strands most of the expert arena at long context.
    #[test]
    fn ling_mla_reserve_omits_ordinary_attention_pools() {
        let mut mla_layers = vec![false; 42];
        for il in [5, 11, 17, 23, 29, 35, 41] {
            mla_layers[il] = true;
        }
        let cfg = Config {
            n_layer: 42,
            n_head: 32,
            head_dim: 192,
            n_embd: 2560,
            n_ff: 6144,
            bailingmoe3: true,
            bailing_mla_layers: mla_layers,
            kda_head_dim: 128,
            ssm_d_conv: 4,
            moe: Some(crate::MoeConfig {
                n_expert: 512,
                n_used: 8,
                n_ff_exp: 768,
                scale: 1.0,
                gating: infr_core::graph::MoeGating::Sigmoid,
                norm_w: true,
                weight_before: false,
                n_expert_groups: 16,
                n_expert_groups_used: 4,
            }),
            ..Default::default()
        };
        let (ctx, ubatch) = (65_681usize, 1024usize);
        let got = super::dense_act_reserve_at(&cfg, &conservative_caps(), ctx, ubatch);
        let m = cfg.moe.expect("Ling is MoE");
        let per_pair = 3 * m.n_ff_exp * 4 + cfg.n_embd * 4 + m.n_ff_exp + cfg.n_embd;
        let moe = m.n_used * per_pair + 48 * cfg.n_embd;
        let per_row = 12 * cfg.n_ff + 96 * cfg.n_embd + moe;
        let expected = ubatch as u64 * per_row as u64 * super::ACT_RESERVE_PAD.0
            / super::ACT_RESERVE_PAD.1
            + super::DENSE_ACT_RESERVE_SLACK;
        assert_eq!(got, expected);
        assert_eq!(got, 959_447_040 + super::DENSE_ACT_RESERVE_SLACK);

        let mut ordinary = cfg.clone();
        ordinary.bailingmoe3 = false;
        ordinary.bailing_mla_layers.clear();
        let ordinary_got =
            super::dense_act_reserve_at(&ordinary, &conservative_caps(), ctx, ubatch);
        let attn_pv = 32 * cfg.n_head * cfg.head_dim;
        let attn_s = 2 * cfg.n_head * ctx.next_multiple_of(256);
        let ordinary_pools = ubatch as u64 * (attn_pv + attn_s) as u64 * super::ACT_RESERVE_PAD.0
            / super::ACT_RESERVE_PAD.1;
        assert_eq!(ordinary_got - got, ordinary_pools);
        assert_eq!(ordinary_pools, 6_769_606_656);
    }

    /// M4: M2's hd256 FlashAttention no longer materializes `nonfa_s`, so the placement/context
    /// budget must not keep charging for that multi-GiB score tile. The discount is capability-
    /// exact: one byte less shared memory, no 16x16 f16 coopmat, hd512, or an SWA layer all keep
    /// the corresponding non-flash reserve.
    #[test]
    fn hd256_flash_reserve_drops_only_the_score_tile_it_avoids() {
        let cfg = Config {
            n_layer: 40,
            n_head: 16,
            n_kv: 2,
            head_dim: 256,
            head_dim_swa: 256,
            n_embd: 4096,
            n_ff: 11008,
            ..Default::default()
        };
        let (ctx, ubatch) = (200_000usize, 1024usize);
        let conservative = super::dense_act_reserve_at(&cfg, &conservative_caps(), ctx, ubatch);
        let flash = super::dense_act_reserve_at(&cfg, &hd256_flash_caps(), ctx, ubatch);
        let automatic_conservative = super::runtime_reserve_at(
            &cfg,
            &conservative_caps(),
            ctx,
            false,
            ubatch,
            DType::Q8_0,
            DType::Q8_0,
        );
        let automatic_flash = super::runtime_reserve_at(
            &cfg,
            &hd256_flash_caps(),
            ctx,
            false,
            ubatch,
            DType::Q8_0,
            DType::Q8_0,
        );
        assert_eq!(
            super::estimate_runtime_reserve_bytes_for_device(&cfg, ctx, ubatch, false),
            automatic_conservative,
        );
        assert_eq!(
            super::estimate_runtime_reserve_bytes_for_device(&cfg, ctx, ubatch, true),
            automatic_flash,
        );
        let rows = ubatch as u64;
        let score_per_row = (2 * cfg.n_head * ctx.next_multiple_of(256)) as u64;
        let expected_score =
            rows * score_per_row * super::ACT_RESERVE_PAD.0 / super::ACT_RESERVE_PAD.1;
        assert_eq!(
            conservative - flash,
            expected_score,
            "the only removed bytes are the padded nonfa_s score pool"
        );

        let mut moe_cfg = cfg.clone();
        moe_cfg.moe = Some(crate::MoeConfig {
            n_expert: 256,
            n_used: 8,
            n_ff_exp: 512,
            scale: 1.0,
            gating: infr_core::graph::MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
        });
        let moe_reserve = super::dense_act_reserve_at(&moe_cfg, &hd256_flash_caps(), ctx, ubatch);
        let m = moe_cfg.moe.expect("just installed");
        let per_pair = 3 * m.n_ff_exp * 4 + moe_cfg.n_embd * 4 + m.n_ff_exp + moe_cfg.n_embd;
        let moe_per_row = m.n_used * per_pair + 48 * moe_cfg.n_embd;
        let expected_moe =
            rows * moe_per_row as u64 * super::ACT_RESERVE_PAD.0 / super::ACT_RESERVE_PAD.1;
        assert_eq!(
            moe_reserve - flash,
            expected_moe,
            "MoE keeps its pair scratch plus the measured executor envelope"
        );

        let mut short_shared = hd256_flash_caps();
        short_shared.max_shared_memory_bytes = infr_vulkan::FLASH_HD256_BM16_SHARED - 1;
        assert_eq!(
            super::dense_act_reserve_at(&cfg, &short_shared, ctx, ubatch),
            conservative,
            "a device that cannot fit the shader must retain non-FA scratch"
        );
        let mut no_coopmat = hd256_flash_caps();
        no_coopmat.coopmat_f16 = None;
        assert_eq!(
            super::dense_act_reserve_at(&cfg, &no_coopmat, ctx, ubatch),
            conservative,
            "scalar/f16 fallback devices must retain non-FA scratch"
        );

        let mut hd512 = cfg.clone();
        hd512.head_dim = 512;
        hd512.head_dim_swa = 512;
        assert_eq!(
            super::dense_act_reserve_at(&hd512, &hd256_flash_caps(), ctx, ubatch),
            super::dense_act_reserve_at(&hd512, &conservative_caps(), ctx, ubatch),
            "M4 must not discount an unsupported head dimension"
        );

        let mut mixed_swa = cfg.clone();
        mixed_swa.swa_window = 1024;
        mixed_swa.swa_pattern = 6;
        let mixed = super::dense_act_reserve_at(&mixed_swa, &hd256_flash_caps(), ctx, ubatch);
        let no_score = flash;
        let swa_span = (mixed_swa.swa_window + ubatch).next_multiple_of(256);
        let swa_score = rows * (2 * mixed_swa.n_head * swa_span) as u64 * super::ACT_RESERVE_PAD.0
            / super::ACT_RESERVE_PAD.1;
        assert_eq!(
            mixed - no_score,
            swa_score,
            "SWA remains non-flash but reserves only its bounded window"
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

    /// **Drift guard (backlog B11).** Every VRAM budget in this file — the residency predicate, the
    /// streaming budget, the MoE expert budget — must be taken against the ALLOCATOR's ceiling
    /// (`VramInfo::alloc_room` = free minus the guard's 256 MiB headroom), never the raw free
    /// figure. The placement sweeps used to compare against `vram.available`, so they could declare
    /// a model resident, or hand a pager an arena, 256 MiB past anything `check_vram_budget` will
    /// ever allocate — which surfaces as a failed activation alloc mid-prefill.
    ///
    /// Each assertion below is placed ONE BYTE either side of the ceiling, so restoring any
    /// `vram.available` comparison flips it: the raw figure accepts every "must not" case here.
    #[test]
    fn budgets_agree_with_the_allocator_ceiling() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let vram = xtx(XTX_FREE);
        const GUARD: u64 = 256 * 1024 * 1024;
        assert_eq!(vram.alloc_room(), XTX_ROOM);
        assert_eq!(
            vram.available - vram.alloc_room(),
            GUARD,
            "the ceiling is the free figure minus the guard headroom"
        );

        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        let (ctx, ub) = (32768usize, 256usize);
        let f16 = (DType::F16, DType::F16);
        // Weights that make a resident session land EXACTLY on the ceiling (KV + reserve priced
        // from the shared primitives, weights = whatever is left of the ceiling).
        let kv_and_act =
            super::dense_resident_need(&cfg, &conservative_caps(), 0, ctx, true, ub, f16.0, f16.1);
        let exact = XTX_ROOM - kv_and_act;
        let fits = |w: u64| {
            super::dense_placement_fits(
                &cfg,
                &conservative_caps(),
                &ec,
                w,
                &vram,
                ctx,
                ub,
                f16.0,
                f16.1,
            )
        };
        assert!(
            fits(exact),
            "a session that exactly fills the ceiling is resident"
        );
        assert!(
            !fits(exact + 1),
            "one byte PAST the guard's ceiling must not be placed resident"
        );
        assert!(
            exact + 1 + kv_and_act <= vram.available,
            "…and that byte is one the raw free figure would have accepted, which is the bug"
        );

        // Streaming budget: exhausted at the ceiling, and it never offers the guard's headroom.
        // Chunk-major is the default and needs no cross-layer residual reservation.
        let lm = super::layer_major_act_bytes(&cfg, ctx, ub);
        assert!(
            lm > 0,
            "the layer-major term must be a real subtraction here"
        );
        let budget = |w: u64| {
            super::dense_stream_budget_at(
                &cfg,
                &conservative_caps(),
                &ec,
                w,
                &vram,
                ctx,
                ub,
                f16.0,
                f16.1,
            )
        };
        assert_eq!(
            budget(exact),
            0,
            "nothing left to stream into at the ceiling"
        );
        assert_eq!(
            budget(exact - (1 << 30)),
            1 << 30,
            "the default chunk-major order keeps the whole room below the ceiling"
        );
        // Explicit layer-major is the only mode that holds the cross-layer residual stream back.
        let layer_major = EngineConfig {
            paging: infr_core::config::PagingCfg {
                layer_major: Some(true),
                ..ec.paging.clone()
            },
            ..ec.clone()
        };
        assert_eq!(
            super::dense_stream_budget_at(
                &cfg,
                &conservative_caps(),
                &layer_major,
                exact - (1 << 30),
                &vram,
                ctx,
                ub,
                f16.0,
                f16.1
            ),
            (1 << 30) - lm,
            "paging.layer_major = true reserves the residual stream explicitly"
        );

        // MoE expert placement: same ceiling, minus the model/shape-derived phase workspace.
        let workspace = super::dense_act_reserve_at(&cfg, &conservative_caps(), ctx, ub);
        let empty = super::ModelMemoryPlan::new(XTX_ROOM, 0, 0, workspace).expect("empty plan");
        assert_eq!(empty.expert_cache_bytes, XTX_ROOM - workspace);
        assert_eq!(empty.minimum_required_bytes(), workspace);
        let full = super::ModelMemoryPlan::new(XTX_ROOM, XTX_ROOM, 0, 0).expect("full plan");
        assert_eq!(full.expert_cache_bytes, 0);
        assert_eq!(
            super::ModelMemoryPlan::new(XTX_ROOM, XTX_ROOM + 1, 0, 0),
            None,
            "a dense half past the ceiling is a hard error, not a 256 MiB overdraft"
        );
        assert!(
            XTX_ROOM < vram.available,
            "…again a case the raw free figure would have waved through"
        );
    }

    #[test]
    fn moe_weight_packing_margin_is_block_rounded_and_budgeted() {
        const MIB: u64 = 1024 * 1024;
        const GIB: u64 = 1024 * MIB;
        assert_eq!(super::resident_weight_packing_margin(1), 256 * MIB);
        assert_eq!(super::resident_weight_packing_margin(16 * GIB), 512 * MIB);

        let plan = super::ModelMemoryPlan::new_with_reserves(
            20 * GIB,
            4 * GIB,
            2 * GIB,
            GIB,
            512 * MIB,
            0,
            256 * MIB,
        )
        .expect("plan");
        assert_eq!(plan.expert_cache_bytes, 12 * GIB + 256 * MIB);
        assert_eq!(plan.minimum_required_bytes(), 7 * GIB + 768 * MIB);
        assert_eq!(
            plan.elastic_pool_bytes(plan.expert_cache_bytes),
            13 * GIB + 256 * MIB
        );
        assert_eq!(plan.elastic_pool_bytes(4 * GIB), 5 * GIB);
        assert_eq!(
            plan.elastic_pool_bytes(plan.expert_cache_bytes)
                .saturating_add(plan.fixed_weight_bytes)
                .saturating_add(plan.persistent_state_bytes)
                .saturating_add(plan.weight_packing_margin_bytes)
                .saturating_add(plan.load_driver_reserve_bytes)
                .saturating_add(plan.post_load_reserve_bytes),
            plan.total_room_bytes,
            "the shared runtime reserve must be returned to the physical elastic arena exactly once"
        );

        let dsv4_plan = super::ModelMemoryPlan::new_with_reserves(
            20 * GIB,
            4 * GIB,
            2 * GIB,
            GIB,
            512 * MIB,
            1536 * MIB,
            256 * MIB,
        )
        .expect("DeepSeek V4 plan");
        assert_eq!(dsv4_plan.expert_cache_bytes, 10 * GIB + 768 * MIB);
        assert_eq!(dsv4_plan.minimum_required_bytes(), 9 * GIB + 256 * MIB);

        let mut cfg = Config::default();
        assert_eq!(super::load_driver_reserve(&cfg), 0);
        let estimated = super::estimate_model_memory_plan(&cfg, 4 * GIB, 20 * GIB, 2 * GIB, GIB)
            .expect("control-plane estimate");
        assert_eq!(estimated.weight_packing_margin_bytes, 256 * MIB);
        assert_eq!(estimated.post_load_reserve_bytes, 256 * MIB);
        cfg.deepseek4 = true;
        assert_eq!(super::load_driver_reserve(&cfg), 1536 * MIB);
        let estimated = super::estimate_model_memory_plan(&cfg, 4 * GIB, 20 * GIB, 2 * GIB, GIB)
            .expect("DeepSeek control-plane estimate");
        assert_eq!(estimated.load_driver_reserve_bytes, 1536 * MIB);

        cfg.deepseek4 = false;
        cfg.qwen35 = true;
        assert_eq!(
            super::load_driver_reserve(&cfg),
            if cfg!(windows) { 2 * GIB } else { 0 }
        );

        let automatic = EngineConfig::default();
        assert_eq!(
            super::session_load_driver_reserve(&cfg, &automatic),
            if cfg!(windows) {
                2 * GIB + 512 * MIB
            } else {
                0
            }
        );
        let mut explicit = EngineConfig::default();
        explicit.device.vram_reserve = Some(infr_core::SizeSpec::Bytes(512 * MIB));
        assert_eq!(
            super::session_load_driver_reserve(&cfg, &explicit),
            if cfg!(windows) { 2 * GIB } else { 0 }
        );

        cfg.qwen35 = false;
        cfg.bailingmoe3 = true;
        assert_eq!(
            super::load_driver_reserve(&cfg),
            if cfg!(windows) { 2 * GIB } else { 0 }
        );
        assert_eq!(
            super::session_load_driver_reserve(&cfg, &automatic),
            if cfg!(windows) {
                2 * GIB + 512 * MIB
            } else {
                0
            }
        );
        assert_eq!(
            super::session_load_driver_reserve(&cfg, &explicit),
            if cfg!(windows) { 2 * GIB } else { 0 }
        );
    }

    #[test]
    fn dynamic_kv_reserve_borrows_the_expert_arena_without_double_counting() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let plan = super::ModelMemoryPlan::new_with_dynamic_reserve(
            24 * GIB,
            5 * GIB,
            GIB,
            2 * GIB,
            4 * GIB,
            GIB,
            0,
            GIB,
        )
        .expect("dynamic KV plan");

        assert_eq!(plan.expert_cache_bytes, 10 * GIB);
        assert_eq!(plan.minimum_required_bytes(), 14 * GIB);
        assert_eq!(plan.elastic_reserve_bytes(), 6 * GIB);
        assert_eq!(plan.elastic_pool_bytes(plan.expert_cache_bytes), 16 * GIB);
        assert_eq!(
            plan.fixed_weight_bytes
                + plan.persistent_state_bytes
                + plan.weight_packing_margin_bytes
                + plan.post_load_reserve_bytes
                + plan.elastic_pool_bytes(plan.expert_cache_bytes),
            plan.total_room_bytes,
            "runtime and maximum KV return to the shared arena exactly once"
        );
    }

    #[test]
    fn moe_pool_split_never_exceeds_budget_and_keeps_a_prefill_floor() {
        const MIB: usize = 1024 * 1024;
        let pools = [
            (MIB, 240usize, [80, 80, 80]),
            (2 * MIB, 120usize, [40, 40, 40]),
        ];
        let batch_floor_bytes = 24 * MIB + 24 * 2 * MIB;
        let physical_floor_bytes = 25 * MIB + 25 * 2 * MIB;
        assert_eq!(
            super::moe_pool_floor_bytes(&pools, 8),
            Some(batch_floor_bytes as u64)
        );
        assert!(
            super::moe_pool_slot_counts(&pools, (physical_floor_bytes - 1) as u64, 8, 2.0)
                .is_none()
        );
        assert_eq!(
            super::moe_pool_slot_counts(&pools, physical_floor_bytes as u64, 8, 2.0),
            Some(vec![25, 25]),
        );

        let budget = 512 * MIB as u64;
        let slots = super::moe_pool_slot_counts(&pools, budget, 8, 2.0).unwrap();
        assert!(slots.iter().all(|&n| n >= 25));
        let allocated = pools
            .iter()
            .zip(&slots)
            .map(|(&(slot_bytes, _, _), &n)| slot_bytes as u64 * n as u64)
            .sum::<u64>();
        assert!(allocated <= budget);
        assert!(pools
            .iter()
            .zip(&slots)
            .all(|(&(_, blocks, _), &n)| n <= blocks + 1));
        assert_eq!(super::moe_pool_capacity_bytes(&pools, &slots), allocated);
    }

    #[test]
    fn automatic_moe_arena_retry_is_bounded_and_honors_the_floor() {
        const MIB: u64 = 1024 * 1024;
        const GIB64: u64 = 1024 * MIB;
        let current = 16 * GIB64;
        assert_eq!(
            super::next_auto_moe_arena_budget(current, 2 * GIB64, 0),
            Some(current - current / 20),
            "an unknown allocation failure retires five percent"
        );
        assert_eq!(
            super::next_auto_moe_arena_budget(current, 2 * GIB64, 1537 * MIB),
            Some(current - 1600 * MIB),
            "a measured deficit rounds up and skips directly past it"
        );
        assert_eq!(
            super::next_auto_moe_arena_budget(2 * GIB64, 2 * GIB64, 0),
            None,
            "the runtime plus one-layer floor is never crossed"
        );
    }

    #[test]
    fn moe_context_fit_reserves_fixed_bytes_not_the_pageable_payload() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = qwen3_14b();
        let ec = EngineConfig::default();
        let vram = xtx(XTX_FREE);
        let (k, v) = (DType::F16, DType::F16);

        let resident =
            super::kv_fit_ctx_for(&cfg, &conservative_caps(), &ec, 30 * GIB, &vram, k, v)
                .expect("has KV");
        assert!(
            resident < super::MIN_SESSION_CTX,
            "an over-VRAM all-resident footprint must not pretend to fit"
        );

        let fixed = 5 * GIB;
        let expert_floor = 2 * GIB;
        let fit = super::kv_fit_ctx_for_moe(
            &cfg,
            &conservative_caps(),
            &ec,
            fixed,
            expert_floor,
            &vram,
            k,
            v,
        )
        .expect("has KV");
        assert!(
            fit >= super::MIN_SESSION_CTX,
            "paged MoE must remain usable"
        );

        let ring = super::kv_ring_wanted(&cfg, &ec);
        let need = |ctx: usize, ub: usize| {
            let kv = super::kv_bytes_estimate_fmt(&cfg, ctx, ring, ub, k, v);
            let runtime = super::dense_act_reserve_at(&cfg, &conservative_caps(), ctx, ub);
            fixed + kv + runtime.max(expert_floor)
        };
        let cands = super::ubatch_candidates(&ec);
        assert!(cands.iter().any(|&ub| need(fit, ub) <= XTX_ROOM));
        assert!(cands.iter().all(|&ub| need(fit + 1, ub) > XTX_ROOM));
    }

    /// **Drift guard (backlog B11), the rung half.** The shared `ubatch_candidates` ladder exists so
    /// the context-fit math and the placement sweep settle on the SAME prefill chunk. They only do
    /// while both budget against the same ceiling: with the sweep on `vram.available` and the fit
    /// math on `alloc_room()`, gemma-3-12b @131072 was validated at one rung and placed at a taller
    /// one (the reported case). Re-derives the expected rung from the primitives — weights + KV +
    /// reserve against `XTX_ROOM` — rather than from the function under test.
    #[test]
    fn fit_math_and_placement_pick_the_same_rung() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        let vram = xtx(XTX_FREE);
        let (k, v) = (DType::F16, DType::F16);
        let cands = super::ubatch_candidates(&ec);
        let need = |ctx: usize, ub: usize| {
            GEMMA3_12B_WEIGHTS
                + super::kv_bytes_estimate_fmt(&cfg, ctx, true, ub, k, v)
                + super::dense_act_reserve_at(&cfg, &conservative_caps(), ctx, ub)
        };
        let expect_rung = |ctx: usize| cands.iter().copied().find(|&ub| need(ctx, ub) <= XTX_ROOM);

        // A shape where the ladder is genuinely WALKED — otherwise "they agree" would be satisfied
        // by both picking the default rung and the guard would prove nothing. Derived, not
        // hardcoded: the heaviest weights that still fit at the 512-row rung, which by construction
        // cannot fit at 1024. (The reported gemma-3-12b @131072 case no longer needs a shorter rung
        // at all — the measured reserve is small enough that the default chunk holds it — so the
        // agreement is checked here and the trained-window outcome in
        // `kv_fit_walks_the_placement_chunk_ladder_gemma3_12b`.)
        let want = cfg.n_ctx_train;
        let heavy = XTX_ROOM
            - super::kv_bytes_estimate_fmt(&cfg, want, true, 512, k, v)
            - super::dense_act_reserve_at(&cfg, &conservative_caps(), want, 512);
        let need_heavy = |ub: usize| {
            heavy
                + super::kv_bytes_estimate_fmt(&cfg, want, true, ub, k, v)
                + super::dense_act_reserve_at(&cfg, &conservative_caps(), want, ub)
        };
        assert!(need_heavy(1024) > XTX_ROOM, "the default chunk must miss");
        assert!(
            need_heavy(512) <= XTX_ROOM,
            "…and the 512-row rung must save it"
        );
        assert_eq!(
            super::dense_resident_rung(&cfg, &conservative_caps(), &ec, heavy, &vram, want, k, v,),
            Some(512),
            "placement must settle on the rung the fit math priced"
        );

        // And at the exact boundary the fit math hands out: placement takes it resident, and one
        // token past it neither of them accepts.
        let fit = super::kv_fit_ctx_for(
            &cfg,
            &conservative_caps(),
            &ec,
            GEMMA3_12B_WEIGHTS,
            &vram,
            k,
            v,
        )
        .expect("has KV");
        assert_eq!(
            super::dense_resident_rung(
                &cfg,
                &conservative_caps(),
                &ec,
                GEMMA3_12B_WEIGHTS,
                &vram,
                fit,
                k,
                v,
            ),
            expect_rung(fit),
        );
        assert!(
            super::dense_resident_rung(
                &cfg,
                &conservative_caps(),
                &ec,
                GEMMA3_12B_WEIGHTS,
                &vram,
                fit,
                k,
                v,
            )
            .is_some(),
            "the advertised context must be one placement can hold resident"
        );
        assert_eq!(
            super::dense_resident_rung(
                &cfg,
                &conservative_caps(),
                &ec,
                GEMMA3_12B_WEIGHTS,
                &vram,
                fit + 1,
                k,
                v,
            ),
            None,
            "one token past the advertised context must miss at EVERY rung — a placement that \
             still says yes here is budgeting against a wider ceiling than the fit math"
        );
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
            &conservative_caps(),
            &ec,
            XTX_ROOM - 64 * 1024 * 1024,
            &xtx(XTX_FREE),
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
            &conservative_caps(),
            &EngineConfig::default(),
            1 << 30,
            &xtx(XTX_FREE),
            DType::F16,
            DType::F16,
        )
        .is_none());
    }

    /// A `Backend` that reports nothing but a live allocation budget — enough to drive
    /// [`super::reclamp_ctx_to_live_room`], which touches no other method. `room: None` stands in
    /// for the CPU/Metal backends, which have no budget to report.
    struct RoomOnly(Option<u64>, Option<u64>);

    impl infr_core::backend::Backend for RoomOnly {
        fn name(&self) -> &str {
            "room-only"
        }
        fn capabilities(&self) -> infr_core::backend::Capabilities {
            infr_core::backend::Capabilities::default()
        }
        fn alloc(
            &self,
            _bytes: usize,
            _usage: infr_core::backend::BufferUsage,
        ) -> infr_core::Result<Box<dyn infr_core::backend::Buffer>> {
            unreachable!("the re-clamp allocates nothing")
        }
        fn upload(
            &self,
            _dst: &dyn infr_core::backend::Buffer,
            _src: &[u8],
        ) -> infr_core::Result<()> {
            unreachable!("the re-clamp uploads nothing")
        }
        fn download(
            &self,
            _src: &dyn infr_core::backend::Buffer,
            _dst: &mut [u8],
        ) -> infr_core::Result<()> {
            unreachable!("the re-clamp downloads nothing")
        }
        fn compile(
            &self,
            _graph: &infr_core::graph::Graph,
        ) -> infr_core::Result<Box<dyn infr_core::backend::Plan>> {
            unreachable!("the re-clamp compiles nothing")
        }
        fn execute(
            &self,
            _plan: &dyn infr_core::backend::Plan,
            _bindings: &infr_core::backend::Bindings,
        ) -> infr_core::Result<()> {
            unreachable!("the re-clamp executes nothing")
        }
        fn sync(&self) -> infr_core::Result<()> {
            Ok(())
        }
        fn device_alloc_room(&self) -> Option<u64> {
            self.0
        }
        fn device_elastic_activation_room(&self) -> Option<u64> {
            self.1
        }
    }

    /// The post-load re-clamp only ever SHRINKS, and only a context the session chose. Each arm is
    /// a decision someone could quietly invert: a backend that cannot report a budget must leave
    /// the window alone (CPU/Metal), a roomy device must not touch it, a tight one must cut it to
    /// something that fits its OWN measured budget, and a user-pinned `--ctx` is documented as
    /// verbatim — the alloc-time guard is its backstop, not this.
    #[test]
    fn reclamp_only_shrinks_and_never_a_pinned_ctx() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        let (k, v) = (DType::F16, DType::F16);
        let want = 32768;
        let call = |be: &dyn infr_core::backend::Backend, ec: &EngineConfig| {
            super::reclamp_ctx_to_live_room(be, &cfg, ec, want, k, v)
        };

        // No budget to report: the caller's window survives untouched.
        assert_eq!(call(&RoomOnly(None, None), &ec), want);

        // Roomy: the fit is past `want`, so the window is kept rather than raised.
        let roomy = super::kv_bytes_estimate_fmt(&cfg, want, true, 1024, k, v)
            + super::dense_act_reserve_at(&cfg, &conservative_caps(), want, 1024)
            + super::POST_KV_DEVICE_RESERVE
            + (1 << 30);
        assert_eq!(call(&RoomOnly(Some(roomy), None), &ec), want);

        // Tight: cut, and cut to a window that really fits the budget it was given.
        let tight = roomy / 3;
        let got = call(&RoomOnly(Some(tight), None), &ec);
        assert!(got < want, "a tight device must shrink the window: {got}");
        let ub = super::ubatch_rows(&ec);
        let need = super::kv_bytes_estimate_fmt(&cfg, got, true, ub, k, v)
            + super::dense_act_reserve_at(&cfg, &conservative_caps(), got, ub);
        assert!(
            need <= tight - super::POST_KV_DEVICE_RESERVE,
            "the clamped window must fit the measured budget: {need} > {tight}"
        );

        // Pinned by the user: honored verbatim, however tight the device is.
        let pinned = EngineConfig {
            device: infr_core::config::DeviceCfg {
                ctx: Some(infr_core::SizeSpec::Bytes(want as u64)),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(call(&RoomOnly(Some(tight), None), &pinned), want);
    }

    #[test]
    fn reclamp_accounts_elastic_activation_separately_from_kv_room() {
        let _scope = PlacementScope::enter(std::sync::Arc::new(PlacementPins::default()));
        let cfg = gemma3_12b();
        let ec = EngineConfig::default();
        let caps = infr_core::backend::Capabilities::default();
        let (k, v) = (DType::F16, DType::F16);
        let want = 32768;
        let ubatch = super::ubatch_rows(&ec);
        let kv = super::kv_bytes_estimate_fmt(&cfg, want, true, ubatch, k, v);
        let activation = super::dense_act_reserve_at(&cfg, &caps, want, ubatch);
        let physical_room = kv + super::POST_KV_DEVICE_RESERVE;

        let without_elastic = super::reclamp_ctx_to_live_room(
            &RoomOnly(Some(physical_room), None),
            &cfg,
            &ec,
            want,
            k,
            v,
        );
        assert!(
            without_elastic < want,
            "ordinary room cannot be spent twice on KV and activation"
        );

        let with_elastic = super::reclamp_ctx_to_live_room(
            &RoomOnly(Some(physical_room), Some(activation)),
            &cfg,
            &ec,
            want,
            k,
            v,
        );
        assert_eq!(
            with_elastic, want,
            "an already-committed elastic arena must cover activation without hiding KV room"
        );
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
            assert_eq!(a.ubatch.load(std::sync::atomic::Ordering::Relaxed), 512);
        }
        {
            let _sb = PlacementScope::enter(b.clone());
            assert!(!super::kv_auto_q8(), "B must NOT inherit A's q8 pin");
            assert_eq!(
                b.ubatch.load(std::sync::atomic::Ordering::Relaxed),
                0,
                "B must NOT inherit A's chunk"
            );
        }
    }
}
