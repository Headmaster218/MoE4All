//! The typed, immutable, explicitly-passed engine configuration — the replacement for the ~174
//! `INFR_*` environment gates scattered across the backends (see `docs/config-plan.md`).
//!
//! Four layers, later wins: `Default` (today's shipped behaviour, reproduced EXACTLY) < config
//! file (TOML) < environment (`INFR_*`, names frozen) < CLI flags. Each layer parses into a
//! [`PartialConfig`] — every leaf `Option`-wrapped — and the layers are folded, so a layer only
//! overrides a field it actually specifies.
//!
//! ```no_run
//! # use infr_core::config::{Config, ConfigOverrides};
//! let cfg = std::sync::Arc::new(Config::load(&ConfigOverrides::default())?);
//! # Ok::<(), infr_core::config::ConfigError>(())
//! ```
//!
//! **Scaffold only (slice S0).** Nothing in the engine reads this yet; every `INFR_*` read site is
//! still where it was, so the tree behaves exactly as before. The per-subsystem slices (S2–S7)
//! move the read sites onto `Arc<Config>` one crate at a time, flipping
//! [`manifest::KnobKey::migrated`] as they go.
//!
//! Module map (`docs/config-plan.md` §4):
//! - [`partial`] — the `cfg_struct!` macro that generates both halves, plus the value grammar.
//! - [`env`] — the ONLY place the process environment is read (through an INJECTED reader).
//! - [`file`] — TOML → `PartialConfig`, plus the 3-step path lookup.
//! - [`cli`] — [`ConfigOverrides`] (clap flags + `--set`) → `PartialConfig`.
//! - [`manifest`] — the checked-in key table: env name → config path → grammar → `migrated`.
//!
//! Invariants (`docs/config-plan.md` §3): no globals (R4) — `Config::load` returns a VALUE; no
//! policy in the layer parsers (R5) — clamping and defaulting live in `Default` and the typed
//! accessors; field names are POSITIVE and the env layer inverts the `INFR_NO_*` spellings.

use std::path::{Path, PathBuf};

use crate::{DType, SizeSpec};

pub mod cli;
pub mod env;
pub mod file;
pub mod manifest;
pub mod partial;

#[cfg(test)]
mod tests;

pub use cli::ConfigOverrides;
pub use partial::{ConfigValue, SetPathError};

use partial::cfg_struct;

// ── sections ─────────────────────────────────────────────────────────────────

cfg_struct! {
    /// Device selection and batch shape (`INFR_DEV`, `INFR_CTX`, `INFR_UBATCH`, threads).
    DeviceCfg / PartialDeviceCfg {
        /// `INFR_DEV`: the raw device spec. TWO parsers consume it and they are NOT shared —
        /// `infr-cli`'s `parse_dev_spec` (→ a `Backend`) and `infr-vulkan`'s
        /// `resolve_infr_dev_index` (→ a physical-device index). The config carries the STRING.
        dev: Option<String> = None,
        /// `INFR_CTX`: context length (the shared size grammar).
        ctx: Option<SizeSpec> = None,
        /// `INFR_UBATCH`: prefill micro-batch rows. `None` = no usable value; the 1024 /
        /// iGPU-adaptive fallback chain stays at its call site (R5).
        ubatch: Option<usize> = None,
        /// Was `INFR_UBATCH` / `--ubatch` supplied AT ALL, usable or not? The PRESENCE half of
        /// §6.12's two-consumer knob, and — like [`KvCfg::type_k_specified`] — a SEPARATE flag
        /// rather than `ubatch.is_some()`, because the two consumers disagree about an unusable
        /// value: `INFR_UBATCH=0` and `INFR_UBATCH=abc` yield NO chunk height (the value site is
        /// `.parse().ok().filter(|&v| v > 0)`) yet still count as "the user pinned a height", which
        /// is what the seam's dense placement sweeps read to skip themselves (`is_err()` at
        /// `seam/mod.rs`'s three sweep gates). Collapsing the two onto one `Option` would silently
        /// re-enable the sweep for `-u 0`, a shipped CLI spelling (R1). S0 left this open; S4
        /// resolved it here.
        ubatch_specified: bool = false,
        /// `INFR_UBATCH_PARALLEL`: prefill chunk for a sequence sharing the GPU (`serve -np N`).
        ubatch_parallel: usize = 256,
        /// `INFR_SUBMIT_DISPATCHES`: submit-splitter cap (`0` = never split). `None` = the
        /// measured `initial_submit_dispatch_cap(integrated)` default.
        submit_dispatches: Option<usize> = None,
        /// `INFR_SG`: subgroup-size preference, exactly 16 or 32. `None` = vendor-derived.
        subgroup_pref: Option<u32> = None,
        /// Worker threads. Sourced from the CLI's `-t`; still PUBLISHED as `RAYON_NUM_THREADS`
        /// because rayon has no other input. Not an `INFR_*` knob.
        threads: Option<usize> = None,
    }
}

cfg_struct! {
    /// Process-default sampling. A server request's `RequestCtx` still layers OVER this, exactly
    /// as it layered over `Sampler::from_env()` before S4 (§5.1) — that precedence is unchanged.
    SamplingCfg / PartialSamplingCfg {
        /// `INFR_TEMP`. `0.0` = greedy: the doc contract that keeps library callers and the
        /// goldens deterministic.
        temp: f32 = 0.0,
        /// `INFR_TOP_K`.
        top_k: usize = 20,
        /// `INFR_TOP_P`.
        top_p: f32 = 0.95,
        /// `INFR_SEED`. Deliberately `Option`: the knob has TWO defaults in the tree (wall-clock
        /// nanos in `seed_rng`, `42` in the CLI/diffusion paths) and BOTH stay at their sites.
        seed: Option<u64> = None,
        /// `INFR_MAX_NEW`.
        max_new: usize = 2048,
        /// `INFR_IGNORE_EOS`.
        ignore_eos: bool = false,
        /// `INFR_NO_THINK`: suppress the model's thinking block. Set AND `!= "0"`.
        no_think: bool = false,
    }
}

cfg_struct! {
    /// KV cache format, slots and host-overflow.
    KvCfg / PartialKvCfg {
        /// `INFR_KV_TYPE_K`: requested K-cache dtype. The runner's backend/alignment GATES stay at
        /// the call site (a gated-out request silently falls back to f16) — R5.
        type_k: Option<DType> = None,
        /// Was `INFR_KV_TYPE_K` supplied AT ALL, parseable or not (§11 [DECIDE-8])? Today an
        /// unparseable name is `is_ok()` for the auto-q8 suppression check but falls through to
        /// f16 for the dtype; this flag preserves that asymmetry exactly.
        type_k_specified: bool = false,
        /// `INFR_KV_TYPE_V`: requested V-cache dtype.
        type_v: Option<DType> = None,
        /// Was `INFR_KV_TYPE_V` supplied at all (see [`KvCfg::type_k_specified`])?
        type_v_specified: bool = false,
        /// `INFR_KV_Q8`: the legacy both-sides-q8 alias.
        force_q8: bool = false,
        /// `INFR_KV_SLOTS`: prefix-cache slots.
        slots: usize = 4,
        /// `INFR_NO_KV_RING` (inverted): the SWA ring cache.
        ring: bool = true,
        /// `INFR_KV_INLINE`.
        inline_decode: bool = false,
        /// `INFR_KV_COOPMAT_BDA`.
        coopmat_bda: bool = false,
        /// `INFR_KV_OVERFLOW`: spill KV to host RAM. `budget::flag_from` grammar — `""` and `"0"`
        /// are OFF here, unlike every `is_ok()` knob (§10.5).
        overflow: bool = false,
        /// `INFR_KV_OVERFLOW_VRAM_MB`, in MiB (the byte conversion is the accessor's).
        overflow_vram_mb: Option<u64> = None,
        /// `INFR_KV_OVERFLOW_RESERVE_MB`, in MiB. `None` = `max(12% of VRAM, 2 GiB)`.
        overflow_reserve_mb: Option<u64> = None,
    }
}

cfg_struct! {
    /// Weight/expert paging and the VRAM-first spill budgets.
    PagingCfg / PartialPagingCfg {
        /// `INFR_CACHE`: paged expert-cache budget.
        cache: Option<SizeSpec> = None,
        /// `INFR_PAGER_RING`: upload staging ring. `None` = the budget-fraction policy.
        ring: Option<SizeSpec> = None,
        /// `INFR_PAGER_STATS`.
        stats: bool = false,
        /// `INFR_DRAM_CACHE`: the HOST weight-cache budget — the DRAM tier
        /// (`infr_core::hostpager`), which reads weights from the model file itself instead of
        /// leaving residency to the OS page cache.
        ///
        /// `None` (the default) keeps the zero-copy mmap path, which costs nothing and is right
        /// whenever the weights fit. This budget is anonymous, non-evictable memory, so it is only
        /// ever what the user asked for — never a fraction picked on their behalf.
        dram: Option<SizeSpec> = None,
        /// `INFR_DRAM_BYPASS`: read paged blocks straight from disk into GPU memory, with NO host
        /// cache in between — the shape a unified-memory device takes automatically, because there
        /// its streaming arena is already GPU-accessible RAM and a cache beneath it would be a
        /// second copy the device cannot read in place.
        ///
        /// Exists as a flag so that behaviour can be exercised on a DISCRETE GPU, which is the
        /// only hardware most of this is developed and tested on. Also the honest choice on a
        /// machine whose RAM is better spent elsewhere.
        dram_bypass: bool = false,
        /// `INFR_LAYER_MAJOR`, TRI-state: `None` = decide from the placement (layer-major whenever
        /// the model's weights STREAM, chunk-major when they are resident), `Some(true)` = force it
        /// on, `Some(false)` = force it off.
        ///
        /// Layer-major prefill runs the chunk loop INSIDE the layer loop, so a prompt sweeps the
        /// weight set once instead of once per `device.ubatch` chunk. That is the whole prefill
        /// cost of a streamed model and nothing at all when the weights are resident, where it
        /// would only add activation residency — hence the auto rule. Both overrides exist for A/B:
        /// forcing it ON is how a resident model can be diffed against the chunk-major order.
        layer_major: Option<bool> = None,
    }
}

cfg_struct! {
    /// The native GEMV knob family (`recorder.rs`'s `GemvKnobs`, today `OnceLock`-memoized).
    GemvCfg / PartialGemvCfg {
        /// `INFR_NO_GEMV_RM`: force the RM=1 path.
        no_rm: bool = false,
        /// `INFR_GEMV_RM`: the RM factor.
        rm: u32 = 2,
        /// `INFR_GEMV_RM_MAXOUT`.
        rm_maxout: usize = usize::MAX,
        /// `INFR_GEMV_RM_MINOUT`.
        rm_minout: usize = 2048,
        /// `INFR_NO_GEMV_SG`.
        no_sg: bool = false,
        /// `INFR_NO_GEMV_ID_SG`.
        no_id_sg: bool = false,
        /// `INFR_GEMV_SG_MINOUT`.
        sg_minout: usize = 2048,
        /// `INFR_GEMV_SG_MAXOUT`.
        sg_maxout: usize = 8192,
        /// `INFR_GEMV_SG_NR`.
        sg_nr: u32 = 2,
        /// The selected GEMV variant, COMPUTED from two keys exactly as `GemvKnobs::resolve`
        /// does it: `INFR_NO_GEMV_REG` present ⇒ `None` (and it silently wins over
        /// `INFR_GEMV_VARIANT`); otherwise `INFR_GEMV_VARIANT`, else `Some("reg")`.
        variant: Option<String> = Some("reg".to_string()),
    }
}

cfg_struct! {
    /// Vulkan kernel-tier selection. Every field is named for the thing being ENABLED and is
    /// `true` when the feature is on; the env layer inverts the `INFR_NO_*` spellings. The four
    /// `no_*` fields are the sanctioned exceptions (§4) — their EFFECT is negative, so a positive
    /// name would be a lie.
    VulkanCfg / PartialVulkanCfg {
        subs {
            /// The GEMV family (§6.5b).
            gemv: GemvCfg => PartialGemvCfg,
        }
        leaves {
        /// `INFR_NO_COOPMAT` (inverted). ANDed with [`VulkanCfg::f16`] at the probe: with
        /// `INFR_NO_F16` set, coopmat is off regardless — preserve that AND (§5.2).
        coopmat: bool = true,
        /// `INFR_CM_8X8`: the 8x8x16 coopmat shape (Intel Arc XMX opt-in).
        coopmat_8x8: bool = false,
        /// `INFR_BF16_COOPMAT`.
        bf16_coopmat: bool = false,
        /// `INFR_F8_COOPMAT`.
        f8_coopmat: bool = false,
        /// `INFR_F8_PREPACK`.
        f8_prepack: bool = false,
        /// `INFR_I8_COOPMAT`.
        i8_coopmat: bool = false,
        /// `INFR_I8_ROW_SCALE`.
        i8_row_scale: bool = false,
        /// `INFR_NO_F16` (inverted).
        f16: bool = true,
        /// `INFR_NO_I8DOT` (inverted).
        i8_dot: bool = true,

        /// `INFR_NO_GEMM_WARP` (inverted).
        gemm_warp: bool = true,
        /// `INFR_GEMM_WIDE_TILE`.
        gemm_wide_tile: bool = false,
        /// `INFR_GEMM_DIRECT_A` — fuse the store_f16 pass into the A_GLOBAL warp GEMM by
        /// reading f32 activations directly and casting to f16 in As staging (opt-in).
        gemm_direct_a: bool = false,
        /// `INFR_NO_SMALL_BM` (inverted).
        small_bm: bool = true,
        /// `INFR_NO_BM16` (inverted).
        bm16: bool = true,
        /// `INFR_NO_MMQ` (inverted).
        mmq: bool = true,
        /// `INFR_NO_MMQ_FALLBACK` (inverted).
        mmq_fallback: bool = true,
        /// `INFR_NO_MMV` (inverted). NOT a presence knob — both sites are `is_err()`. Getting
        /// this backwards disables the mmv tier for everyone (§10.2).
        mmv: bool = true,
        /// `INFR_MMV_DECODE`. The site is `mmv_decode && mmv`.
        mmv_decode: bool = false,
        /// `INFR_NO_MMV_M4` (inverted).
        mmv_m4: bool = true,
        /// `INFR_NO_MMV_O4` (inverted).
        mmv_o4: bool = true,
        /// `INFR_MMV_MW`, TRI-state: `None` = vendor default, `Some(false)` = `"0"` (force off),
        /// `Some(true)` = any other value (force on). Three different dtype lists (§10.3).
        mmv_mw: Option<bool> = None,
        /// `INFR_MMV_MW_WARPS`.
        mmv_mw_warps: Option<usize> = None,
        /// `INFR_MMV_FUSE_QUANT` — fuse the per-Linear `quant_q8` dispatch into the decode
        /// GEMV kernel (inline f32→int8 quantization per 32-block). Q4_K/Q6_K only, opt-in.
        mmv_fuse_quant: bool = false,
        /// `INFR_NO_MROW` (inverted).
        mrow: bool = true,
        /// `INFR_NO_MROW16` (inverted).
        mrow16: bool = true,
        /// `INFR_NO_F32_MROW` (inverted).
        f32_mrow: bool = true,
        /// `INFR_NO_F32_V4` (inverted).
        f32_v4: bool = true,
        /// `INFR_MOE_SMALL_M` (`tier::EnvRows`, clamped 0..=64 by the accessor — an unclamped
        /// override trips the amdgpu ring watchdog).
        moe_small_m: usize = 8,
        /// `INFR_CANVAS_CHUNK_N` (`tier::EnvRows`, floored at 1).
        canvas_chunk_n: usize = 3,

        /// `INFR_FLASH_SPLITS`.
        flash_splits: Option<usize> = None,
        /// `INFR_FLASH_BM`: compared to the LITERAL `"32"`, not parsed as an int.
        flash_bm32: bool = false,
        /// `INFR_FLASH_MIN_ROWS`.
        flash_min_rows: usize = 24,
        /// `INFR_FLASH_STAGE`.
        flash_stage: bool = false,
        /// `INFR_FLASH_DEQUANT`.
        flash_dequant: bool = false,
        /// `INFR_NO_FLASH_WARP` (inverted).
        flash_warp: bool = true,
        /// `INFR_NO_NC_FA` (inverted).
        nc_fa: bool = true,
        /// `INFR_NO_QK_WARP` (inverted).
        qk_warp: bool = true,
        /// `INFR_NO_PV_WARP` (inverted).
        pv_warp: bool = true,
        /// `INFR_PV_SPLITS`.
        pv_splits: Option<usize> = None,
        /// `INFR_NO_ATTN_HD`: a `presence` knob despite the `NO_` spelling — sanctioned negative
        /// field (§4).
        no_attn_hd_spec: bool = false,
        /// `INFR_NO_ATTN_DECODE` (inverted) — the decode-only split-K pass-1 specialization
        /// (`attn_decode.comp`). Setting the env var keeps the general `attn_partial` builds on
        /// the decode fast path; the output is bit-identical either way, so this is an A/B knob.
        attn_decode: bool = true,
        /// `INFR_NO_Q8_DECODE_D8` (inverted) — use the historical full-wave QK reduction instead
        /// of the 8-lane clustered Q8 hd256 decode kernel. Kept as a narrow A/B/fallback knob;
        /// f16 and every other KV/head-dim shape ignore it.
        q8_decode_d8: bool = true,
        /// `INFR_NO_MROWS_ATTN` / `INFR_MROWS_ATTN`, an ASYMMETRIC tri-state: `Some(false)` (the
        /// `NO_` key) wins unconditionally; `Some(true)` bypasses the rows/kv_len heuristic;
        /// `None` lets the heuristic decide.
        mrows_attn: Option<bool> = None,
        /// `INFR_NO_MLA_SG` (inverted) — the subgroup single-pass MLA tier (`mla.comp`'s
        /// `-DMLA_SG` builds). Setting the env var drops DeepSeek attention back to the scalar
        /// two-pass builds. NOT bit-identical either way: the tier carries an online softmax
        /// (running max, rescaled accumulator) where the scalar builds take two full passes, so
        /// the summation order differs.
        mla_sg: bool = true,

        /// `INFR_DN_CHUNK_SCAN` — spelled POSITIVELY but read with `.is_err()`: setting it
        /// DISABLES the chunked scan. The name lies about its polarity; R2 freezes it (§10.11).
        dn_chunk_scan: bool = true,
        /// `INFR_NO_DN_CHUNK` (inverted).
        dn_chunk: bool = true,
        /// `INFR_NO_DN_SPLIT` (inverted).
        dn_split: bool = true,
        /// Direct interleaved q/k/v reads for single-token Vulkan DeltaNet decode.  Enabled by
        /// default after dense-vs-strided output/state parity; `INFR_NO_DELTA_STRIDED` disables it
        /// and the historical `INFR_DELTA_STRIDED` opt-in remains accepted.
        delta_strided: bool = true,
        /// `INFR_NO_PUSH_DESC` (inverted).
        push_desc: bool = true,
        /// `INFR_NO_PIPELINE_CACHE` (inverted).
        pipeline_cache_disk: bool = true,
        /// `INFR_NO_VRAM_GUARD` — sanctioned negative field (§4).
        no_vram_guard: bool = false,
        /// `INFR_NO_MOE_SM_POOL` — sanctioned negative field (§4).
        no_moe_sm_pool: bool = false,
        /// `INFR_SEAM_NO_REPLAY` — sanctioned negative field (§4). Read `is_ok()` in the Vulkan
        /// adapter and `is_err()` in the llama runner; ONE field, the llama site becomes
        /// `!cfg.no_replay`.
        no_replay: bool = false,
        /// `INFR_NO_GPU_POS` (inverted). Also read in `infr-llama`'s runner.
        gpu_pos: bool = true,
        /// `INFR_NO_FUSE_ADD` (inverted) — reaches the shared fusion pass as
        /// `FusionCfg.linear_add.enabled`, not as a direct read.
        fuse_add: bool = true,

        /// `INFR_BDA_CHUNK_ELEMS`: element cap per BDA addressing unit. `None` =
        /// `BDA_CHUNK_UNIT_MAX`; a value below 2 is treated as unset.
        bda_chunk_elems: Option<u64> = None,
        /// `INFR_BDA_CHUNK_BYTES`: byte cap per BDA addressing unit.
        bda_chunk_bytes: Option<u64> = None,
        }
    }
}

cfg_struct! {
    /// Metal kernel-tier selection. Fifteen of these are `INFR_METAL_NO_*` disable-switches read
    /// with `.is_err()`; the fields are POSITIVE and default `true`.
    ///
    /// Metal cannot be run on the dev box — its slice is compile-checked
    /// (`cargo check -p infr-metal --target x86_64-apple-darwin`) and validated by macOS CI.
    MetalCfg / PartialMetalCfg {
        /// `INFR_METAL_NO_F16_NATIVE` (inverted).
        f16_native: bool = true,
        /// `INFR_METAL_NO_F32_NATIVE` (inverted).
        f32_native: bool = true,
        /// `INFR_METAL_NO_BF16_NATIVE` (inverted).
        bf16_native: bool = true,
        /// `INFR_METAL_NO_F16_CMM` (inverted).
        f16_cmm: bool = true,
        /// `INFR_METAL_NO_BF16_CMM` (inverted).
        bf16_cmm: bool = true,
        /// `INFR_METAL_NO_F32_CMM` (inverted).
        f32_cmm: bool = true,
        /// `INFR_METAL_NO_F16_RT` (inverted).
        f16_rt: bool = true,
        /// `INFR_METAL_NO_BF16_RT` (inverted).
        bf16_rt: bool = true,
        /// `INFR_METAL_NO_F32_RT` (inverted).
        f32_rt: bool = true,
        /// `INFR_METAL_NO_KQUANT_NATIVE` (inverted).
        kquant_native: bool = true,
        /// `INFR_METAL_NO_Q5K_RT` (inverted).
        q5k_rt: bool = true,
        /// `INFR_METAL_NO_RMSNORM_VEC4` (inverted).
        rmsnorm_vec4: bool = true,
        /// `INFR_METAL_NO_CONV1D_PAR` (inverted).
        conv1d_par: bool = true,
        /// `INFR_METAL_NO_DN_GATE_PREP` (inverted).
        dn_gate_prep: bool = true,
        /// `INFR_METAL_NO_DN_NORM_PREP` (inverted).
        dn_norm_prep: bool = true,
        /// `INFR_METAL_LMHEAD_MRV`: PRESENCE lifts the mrv `out_f` ceiling (the site reads
        /// `.is_err()` to keep the cap).
        lmhead_mrv_uncapped: bool = false,
        /// `INFR_METAL_NODELTA`: read BOTH ways in `exec.rs` (`is_ok()` at three sites,
        /// `is_err()` at two) with the SAME meaning — set ⇒ DeltaNet off. One positive field.
        deltanet: bool = true,
        /// `INFR_METAL_NOMOE`: as [`MetalCfg::deltanet`].
        moe: bool = true,
        /// Persist the compiled `MTLComputePipelineState`s to disk as an `MTLBinaryArchive`
        /// (`infr_core::kernel_cache`, `~/.cache/infr/metal-pipelines-<device>.bin`) and seed
        /// pipeline creation from it on later launches instead of re-running the driver's
        /// AIR → GPU-ISA back end for every kernel, every launch.
        ///
        /// This caches the BACK end only: `MTLLibrary` exposes no way to serialize the AIR it
        /// compiled from MSL, so the ~340 KB front-end compile in `Pipelines::build` still runs
        /// every launch (see `infr-metal/src/pcache.rs`).
        ///
        /// Has NO env key, like [`CpuCfg::reference`] — settable
        /// with `--set kernels.metal.pipeline_cache=false` or the TOML file. Clear it to force
        /// cold pipeline creation (bisecting a suspected cache bug); the cache is otherwise
        /// self-invalidating on any source / device / OS / compile-option change.
        pipeline_cache: bool = true,
    }
}

cfg_struct! {
    /// CPU backend knobs.
    CpuCfg / PartialCpuCfg {
        /// `INFR_CPU_SPIN`: spin-pool idle ceiling. Was a `OnceLock` memo before S3 (§10.6); it is
        /// now hoisted to a `SpinPool` field at construction, so it costs nothing per call AND a
        /// second pool can hold a different value.
        spin: u32 = 32768,
        /// `INFR_CPU_NO_SPINPOOL` (inverted): set AND `!= "0"` routes jobs through rayon.
        spinpool: bool = true,
        /// `INFR_CPU_REPACK_MB`: repack-cache byte budget, in MiB.
        repack_mb: usize = 4096,
        /// The bit-reference kernel path (`CpuBackend::reference()`). Has NO env key and never
        /// had one — it is the model the rest of the campaign follows: a typed field on the
        /// owning struct, chosen by the caller.
        reference: bool = false,
    }
}

cfg_struct! {
    /// Per-backend kernel tiers, plus the two backend-independent graph-shape gates.
    KernelCfg / PartialKernelCfg {
        subs {
            /// Vulkan (`infr-vulkan`).
            vulkan: VulkanCfg => PartialVulkanCfg,
            /// Metal (`infr-metal`).
            metal: MetalCfg => PartialMetalCfg,
            /// CPU (`infr-cpu`).
            cpu: CpuCfg => PartialCpuCfg,
        }
        leaves {
            /// `INFR_NO_QKV_FUSE` (inverted). A GRAPH-shape knob in `infr-llama`, not a Vulkan one.
            qkv_fuse: bool = true,
            /// `INFR_NO_GATED_RMSNORM` (inverted), ANDed with `caps.gated_rmsnorm` at the site.
            gated_rmsnorm: bool = true,
        }
    }
}

cfg_struct! {
    /// Speculative decode / MTP.
    SpecCfg / PartialSpecCfg {
        /// `INFR_MTP`: the EXACT string `"1"`. `INFR_MTP=true` does nothing today; the `== "1"`
        /// mapping is the behaviour-preserving one.
        mtp: bool = false,
        /// `INFR_NO_MTP_CKPT` (inverted).
        mtp_ckpt: bool = true,
        /// `INFR_NO_MTP_REPRIME` (inverted). ANDed with [`SpecCfg::mtp_ckpt`] at the site (R5).
        mtp_reprime: bool = true,
        /// `INFR_NO_MTP_DRAFT_CHAIN` (inverted).
        mtp_draft_chain: bool = true,
        /// `INFR_SPEC_DRAFT`: draft-model path.
        draft: Option<PathBuf> = None,
        /// `INFR_SPEC_K`: draft-length upper bound.
        k: usize = 6,
        /// `INFR_SPEC_DEBUG`.
        debug: bool = false,
        /// `INFR_DECODE_CHAIN`: decode iterations per submission.
        decode_chain: usize = 8,
        /// `INFR_NO_GPU_DRAFT_PROB` (inverted).
        gpu_draft_prob: bool = true,
        /// `INFR_NO_GPU_MTP_ACCEPT` (inverted).
        gpu_mtp_accept: bool = true,
        /// `INFR_NO_GPU_ARGMAX` (inverted).
        gpu_argmax: bool = true,
        /// `INFR_NO_GPU_SAMPLE` (inverted).
        gpu_sample: bool = true,
        /// `INFR_NO_GPU_EMBED` (inverted).
        gpu_embed: bool = true,
    }
}

cfg_struct! {
    /// Multi-GPU splits and their host/P2P transport. Per-device configs are OUT OF SCOPE for
    /// this campaign — every backend in a multi-GPU process gets the SAME `Arc<Config>` (§5.1).
    MultiCfg / PartialMultiCfg {
        /// `INFR_PIPELINE`: layer-split device list, minimum 2 devices.
        pipeline: Option<Vec<usize>> = None,
        /// `INFR_TENSOR_PARALLEL`: minimum 2 devices.
        tensor_parallel: Option<Vec<usize>> = None,
        /// `INFR_EXPERT_PARALLEL`: minimum **1** device — the three minimums differ, preserve them.
        expert_parallel: Option<Vec<usize>> = None,
        /// `INFR_PIPELINE_HOST` (inverted): its ABSENCE selects the P2P transport.
        pipeline_p2p: bool = true,
        /// `INFR_TP_HOST` (inverted).
        tp_p2p: bool = true,
        /// `INFR_EP_HOST` (inverted).
        ep_p2p: bool = true,
    }
}

/// How the Metal backend obtains **device** time per op.
///
/// Metal batches every op into one command buffer, so — unlike vulkan's timestamp queries —
/// per-op device time is never free there. Both non-`Off` modes cost
/// something, which is why this is a separate axis from [`ProfCfg::ops`] rather than implied by
/// it: turning per-op profiling on gets you host ENCODE time for free, and you opt in to paying
/// for device time.
///
/// This replaces `INFR_METAL_PROFILE`, which was three exact string literals masquerading as an
/// integer level (`=1` on, `=2` flush, `=3` counters — so `=20` meant "on, level unrecognized",
/// not "≥ 2"). Named modes cannot be misread that way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetalDeviceTime {
    /// No device timing: the profile reports host encode wall only. (Still a finding — a large
    /// encode share IS the per-op command-buffer overhead.)
    #[default]
    Off,
    /// Flush the batch after each op so its GPU wall is isolable. Costs the batching, and the
    /// decode replay tape re-executes tokens without walking ops, so this silently attributes a
    /// sample of one token on the decode path — prefer [`Counters`](Self::Counters).
    Flush,
    /// Stage-boundary timestamp counter samples: one encoder per op inside ONE command buffer, so
    /// the numbers are in-context with no per-op flush distortion. Disables the replay tape for
    /// the run. The honest per-op device mode.
    Counters,
}

impl std::str::FromStr for MetalDeviceTime {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "false" | "0" => Ok(Self::Off),
            "flush" => Ok(Self::Flush),
            "counters" | "counter" => Ok(Self::Counters),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for MetalDeviceTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Flush => "flush",
            Self::Counters => "counters",
        })
    }
}

cfg_struct! {
    /// Profiling and timing output. Setting any of these from the config FILE is announced at
    /// startup (§11 [DECIDE-7]).
    ///
    /// These names describe what each knob DOES. They used to not: `prof`, `prof2`,
    /// `prof2_shapes`, `prof_dec`, `prof_ops`, `prof_pf` were six names carrying no information
    /// about which of three unrelated features they belonged to, and five separate knobs
    /// (`prof`, `prof_dec`, `prof_pf`, `mtp_time`, `diffusion_time`) each turned on host-side
    /// stage timing for ONE pipeline — so "where does the wall time go" meant discovering four
    /// more switches after you found the first. Every key is now `INFR_PROF_*`, and the three
    /// features are three knobs: [`ops`](Self::ops) (per-op device profile),
    /// [`stages`](Self::stages) (host-side stage timing), [`vram`](Self::vram) (memory).
    ProfCfg / PartialProfCfg {
        /// `INFR_PROF_OPS`: per-op profiling on every backend — vulkan, metal and cpu. THE
        /// switch the backends gate on, through [`crate::prof::enabled`] (which also ANDs the
        /// warmup-suppression flag). See `docs/perf/playbook.md` § Profiling.
        ops: bool = false,
        /// `INFR_PROF_OP_SHAPES`: itemize per-op labels by SHAPE rather than folding a kind into
        /// one row. Vulkan-only refinement (it labels by kernel name, so the shared
        /// `op_label` shape itemizer does not apply); ANDed with [`ops`](Self::ops) at the site.
        op_shapes: bool = false,
        /// `INFR_PROF_STAGES`: host-side stage timing — prompt/decode throughput, the decode
        /// setup-vs-execute split, the per-chunk prefill build/compile/execute split, MTP verify
        /// time, and diffusion per-step phases. Each section prints only when its pipeline
        /// actually ran, so one switch covers every model class.
        stages: bool = false,
        /// `INFR_PROF_VRAM`: log live VRAM in use after weight load. Memory hygiene, not timing.
        vram: bool = false,
        /// `INFR_PROF_DIFFUSION_TRACE`: per-step schedule/entropy trace for diffusion models.
        /// A sampler trace, not a timing measurement — [`stages`](Self::stages) covers the timing.
        diffusion_trace: bool = false,
        /// `INFR_PAGER_PROFILE`: aggregate pager/cache/upload/submit timings at process exit.
        /// Runtime-only wall timing, deliberately separate from `INFR_PROF_OPS` because Vulkan
        /// timestamp profiling changes the paged ring's nowait behaviour.
        pager_profile: bool = false,
        /// `INFR_PROF_OUT`: write the exit report as JSON to this path, in addition to stderr.
        /// (`INFR_PROFILE` — the BUILD-time host-instrumentation input read by `build.rs` — is
        /// deliberately NOT config; see [`manifest::NOT_MIGRATED`].)
        out: Option<PathBuf> = None,
        /// `INFR_PROF_METAL_DEVICE_TIME`: how Metal obtains per-op DEVICE time. See
        /// [`MetalDeviceTime`].
        metal_device_time: MetalDeviceTime = MetalDeviceTime::Off,
        /// `INFR_PROF_METAL_DEBUG`: extra Metal profiler debug output.
        metal_debug: bool = false,
    }
}

impl ProfCfg {
    /// Does Metal need to flush after each op to isolate its GPU wall?
    pub fn metal_flush_per_op(&self) -> bool {
        self.metal_device_time == MetalDeviceTime::Flush
    }

    /// Does Metal need the stage-boundary timestamp counter set?
    pub fn metal_counters(&self) -> bool {
        self.metal_device_time == MetalDeviceTime::Counters
    }
}

cfg_struct! {
    /// Model acquisition from the HuggingFace hub (`infr pull`, and the auto-pull `infr run` and
    /// `infr serve` do when a model is missing).
    HubCfg / PartialHubCfg {
        /// `INFR_PULL_JOBS`: how many CONNECTIONS one model's download may use at the same time.
        ///
        /// A single HTTPS connection to the HF CDN is the ceiling, not the link, so a pull is
        /// worth spreading over several. Measured against `unsloth/DeepSeek-V3.2-GGUF`'s Q2_K
        /// shards (`curl` to `/dev/null`, 25 s per run): one connection sustained 8.8 MB/s, five
        /// sustained 78.7 MB/s — 15.7 MB/s EACH, so the per-connection rate went UP rather than
        /// down. Five were nowhere near the point where they start competing, which is why the
        /// default sits above five.
        ///
        /// The connections are spent on whatever the model gives them to do, and the setting
        /// deliberately does NOT say which:
        ///
        /// * a split model is `-NNNNN-of-MMMMM` shards, fetched one file per connection;
        /// * a model shipped as ONE file — `unsloth/DeepSeek-V3.2-GGUF`'s `UD-TQ1_0` is a single
        ///   161 GB GGUF — is split into byte ranges instead, fetched several at a time and
        ///   reassembled (measured on that object through `infr pull` itself, same 60-second
        ///   window: 11.4 MB/s at `1`, 80.5 MB/s at the default `8`; the whole 161 GB pull ran at
        ///   80.0 MB/s end to end);
        /// * and the tail of a shard set, where fewer files remain than there are connections,
        ///   gives the leftovers to the files still going.
        ///
        /// ONE number for both because the axes MULTIPLY: a per-file bound and a per-range bound
        /// of 8 each is 64 sockets on a repo whose file count is the publisher's choice
        /// (`DeepSeek-V3.2-REAP` ships 236 shards). Every in-flight transfer holds a socket, and
        /// each file holds a temp file and a progress LINE — a bar block taller than the terminal
        /// is the first of those to stop being free.
        ///
        /// `0` and `1` both mean strictly one connection: one file at a time, never split. That is
        /// the pre-concurrency behaviour, and the value to set when a proxy or a metered link wants
        /// exactly one connection.
        pull_jobs: usize = 8,
    }
}

cfg_struct! {
    /// Debug/poison/barrier switches. Setting any of these from the config FILE is announced at
    /// startup (§11 [DECIDE-7]).
    DebugCfg / PartialDebugCfg {
        /// `INFR_DEBUG_BDA_CHUNK`.
        bda_chunk: bool = false,
        /// `INFR_DEBUG_COOPMAT`.
        coopmat: bool = false,
        /// `INFR_DEBUG_WIDE_DISPATCH`.
        wide_dispatch: bool = false,
        /// `INFR_DEBUG_CHAT`.
        chat: bool = false,
        /// `INFR_MOE_COUNTS_DEBUG`.
        moe_counts: bool = false,
        /// `INFR_MOE_COUNTS_DUMP`.
        moe_counts_dump: bool = false,
        /// `INFR_POISON_UNINIT`.
        poison_uninit: bool = false,
        /// `INFR_NOBARRIER`.
        no_barrier: bool = false,
        /// `INFR_FULLBARRIER`.
        full_barrier: bool = false,
    }
}

cfg_struct! {
    /// `infr serve` process-level knobs. Per-request sampling is NOT here — it stays on
    /// `RequestCtx` (§5.1).
    ServeCfg / PartialServeCfg {
        /// `INFR_API_KEY`: bearer token. An EMPTY value means NO auth — the opposite of the
        /// `is_ok()` presence grammar. Do not unify them (§10.5).
        api_key: Option<String> = None,
        /// `INFR_MAX_TOKENS_CAP`: must be a positive integer, else the default.
        max_tokens_cap: u32 = 131_072,
        /// `INFR_REQUEST_TIMEOUT_SECS`: per-request wall-clock deadline, in seconds. `0` (the
        /// default) means NO deadline.
        ///
        /// OFF by default on purpose. `max_tokens_cap` bounds a request in TOKENS, which is a
        /// budget the client asked for; this bounds it in TIME, which the client did not. A
        /// deadline that fires mid-generation truncates a legitimate long reply — a slow model
        /// answering a hard prompt looks exactly like a stuck one from out here — so shipping a
        /// default would silently cut off answers that work today. It is the operator's explicit
        /// decision, made once they know their model's token rate and what a `--parallel` slot is
        /// worth to them.
        request_timeout_secs: u64 = 0,
        /// `INFR_SERVE_STATS_SECS`: how often `infr serve` logs its throughput line, in seconds.
        /// `0` switches the line off entirely.
        ///
        /// ON by default, unlike `request_timeout_secs`, because it changes nothing a client can
        /// observe — it is a log line, not a policy — and a server with no throughput signal is the
        /// state this knob exists to end. It is ACTIVITY-ONLY: an interval in which nothing
        /// happened emits nothing, so an idle server's log stays clean and the knob is only worth
        /// reaching for when the log itself is expensive.
        stats_interval_secs: u64 = 5,
    }
}

cfg_struct! {
    /// The whole resolved configuration. Build it once (in `main()`, or by a library caller),
    /// wrap it in an `Arc`, and hand it to the backends and sessions. There is deliberately NO
    /// global (R4) and no interior mutability: a `Config` is a value.
    Config / PartialConfig {
        subs {
            /// Device selection and batch shape.
            device: DeviceCfg => PartialDeviceCfg,
            /// Process-default sampling.
            sampling: SamplingCfg => PartialSamplingCfg,
            /// KV cache.
            kv: KvCfg => PartialKvCfg,
            /// Weight/expert paging.
            paging: PagingCfg => PartialPagingCfg,
            /// Per-backend kernel tiers.
            kernels: KernelCfg => PartialKernelCfg,
            /// Speculative decode / MTP.
            spec: SpecCfg => PartialSpecCfg,
            /// Multi-GPU.
            multi: MultiCfg => PartialMultiCfg,
            /// Profiling.
            prof: ProfCfg => PartialProfCfg,
            /// Server.
            serve: ServeCfg => PartialServeCfg,
            /// Model acquisition from the HuggingFace hub.
            hub: HubCfg => PartialHubCfg,
            /// Debug switches.
            debug: DebugCfg => PartialDebugCfg,
        }
        leaves {}
    }
}

// ── loading ──────────────────────────────────────────────────────────────────

/// Anything that can stop a `Config` from resolving.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The config file could not be read.
    #[error("config file {path}: {source}")]
    Io {
        /// The file we tried to read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// `--config <PATH>` named a file that does not exist. (An ABSENT default-location file is a
    /// no-op, never an error — only an explicitly-requested one is.)
    #[error("config file {0} does not exist")]
    Missing(PathBuf),
    /// The file is not valid TOML.
    #[error("config file {path}: {message}")]
    Toml {
        /// The file we tried to parse.
        path: PathBuf,
        /// The parser's message.
        message: String,
    },
    /// A KNOWN key was given a value that does not parse into its type. Hard error in the file
    /// layer and in `--set`; the env layer's per-key behaviour is recorded in
    /// [`manifest::KnobKey::bad_value`] and mostly falls back instead (R1).
    #[error("config `{path}`: {message}")]
    Value {
        /// The dotted config path.
        path: String,
        /// What was wrong with the value.
        message: String,
    },
    /// `--set` named a path that does not exist. Unlike the file layer (which warns and ignores),
    /// this is a HARD error: it was typed for this run, and ignoring it silently would give a
    /// wrong result with no second chance to notice (§11 [DECIDE-5]).
    #[error("unknown config path `{path}`{}", .suggestion.as_ref().map(|s| format!(" — did you mean `{s}`?")).unwrap_or_default())]
    UnknownPath {
        /// The path the user typed.
        path: String,
        /// The nearest known path, if one is close enough.
        suggestion: Option<String>,
    },
    /// The same `--set` path was given twice. Not a silent last-wins (§11 [DECIDE-3]).
    #[error("`--set {path}=` given more than once")]
    DuplicateSet {
        /// The repeated path.
        path: String,
    },
    /// An environment variable held a value the engine rejects LOUDLY today (`INFR_SG`,
    /// `INFR_SUBMIT_DISPATCHES`, the three multi-GPU device lists). Those keep erroring — that is
    /// the behaviour being preserved, not an `unwrap_or` (R1).
    #[error("{key}: {message}")]
    Env {
        /// The `INFR_*` variable.
        key: &'static str,
        /// Why it was rejected — the wording matches the pre-config error text.
        message: String,
    },
}

impl Config {
    /// Fold `layers` (LOWEST precedence first) over [`Config::default`].
    ///
    /// `load_from_layers(&[])` is exactly `Config::default()` — the property §8.1 pins.
    pub fn load_from_layers(layers: &[PartialConfig]) -> Config {
        let mut merged = PartialConfig::default();
        for layer in layers {
            merged.merge(layer.clone());
        }
        let mut cfg = Config::default();
        merged.apply(&mut cfg);
        cfg
    }

    /// Resolve the four layers: `Default` < file < env < CLI.
    ///
    /// Reads the process environment (through [`ConfigLayer::env`]) and the filesystem; every
    /// other entry point in this module is pure, so the tests never need either.
    pub fn load(overrides: &ConfigOverrides) -> Result<Config, ConfigError> {
        let (file_layer, file_path) = ConfigLayer::file(overrides.config_path.as_deref())?;
        let env_layer = ConfigLayer::env()?;
        let cli_layer = ConfigLayer::cli(overrides)?;
        let cfg = Config::load_from_layers(&[file_layer.clone(), env_layer, cli_layer]);
        // §11 [DECIDE-7]: the file MAY set the diagnostic knobs, but it has to say so — otherwise
        // "why is my server printing timings" is unanswerable from the command line.
        if let Some(path) = &file_path {
            if let Some(line) = announce_file_diagnostics(&file_layer, path) {
                eprintln!("{line}");
            }
        }
        Ok(cfg)
    }

    /// The `Default` < environment fold, with no file and no CLI layer.
    ///
    /// No longer transitional: `SeamModel::load` — the last bridge caller — was deleted in S8, and
    /// what remains is `CpuBackend::new` / `CpuBackend::reference`, the env-sourced TWINS of
    /// `new_with`/`reference_with` that a library caller may legitimately want (infallible is
    /// right there — CPU owns none of the loud keys). The Vulkan bridge went away in S5a and the
    /// Metal one similarly, when `MetalBackend::new_with(device_id,
    /// cfg)` became the real constructors.
    ///
    /// It is NOT `Config::load`: no file layer (these knobs were env-only before the migration, so
    /// reading a file here would ADD behaviour, and the diagnostics banner would print twice), and
    /// deliberately INFALLIBLE, so a layer error can neither pre-empt nor duplicate the error text
    /// an unmigrated read site still produces for itself (R1) — a rejected layer falls back to
    /// `Config::default()`.
    ///
    /// That infallibility is exactly why `VulkanBackend::new` and
    /// `MetalBackend::new` do NOT use this: all five loud keys are `Config`-sourced now, so
    /// swallowing the layer error there would silently drop an `INFR_SG` /
    /// `INFR_SUBMIT_DISPATCHES` / device-list rejection a caller gets today. They build the same
    /// fold with `ConfigLayer::env()?` instead.
    pub fn load_from_env() -> Config {
        let layer = ConfigLayer::env().unwrap_or_default();
        Config::load_from_layers(&[layer])
    }

    /// Every dotted config path, in declaration order. The TOML schema, `--set`'s path set and
    /// the manifest's `path` column are all checked against THIS list.
    pub fn all_paths() -> Vec<String> {
        let mut out = Vec::new();
        Config::collect_paths("", &mut out);
        out
    }
}

/// The one line printed when a config FILE turned on a `prof.*` / `debug.*` knob (§11
/// [DECIDE-7]). Returns `None` when the file set no diagnostics. Split out from
/// [`Config::load`] so it is testable without capturing stderr.
pub fn announce_file_diagnostics(file_layer: &PartialConfig, path: &Path) -> Option<String> {
    let defaults = Config::default();
    let mut applied = Config::default();
    file_layer.clone().apply(&mut applied);
    let changed: Vec<String> = Config::all_paths()
        .into_iter()
        .filter(|p| p.starts_with("prof.") || p.starts_with("debug."))
        .filter(|p| file_layer.is_path_set(p) && applied.get_path(p) != defaults.get_path(p))
        .collect();
    if changed.is_empty() {
        return None;
    }
    Some(format!(
        "[infr] config: {} enabled diagnostics: {}",
        path.display(),
        changed.join(", ")
    ))
}

/// The four layer parsers. Each returns a [`PartialConfig`] — never a resolved `Config` — so the
/// fold, not the parser, decides precedence.
pub struct ConfigLayer;

impl ConfigLayer {
    /// The config-file layer, resolved through the 3-step lookup (`--config <PATH>` →
    /// `./infr.toml` → `$XDG_CONFIG_HOME/infr/config.toml`, else `~/.config/infr/config.toml`).
    /// First existing file wins; there is no merging across files.
    ///
    /// Returns the layer AND the file it came from (`None` = no file found, which is a no-op, not
    /// an error). Unknown keys are warned about on stderr and ignored (§11 [DECIDE-5]).
    pub fn file(explicit: Option<&Path>) -> Result<(PartialConfig, Option<PathBuf>), ConfigError> {
        file::load(explicit)
    }

    /// The environment layer, over the REAL process environment.
    ///
    /// This thin wrapper is the only impure entry point; [`env::parse`] takes an injected reader,
    /// so tests drive it with a `HashMap` and never touch the process environment (§8.8).
    pub fn env() -> Result<PartialConfig, ConfigError> {
        env::parse(&|k| std::env::var(k).ok())
    }

    /// The CLI layer: bespoke flags plus `--set <config.path>=<value>`.
    pub fn cli(overrides: &ConfigOverrides) -> Result<PartialConfig, ConfigError> {
        cli::parse(overrides)
    }
}

/// Parse a multi-GPU device spec (`Vulkan0,Vulkan1,…`; a bare index `N` is also accepted) into
/// physical device indices, requiring at least `min` of them.
///
/// **The ONE copy of this grammar.** It existed twice until S4 — here (behind
/// `ConfigValue for Vec<usize>`) and as `infr_llama::seam::parse_device_spec` — which is precisely
/// the drift class this campaign removes. The seam's copy, and the `parse_device_list` `var_os`
/// wrapper around it, are deleted; the env layer (§6.11), the TOML file and `--set` all come here.
///
/// `min` is the per-knob minimum device count — `INFR_PIPELINE`/`INFR_TENSOR_PARALLEL` need 2,
/// `INFR_EXPERT_PARALLEL` needs 1, and those three minimums are NOT the same (R1). `0` skips the
/// check entirely, which is what the `--set`/TOML value grammar wants: a count bound is policy and
/// belongs to the consumer (R5), while the env layer applies it because it must reproduce today's
/// loud error.
///
/// Empty and whitespace-only segments are skipped (`" 0 , 1 ,2, "` is three devices), matching the
/// pre-config seam parser byte for byte. The error text names no key: the caller prefixes it with
/// the `INFR_*` variable or the dotted path the spec came from.
pub fn parse_device_spec(spec: &str, min: usize) -> Result<Vec<usize>, String> {
    let devs: Vec<usize> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|part| {
            part.strip_prefix("Vulkan")
                .unwrap_or(part)
                .parse::<usize>()
                .map_err(|_| format!("'{part}' is not VulkanN or a device index"))
        })
        .collect::<Result<_, _>>()?;
    if devs.len() < min {
        return Err(format!(
            "needs >={min} device(s) for a real split (got '{spec}'); unset it to run single-device"
        ));
    }
    Ok(devs)
}

/// The closest known config path to `typo`, for a did-you-mean suggestion. `None` when nothing is
/// close enough to be worth guessing.
pub(crate) fn suggest_path(typo: &str) -> Option<String> {
    let paths = Config::all_paths();
    let mut best: Option<(usize, String)> = None;
    for p in paths {
        let d = edit_distance(typo, &p);
        if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
            best = Some((d, p));
        }
    }
    // A third of the typed length, floored at 2 — enough for a transposed pair or a dropped
    // segment character, not enough to suggest an unrelated knob.
    let budget = (typo.len() / 3).max(2);
    best.filter(|(d, _)| *d <= budget).map(|(_, p)| p)
}

/// Plain Levenshtein distance over bytes (paths are ASCII).
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}
