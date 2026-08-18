//! The `INFR_*` environment layer — the ONLY place in the engine that reads a knob out of the
//! process environment once the campaign lands (R3).
//!
//! [`parse`] takes an INJECTED reader (`Fn(&str) -> Option<String>`) rather than calling
//! `std::env::var` itself. That is the whole point: a test drives it with a `HashMap`, never
//! mutates the process environment, and therefore never races another test in the same binary
//! (which is the bug the deleted `infr_core::test_env::EnvGuard` existed to work around).
//! `ConfigLayer::env()`
//! is the thin impure wrapper that passes `|k| std::env::var(k).ok()`.
//!
//! **R5 — no behaviour here.** This module turns strings into `Option<T>` and nothing else. No
//! clamping, no defaulting, no policy. The only "logic" is per-KEY grammar (presence vs
//! `presence-inv` vs `flag_from` vs an exact string literal), which IS the string→value step, and
//! the handful of keys that reject a bad value LOUDLY today and must keep doing so.
//!
//! Polarity: a config field is named for the thing being ENABLED. An `INFR_NO_*` key therefore
//! emits `Some(false)` when it is PRESENT and `None` when it is absent — never `Some(true)`. Note
//! `INFR_NO_FOO=0` still turns the feature OFF: only presence matters (§7.0's truth table). The
//! four `budget::flag_from` keys are the exception, where `"0"` and `""` mean off.

use std::path::PathBuf;

use super::{ConfigError, PartialConfig};
use crate::{DType, SizeSpec};

/// An environment reader: variable name → value, `None` when unset. An empty value is
/// `Some("")` — "set to the empty string" is DISTINCT from "unset" for every presence knob.
pub type Get<'a> = &'a dyn Fn(&str) -> Option<String>;

// ── per-grammar readers ──────────────────────────────────────────────────────

/// `presence`: set to anything (including empty) ⇒ the feature is ON.
fn presence(get: Get, key: &str) -> Option<bool> {
    get(key).map(|_| true)
}

/// `presence-inv`: the key's PRESENCE disables a feature whose field is positive.
fn presence_inv(get: Get, key: &str) -> Option<bool> {
    get(key).map(|_| false)
}

/// `budget::flag_from`: set and neither `""` nor `"0"` ⇒ on. NOT the same as `presence`.
fn flag(get: Get, key: &str) -> Option<bool> {
    get(key).map(|v| crate::budget::flag_from(Some(&v)))
}

/// Set AND `!= "0"` ⇒ the (positive) field is true.
fn set_not_zero(get: Get, key: &str) -> Option<bool> {
    get(key).map(|v| v != "0")
}

/// Set AND `!= "0"` ⇒ the (positive) field is FALSE — the `INFR_CPU_NO_SPINPOOL` spelling.
fn set_not_zero_inv(get: Get, key: &str) -> Option<bool> {
    get(key).map(|v| v == "0")
}

/// A numeric knob whose bad value falls back to the default today: unparseable ⇒ this layer did
/// not specify it, which is exactly what `.and_then(parse).ok().unwrap_or(default)` does.
fn num<T: std::str::FromStr>(get: Get, key: &str) -> Option<T> {
    get(key).and_then(|v| v.parse().ok())
}

/// [`num`] for the sites that additionally reject `0` (`.filter(|&v| v > 0)`).
fn num_pos(get: Get, key: &str) -> Option<usize> {
    num::<usize>(get, key).filter(|&v| v > 0)
}

/// A numeric knob behind an `Option<T>` FIELD (partial = `Option<Option<T>>`).
fn opt_num<T: std::str::FromStr>(get: Get, key: &str) -> Option<Option<T>> {
    num::<T>(get, key).map(Some)
}

/// A MiB-valued knob: trimmed, then a plain `u64` count — so a float, a negative or an empty value
/// leaves the knob unspecified (`mib_and_size_string_spellings`). Stored in MiB; the byte
/// conversion belongs to the accessor ([`crate::budget::mib_bytes`], R5).
fn opt_mib(get: Get, key: &str) -> Option<Option<u64>> {
    get(key)
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Some)
}

/// The shared size/count grammar (`crate::parse_size`) behind an `Option<SizeSpec>` field.
fn opt_size(get: Get, key: &str) -> Option<Option<SizeSpec>> {
    get(key).and_then(|v| crate::parse_size(&v)).map(Some)
}

/// A free-string knob behind an `Option<String>` field.
fn opt_text(get: Get, key: &str) -> Option<Option<String>> {
    get(key).map(Some)
}

/// A filesystem-path knob behind an `Option<PathBuf>` field.
fn opt_path(get: Get, key: &str) -> Option<Option<PathBuf>> {
    get(key).map(|v| Some(PathBuf::from(v)))
}

/// A KV format name behind an `Option<DType>` field. An unrecognized name still COUNTS as
/// specified (the caller records that separately) but yields no dtype — today's asymmetry.
fn opt_kv_dtype(get: Get, key: &str) -> Option<Option<DType>> {
    get(key).map(|v| crate::budget::parse_kv_dtype(&v))
}

/// A multi-GPU device list. This is one of the five knobs that ERROR on a bad value today, so the
/// error propagates out of the layer instead of falling back (§6.1, §6.11). The per-knob minimum
/// device counts differ (2 / 2 / 1) and are passed straight through to the shared parser.
///
/// The grammar itself is [`super::parse_device_spec`] — the ONE copy since S4, shared with the
/// `--set`/TOML value grammar and with what used to be `infr_llama::seam::parse_device_spec`.
fn device_list(
    get: Get,
    key: &'static str,
    min: usize,
) -> Result<Option<Option<Vec<usize>>>, ConfigError> {
    let Some(spec) = get(key) else {
        return Ok(None);
    };
    let devs = super::parse_device_spec(&spec, min)
        .map_err(|message| ConfigError::Env { key, message })?;
    Ok(Some(Some(devs)))
}

// ── the layer ────────────────────────────────────────────────────────────────

/// Read every `INFR_*` knob through `get` and return what this layer specifies.
///
/// Absent keys leave their field `None`, so the fold never lets the environment clobber a config
/// file or a CLI flag it did not actually mention.
#[allow(clippy::too_many_lines)] // one straight-line statement per knob; splitting it hides keys
pub fn parse(get: Get) -> Result<PartialConfig, ConfigError> {
    let mut p = PartialConfig::default();

    // ── device ───────────────────────────────────────────────────────────────
    // The raw spec, unfiltered: the CLI's `.filter(|s| !s.trim().is_empty())` is that consumer's
    // policy and stays at its site, and `resolve_infr_dev_index` tolerates an empty value.
    p.device.dev = opt_text(get, "INFR_DEV");
    p.device.ctx = opt_size(get, "INFR_CTX");
    // §6.12: the VALUE (`>0`, else fall through) and the PRESENCE (the placement sweeps' "the user
    // pinned a height") are recorded separately — `INFR_UBATCH=0`/`=abc` is specified-but-unusable.
    p.device.ubatch = num_pos(get, "INFR_UBATCH").map(Some);
    p.device.ubatch_specified = presence(get, "INFR_UBATCH");
    p.device.ubatch_parallel = num_pos(get, "INFR_UBATCH_PARALLEL");
    // Rejects a bad value LOUDLY today (`VulkanBackend::new` returns an error) — keep erroring.
    if let Some(v) = get("INFR_SUBMIT_DISPATCHES") {
        let n = v.parse::<usize>().map_err(|e| ConfigError::Env {
            key: "INFR_SUBMIT_DISPATCHES",
            message: format!("expected a dispatch count (0 = never split): {e}"),
        })?;
        p.device.submit_dispatches = Some(Some(n));
    }
    // Exactly "16" or "32"; anything else is an error, not a silent fallback.
    if let Some(v) = get("INFR_SG") {
        let sg = match v.as_str() {
            "16" => 16u32,
            "32" => 32,
            other => {
                return Err(ConfigError::Env {
                    key: "INFR_SG",
                    message: format!(
                        "must be 16 or 32 (got {other:?}) — the decode GEMV family only has \
                         subgroup-16 and subgroup-32 builds"
                    ),
                })
            }
        };
        p.device.subgroup_pref = Some(Some(sg));
    }

    // ── sampling ─────────────────────────────────────────────────────────────
    p.sampling.temp = num(get, "INFR_TEMP");
    p.sampling.top_k = num(get, "INFR_TOP_K");
    p.sampling.top_p = num(get, "INFR_TOP_P");
    p.sampling.seed = opt_num(get, "INFR_SEED");
    p.sampling.max_new = num(get, "INFR_MAX_NEW");
    p.sampling.ignore_eos = presence(get, "INFR_IGNORE_EOS");
    p.sampling.no_think = set_not_zero(get, "INFR_NO_THINK");

    // ── kv ───────────────────────────────────────────────────────────────────
    // §11 [DECIDE-8]: an unparseable name still counts as SPECIFIED (it suppresses auto-q8) while
    // yielding no dtype (the runner falls through to f16). Both halves are recorded.
    p.kv.type_k = opt_kv_dtype(get, "INFR_KV_TYPE_K");
    p.kv.type_k_specified = presence(get, "INFR_KV_TYPE_K");
    p.kv.type_v = opt_kv_dtype(get, "INFR_KV_TYPE_V");
    p.kv.type_v_specified = presence(get, "INFR_KV_TYPE_V");
    p.kv.force_q8 = presence(get, "INFR_KV_Q8");
    p.kv.slots = num_pos(get, "INFR_KV_SLOTS");
    p.kv.ring = presence_inv(get, "INFR_NO_KV_RING");
    p.kv.inline_decode = presence(get, "INFR_KV_INLINE");
    p.kv.coopmat_bda = presence(get, "INFR_KV_COOPMAT_BDA");
    p.kv.overflow = flag(get, "INFR_KV_OVERFLOW");
    p.kv.overflow_vram_mb = opt_mib(get, "INFR_KV_OVERFLOW_VRAM_MB");
    p.kv.overflow_reserve_mb = opt_mib(get, "INFR_KV_OVERFLOW_RESERVE_MB");

    // ── paging ───────────────────────────────────────────────────────────────
    p.paging.cache = opt_size(get, "INFR_CACHE");
    p.paging.ring = opt_size(get, "INFR_PAGER_RING");
    p.paging.dram = opt_size(get, "INFR_DRAM_CACHE");
    p.paging.dram_bypass = presence(get, "INFR_DRAM_BYPASS");
    p.paging.stats = presence(get, "INFR_PAGER_STATS");
    // Tri-state: "0" ⇒ force chunk-major, any other value ⇒ force layer-major, unset ⇒ decide
    // from the placement (see `PagingCfg::layer_major`).
    p.paging.layer_major = get("INFR_LAYER_MAJOR").map(|s| Some(s != "0"));

    // ── kernels.vulkan ───────────────────────────────────────────────────────
    let v = &mut p.kernels.vulkan;
    v.coopmat = presence_inv(get, "INFR_NO_COOPMAT");
    v.coopmat_8x8 = presence(get, "INFR_CM_8X8");
    v.bf16_coopmat = presence(get, "INFR_BF16_COOPMAT");
    v.f8_coopmat = presence(get, "INFR_F8_COOPMAT");
    v.f8_prepack = presence(get, "INFR_F8_PREPACK");
    v.i8_coopmat = presence(get, "INFR_I8_COOPMAT");
    v.i8_row_scale = presence(get, "INFR_I8_ROW_SCALE");
    v.f16 = presence_inv(get, "INFR_NO_F16");
    v.i8_dot = presence_inv(get, "INFR_NO_I8DOT");

    v.gemm_warp = presence_inv(get, "INFR_NO_GEMM_WARP");
    v.gemm_wide_tile = presence(get, "INFR_GEMM_WIDE_TILE");
    v.gemm_direct_a = presence(get, "INFR_GEMM_DIRECT_A");
    v.small_bm = presence_inv(get, "INFR_NO_SMALL_BM");
    v.bm16 = presence_inv(get, "INFR_NO_BM16");
    v.mmq = presence_inv(get, "INFR_NO_MMQ");
    v.mmq_fallback = presence_inv(get, "INFR_NO_MMQ_FALLBACK");
    v.mmv = presence_inv(get, "INFR_NO_MMV");
    v.mmv_decode = presence(get, "INFR_MMV_DECODE");
    v.mmv_m4 = presence_inv(get, "INFR_NO_MMV_M4");
    v.mmv_o4 = presence_inv(get, "INFR_NO_MMV_O4");
    // Tri-state: "0" ⇒ force off, any other value ⇒ force on, unset ⇒ the vendor default.
    v.mmv_mw = get("INFR_MMV_MW").map(|s| Some(s != "0"));
    v.mmv_mw_warps = opt_num(get, "INFR_MMV_MW_WARPS");
    v.mmv_fuse_quant = presence(get, "INFR_MMV_FUSE_QUANT");
    v.mrow = presence_inv(get, "INFR_NO_MROW");
    v.mrow16 = presence_inv(get, "INFR_NO_MROW16");
    v.f32_mrow = presence_inv(get, "INFR_NO_F32_MROW");
    v.f32_v4 = presence_inv(get, "INFR_NO_F32_V4");
    // `tier::EnvRows` clamps (0..=64 / 1..) — that is policy and stays in the accessor (R5).
    v.moe_small_m = num(get, "INFR_MOE_SMALL_M");
    v.canvas_chunk_n = num(get, "INFR_CANVAS_CHUNK_N");

    v.flash_splits = opt_num(get, "INFR_FLASH_SPLITS");
    // Compared to the LITERAL "32"; any other value reads as unset at the site, so it maps to
    // `Some(false)` — same resolved value, and the layer still records that it was mentioned.
    v.flash_bm32 = get("INFR_FLASH_BM").map(|s| s == "32");
    v.flash_min_rows = num(get, "INFR_FLASH_MIN_ROWS");
    v.flash_stage = presence(get, "INFR_FLASH_STAGE");
    v.flash_dequant = presence(get, "INFR_FLASH_DEQUANT");
    v.flash_warp = presence_inv(get, "INFR_NO_FLASH_WARP");
    v.nc_fa = presence_inv(get, "INFR_NO_NC_FA");
    v.qk_warp = presence_inv(get, "INFR_NO_QK_WARP");
    v.pv_warp = presence_inv(get, "INFR_NO_PV_WARP");
    v.pv_splits = opt_num(get, "INFR_PV_SPLITS");
    v.no_attn_hd_spec = presence(get, "INFR_NO_ATTN_HD");
    v.attn_decode = presence_inv(get, "INFR_NO_ATTN_DECODE");
    v.q8_decode_d8 = presence_inv(get, "INFR_NO_Q8_DECODE_D8");
    v.mla_sg = presence_inv(get, "INFR_NO_MLA_SG");
    // Asymmetric pair: the `NO_` key wins over the opt-in when BOTH are set.
    v.mrows_attn = if get("INFR_NO_MROWS_ATTN").is_some() {
        Some(Some(false))
    } else if get("INFR_MROWS_ATTN").is_some() {
        Some(Some(true))
    } else {
        None
    };

    // Spelled positively, read with `.is_err()` — setting it DISABLES the chunked scan (§10.11).
    v.dn_chunk_scan = presence_inv(get, "INFR_DN_CHUNK_SCAN");
    v.dn_chunk = presence_inv(get, "INFR_NO_DN_CHUNK");
    v.dn_split = presence_inv(get, "INFR_NO_DN_SPLIT");
    // Default-on since M6.  The new NO_ spelling wins when both are present; retain the historical
    // positive opt-in so old launchers/configuration continue to resolve to the same value.
    v.delta_strided = if get("INFR_NO_DELTA_STRIDED").is_some() {
        Some(false)
    } else {
        presence(get, "INFR_DELTA_STRIDED")
    };
    v.push_desc = presence_inv(get, "INFR_NO_PUSH_DESC");
    v.pipeline_cache_disk = presence_inv(get, "INFR_NO_PIPELINE_CACHE");
    v.no_vram_guard = presence(get, "INFR_NO_VRAM_GUARD");
    v.no_moe_sm_pool = presence(get, "INFR_NO_MOE_SM_POOL");
    v.no_replay = presence(get, "INFR_SEAM_NO_REPLAY");
    v.gpu_pos = presence_inv(get, "INFR_NO_GPU_POS");
    v.fuse_add = presence_inv(get, "INFR_NO_FUSE_ADD");

    // `cap_from_env`: trimmed, and a cap below 2 can never fit one row ⇒ treated as unset.
    v.bda_chunk_elems = get("INFR_BDA_CHUNK_ELEMS")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n >= 2)
        .map(Some);
    v.bda_chunk_bytes = get("INFR_BDA_CHUNK_BYTES")
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n >= 2)
        .map(Some);

    // `GemvKnobs::resolve`, verbatim (§6.5b) — this whole group is already a pure resolver today.
    let g = &mut v.gemv;
    g.no_rm = presence(get, "INFR_NO_GEMV_RM");
    g.rm = num(get, "INFR_GEMV_RM");
    g.rm_maxout = num(get, "INFR_GEMV_RM_MAXOUT");
    g.rm_minout = num(get, "INFR_GEMV_RM_MINOUT");
    g.no_sg = presence(get, "INFR_NO_GEMV_SG");
    g.no_id_sg = presence(get, "INFR_NO_GEMV_ID_SG");
    g.sg_minout = num(get, "INFR_GEMV_SG_MINOUT");
    g.sg_maxout = num(get, "INFR_GEMV_SG_MAXOUT");
    g.sg_nr = num(get, "INFR_GEMV_SG_NR");
    // `INFR_NO_GEMV_REG` silently WINS over `INFR_GEMV_VARIANT` — R1-frozen (§10.11).
    if get("INFR_NO_GEMV_REG").is_some() {
        g.variant = Some(None);
    } else if let Some(s) = get("INFR_GEMV_VARIANT") {
        g.variant = Some(Some(s));
    }

    // ── kernels.metal ────────────────────────────────────────────────────────
    let m = &mut p.kernels.metal;
    m.f16_native = presence_inv(get, "INFR_METAL_NO_F16_NATIVE");
    m.f32_native = presence_inv(get, "INFR_METAL_NO_F32_NATIVE");
    m.bf16_native = presence_inv(get, "INFR_METAL_NO_BF16_NATIVE");
    m.f16_cmm = presence_inv(get, "INFR_METAL_NO_F16_CMM");
    m.bf16_cmm = presence_inv(get, "INFR_METAL_NO_BF16_CMM");
    m.f32_cmm = presence_inv(get, "INFR_METAL_NO_F32_CMM");
    m.f16_rt = presence_inv(get, "INFR_METAL_NO_F16_RT");
    m.bf16_rt = presence_inv(get, "INFR_METAL_NO_BF16_RT");
    m.f32_rt = presence_inv(get, "INFR_METAL_NO_F32_RT");
    m.kquant_native = presence_inv(get, "INFR_METAL_NO_KQUANT_NATIVE");
    m.q5k_rt = presence_inv(get, "INFR_METAL_NO_Q5K_RT");
    m.rmsnorm_vec4 = presence_inv(get, "INFR_METAL_NO_RMSNORM_VEC4");
    m.conv1d_par = presence_inv(get, "INFR_METAL_NO_CONV1D_PAR");
    m.dn_gate_prep = presence_inv(get, "INFR_METAL_NO_DN_GATE_PREP");
    m.dn_norm_prep = presence_inv(get, "INFR_METAL_NO_DN_NORM_PREP");
    m.lmhead_mrv_uncapped = presence(get, "INFR_METAL_LMHEAD_MRV");
    m.deltanet = presence_inv(get, "INFR_METAL_NODELTA");
    m.moe = presence_inv(get, "INFR_METAL_NOMOE");

    // ── kernels.cpu ──────────────────────────────────────────────────────────
    p.kernels.cpu.spin = num(get, "INFR_CPU_SPIN");
    p.kernels.cpu.spinpool = set_not_zero_inv(get, "INFR_CPU_NO_SPINPOOL");
    p.kernels.cpu.repack_mb = num(get, "INFR_CPU_REPACK_MB");

    // ── kernels (graph shape, `infr-llama`) ──────────────────────────────────
    p.kernels.qkv_fuse = presence_inv(get, "INFR_NO_QKV_FUSE");
    p.kernels.gated_rmsnorm = presence_inv(get, "INFR_NO_GATED_RMSNORM");

    // ── spec ─────────────────────────────────────────────────────────────────
    // The EXACT string "1"; `INFR_MTP=true` does nothing today.
    p.spec.mtp = get("INFR_MTP").map(|s| s == "1");
    p.spec.mtp_ckpt = presence_inv(get, "INFR_NO_MTP_CKPT");
    p.spec.mtp_reprime = presence_inv(get, "INFR_NO_MTP_REPRIME");
    p.spec.mtp_draft_chain = presence_inv(get, "INFR_NO_MTP_DRAFT_CHAIN");
    p.spec.draft = opt_path(get, "INFR_SPEC_DRAFT");
    p.spec.k = num(get, "INFR_SPEC_K");
    p.spec.debug = presence(get, "INFR_SPEC_DEBUG");
    p.spec.decode_chain = num(get, "INFR_DECODE_CHAIN");
    p.spec.gpu_draft_prob = presence_inv(get, "INFR_NO_GPU_DRAFT_PROB");
    p.spec.gpu_mtp_accept = presence_inv(get, "INFR_NO_GPU_MTP_ACCEPT");
    p.spec.gpu_argmax = presence_inv(get, "INFR_NO_GPU_ARGMAX");
    p.spec.gpu_sample = presence_inv(get, "INFR_NO_GPU_SAMPLE");
    p.spec.gpu_embed = presence_inv(get, "INFR_NO_GPU_EMBED");

    // ── multi ────────────────────────────────────────────────────────────────
    p.multi.pipeline = device_list(get, "INFR_PIPELINE", 2)?;
    p.multi.tensor_parallel = device_list(get, "INFR_TENSOR_PARALLEL", 2)?;
    p.multi.expert_parallel = device_list(get, "INFR_EXPERT_PARALLEL", 1)?;
    p.multi.pipeline_p2p = presence_inv(get, "INFR_PIPELINE_HOST");
    p.multi.tp_p2p = presence_inv(get, "INFR_TP_HOST");
    p.multi.ep_p2p = presence_inv(get, "INFR_EP_HOST");

    // ── prof ─────────────────────────────────────────────────────────────────
    // Every key is `INFR_PROF_*`. The old spellings (INFR_PROF, INFR_PROF2, INFR_PROF2_SHAPES,
    // INFR_PROF_DEC, INFR_PROF_PF, INFR_PROFILE_OUT, INFR_VRAM_LOG, INFR_MTP_TIME,
    // INFR_DIFFUSION_TIME, INFR_EB_TRACE, INFR_METAL_PROFILE, INFR_METAL_PROF_DEBUG) are dropped
    // CLEANLY per the project's env policy — no aliases, no warnings, they simply stop being read.
    p.prof.ops = presence(get, "INFR_PROF_OPS");
    p.prof.op_shapes = presence(get, "INFR_PROF_OP_SHAPES");
    p.prof.stages = presence(get, "INFR_PROF_STAGES");
    p.prof.vram = presence(get, "INFR_PROF_VRAM");
    p.prof.diffusion_trace = presence(get, "INFR_PROF_DIFFUSION_TRACE");
    p.prof.pager_profile = presence(get, "INFR_PAGER_PROFILE");
    p.prof.out = opt_path(get, "INFR_PROF_OUT");
    // Unlike its predecessor's exact-literal `"2"`/`"3"` compare, an unrecognized mode here falls
    // back to the default rather than silently meaning "on, level unrecognized".
    p.prof.metal_device_time = opt_text(get, "INFR_PROF_METAL_DEVICE_TIME")
        .flatten()
        .and_then(|s| s.parse::<super::MetalDeviceTime>().ok());
    p.prof.metal_debug = presence(get, "INFR_PROF_METAL_DEBUG");

    // ── debug ────────────────────────────────────────────────────────────────
    p.debug.bda_chunk = presence(get, "INFR_DEBUG_BDA_CHUNK");
    p.debug.coopmat = presence(get, "INFR_DEBUG_COOPMAT");
    p.debug.wide_dispatch = presence(get, "INFR_DEBUG_WIDE_DISPATCH");
    p.debug.chat = presence(get, "INFR_DEBUG_CHAT");
    p.debug.moe_counts = presence(get, "INFR_MOE_COUNTS_DEBUG");
    p.debug.moe_counts_dump = presence(get, "INFR_MOE_COUNTS_DUMP");
    p.debug.poison_uninit = presence(get, "INFR_POISON_UNINIT");
    p.debug.no_barrier = presence(get, "INFR_NOBARRIER");
    p.debug.full_barrier = presence(get, "INFR_FULLBARRIER");

    // ── serve ────────────────────────────────────────────────────────────────
    // An EMPTY value means "no auth", not "unset" — the site is `.filter(|k| !k.is_empty())`.
    p.serve.api_key = get("INFR_API_KEY").map(|k| if k.is_empty() { None } else { Some(k) });
    p.serve.max_tokens_cap = num::<u32>(get, "INFR_MAX_TOKENS_CAP").filter(|&v| v > 0);
    // NO `> 0` filter here, unlike the cap above: `0` is a MEANINGFUL value for this knob ("no
    // deadline"), not a bad one. Dropping it would make `INFR_REQUEST_TIMEOUT_SECS=0` mean "the
    // layer said nothing", so it could not turn a deadline back OFF over a config file that set one.
    p.serve.request_timeout_secs = num::<u64>(get, "INFR_REQUEST_TIMEOUT_SECS");
    // Ditto: `0` means "no periodic throughput line", a value the operator may need to set OVER a
    // config file that enabled it, so it must survive this layer rather than be filtered as junk.
    p.serve.stats_interval_secs = num::<u64>(get, "INFR_SERVE_STATS_SECS");

    // ── hub ──────────────────────────────────────────────────────────────────
    // NO `> 0` filter: `0` is a MEANINGFUL value ("fetch strictly one file at a time"), so it must
    // survive this layer to be able to turn concurrency back off over a config file that raised it.
    p.hub.pull_jobs = num::<usize>(get, "INFR_PULL_JOBS");

    Ok(p)
}
