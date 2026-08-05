//! The checked-in `INFR_*` key table: env name → config path → grammar → `migrated`.
//!
//! This — not `docs/config-plan.md` — is the authority on what the campaign has to move, because
//! the compiler and the tests read it. It is seeded from the `docs/config-plan.md` §6.0 command:
//!
//! ```bash
//! grep -rhoE '"INFR_[A-Z_0-9]*"' crates/*/src crates/*/build.rs | tr -d '"' | sort -u \
//!   | grep -vE '_TEST_|^INFR_(TP|EP|CPU|METAL)$'
//! ```
//!
//! which yields **177** keys at `2dd0c5a`: [`KEYS`]'s 174 plus [`NOT_MIGRATED`]'s 3. (A naive
//! `grep 'env::var("INFR_'` found only 153 of them — 205 literal call sites — because ~27 reads
//! went through a helper that took the key name as a `&str`: `budget::env_flag`/`env_mib`/
//! `overflow_vram_reserve`, `pager::ring_bytes_policy`, `tier::EnvRows::get`, `fusion`'s
//! `disable_env`, `seam::parse_device_list`, `recorder::GemvKnobs::resolve` and
//! `recorder::cap_from_env`. Any inventory built from that one grep is wrong by ~15%. The first
//! four of those wrappers are gone as of S2; the rest go with their own slices.)
//!
//! Three things read this table:
//! - `config::tests::env_layer_reads_every_key` (§8.8) — set each key through the injected reader
//!   and assert the layer noticed. This is what stops a knob being silently dropped mid-campaign.
//! - `config::tests::presence_inverted_knobs_have_the_right_polarity` (§8.7) — the `""`/`"0"`/
//!   `"1"` truth table for every `presence-inv` entry.
//! - `config::tests::manifest_matches_the_tree` (§8.9) — a new `INFR_*` literal in `crates/*/src`
//!   with no entry here fails the build's tests, so a feature branch cannot re-introduce an
//!   ungoverned knob.
//!
//! A slice flips its knobs' `pending` → `migrated` as it moves their read sites.

/// How an env VALUE maps onto its config field. This is the per-KEY grammar; the per-TYPE
/// spelling used by the TOML file and `--set` lives in [`super::ConfigValue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grammar {
    /// Set to anything (including empty) ⇒ the field is `true`.
    Presence,
    /// Set to anything ⇒ the (positive) field is `false`. `INFR_NO_FOO=0` still means OFF.
    PresenceInv,
    /// Set ⇒ `Some(true)` in an `Option<bool>` field.
    PresenceOptTrue,
    /// Set ⇒ `Some(false)` in an `Option<bool>` field, and it WINS over its opt-in sibling.
    PresenceOptFalse,
    /// Set ⇒ the `Option<_>` field is cleared to `None` (`INFR_NO_GEMV_REG`).
    PresenceClears,
    /// `budget::flag_from`: set and neither `""` nor `"0"` ⇒ `true`.
    Flag,
    /// Set AND `!= "0"` ⇒ the field is `true`.
    SetNotZero,
    /// Set AND `!= "0"` ⇒ the (positive) field is `false`.
    SetNotZeroInv,
    /// `"0"` ⇒ `Some(false)`, any other value ⇒ `Some(true)`, unset ⇒ `None`.
    TriState,
    /// An integer.
    Int,
    /// A float.
    Float,
    /// A free string.
    Text,
    /// A filesystem path.
    Path,
    /// The shared size/count grammar (`crate::parse_size`).
    Size,
    /// A MiB count.
    Mib,
    /// A KV format name (`budget::parse_kv_dtype`).
    Dtype,
    /// A multi-GPU device list (`Vulkan0,Vulkan1,…`).
    DeviceList,
    /// Compared against an exact string literal, NOT parsed.
    Literal,
}

/// What a BAD value does in the ENVIRONMENT layer today — which R1 freezes.
///
/// Most keys swallow it (`.and_then(parse).ok().unwrap_or(default)`); five reject it loudly and
/// must keep doing so. The FILE layer is stricter for everyone: a value that does not parse into
/// a known key's type is always a hard error there (§11 [DECIDE-5] — "a value of the wrong type
/// for a known key stays a hard error"), which is why `ctx = "banana"` fails to load from a file
/// while `INFR_CTX=banana` is still silently ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadValue {
    /// Falls back to the default. The layer reports "not specified".
    Ignored,
    /// Errors out of `ConfigLayer::env()`.
    Error,
}

/// One `INFR_*` knob.
#[derive(Debug, Clone, Copy)]
pub struct KnobKey {
    /// The environment variable. FROZEN — this campaign changes where the value goes, not what
    /// users type (R2).
    pub env: &'static str,
    /// The dotted config path it lands on. Checked against `Config::all_paths()` by a test, so it
    /// cannot drift from the TOML schema or from `--set`'s path set.
    pub path: &'static str,
    /// How the env VALUE maps onto the field.
    pub grammar: Grammar,
    /// What a bad value does in the env layer today.
    pub bad_value: BadValue,
    /// A value that provably makes the env layer specify this knob — the §8.8 probe.
    pub sample: &'static str,
    /// Has the read site moved onto `Config` yet? Flipped by the slice that migrates it.
    pub migrated: bool,
}

macro_rules! knobs {
    (@mig pending) => { false };
    (@mig migrated) => { true };
    ($($env:literal => $path:literal, $g:ident, $bv:ident, $sample:literal, $mig:ident;)*) => {
        /// Every `INFR_*` knob that becomes configuration. See [`NOT_MIGRATED`] for the rest.
        pub const KEYS: &[KnobKey] = &[$(
            KnobKey {
                env: $env,
                path: $path,
                grammar: Grammar::$g,
                bad_value: BadValue::$bv,
                sample: $sample,
                migrated: knobs!(@mig $mig),
            }
        ),*];
    };
}

knobs! {
    // ── device (§6.1) ────────────────────────────────────────────────────────
    "INFR_DEV"                  => "device.dev",               Text, Ignored, "vulkan0", migrated;
    "INFR_CTX"                  => "device.ctx",               Size, Ignored, "32k",     migrated;
    "INFR_UBATCH"               => "device.ubatch",            Int,  Ignored, "512",     migrated;
    "INFR_UBATCH_PARALLEL"      => "device.ubatch_parallel",   Int,  Ignored, "128",     migrated;
    "INFR_SUBMIT_DISPATCHES"    => "device.submit_dispatches", Int,  Error,   "64",      migrated;
    "INFR_SG"                   => "device.subgroup_pref",     Literal, Error, "16",     migrated;

    // ── sampling (§6.2) ──────────────────────────────────────────────────────
    "INFR_TEMP"                 => "sampling.temp",       Float,      Ignored, "0.6",  migrated;
    "INFR_TOP_K"                => "sampling.top_k",      Int,        Ignored, "40",   migrated;
    "INFR_TOP_P"                => "sampling.top_p",      Float,      Ignored, "0.9",  migrated;
    "INFR_SEED"                 => "sampling.seed",       Int,        Ignored, "42",   migrated;
    "INFR_MAX_NEW"              => "sampling.max_new",    Int,        Ignored, "256",  migrated;
    "INFR_IGNORE_EOS"           => "sampling.ignore_eos", Presence,   Ignored, "1",    migrated;
    "INFR_NO_THINK"             => "sampling.no_think",   SetNotZero, Ignored, "1",    migrated;

    // ── kv (§6.3) ────────────────────────────────────────────────────────────
    "INFR_KV_TYPE_K"            => "kv.type_k",              Dtype,       Ignored, "q8_0", migrated;
    "INFR_KV_TYPE_V"            => "kv.type_v",              Dtype,       Ignored, "q8_0", migrated;
    "INFR_KV_Q8"                => "kv.force_q8",            Presence,    Ignored, "1",    migrated;
    "INFR_KV_SLOTS"             => "kv.slots",               Int,         Ignored, "8",    migrated;
    "INFR_NO_KV_RING"           => "kv.ring",                PresenceInv, Ignored, "1",    migrated;
    "INFR_KV_INLINE"            => "kv.inline_decode",       Presence,    Ignored, "1",    migrated;
    "INFR_KV_COOPMAT_BDA"       => "kv.coopmat_bda",         Presence,    Ignored, "1",    migrated;
    "INFR_KV_OVERFLOW"          => "kv.overflow",            Flag,        Ignored, "1",    migrated;
    "INFR_KV_OVERFLOW_VRAM_MB"  => "kv.overflow_vram_mb",    Mib,         Ignored, "512",  migrated;
    "INFR_KV_OVERFLOW_RESERVE_MB" => "kv.overflow_reserve_mb", Mib,       Ignored, "128",  migrated;

    // ── paging (§6.4) ────────────────────────────────────────────────────────
    "INFR_CACHE"                => "paging.cache",                     Size,     Ignored, "8g",  migrated;
    "INFR_PAGER_RING"           => "paging.ring",                      Size,     Ignored, "1g",  migrated;
    "INFR_PAGER_STATS"          => "paging.stats",                     Presence, Ignored, "1",   migrated;
    "INFR_DRAM_CACHE"           => "paging.dram",                      Size,     Ignored, "8g",  migrated;
    "INFR_DRAM_BYPASS"          => "paging.dram_bypass",               Flag,     Ignored, "1",   migrated;
    "INFR_LAYER_MAJOR"          => "paging.layer_major",               TriState, Ignored, "1",   migrated;

    // ── kernels.vulkan — coopmat / capability masking (§6.5, §5.2) ───────────
    "INFR_NO_COOPMAT"   => "kernels.vulkan.coopmat",      PresenceInv, Ignored, "1", migrated;
    "INFR_CM_8X8"       => "kernels.vulkan.coopmat_8x8",  Presence,    Ignored, "1", migrated;
    "INFR_BF16_COOPMAT" => "kernels.vulkan.bf16_coopmat", Presence,    Ignored, "1", migrated;
    "INFR_F8_COOPMAT"   => "kernels.vulkan.f8_coopmat",   Presence,    Ignored, "1", migrated;
    "INFR_F8_PREPACK"   => "kernels.vulkan.f8_prepack",   Presence,    Ignored, "1", migrated;
    "INFR_I8_COOPMAT"   => "kernels.vulkan.i8_coopmat",   Presence,    Ignored, "1", migrated;
    "INFR_I8_ROW_SCALE" => "kernels.vulkan.i8_row_scale", Presence,    Ignored, "1", migrated;
    "INFR_NO_F16"       => "kernels.vulkan.f16",          PresenceInv, Ignored, "1", migrated;
    "INFR_NO_I8DOT"     => "kernels.vulkan.i8_dot",       PresenceInv, Ignored, "1", migrated;

    // ── kernels.vulkan — GEMM / GEMV tiers (§6.5) ────────────────────────────
    "INFR_NO_GEMM_WARP"     => "kernels.vulkan.gemm_warp",      PresenceInv, Ignored, "1",  migrated;
    "INFR_GEMM_WIDE_TILE"   => "kernels.vulkan.gemm_wide_tile", Presence,    Ignored, "1",  migrated;
    "INFR_GEMM_DIRECT_A"    => "kernels.vulkan.gemm_direct_a",  Presence,    Ignored, "1",  migrated;
    "INFR_NO_SMALL_BM"      => "kernels.vulkan.small_bm",       PresenceInv, Ignored, "1",  migrated;
    "INFR_NO_BM16"          => "kernels.vulkan.bm16",           PresenceInv, Ignored, "1",  migrated;
    "INFR_NO_MMQ"           => "kernels.vulkan.mmq",            PresenceInv, Ignored, "1",  migrated;
    "INFR_NO_MMQ_FALLBACK"  => "kernels.vulkan.mmq_fallback",   PresenceInv, Ignored, "1",  migrated;
    "INFR_NO_MMV"           => "kernels.vulkan.mmv",            PresenceInv, Ignored, "1",  migrated;
    "INFR_MMV_DECODE"       => "kernels.vulkan.mmv_decode",     Presence,    Ignored, "1",  migrated;
    "INFR_NO_MMV_M4"        => "kernels.vulkan.mmv_m4",         PresenceInv, Ignored, "1",  migrated;
    "INFR_NO_MMV_O4"        => "kernels.vulkan.mmv_o4",         PresenceInv, Ignored, "1",  migrated;
    "INFR_MMV_MW"           => "kernels.vulkan.mmv_mw",         TriState,    Ignored, "0",  migrated;
    "INFR_MMV_MW_WARPS"     => "kernels.vulkan.mmv_mw_warps",   Int,         Ignored, "4",  migrated;
    "INFR_MMV_FUSE_QUANT"   => "kernels.vulkan.mmv_fuse_quant", Presence,    Ignored, "1",  migrated;
    "INFR_NO_MROW"          => "kernels.vulkan.mrow",           PresenceInv, Ignored, "1",  migrated;
    "INFR_NO_MROW16"        => "kernels.vulkan.mrow16",         PresenceInv, Ignored, "1",  migrated;
    "INFR_NO_F32_MROW"      => "kernels.vulkan.f32_mrow",       PresenceInv, Ignored, "1",  migrated;
    "INFR_NO_F32_V4"        => "kernels.vulkan.f32_v4",         PresenceInv, Ignored, "1",  migrated;
    "INFR_MOE_SMALL_M"      => "kernels.vulkan.moe_small_m",    Int,         Ignored, "16", migrated;
    "INFR_CANVAS_CHUNK_N"   => "kernels.vulkan.canvas_chunk_n", Int,         Ignored, "5",  migrated;

    // ── kernels.vulkan — attention (§6.5) ────────────────────────────────────
    "INFR_FLASH_SPLITS"   => "kernels.vulkan.flash_splits",     Int,              Ignored, "2",  migrated;
    "INFR_FLASH_BM"       => "kernels.vulkan.flash_bm32",       Literal,          Ignored, "32", migrated;
    "INFR_FLASH_MIN_ROWS" => "kernels.vulkan.flash_min_rows",   Int,              Ignored, "8",  migrated;
    "INFR_FLASH_STAGE"    => "kernels.vulkan.flash_stage",      Presence,         Ignored, "1",  migrated;
    "INFR_FLASH_DEQUANT"  => "kernels.vulkan.flash_dequant",    Presence,         Ignored, "1",  migrated;
    "INFR_NO_FLASH_WARP"  => "kernels.vulkan.flash_warp",       PresenceInv,      Ignored, "1",  migrated;
    "INFR_NO_NC_FA"       => "kernels.vulkan.nc_fa",            PresenceInv,      Ignored, "1",  migrated;
    "INFR_NO_QK_WARP"     => "kernels.vulkan.qk_warp",          PresenceInv,      Ignored, "1",  migrated;
    "INFR_NO_PV_WARP"     => "kernels.vulkan.pv_warp",          PresenceInv,      Ignored, "1",  migrated;
    "INFR_PV_SPLITS"      => "kernels.vulkan.pv_splits",        Int,              Ignored, "2",  migrated;
    "INFR_NO_ATTN_HD"     => "kernels.vulkan.no_attn_hd_spec",  Presence,         Ignored, "1",  migrated;
    "INFR_NO_ATTN_DECODE" => "kernels.vulkan.attn_decode",      PresenceInv,      Ignored, "1",  migrated;
    "INFR_NO_MROWS_ATTN"  => "kernels.vulkan.mrows_attn",       PresenceOptFalse, Ignored, "1",  migrated;
    "INFR_MROWS_ATTN"     => "kernels.vulkan.mrows_attn",       PresenceOptTrue,  Ignored, "1",  migrated;

    // ── kernels.vulkan — DeltaNet + misc (§6.5) ──────────────────────────────
    "INFR_DN_CHUNK_SCAN"      => "kernels.vulkan.dn_chunk_scan",      PresenceInv, Ignored, "1", migrated;
    "INFR_NO_DN_CHUNK"        => "kernels.vulkan.dn_chunk",           PresenceInv, Ignored, "1", migrated;
    "INFR_NO_DN_SPLIT"        => "kernels.vulkan.dn_split",           PresenceInv, Ignored, "1", migrated;
    "INFR_DELTA_STRIDED"      => "kernels.vulkan.delta_strided",      Presence,    Ignored, "1", migrated;
    "INFR_NO_PUSH_DESC"       => "kernels.vulkan.push_desc",          PresenceInv, Ignored, "1", migrated;
    "INFR_NO_PIPELINE_CACHE"  => "kernels.vulkan.pipeline_cache_disk", PresenceInv, Ignored, "1", migrated;
    "INFR_NO_VRAM_GUARD"      => "kernels.vulkan.no_vram_guard",      Presence,    Ignored, "1", migrated;
    "INFR_NO_MOE_SM_POOL"     => "kernels.vulkan.no_moe_sm_pool",     Presence,    Ignored, "1", migrated;
    "INFR_SEAM_NO_REPLAY"     => "kernels.vulkan.no_replay",          Presence,    Ignored, "1", migrated;
    "INFR_NO_GPU_POS"         => "kernels.vulkan.gpu_pos",            PresenceInv, Ignored, "1", migrated;
    "INFR_NO_FUSE_ADD"        => "kernels.vulkan.fuse_add",           PresenceInv, Ignored, "1", migrated;

    // ── kernels.vulkan — BDA chunk caps (§6.5c) ──────────────────────────────
    "INFR_BDA_CHUNK_ELEMS" => "kernels.vulkan.bda_chunk_elems", Int, Ignored, "1024", migrated;
    "INFR_BDA_CHUNK_BYTES" => "kernels.vulkan.bda_chunk_bytes", Int, Ignored, "4096", migrated;

    // ── kernels.vulkan.gemv (§6.5b) ──────────────────────────────────────────
    "INFR_NO_GEMV_RM"     => "kernels.vulkan.gemv.no_rm",     Presence,       Ignored, "1",     migrated;
    "INFR_GEMV_RM"        => "kernels.vulkan.gemv.rm",        Int,            Ignored, "4",     migrated;
    "INFR_GEMV_RM_MAXOUT" => "kernels.vulkan.gemv.rm_maxout", Int,            Ignored, "16384", migrated;
    "INFR_GEMV_RM_MINOUT" => "kernels.vulkan.gemv.rm_minout", Int,            Ignored, "1024",  migrated;
    "INFR_NO_GEMV_SG"     => "kernels.vulkan.gemv.no_sg",     Presence,       Ignored, "1",     migrated;
    "INFR_NO_GEMV_ID_SG"  => "kernels.vulkan.gemv.no_id_sg",  Presence,       Ignored, "1",     migrated;
    "INFR_GEMV_SG_MINOUT" => "kernels.vulkan.gemv.sg_minout", Int,            Ignored, "1024",  migrated;
    "INFR_GEMV_SG_MAXOUT" => "kernels.vulkan.gemv.sg_maxout", Int,            Ignored, "4096",  migrated;
    "INFR_GEMV_SG_NR"     => "kernels.vulkan.gemv.sg_nr",     Int,            Ignored, "4",     migrated;
    "INFR_NO_GEMV_REG"    => "kernels.vulkan.gemv.variant",   PresenceClears, Ignored, "1",     migrated;
    "INFR_GEMV_VARIANT"   => "kernels.vulkan.gemv.variant",   Text,           Ignored, "rm",    migrated;

    // ── kernels.metal (§6.6) ─────────────────────────────────────────────────
    "INFR_METAL_NO_F16_NATIVE"    => "kernels.metal.f16_native",    PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_F32_NATIVE"    => "kernels.metal.f32_native",    PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_BF16_NATIVE"   => "kernels.metal.bf16_native",   PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_F16_CMM"       => "kernels.metal.f16_cmm",       PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_BF16_CMM"      => "kernels.metal.bf16_cmm",      PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_F32_CMM"       => "kernels.metal.f32_cmm",       PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_F16_RT"        => "kernels.metal.f16_rt",        PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_BF16_RT"       => "kernels.metal.bf16_rt",       PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_F32_RT"        => "kernels.metal.f32_rt",        PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_KQUANT_NATIVE" => "kernels.metal.kquant_native", PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_Q5K_RT"        => "kernels.metal.q5k_rt",        PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_RMSNORM_VEC4"  => "kernels.metal.rmsnorm_vec4",  PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_CONV1D_PAR"    => "kernels.metal.conv1d_par",    PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_DN_GATE_PREP"  => "kernels.metal.dn_gate_prep",  PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NO_DN_NORM_PREP"  => "kernels.metal.dn_norm_prep",  PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_LMHEAD_MRV"       => "kernels.metal.lmhead_mrv_uncapped", Presence, Ignored, "1", migrated;
    "INFR_METAL_NODELTA"          => "kernels.metal.deltanet",      PresenceInv, Ignored, "1", migrated;
    "INFR_METAL_NOMOE"            => "kernels.metal.moe",           PresenceInv, Ignored, "1", migrated;

    // ── kernels.cpu (§6.7) ───────────────────────────────────────────────────
    "INFR_CPU_SPIN"        => "kernels.cpu.spin",      Int,           Ignored, "4096", migrated;
    "INFR_CPU_NO_SPINPOOL" => "kernels.cpu.spinpool",  SetNotZeroInv, Ignored, "1",    migrated;
    "INFR_CPU_REPACK_MB"   => "kernels.cpu.repack_mb", Int,           Ignored, "1024", migrated;

    // ── kernels — graph shape, `infr-llama` (§6.9) ───────────────────────────
    "INFR_NO_QKV_FUSE"      => "kernels.qkv_fuse",      PresenceInv, Ignored, "1", migrated;
    "INFR_NO_GATED_RMSNORM" => "kernels.gated_rmsnorm", PresenceInv, Ignored, "1", migrated;

    // ── spec (§6.8) ──────────────────────────────────────────────────────────
    "INFR_MTP"                => "spec.mtp",             Literal,     Ignored, "1",         migrated;
    "INFR_NO_MTP_CKPT"        => "spec.mtp_ckpt",        PresenceInv, Ignored, "1",         migrated;
    "INFR_NO_MTP_REPRIME"     => "spec.mtp_reprime",     PresenceInv, Ignored, "1",         migrated;
    "INFR_NO_MTP_DRAFT_CHAIN" => "spec.mtp_draft_chain", PresenceInv, Ignored, "1",         migrated;
    "INFR_SPEC_DRAFT"         => "spec.draft",           Path,        Ignored, "/tmp/d.gguf", migrated;
    "INFR_SPEC_K"             => "spec.k",               Int,         Ignored, "4",         migrated;
    "INFR_SPEC_DEBUG"         => "spec.debug",           Presence,    Ignored, "1",         migrated;
    "INFR_DECODE_CHAIN"       => "spec.decode_chain",    Int,         Ignored, "4",         migrated;
    "INFR_NO_GPU_DRAFT_PROB"  => "spec.gpu_draft_prob",  PresenceInv, Ignored, "1",         migrated;
    "INFR_NO_GPU_MTP_ACCEPT"  => "spec.gpu_mtp_accept",  PresenceInv, Ignored, "1",         migrated;
    "INFR_NO_GPU_ARGMAX"      => "spec.gpu_argmax",      PresenceInv, Ignored, "1",         migrated;
    "INFR_NO_GPU_SAMPLE"      => "spec.gpu_sample",      PresenceInv, Ignored, "1",         migrated;
    "INFR_NO_GPU_EMBED"       => "spec.gpu_embed",       PresenceInv, Ignored, "1",         migrated;

    // ── multi (§6.11) ────────────────────────────────────────────────────────
    "INFR_PIPELINE"        => "multi.pipeline",        DeviceList,  Error,   "0,1", migrated;
    "INFR_TENSOR_PARALLEL" => "multi.tensor_parallel", DeviceList,  Error,   "0,1", migrated;
    "INFR_EXPERT_PARALLEL" => "multi.expert_parallel", DeviceList,  Error,   "0",   migrated;
    "INFR_PIPELINE_HOST"   => "multi.pipeline_p2p",    PresenceInv, Ignored, "1",   migrated;
    "INFR_TP_HOST"         => "multi.tp_p2p",          PresenceInv, Ignored, "1",   migrated;
    "INFR_EP_HOST"         => "multi.ep_p2p",          PresenceInv, Ignored, "1",   migrated;

    // ── prof (§6.9) ──────────────────────────────────────────────────────────
    // One `INFR_PROF_*` prefix, and each name says what it does. Thirteen keys became eight: the
    // per-op profile had four (INFR_PROF2 / INFR_PROF_OPS / INFR_PROF2_SHAPES / INFR_METAL_PROFILE)
    // and host-side stage timing had five, one per pipeline (INFR_PROF, INFR_PROF_DEC,
    // INFR_PROF_PF, INFR_MTP_TIME, INFR_DIFFUSION_TIME). Old spellings are dropped CLEANLY.
    "INFR_PROF_OPS"                => "prof.ops",               Presence, Ignored, "1",           migrated;
    "INFR_PROF_OP_SHAPES"          => "prof.op_shapes",         Presence, Ignored, "1",           migrated;
    "INFR_PROF_STAGES"             => "prof.stages",            Presence, Ignored, "1",           migrated;
    "INFR_PROF_VRAM"               => "prof.vram",              Presence, Ignored, "1",           migrated;
    "INFR_PROF_DIFFUSION_TRACE"    => "prof.diffusion_trace",   Presence, Ignored, "1",           migrated;
    "INFR_PROF_OUT"                => "prof.out",               Path,     Ignored, "/tmp/p.json", migrated;
    "INFR_PROF_METAL_DEVICE_TIME"  => "prof.metal_device_time", Text,     Ignored, "counters",    migrated;
    "INFR_PROF_METAL_DEBUG"        => "prof.metal_debug",       Presence, Ignored, "1",           migrated;

    // ── debug (§6.9) ─────────────────────────────────────────────────────────
    "INFR_DEBUG_BDA_CHUNK"     => "debug.bda_chunk",       Presence, Ignored, "1", migrated;
    "INFR_DEBUG_COOPMAT"       => "debug.coopmat",         Presence, Ignored, "1", migrated;
    "INFR_DEBUG_WIDE_DISPATCH" => "debug.wide_dispatch",   Presence, Ignored, "1", migrated;
    "INFR_DEBUG_CHAT"          => "debug.chat",            Presence, Ignored, "1", migrated;
    "INFR_MOE_COUNTS_DEBUG"    => "debug.moe_counts",      Presence, Ignored, "1", migrated;
    "INFR_MOE_COUNTS_DUMP"     => "debug.moe_counts_dump", Presence, Ignored, "1", migrated;
    "INFR_POISON_UNINIT"       => "debug.poison_uninit",   Presence, Ignored, "1", migrated;
    "INFR_NOBARRIER"           => "debug.no_barrier",      Presence, Ignored, "1", migrated;
    "INFR_FULLBARRIER"         => "debug.full_barrier",    Presence, Ignored, "1", migrated;

    // ── serve (§6.9) ─────────────────────────────────────────────────────────
    "INFR_API_KEY"        => "serve.api_key",        Text, Ignored, "hunter2", migrated;
    "INFR_MAX_TOKENS_CAP" => "serve.max_tokens_cap", Int,  Ignored, "4096",    migrated;
    // `0`/unset = no deadline, so the env layer does NOT filter non-positive values the way
    // `INFR_MAX_TOKENS_CAP` does — see `ServeCfg::request_timeout_secs`.
    "INFR_REQUEST_TIMEOUT_SECS" => "serve.request_timeout_secs", Int, Ignored, "300", migrated;
    // Same "`0` is a VALUE, not a bad one" grammar as the deadline above: `0` disables the periodic
    // throughput line, so the env layer must not filter it out.
    "INFR_SERVE_STATS_SECS" => "serve.stats_interval_secs", Int, Ignored, "10", migrated;
}

/// `INFR_*` keys that exist in `crates/*/src` (or a `build.rs`) and deliberately do NOT become
/// configuration (§6.10). Listed here so the drift test can tell "excluded on purpose" from
/// "someone added a knob and forgot the manifest".
pub const NOT_MIGRATED: &[(&str, &str)] = &[
    (
        "INFR_PROFILE",
        "build-time input: read by build.rs in core/cpu/gguf/llama/vulkan to set cfg(infr_profile). \
         A runtime Config cannot exist when it is read (§5.3).",
    ),
    (
        "INFR_LLAMA_DIFFUSION_CLI",
        "test/dev fixture: points at a llama.cpp binary on disk (cli/main.rs).",
    ),
    (
        "INFR_DIFFUSION_VISUAL",
        "CLI presentation only, NOT a Config field: since S7 it is `infr run --diffusion-visual`, \
         a plain clap flag whose `env =` fallback keeps the old spelling working. The literal that \
         remains in `cli/main.rs` is that clap attribute, not a `std::env::var` read.",
    ),
];

/// `INFR_*` spellings that the §6.0 filter drops and that are NOT knobs at all.
///
/// - `INFR_*_TEST_*` — `#[cfg(test)]` fixtures that exist only to prove a wrapper reads its own
///   variable; they vanish with the wrapper.
/// - `INFR_TP` / `INFR_EP` — `label` strings passed to `parse_device_spec` for error messages, not
///   variables. The real keys are `INFR_TENSOR_PARALLEL` / `INFR_EXPERT_PARALLEL`.
/// - `INFR_CPU` / `INFR_METAL` — DEAD. Removed as backend selectors; the only references left are
///   `cli/main.rs`'s `legacy_metal_cpu_flags_are_no_longer_read` regression test and doc comments.
///   Do not add them to `Config`; do not remove that test.
pub const NOT_KNOBS: &[&str] = &["INFR_TP", "INFR_EP", "INFR_CPU", "INFR_METAL"];
