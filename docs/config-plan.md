# config-plan.md — replace the `INFR_*` env gates with a layered `Config`

**Status: PLAN ONLY. No code has been written for this yet.** Every section
below is prescriptive: follow it literally. Where a decision is still open it is
marked **[DECIDE]** and must be answered by the repo owner before the slice that
needs it starts.

## 1. The problem

Runtime behaviour is currently steered by **138 distinct `INFR_*` environment
variables**, read by ~155 `std::env::var` call sites spread over `infr-vulkan`
(~74), `infr-llama` (~54), `infr-metal` (~26), `infr-cli` (~21), `infr-core`
(~15), `infr-cpu` (~7), `infr-rocm` (~4), plus `infr-server`, `infr-chat`,
`infr-gguf`, `infr-hub`, `infr-prof-rt`. The full inventory is §6.

Concrete consequences, all observed in this repo:

1. **Tests cannot set behaviour without mutating global state.** `cargo test`
   runs a binary's tests on a thread pool; two tests writing the same variable
   interleave. Commit `b9069a3` had to add a process-wide lock + restore guard
   (`infr_core::test_env::EnvGuard`) to 11 files because `infr-vulkan`'s
   `attn_flash_stage_dequant_parity` and `attn_flash_warp_dequant_parity` both
   drive `INFR_FLASH_SPLITS` and clobbered each other. That guard is a
   **workaround**; this plan removes the need for it.
2. **A memoized read is unsettable.** Two knobs cached their value in a
   `OnceLock` (`INFR_PAGER_STATS`, `INFR_NO_ATTN_HD`), so the first reader
   pinned the value process-wide and any later test silently measured nothing.
   Both were de-memoized in `b9069a3` — at the cost of a `getenv` on a selection
   path.
3. **No discoverability.** There is no `--help` listing, no config file, no way
   to see what is set. The knobs are documented in scattered doc comments and
   `README.md`.
4. **No validation.** A typo (`INFR_FLASH_SPLIT=2`) is silently ignored. A bad
   value (`INFR_MOE_SMALL_M=100000`) once hung the GPU; it now clamps, but only
   because someone remembered to clamp it at that one site.
5. **The CLI already fights this.** `DeviceOpts::resolve` /
   `SamplingOpts::resolve` in `crates/infr-cli/src/main.rs` take clap flags and
   **write them back into the process env** (`std::env::set_var("INFR_CTX", …)`)
   purely so that code deep in the seam can read them. That is the tell: the
   value already exists as a typed thing at startup and is being laundered
   through a string table to cross an API boundary.

## 2. The target

One typed, immutable, explicitly-passed configuration value, resolved once, from
four layers.

**Precedence (later wins):**

| #           | Layer       | Source                    | Notes                                                                      |
| ----------- | ----------- | ------------------------- | -------------------------------------------------------------------------- |
| 4 (lowest)  | `Default`   | `impl Default for Config` | The shipped behaviour. Must reproduce today's defaults EXACTLY.            |
| 3           | Config file | TOML (§4)                 | Absent file = no-op, never an error. A malformed file IS an error.         |
| 2           | Environment | `INFR_*`                  | Same variable names as today, unchanged, so existing scripts keep working. |
| 1 (highest) | CLI flags   | clap                      | An explicitly-passed flag always wins.                                     |

**Rule:** a layer only overrides a field it actually specifies. Layers are
parsed into `Option<T>`-shaped "partial" structs and folded; a `None` never
overwrites a `Some`.

```rust
let cfg = Config::builder()
    .layer(ConfigLayer::file(path)?)   // may be empty
    .layer(ConfigLayer::env())         // reads INFR_*
    .layer(ConfigLayer::cli(&args))    // clap → partial
    .build();                          // fills the rest from Default
```

**Consumption:**

```rust
// production
let cfg = Arc::new(Config::load(&cli_args)?);   // once, in main()
let be  = VulkanBackend::new_with(cfg.clone())?;

// tests — no env, no lock, no leak, no ordering hazard
let cfg = Config { kernels: KernelCfg { vulkan: VulkanCfg { flash_splits: Some(2), ..Default::default() },
                                        ..Default::default() }, ..Default::default() };
let be  = VulkanBackend::new_with(Arc::new(cfg))?;
```

## 3. Non-negotiable rules

These are the guardrails for every slice. A slice that breaks one is wrong even
if it compiles.

- **R1 — Behaviour-preserving.** This is a refactor. Every default must
  reproduce today's behaviour bit-for-bit. The goldens must not move: qwen3 CPU
  golden `0xfd63781ea3bfa785`, `gpu_seam` 27/27, `rocm_seam` 9/9,
  `cpu_golden_qwen3_quants` (8 hashes).
- **R2 — Env names are frozen.** `INFR_FLASH_SPLITS` stays spelled
  `INFR_FLASH_SPLITS`. This plan changes _where the value goes_, not what users
  type. No renames, no deprecations, in any slice. (A rename campaign, if
  wanted, is a separate plan AFTER this one lands.)
- **R3 — One read point per knob.** After a slice, the knob's `std::env::var`
  appears exactly once, inside `infr-core/src/config/env.rs`. Nowhere else.
  `grep -rn 'env::var("INFR_' crates/*/src` is the check.
- **R4 — No new globals.** Do not replace `std::env::var` with a
  `static CONFIG: OnceLock`. The point is to make configuration an explicit
  value. (See §5 for the one narrowly-scoped exception and why it is temporary.)
- **R5 — No behaviour in the layer parsers.** `env.rs` only turns strings into
  `Option<T>`. Clamping, defaulting and policy live in the typed accessor (e.g.
  `tier::EnvRows::resolve` already has exactly this shape — keep it).
- **R6 — Hot paths take a borrow, not a clone.** `&Config` or `&VulkanCfg`
  threaded from the owning struct. Never `cfg.clone()` inside a per-op or
  per-dispatch path.
- **R7 — Tests never touch the environment.** After a knob migrates, every test
  that drove it via `EnvGuard` must be rewritten to build a `Config`. The
  `EnvGuard` use for that knob is deleted in the same slice.
  `infr_core::test_env` itself is deleted in the final slice (S9).
- **R8 — One slice, one commit, one crate (or one subsystem).** Conventional
  Commits. Stage explicit files. No AI attribution.

## 4. Module layout and file format

New module tree in `infr-core` (chosen because every other crate already depends
on it, and because `DType`/`SizeSpec`/`parse_size` — the value types the knobs
parse into — live there):

```
crates/infr-core/src/config/
    mod.rs        // `Config`, the section structs, `Default`, `Config::load`, the builder/fold
    partial.rs    // `PartialConfig` + per-section partials (every field Option<T>), `merge()`
    env.rs        // THE ONLY place `std::env::var("INFR_*")` appears; env → PartialConfig
    file.rs       // TOML → PartialConfig; path discovery
    cli.rs        // a `ConfigOverrides` struct the CLI fills; → PartialConfig
    tests.rs      // precedence tests (§8)
```

Section structs (one per subsystem; names are fixed by this plan so slices do
not diverge):

```rust
pub struct Config {
    pub device:   DeviceCfg,    // INFR_DEV, INFR_CTX, INFR_UBATCH, threads
    pub sampling: SamplingCfg,  // INFR_TEMP, INFR_TOP_K, INFR_TOP_P, INFR_SEED, INFR_MAX_NEW, ...
    pub kv:       KvCfg,        // INFR_KV_TYPE_K/V, INFR_KV_Q8, INFR_KV_SLOTS, INFR_KV_INLINE, ...
    pub paging:   PagingCfg,    // INFR_CACHE, INFR_PAGER_RING, INFR_PAGER_STATS, ROCm prefetch
    pub kernels:  KernelCfg,    // per-backend kernel tier knobs (see below)
    pub spec:     SpecCfg,      // MTP / speculative decode
    pub prof:     ProfCfg,      // INFR_PROF*, INFR_VRAM_LOG, INFR_*_DEBUG_*
    pub serve:    ServeCfg,     // INFR_API_KEY, INFR_MAX_TOKENS_CAP
    pub debug:    DebugCfg,     // poison/barrier/dump switches
}
pub struct KernelCfg {
    pub vulkan: VulkanCfg,
    pub metal:  MetalCfg,
    pub rocm:   RocmCfg,
    pub cpu:    CpuCfg,
}
```

**File format: TOML.** Section path = struct path.

```toml
[device]
dev = "vulkan1"
ctx = "32k"          # the shared size grammar: 8192 / 256k / 50%

[kv]
type_k = "q8_0"

[kernels.vulkan]
flash_splits = 2
no_gemm_warp = false
```

**Lookup order for the file layer (first existing file wins, no merging of
multiple files):**

1. `--config <PATH>` (error if the path does not exist)
2. `./infr.toml`
3. `$XDG_CONFIG_HOME/infr/config.toml`, else `~/.config/infr/config.toml`

**[DECIDE-1]** Confirm TOML + this 3-step lookup. Alternative considered: global
path only (one fewer precedence rung to explain).

## 5. How `Config` reaches the reader

This is the one genuinely invasive part, because knobs are read deep inside hot
paths (`recorder.rs`, `gemm.rs`, `adapter.rs`, `exec.rs`).

**Chosen design: ownership by the backend, borrow at the read site.**

- `VulkanBackend`, `CpuBackend`, `MetalBackend`, `RocmBackend` each gain a
  private `cfg: Arc<Config>` field, set by a new constructor
  `new_with(cfg: Arc<Config>)`. The existing `new()` becomes
  `new_with(Arc::new(Config::default()))` — so every current caller keeps
  compiling and gets today's behaviour.
- `Recorder` (Vulkan) and the per-forward exec state (Metal/ROCm/CPU) borrow
  `&Config` from the backend that created them; they already hold a backend
  reference.
- `SeamModel` / `DenseSession<B, X>` gain the same `Arc<Config>` so the seam
  runner (`infr-llama/src/seam/runner.rs`, `mod.rs`) can read `cfg.kv`,
  `cfg.spec`, `cfg.device.ubatch` without an env read.
- `infr-cli` builds the `Config` in `main()` and passes it down.
  `DeviceOpts::resolve` / `SamplingOpts::resolve` **stop writing env vars** and
  instead fill a `ConfigOverrides`.

**Transitional exception to R4 (delete before the campaign closes):** during
S2–S7 a knob that has not been migrated yet still reads env at its old site.
That is fine and expected — do NOT introduce a global to bridge them. If a slice
finds a read site with no plausible owner in scope, STOP and record it in §10
"blocked sites" rather than inventing a global.

**[DECIDE-2]** Confirm the threading approach over the cheaper alternative (a
single `OnceLock<Config>` + a thread-local test override). Threading is more
work (~155 sites plus constructors) but is the only version that makes
configuration a value rather than ambient state.

## 6. Knob inventory

Every `INFR_*` key read outside `tests/`, its grammar as implemented today, and
its destination. `presence` = `is_ok()` (set to anything, including empty, ⇒
on). `presence-inv` = `is_err()` (the _absence_ enables the feature). Both map
to `bool` in the config, with the polarity noted so the env layer inverts where
needed.

**Migrating a `presence-inv` knob is the single most likely place to introduce a
behaviour change. Write the truth table in the PR description before changing
the code.**

### 6.1 `device` — `DeviceCfg`

| Env                      | Grammar            | Config path                               | Default           | Read sites                                                           |
| ------------------------ | ------------------ | ----------------------------------------- | ----------------- | -------------------------------------------------------------------- |
| `INFR_DEV`               | string             | `device.dev: Option<String>`              | `None`            | cli/main.rs:91,3845,3859; vulkan/lib.rs                              |
| `INFR_CTX`               | size               | `device.ctx: Option<SizeSpec>`            | `None`            | cli/main.rs:3481; llama/chat/mod.rs:164; llama/chat/diffusion.rs:140 |
| `INFR_UBATCH`            | int                | `device.ubatch: Option<usize>`            | `None` (adaptive) | llama/seam/mod.rs:1254,1292,1354                                     |
| `INFR_UBATCH_PARALLEL`   | int, default 256   | `device.ubatch_parallel: usize`           | `256`             | llama/seam/mod.rs:613                                                |
| `INFR_SUBMIT_DISPATCHES` | int-ish string     | `device.submit_dispatches: Option<usize>` | `None`            | vulkan/lib.rs:1980                                                   |
| `INFR_SG`                | int-ish string     | `device.subgroup_pref: Option<u32>`       | `None`            | vulkan/lib.rs:1852                                                   |
| `RAYON_NUM_THREADS`      | int (NOT `INFR_*`) | `device.threads: Option<usize>`           | `None`            | set by cli; read by rayon                                            |

`RAYON_NUM_THREADS` is third-party: keep publishing it as an env var from the
CLI (rayon has no other input), but source the value from `cfg.device.threads`.

### 6.2 `sampling` — `SamplingCfg`

| Env               | Grammar           | Config path                 | Default        | Read sites                                                        |
| ----------------- | ----------------- | --------------------------- | -------------- | ----------------------------------------------------------------- |
| `INFR_TEMP`       | f32, default 0.0  | `sampling.temp: f32`        | `0.0` (greedy) | llama/sampling.rs:44; cli/main.rs:3359,3447                       |
| `INFR_TOP_K`      | usize, default 20 | `sampling.top_k: usize`     | `20`           | llama/sampling.rs:45                                              |
| `INFR_TOP_P`      | f32, default 0.95 | `sampling.top_p: f32`       | `0.95`         | llama/sampling.rs:46                                              |
| `INFR_SEED`       | int, default 42   | `sampling.seed: u64`        | `42`           | llama/sampling.rs:383; cli/main.rs:2306                           |
| `INFR_MAX_NEW`    | int, default 2048 | `sampling.max_new: usize`   | `2048`         | cli/main.rs:1657                                                  |
| `INFR_IGNORE_EOS` | presence          | `sampling.ignore_eos: bool` | `false`        | cli/main.rs:2301; llama/seam/model.rs:1593; llama/mtp/mod.rs:2380 |
| `INFR_NO_THINK`   | `!=0`             | `sampling.no_think: bool`   | `false`        | chat/template.rs:220                                              |

`Sampler::from_env()` (llama/sampling.rs) becomes
`Sampler::from_cfg(&SamplingCfg)`. Its per-request override path
(`Sampler::resolve(req)`) is unchanged — `RequestCtx` still wins over config,
exactly as it wins over env today.

### 6.3 `kv` — `KvCfg`

| Env                        | Grammar        | Config path                        | Default | Read sites                                         |
| -------------------------- | -------------- | ---------------------------------- | ------- | -------------------------------------------------- |
| `INFR_KV_TYPE_K`           | dtype name     | `kv.type_k: Option<DType>`         | `None`  | llama/seam/mod.rs:725; runner.rs                   |
| `INFR_KV_TYPE_V`           | dtype name     | `kv.type_v: Option<DType>`         | `None`  | llama/seam/mod.rs:726                              |
| `INFR_KV_Q8`               | presence       | `kv.force_q8: bool`                | `false` | llama/seam/{mod.rs:727,model.rs:341,runner.rs:464} |
| `INFR_KV_SLOTS`            | int, default 4 | `kv.slots: usize`                  | `4`     | llama/seam/model.rs:200                            |
| `INFR_NO_KV_RING`          | presence-inv   | `kv.ring: bool`                    | `true`  | llama/seam/mod.rs:828                              |
| `INFR_KV_INLINE`           | presence       | `kv.inline_decode: bool`           | `false` | vulkan/adapter.rs:2746                             |
| `INFR_KV_COOPMAT_BDA`      | presence       | `kv.coopmat_bda: bool`             | `false` | vulkan/adapter.rs:2883                             |
| `INFR_KV_OVERFLOW`         | flag           | `kv.overflow: bool`                | `false` | via `budget::env_flag`                             |
| `INFR_KV_OVERFLOW_VRAM_MB` | MiB            | `kv.overflow_vram_mb: Option<u64>` | `None`  | via `budget::env_mib`                              |

The K/V dtype parser already exists twice (`seam/runner.rs::parse_kv_fmt`,
`budget::parse_kv_dtype`); the config layer uses
`infr_core::budget::parse_kv_dtype` — do not write a third.

### 6.4 `paging` — `PagingCfg`

| Env                                     | Grammar  | Config path                                     | Default | Read sites                                           |
| --------------------------------------- | -------- | ----------------------------------------------- | ------- | ---------------------------------------------------- |
| `INFR_CACHE`                            | size     | `paging.cache: Option<SizeSpec>`                | `None`  | llama/seam/mod.rs:907                                |
| `INFR_PAGER_RING`                       | size     | `paging.ring: Option<SizeSpec>`                 | `None`  | core/pager.rs:276 (already split: `ring_bytes_from`) |
| `INFR_PAGER_STATS`                      | presence | `paging.stats: bool`                            | `false` | vulkan/pager.rs:489                                  |
| `INFR_ROCM_EXPERT_BUDGET`               | size     | `paging.rocm_expert_budget: Option<SizeSpec>`   | `None`  | llama/seam/mod.rs:173                                |
| `INFR_ROCM_WEIGHT_PREFETCH_SLOTS`       | int      | `paging.rocm_prefetch_slots: Option<usize>`     | `None`  | rocm/weight_pager.rs:92                              |
| `INFR_ROCM_WEIGHT_PREFETCH_MAX_BANK_MB` | MiB      | `paging.rocm_prefetch_max_bank_mb: Option<u64>` | `None`  | rocm/weight_pager.rs:81                              |
| `INFR_ROCM_WEIGHT_OVERFLOW`             | flag     | `paging.rocm_weight_overflow: bool`             | `false` | via `budget::env_flag`                               |

### 6.5 `kernels.vulkan` — `VulkanCfg` (the biggest group)

Coopmat / tier selection:

| Env                 | Grammar      | Config path          | Default |
| ------------------- | ------------ | -------------------- | ------- |
| `INFR_NO_COOPMAT`   | presence-inv | `coopmat: bool`      | `true`  |
| `INFR_CM_8X8`       | presence     | `coopmat_8x8: bool`  | `false` |
| `INFR_BF16_COOPMAT` | presence     | `bf16_coopmat: bool` | `false` |
| `INFR_F8_COOPMAT`   | presence     | `f8_coopmat: bool`   | `false` |
| `INFR_F8_PREPACK`   | presence     | `f8_prepack: bool`   | `false` |
| `INFR_I8_COOPMAT`   | presence     | `i8_coopmat: bool`   | `false` |
| `INFR_I8_ROW_SCALE` | presence     | `i8_row_scale: bool` | `false` |
| `INFR_NO_F16`       | presence-inv | `f16: bool`          | `true`  |
| `INFR_NO_I8DOT`     | presence-inv | `i8_dot: bool`       | `true`  |

GEMM / GEMV tiers:

| Env                                   | Grammar            | Config path                   | Default                       |
| ------------------------------------- | ------------------ | ----------------------------- | ----------------------------- |
| `INFR_NO_GEMM_WARP`                   | presence-inv       | `gemm_warp: bool`             | `true`                        |
| `INFR_GEMM_WIDE_TILE`                 | presence           | `gemm_wide_tile: bool`        | `false`                       |
| `INFR_NO_SMALL_BM`                    | presence-inv       | `small_bm: bool`              | `true`                        |
| `INFR_NO_BM16`                        | presence-inv       | `bm16: bool`                  | `true`                        |
| `INFR_NO_MMQ`                         | presence-inv       | `mmq: bool`                   | `true`                        |
| `INFR_NO_MMQ_FALLBACK`                | presence-inv       | `mmq_fallback: bool`          | `true`                        |
| `INFR_NO_MMV`                         | presence           | `no_mmv: bool`                | `false`                       |
| `INFR_MMV_DECODE`                     | presence           | `mmv_decode: bool`            | `false`                       |
| `INFR_NO_MMV_M4` / `INFR_NO_MMV_O4`   | presence-inv       | `mmv_m4` / `mmv_o4: bool`     | `true`                        |
| `INFR_MMV_MW`                         | `"0"`/`"1"` string | `mmv_mw: Option<bool>`        | `None` (vendor default)       |
| `INFR_MMV_MW_WARPS`                   | int                | `mmv_mw_warps: Option<usize>` | `None`                        |
| `INFR_NO_MROW` / `INFR_NO_MROW16`     | presence-inv       | `mrow` / `mrow16: bool`       | `true`                        |
| `INFR_NO_F32_MROW` / `INFR_NO_F32_V4` | presence-inv       | `f32_mrow` / `f32_v4: bool`   | `true`                        |
| `INFR_MOE_SMALL_M`                    | int, clamp 0..=64  | `moe_small_m: usize`          | `8` (already `tier::EnvRows`) |
| `INFR_CANVAS_CHUNK_N`                 | int, min 1         | `canvas_chunk_n: usize`       | `3` (already `tier::EnvRows`) |

Attention:

| Env                                      | Grammar             | Config path                   | Default                            |
| ---------------------------------------- | ------------------- | ----------------------------- | ---------------------------------- |
| `INFR_FLASH_SPLITS`                      | int                 | `flash_splits: Option<usize>` | `None`                             |
| `INFR_FLASH_BM`                          | `"32"`              | `flash_bm32: bool`            | `false`                            |
| `INFR_FLASH_MIN_ROWS`                    | int, default 24     | `flash_min_rows: usize`       | `24`                               |
| `INFR_FLASH_STAGE`                       | presence            | `flash_stage: bool`           | `false`                            |
| `INFR_FLASH_DEQUANT`                     | presence            | `flash_dequant: bool`         | `false`                            |
| `INFR_NO_FLASH_WARP`                     | presence-inv        | `flash_warp: bool`            | `true`                             |
| `INFR_NO_NC_FA`                          | presence-inv        | `nc_fa: bool`                 | `true`                             |
| `INFR_NO_QK_WARP` / `INFR_NO_PV_WARP`    | presence-inv        | `qk_warp` / `pv_warp: bool`   | `true`                             |
| `INFR_PV_SPLITS`                         | int                 | `pv_splits: Option<usize>`    | `None`                             |
| `INFR_NO_ATTN_HD`                        | presence            | `no_attn_hd_spec: bool`       | `false` (de-memoized in `b9069a3`) |
| `INFR_MROWS_ATTN` / `INFR_NO_MROWS_ATTN` | presence / presence | `mrows_attn: Option<bool>`    | `None`                             |

DeltaNet + misc:

| Env                                     | Grammar      | Config path                   | Default |
| --------------------------------------- | ------------ | ----------------------------- | ------- |
| `INFR_DN_CHUNK_SCAN`                    | presence-inv | `dn_chunk_scan: bool`         | `true`  |
| `INFR_NO_DN_CHUNK` / `INFR_NO_DN_SPLIT` | presence-inv | `dn_chunk` / `dn_split: bool` | `true`  |
| `INFR_DELTA_STRIDED`                    | presence     | `delta_strided: bool`         | `false` |
| `INFR_NO_PUSH_DESC`                     | presence-inv | `push_desc: bool`             | `true`  |
| `INFR_NO_VRAM_GUARD`                    | presence     | `no_vram_guard: bool`         | `false` |
| `INFR_SEAM_NO_REPLAY`                   | presence     | `no_replay: bool`             | `false` |
| `INFR_NO_GPU_POS`                       | presence-inv | `gpu_pos: bool`               | `true`  |

### 6.6 `kernels.metal` — `MetalCfg`

All 15 are `presence-inv` disable-switches except the two profile knobs. Config
fields are POSITIVE (`f16_cmm: bool`, default `true`); the env layer inverts.

`INFR_METAL_NO_F16_NATIVE`, `_NO_F32_NATIVE`, `_NO_BF16_NATIVE`, `_NO_F16_CMM`,
`_NO_F32_CMM`, `_NO_BF16_CMM`, `_NO_F16_RT`, `_NO_F32_RT`, `_NO_BF16_RT`,
`_NO_KQUANT_NATIVE`, `_NO_Q5K_RT`, `_NO_RMSNORM_VEC4`, `_NO_CONV1D_PAR`,
`_NO_DN_GATE_PREP`, `_NO_DN_NORM_PREP`, `INFR_METAL_LMHEAD_MRV` (presence-inv,
lifts the mrv `out_f` ceiling), `INFR_METAL_NODELTA` (presence),
`INFR_METAL_NOMOE` (presence) → `kernels.metal.*`. `INFR_METAL_PROFILE`
(presence, 3 levels) and `INFR_METAL_PROF_DEBUG` → `prof.metal_*`.

**Metal cannot be run on the dev box.** Its slice is compile-checked with
`cargo check -p infr-metal --all-targets --target x86_64-apple-darwin` and
validated by the macOS CI job. Do not guess at behaviour: keep the polarity
table in the commit message.

### 6.7 `kernels.rocm` / `kernels.cpu`

| Env                    | Grammar            | Config path                              | Default   |
| ---------------------- | ------------------ | ---------------------------------------- | --------- |
| `INFR_ROCM_WMMA_TILE`  | string             | `kernels.rocm.wmma_tile: Option<String>` | `None`    |
| `INFR_ROCM_COOP_TILE`  | string             | `kernels.rocm.coop_tile: Option<String>` | `None`    |
| `INFR_CPU_SPIN`        | int, default 32768 | `kernels.cpu.spin: usize`                | `1 << 15` |
| `INFR_CPU_NO_SPINPOOL` | `!=0`              | `kernels.cpu.spinpool: bool`             | `true`    |
| `INFR_CPU_REPACK_MB`   | int, default 4096  | `kernels.cpu.repack_mb: usize`           | `4096`    |

`CpuBackend::reference()` (landed `a1aed9e`) becomes
`kernels.cpu.reference: bool` in the same slice — it is already a struct field,
so this is the model the rest should follow.

### 6.8 `spec` — `SpecCfg` (MTP / speculative decode)

`INFR_MTP` (string), `INFR_MTP_TIME` (presence), `INFR_NO_MTP_CKPT` /
`INFR_NO_MTP_DRAFT_CHAIN` (presence-inv), `INFR_NO_MTP_REPRIME` (presence),
`INFR_SPEC_DRAFT` (string path), `INFR_SPEC_K` (int, default 6),
`INFR_SPEC_DEBUG` (presence), `INFR_DECODE_CHAIN` (int, default 8),
`INFR_NO_GPU_DRAFT_PROB` / `INFR_NO_GPU_MTP_ACCEPT` / `INFR_NO_GPU_ARGMAX` /
`INFR_NO_GPU_SAMPLE` / `INFR_NO_GPU_EMBED` (all presence-inv).

### 6.9 `prof` / `debug` / `serve`

`prof`: `INFR_PROF`, `INFR_PROF2`, `INFR_PROF2_SHAPES`, `INFR_PROF_DEC`,
`INFR_PROF_OPS`, `INFR_PROF_PF`, `INFR_PROFILE_OUT`, `INFR_VRAM_LOG`,
`INFR_MTP_TIME`, `INFR_DIFFUSION_TIME`, `INFR_METAL_PROFILE`,
`INFR_METAL_PROF_DEBUG`. `debug`: `INFR_DEBUG_BDA_CHUNK`, `INFR_DEBUG_COOPMAT`,
`INFR_DEBUG_WIDE_DISPATCH`, `INFR_DEBUG_CHAT`, `INFR_MOE_COUNTS_DEBUG`,
`INFR_MOE_COUNTS_DUMP`, `INFR_POISON_UNINIT`, `INFR_NOBARRIER`,
`INFR_FULLBARRIER`, `INFR_NO_GATED_RMSNORM`, `INFR_NO_QKV_FUSE`. `serve`:
`INFR_API_KEY` (string), `INFR_MAX_TOKENS_CAP` (int).

### 6.10 Explicitly NOT migrated

- `INFR_PROFILE` — read by **build scripts** (`build.rs` in
  core/cpu/gguf/llama/vulkan) to set a `cfg(infr_profile)`. Build-time input,
  not runtime config. Leave it.
- `INFR_BLESS` — golden re-blessing, test-only.
- `INFR_TEST_GGUF`, `INFR_TEST_MODEL`, `INFR_LLAMA_DIFFUSION_CLI` — test/dev
  fixtures pointing at files on disk. Leave as env; note them in `README.md` as
  test-only.
- `INFR_DIFFUSION_VISUAL` — CLI presentation only; migrate to a plain clap flag
  in S7, not to `Config`.
- `RAYON_NUM_THREADS`, `VK_*`, `MESA_*`, `RADV_*`, `HSA_*` — third-party.

## 7. Slices

Each slice is one commit. Do them in order — later slices depend on the seam
built by earlier ones. Every slice ends with the full verification block from
§9.

### S0 — scaffold (no behaviour change, no migration)

1. Create `crates/infr-core/src/config/` per §4 with: `Config` + all section
   structs, all fields, `impl Default` reproducing today's defaults,
   `PartialConfig` + `merge`, `ConfigLayer::{file, env, cli}`, `Config::load`.
2. `env.rs` reads ALL 138 keys, but nothing consumes the result yet.
3. Add the `toml` crate to `infr-core`'s dependencies (workspace-pinned).
4. Tests in `config/tests.rs`: §8's precedence tests, plus a test asserting
   `Config::default() == Config::load_from_layers(&[])`.
5. **Do not touch any read site.** After S0 the tree behaves identically and
   every env read is still where it was.

### S1 — CLI produces a `Config`

1. `infr-cli/src/main.rs`: add `--config <PATH>`; build `Config` in `main()`
   from file+env+flags; stop `set_var`-ing in `DeviceOpts::resolve` /
   `SamplingOpts::resolve` (fill `ConfigOverrides` instead). Keep publishing
   `RAYON_NUM_THREADS` (§6.1).
2. Thread `Arc<Config>` into the commands (`run`/`serve`/`bench`/`chat`) and
   into `SeamModel::load`.
3. **Until later slices land, the deep readers still read env** — so in S1 ONLY,
   after building the config, the CLI re-publishes the knobs it used to publish
   (`INFR_DEV`, `INFR_CTX`, `INFR_UBATCH`) so behaviour is unchanged. Delete
   that re-publication in S8.
4. Add `--set <path>=<value>` **[DECIDE-3]** (generic override for knobs with no
   dedicated flag), or drop it if the owner prefers file+env only.

### S2 — `infr-core`'s own knobs

`tier::EnvRows` (2 knobs), `budget::{env_flag, env_mib, overflow_vram_reserve}`,
`pager::ring_bytes_policy`. These already have pure `*_from`/`resolve` halves
(landed `b9069a3`), so the slice is: delete the env-reading wrapper, take the
value from `&Config` at the call site. Smallest possible first migration — use
it to establish the pattern the other slices copy.

### S3 — `infr-cpu`

7 read sites. `CpuBackend::new_with(cfg)`; fold `reference: bool` into
`kernels.cpu`. Rewrite `crates/infr-cpu/tests/decode_parity.rs` to build a
`Config` instead of calling `CpuBackend::reference()`.

### S4 — `infr-llama` seam (~54 sites)

`kv`, `spec`, `device.ubatch`, `paging.cache`, `sampling`. `SeamModel` and
`DenseSession<B, X>` gain `Arc<Config>`. `Sampler::from_env` →
`Sampler::from_cfg`. This is the slice that unblocks deleting most `EnvGuard`
uses in `crates/infr-llama/tests/cpu_backend.rs`.

### S5 — `infr-vulkan` (~74 sites, split into S5a/S5b if the diff exceeds ~800 lines)

S5a: `VulkanBackend::new_with` + device/caps knobs read at construction
(`lib.rs`: coopmat, subgroup, push-desc, VRAM guard, submit splitter). S5b:
`Recorder`/`adapter.rs`/`gemm.rs` kernel-tier knobs — the hot paths. Borrow
`&VulkanCfg` from the backend; R6 applies (no clones per dispatch).

### S6 — `infr-metal` (26 sites) and `infr-rocm` (4 sites)

Metal is compile-checked only (§6.6). ROCm is fully runnable on the dev box.

### S7 — `infr-server`, `infr-chat`, `infr-hub`, CLI presentation knobs

### S8 — remove the transitional bridges

Delete the CLI's env re-publication from S1. Assert R3 holds:
`grep -rn 'env::var("INFR_' crates/*/src` returns only
`infr-core/src/config/env.rs` (plus the §6.10 exclusions).

### S9 — delete `infr_core::test_env`

Every remaining `EnvGuard` use should be gone by now; delete the module and the
guard uses. If any test still needs it, that test names a knob that was not
migrated — fix that instead.

### S10 — documentation

`README.md`: replace the env-var tables with a config-file reference + the
precedence rules. Add `docs/config.md` (user-facing) and a commented
`infr.example.toml`.

## 8. Required tests

In `config/tests.rs` (these are the acceptance criteria for S0):

1. `default_config_matches_documented_defaults` — every field's `Default` equals
   the value in §6.
2. `env_overrides_file` — file says `flash_splits = 2`, env says `4` ⇒ `4`.
3. `cli_overrides_env` — env says `4`, CLI says `8` ⇒ `8`.
4. `absent_layer_does_not_clobber` — a file setting only `[kv]` leaves
   `[kernels.vulkan]` at its env/default value.
5. `unknown_toml_key_is_an_error` — typo detection (`deny_unknown_fields`).
6. `bad_value_is_an_error_not_a_silent_default` — `ctx = "banana"` fails to
   load.
7. `presence_inverted_knobs_have_the_right_polarity` — table-driven over every
   `presence-inv` knob in §6: env set ⇒ field `false`; env unset ⇒ field `true`.
8. `env_layer_reads_every_documented_key` — iterate a const list of all 138
   keys, set each one, assert the corresponding field changed. This is what
   stops a knob being silently dropped during migration. (Uses `EnvGuard` until
   S9; afterwards, drive `env.rs`'s parser with an injected
   `HashMap<String, String>` instead of the real environment — prefer that from
   S0.)

Per-slice: every knob a slice migrates must gain (or keep) a test that sets it
via `Config` and asserts the behaviour it gates. If a knob has no observable
behaviour to test, say so explicitly in the commit message rather than skipping
silently.

## 9. Verification block (run for EVERY slice)

```bash
cargo build --workspace --all-targets
cargo build --workspace --all-targets --features rocm
cargo check -p infr-metal --all-targets --target x86_64-apple-darwin   # aarch64 std NOT installed
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features rocm -- -D warnings
cargo fmt --all
cargo test --workspace
cargo test --release -p infr-llama --test cpu_backend gpu_seam -- --include-ignored   # 27/27
cargo test --release -p infr-vulkan -- --include-ignored                              # 210, PARALLEL
cargo test --release -p infr-rocm --features rocm -- --include-ignored                # 34
cargo test --release -p infr-llama --features rocm --test cpu_backend rocm_seam -- --include-ignored  # 9
cargo test --release -p infr-llama --test cpu_backend cpu_golden -- --include-ignored  # 6, hashes unmoved
```

Plus, for S4/S5 (hot paths), an interleaved bench against the previous commit —
decode and prefill, ≥2 reps each, alternating binaries:

```bash
infr bench 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M' -p 512 -n 0 -r 2
infr bench 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M' -p 0 -n 128 -r 2
infr bench 'unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M' -p 0 -n 64 -r 2
```

Bench trap: the first run of a burst gets a thermal boost. Only compare
steady-state pairs.

## 10. Traps (read before writing code)

1. **`presence-inv` polarity.** `INFR_NO_GEMM_WARP` set ⇒ warp GEMM OFF. The
   config field is `gemm_warp: bool` defaulting to `true`. Getting this
   backwards flips a kernel tier and the goldens will NOT catch it if the
   alternate kernel is also correct — only the bench will.
2. **`INFR_NO_MMV` vs `INFR_MMV_DECODE`.** Both are `presence`, they interact at
   `adapter.rs:409`; read that site before migrating either.
3. **`INFR_MMV_MW` is tri-state** (`unset` = vendor default, `"1"` = force on,
   `"0"` = force off). It must map to `Option<bool>`, not `bool`.
4. **`INFR_FLASH_BM` is compared to the literal `"32"`**, not parsed as an int.
5. **Empty string counts as "set"** for `is_ok()` knobs but as "off" for
   `budget::flag_from`. Preserve each site's grammar; do not unify them in this
   campaign.
6. **Do not memoize.** No `OnceLock` around a config read (that is what broke
   `INFR_PAGER_STATS`).
7. **`Sampler::from_env` has a doc contract**: unset ⇒ greedy, so library
   callers and goldens stay deterministic. `SamplingCfg::default()` must be
   `temp: 0.0`.
8. **The CLI's `--dev` parsing (`parse_dev_spec`) is shared** with the deep
   Vulkan reader. Keep one parser; put it in `config/`.
9. **`w_off`-style hot paths**: `adapter.rs` reads knobs inside per-op lowering.
   Hoist the read to the enclosing struct at construction where the value cannot
   change mid-run.
10. **Blocked sites**: if a read site has no owner in scope, record it in this
    file under a new "§11 blocked sites" heading with file:line and why — do not
    invent a global.

## 11. Open decisions

- **[DECIDE-1]** TOML + 3-step lookup (`--config`, `./infr.toml`,
  `~/.config/infr/config.toml`)?
- **[DECIDE-2]** Thread `Arc<Config>` through backends (this plan), or a single
  `OnceLock` + test override (much smaller diff, keeps ambient state)?
- **[DECIDE-3]** Ship a generic `--set kernels.vulkan.flash_splits=2` escape
  hatch, so the ~120 tuning knobs stay reachable from the command line without
  ~120 flags?
- **[DECIDE-4]** Execution: delegate slices to subagents (the flow that landed
  the backend-unification campaign, `docs/backend-unification-plan.md`), or
  inline?
