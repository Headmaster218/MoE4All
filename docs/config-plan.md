# config-plan.md — replace the `INFR_*` env gates with a layered `Config`

**Status: this document holds only PENDING work.** Landed slices are pruned out
after review; the ledger below is the trail. Every section is prescriptive:
follow it literally. All eight decisions are settled (§11) — do not re-open
them.

## Ledger

| Slice | What                                                                                                                                                                                                                                                                                         | Commit    |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| S0    | Config scaffold in `infr-core`: `Config` + sections, partial/merge fold, env(injected reader)/file(TOML)/cli layers, `manifest.rs`, 23 tests                                                                                                                                                 | `a0bff9c` |
| S1    | CLI builds the `Config`: `--config`, `--set`, `DeviceOpts`/`SamplingOpts` fill a `ConfigOverrides` instead of `set_var`, `Arc<Config>` threaded into every command, CLI `mod tests` off its hand-rolled env lock                                                                             | `addc1ac` |
| S2    | `infr-core`'s own knobs (12) read from `Config`: `EnvRows::clamped`, `budget` flag/mib/reserve, `pager::ring_bytes`, `FusionCfg.enabled`. Temporary `Config::load_from_env()` bridge in `VulkanBackend::new_selected` (dies in S5a) and `RocmBackend::new` (dies in S6)                      | `6a8c2cb` |
| S3    | `infr-cpu` (6 knobs) on `Config`: `CpuBackend::new_with(cfg)`, `reference()` → `kernels.cpu.reference`, `spin_limit()`'s `OnceLock` deleted (per-pool field). Crate is `INFR_*`-free. Bridge `Config::load_from_env()` in `CpuBackend::new` dies in **S4**                                   | `b2d6f04` |
| S4    | `infr-llama` seam (35 keys): `SeamModel`/`DenseSession` carry `Arc<Config>`, `Sampler::from_cfg`, `device.ubatch_specified` added, device-list grammar moved into `infr-core`, CPU bridge deleted, 44 test loads + 32 `INFR_TEMP` writes converted                                           | `c29d816` |
| S5a   | `infr-vulkan` construction knobs (13): `VulkanBackend::new_with(cfg)` reached from every seam/CLI caller, capability masks folded into the probe (§5.2), S2's Vulkan `load_from_env()` bridge deleted, `INFR_DEV` dropped from the CLI bridge, S2's `INFR_MOE_SMALL_M` test exception closed | `a481747` |

**Authority:** `crates/infr-core/src/config/manifest.rs` is the knob inventory —
177 keys, their config paths, grammars, and a `migrated` flag — and the tests
enforce it against the tree. This document does NOT duplicate it. Line numbers
quoted below drift; each read site carries enough source text to re-find by
string search.

## 1. The problem

Runtime behaviour is currently steered by **177 distinct `INFR_*` keys** read
across **205 literal `std::env::var` / `env::var_os` call sites** plus the 27
keys that are read ONLY through a helper taking the key name as a `&str`
parameter (§6.0). (An earlier revision said 179/206; re-derived at `2dd0c5a`,
where `git log 6573fb3..HEAD -- crates/` is empty — the tree did not move, the
counts were simply wrong.) They are spread over `infr-vulkan` (64 keys),
`infr-llama` (38), `infr-metal` (20), `infr-cli` (13), `infr-rocm` (13),
`infr-cpu` (6), `infr-core`, `infr-server`, `infr-chat`, `infr-gguf`,
`infr-prof-rt`. The full inventory is §6.

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
   path. **Three memoized families remain** and are still unsettable from a
   test: `recorder.rs`'s `gemv_knobs()` (`OnceLock<GemvKnobs>`, 11 keys),
   `recorder.rs`'s `cap_from_env` (two `AtomicU64` cells, 2 keys), and
   `pool.rs`'s `spin_limit()` (`SPIN_LIMIT.get_or_init`, 1 key). See trap §10.6.
3. **No discoverability.** There is no `--help` listing, no config file, no way
   to see what is set. The knobs are documented in scattered doc comments and
   `README.md`.
4. **No validation.** A typo (`INFR_FLASH_SPLIT=2`) is silently ignored. A bad
   value (`INFR_MOE_SMALL_M=100000`) once hung the GPU; it now clamps, but only
   because someone remembered to clamp it at that one site.
5. **The CLI already fights this.** `DeviceOpts::resolve` /
   `SamplingOpts::resolve` in `crates/infr-cli/src/main.rs` (the `set_var` block
   around `main.rs:197-291`) take clap flags and **write them back into the
   process env** (`std::env::set_var("INFR_CTX", …)`) purely so that code deep
   in the seam can read them. That is the tell: the value already exists as a
   typed thing at startup and is being laundered through a string table to cross
   an API boundary.
6. **The knob set is not greppable.** `grep 'env::var("INFR_'` misses 27 real
   keys, because they are read through `budget::env_flag(var)`,
   `budget::env_mib(var)`, `budget::overflow_vram_reserve(_, env)`,
   `tier::EnvRows { env, .. }.get()`, `fusion::FusionCfg { disable_env, .. }`,
   `seam::parse_device_list(label, min)`, `recorder::GemvKnobs::resolve(get)`
   and `recorder::cap_from_env(cell, var)`. Any inventory built by that one grep
   — as an earlier draft of §6 was — is wrong by ~15%.

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
- **R3 — One read point per knob.** After a slice, the knob is read from the
  process environment exactly once, inside `infr-core/src/config/env.rs`.
  Nowhere else. Because of §1.6 the check is THREE greps, all of which must come
  back clean (modulo the §6.10 exclusions):

  ```bash
  grep -rn 'env::var\(_os\)\?("INFR_' crates/*/src crates/*/build.rs
  grep -rn 'env_flag\|env_mib\|overflow_vram_reserve\|ring_bytes_policy' crates/*/src
  grep -rn 'disable_env\|EnvRows\|parse_device_list\|cap_from_env\|gemv_knobs' crates/*/src
  ```

  A helper that still reads the environment is a violation even if no `INFR_`
  literal is next to it.

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
  `infr_core::test_env` itself is deleted in the final slice (S9). Note
  `crates/infr-cli/src/main.rs`'s own `mod tests` uses a hand-rolled
  `static ENV_LOCK: Mutex<()>` rather than `EnvGuard` — S1 must convert those
  too, or they will keep the CLI's env writes alive.
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
    env.rs        // THE ONLY place the process environment is read; env → PartialConfig
    file.rs       // TOML → PartialConfig; path discovery
    cli.rs        // a `ConfigOverrides` struct the CLI fills; → PartialConfig
    manifest.rs   // GENERATED: `pub const KEYS: &[KnobKey]` — name, section, field, grammar
    tests.rs      // precedence tests (§8)
```

`manifest.rs` is what makes the migration checkable. It is generated once in S0
by the §6.0 command, hand-annotated with grammar + destination, and then frozen:
`env.rs` iterates it, and the §8.8 test iterates it. When a slice migrates a
knob it flips that entry's `migrated: bool`. A knob that exists in the tree but
not in `manifest.rs` fails the S0 test — that is the mechanism that stops a knob
being silently dropped, and it is why this document's tables do not need to be
perfect.

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
    pub multi:    MultiCfg,     // INFR_TENSOR_PARALLEL / _EXPERT_PARALLEL / _PIPELINE + *_HOST (§6.11)
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

**Field-naming rule (enforce it, it is the polarity guard):** a config field is
named for the thing being ENABLED and is `true` when the feature is on —
`gemm_warp`, `mmv`, `coopmat`, `push_desc`. The env layer inverts for the
`INFR_NO_*` spellings. The only exceptions are the four knobs whose _effect_ is
negative rather than a tier choice, where the positive name would be a lie:
`no_vram_guard`, `no_replay`, `no_attn_hd_spec`, `no_moe_sm_pool`. Do not invent
more exceptions.

**File format: TOML.** Section path = struct path.

```toml
[device]
dev = "vulkan1"
ctx = "32k"          # the shared size grammar: 8192 / 256k / 50%

[kv]
type_k = "q8_0"

[kernels.vulkan]
flash_splits = 2
gemm_warp = false    # NOT `no_gemm_warp` — the file speaks the POSITIVE field names
```

**Lookup order for the file layer (first existing file wins, no merging of
multiple files):**

1. `--config <PATH>` (error if the path does not exist)
2. `./infr.toml`
3. `$XDG_CONFIG_HOME/infr/config.toml`, else `~/.config/infr/config.toml`

**[DECIDE-1] — DECIDED (repo owner, 2026-07-26): this section is normative.**
TOML, exactly this 3-step lookup, first existing file wins, no cross-file
merging. The global-path-only alternative is rejected.

**[DECIDE-5] — DECIDED: warn-and-ignore for unknown keys (§11).** Original note:
`deny_unknown_fields` (§8.5) makes a typo an error — but it also makes a config
file written for a NEWER infr fail hard on an OLDER binary, and makes deleting a
knob a breaking change for anyone with it in their file. Pick one: (a) hard
error, (b) warn-to-stderr and ignore, (c) hard error on unknown keys inside a
known section, warn on an unknown section. This blocks S0 — the
`unknown_toml_key_is_an_error` test encodes the answer.

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

### 5.1 Scope: what a `Config` is attached to

The plan is **one `Config` per process, cloned by `Arc` into each backend and
each session.** Concretely:

- **Per-process.** Built once in `main()` (or by the library caller). This is
  what today's env already is, so it is behaviour-preserving by construction.
- **Not per-request.** `infr serve` accepts per-request sampling through
  `RequestCtx` (`infr-llama/src/sampling.rs:143`), and that path is
  **unchanged**: `Sampler::resolve(req)` layers a request's `RequestSampling`
  over the process sampler, exactly as it layers over `from_env()` today. The
  rename is `Sampler::from_env()` → `Sampler::from_cfg(&SamplingCfg)`; `resolve`
  keeps its signature and its precedence (`RequestCtx` > `Config`). No other
  knob becomes per-request in this campaign — `INFR_MAX_TOKENS_CAP` and
  `INFR_API_KEY` stay per-process.
- **Not per-model.** `SeamModel::load` takes the `Arc<Config>` it was given. A
  process that loads a target + a draft model (`INFR_SPEC_DRAFT`,
  `cli/main.rs:1421`) gives both the SAME config, which is what happens today.
- **Per-backend-instance is the seam that matters for multi-device.** Each
  `*Backend::new_with` gets its own `Arc<Config>` handle. Today the multi-GPU
  paths (`PipelineBackend`, TP, EP — §6.11) build several Vulkan backends in one
  process and they all read the same env, so cloning the same `Arc` into all of
  them reproduces current behaviour. Per-device configs are explicitly OUT OF
  SCOPE; do not add a `Vec<Config>` in this campaign.

### 5.2 `Capabilities` is NOT config, but config feeds it

`infr_core::backend::Capabilities` (`backend.rs:43`) is device-probed and must
stay a probe result, not a config section. But today six knobs are folded into
the probe at construction, so the probe is already config-masked:

| Site                 | Read                                       |
| -------------------- | ------------------------------------------ |
| `vulkan/lib.rs:1583` | `has_f16 && INFR_NO_F16.is_err()`          |
| `vulkan/lib.rs:1585` | `has_f16 && INFR_NO_COOPMAT.is_err()`      |
| `vulkan/lib.rs:1615` | `has_i8_dot && INFR_NO_I8DOT.is_err()`     |
| `vulkan/lib.rs:1520` | `INFR_CM_8X8` → the 8x8 coopmat shape      |
| `vulkan/lib.rs:1357` | `INFR_NO_PUSH_DESC` → push-descriptor path |
| `vulkan/lib.rs:1852` | `INFR_SG` → subgroup preference            |

**Rule for S5a:** the capability _probe_ keeps returning what the device
supports; the _masking_ moves into the `Capabilities` constructor, which gains a
`&VulkanCfg` parameter. Do not add these knobs to `Capabilities` as fields, and
do not read them again downstream — everything downstream already reads
`caps.f16_coopmat()` etc. and must keep doing so. `INFR_NO_F16` masking
`has_f16` also gates `INFR_NO_COOPMAT`: with `INFR_NO_F16` set, coopmat is off
regardless. Preserve that AND, it is not a bug.

### 5.3 Values consumed before a `Config` could exist

- `INFR_PROFILE` is read by `build.rs` in core/cpu/gguf/llama/vulkan to set
  `cfg(infr_profile)`. It is a **build-time** input; a runtime `Config` cannot
  exist when it is read. It is out of scope (§6.10) — do not try.
- Everything else is read after `main()` starts. There is no early-startup
  ordering hazard in this tree: the earliest runtime read is
  `selected_backend()` in `cli/main.rs:126-130` (`INFR_DEV`), which is inside
  `main()` and therefore after `Config::load`.

**Transitional exception to R4 (delete before the campaign closes):** during
S2–S7 a knob that has not been migrated yet still reads env at its old site.
That is fine and expected — do NOT introduce a global to bridge them. If a slice
finds a read site with no plausible owner in scope, STOP and record it in §10.10
"blocked sites" rather than inventing a global.

**[DECIDE-2] — DECIDED (repo owner, 2026-07-26): thread `Arc<Config>`, as
described above.** The cheaper alternative (a single `OnceLock<Config>` + a
thread-local test override) is REJECTED. It is more work — ~206 sites plus
constructors — and that cost is accepted deliberately: it is the only version
that makes configuration a value rather than ambient state, which is the point
of the campaign. Do not fall back to a global for an awkward site; that is a
blocked site (§10.10).

## 6. Knob inventory

Subsections: **6.0** re-derivation · 6.1 device · 6.2 sampling · 6.3 kv · 6.4
paging · 6.5 kernels.vulkan · 6.6 kernels.metal · 6.7 kernels.rocm + kernels.cpu
· 6.8 spec · 6.9 prof/debug/serve · **6.10 NOT migrated** · **6.11 multi-GPU** ·
**6.12 knobs with more than one grammar (read this)**.

`presence` = `is_ok()`/`is_some()` (set to anything, including empty, ⇒ on).
`presence-inv` = `is_err()`/`is_none()` (the _absence_ enables the feature).
Both map to `bool` in the config, with the polarity noted so the env layer
inverts where needed.

**Migrating a `presence-inv` knob is the single most likely place to introduce a
behaviour change. Write the truth table in the PR description before changing
the code.**

### 6.0 The inventory lives in code, not here

`crates/infr-core/src/config/manifest.rs` (landed in S0) is the AUTHORITY: every
`INFR_*` key, its config path, its grammar, whether a bad value errors, and a
`migrated` flag. Three tests read it — `env_layer_reads_every_key` (every key
reaches its field), `presence_inverted_knobs_have_the_right_polarity` (the
`""`/`"0"`/`"1"` truth table), and `manifest_matches_the_tree` (a new `INFR_*`
literal anywhere in `crates/*/src` fails the build's tests until it is in the
manifest). The per-knob tables that used to live here were a stale second copy
and are deleted; read the manifest.

Per-slice: find your knobs with
`rg 'migrated: false' crates/infr-core/src/config/manifest.rs` filtered by the
`path` prefix your slice owns, and flip each to `migrated: true` as you move its
read site.

### 6.10 Explicitly NOT migrated

- `INFR_PROFILE` — read by **build scripts** (`build.rs` in
  core/cpu/gguf/llama/vulkan) to set a `cfg(infr_profile)`. Build-time input,
  not runtime config. Leave it (§5.3).
- `INFR_BLESS` — golden re-blessing, read only in
  `crates/infr-llama/tests/cpu_backend.rs:95`. Test-only; never appears in
  `crates/*/src`.
- `INFR_TEST_GGUF` (`gguf/lib.rs:723`), `INFR_TEST_MODEL`
  (`llama/qwen35.rs:163`, `llama/util.rs:150`), `INFR_LLAMA_DIFFUSION_CLI`
  (`cli/main.rs:2960`) — test/dev fixtures pointing at files on disk. Leave as
  env; note them in `README.md` as test-only.
- `INFR_DIFFUSION_VISUAL` (`cli/main.rs:1245`) — CLI presentation only; migrate
  to a plain clap flag in S7, not to `Config`.
- `INFR_CPU`, `INFR_METAL` — dead keys, nothing reads them (§6.0).
- `INFR_BUDGET_TEST_*`, `INFR_TIER_TEST_*`, `INFR_TEST_ENV_GUARD_*` — in-`src`
  `#[cfg(test)]` fixtures. They vanish with `test_env` in S9 (except the
  budget/tier ones, which assert the wrapper reads the right variable and are
  deleted with the wrapper).
- `RAYON_NUM_THREADS`, `VK_*`, `MESA_*`, `RADV_*`, `HSA_*` — third-party.

### 6.11 `multi` — multi-GPU and host transport (6 keys)

Missing from every earlier draft. All read in `infr-llama/src/seam/mod.rs`.

| Env                    | Grammar                                        | Config path                                 | Default | Site                                |
| ---------------------- | ---------------------------------------------- | ------------------------------------------- | ------- | ----------------------------------- |
| `INFR_PIPELINE`        | `VulkanN,VulkanM,…` or bare indices, min **2** | `multi.pipeline: Option<Vec<usize>>`        | `None`  | `mod.rs:1696` (`parse_device_list`) |
| `INFR_TENSOR_PARALLEL` | same grammar, min **2**                        | `multi.tensor_parallel: Option<Vec<usize>>` | `None`  | `mod.rs:1795`                       |
| `INFR_EXPERT_PARALLEL` | same grammar, min **1**                        | `multi.expert_parallel: Option<Vec<usize>>` | `None`  | `mod.rs:2005`                       |
| `INFR_PIPELINE_HOST`   | presence-inv (`var_os(..).is_none()` ⇒ P2P)    | `multi.pipeline_p2p: bool`                  | `true`  | `mod.rs:1757`                       |
| `INFR_TP_HOST`         | presence-inv                                   | `multi.tp_p2p: bool`                        | `true`  | `mod.rs:1968`                       |
| `INFR_EP_HOST`         | presence-inv                                   | `multi.ep_p2p: bool`                        | `true`  | `mod.rs:2144`                       |

`parse_device_spec(spec, min, label)` (`mod.rs:1661`) is already the pure half —
it takes the spec string and the label used in error messages, and it ERRORS on
garbage or on fewer than `min` devices. The env layer calls it and propagates
the error; `parse_device_list` (`mod.rs:1682`, the `var_os` wrapper) is deleted.

Note the three keys have DIFFERENT `min` values (2/2/1). Preserve them.

### 6.12 Knobs with more than one grammar or more than one default

**Read this table before touching any of these.** Each row is a knob where a
single `Config` field is not obviously enough, and where a careless migration
changes behaviour.

| Knob                    | Site A                                                                                   | Site B                                                                                                                                  | How to model it                                                                                                                                                                                                                                                                                                       |
| ----------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INFR_DEV`              | `cli/main.rs:91` → `parse_dev_spec` → a `Backend` enum (`vulkan*`/`metal`/`cpu`/`rocm*`) | `vulkan/lib.rs:1043` → `resolve_infr_dev_index` → a device INDEX; TOLERATES `metal`/`cpu`/empty by falling back to the discrete default | Keep BOTH parsers. `device.dev: Option<String>` carries the raw spec; each consumer keeps its own parse. They are in different crates and cannot be one function.                                                                                                                                                     |
| `INFR_UBATCH`           | `mod.rs:577` — the VALUE (`parse`, `>0`)                                                 | `mod.rs:1254,1292,1354` — PRESENCE (`is_err`), meaning "the user did not pin a ubatch, so the placement sweep may choose one"           | `Option<usize>`: `Some(_)` reproduces `is_ok()`, `None` reproduces `is_err()`. This is why the field must be `Option`, not a defaulted `usize`.                                                                                                                                                                       |
| `INFR_KV_TYPE_K` / `_V` | `runner.rs:451` — the VALUE (dtype, gated)                                               | `mod.rs:725,726` — PRESENCE, feeding `kv_env_unset()` which gates auto-q8                                                               | `Option<DType>`; `kv_env_unset()` becomes `cfg.kv.type_k.is_none() && cfg.kv.type_v.is_none() && !cfg.kv.force_q8`. An unparseable dtype today is `is_ok()` for the presence check but falls through to f16 for the value — preserve that by storing `Option<String>` alongside, or by parsing lazily. **[DECIDE-8]** |
| `INFR_SEED`             | `sampling.rs:383` — default = wall-clock nanos                                           | `cli/main.rs:2306`, `chat/diffusion.rs:268` — default `42`                                                                              | `Option<u64>`; both fallbacks stay at their sites.                                                                                                                                                                                                                                                                    |
| `INFR_METAL_NODELTA`    | `exec.rs:1116,1131,1155` — `is_ok()`                                                     | `exec.rs:4698,4819` — `is_err()`                                                                                                        | Same semantics (set ⇒ DeltaNet off), opposite spellings. One positive field `metal.deltanet: bool` default `true`; site A becomes `!cfg.deltanet`, site B becomes `cfg.deltanet`.                                                                                                                                     |
| `INFR_METAL_NOMOE`      | `exec.rs:1155` — `is_ok()`                                                               | `exec.rs:4314` — `is_err()`                                                                                                             | As above → `metal.moe: bool` default `true`.                                                                                                                                                                                                                                                                          |
| `INFR_SEAM_NO_REPLAY`   | `vulkan/adapter.rs:175` — `is_ok()`                                                      | `llama/seam/runner.rs:3865` — `is_err()`                                                                                                | One field `no_replay: bool` default `false`; the llama site becomes `!cfg.no_replay`. Two crates ⇒ two slices (S4, S5); the field lands in S4 and S5 consumes it.                                                                                                                                                     |
| `INFR_MMV_MW`           | `adapter.rs:509` — `Some("0")` / `Some(_)` / `None` are THREE different dtype lists      | —                                                                                                                                       | `Option<bool>`; `Some(false)`=`"0"`, `Some(true)`=any other value, `None`=vendor default.                                                                                                                                                                                                                             |
| `INFR_METAL_PROFILE`    | `lib.rs:184` presence                                                                    | `lib.rs:185` `=="2"`, `lib.rs:149` `=="3"`                                                                                              | `Option<String>` + three derived booleans in the accessor.                                                                                                                                                                                                                                                            |
| `INFR_KV_OVERFLOW`      | `flag_from` grammar (empty and `"0"` are OFF)                                            | vs every `is_ok()` knob (empty is ON)                                                                                                   | Keep `flag_from` for this key and the other three `budget::env_flag` keys; do NOT route them through the presence parser.                                                                                                                                                                                             |

## 7. Slices

Each slice is one commit. Do them in order — later slices depend on the seam
built by earlier ones. Every slice ends with the full verification block from
§9.

**Exit criteria that apply to EVERY slice, S2 onward:**

1. `cargo test --workspace` green, plus the §9 block for the affected backend.
2. The knobs the slice claims are marked `migrated: true` in
   `config/manifest.rs`, and the three §R3 greps return nothing for those keys.
3. The `EnvGuard` uses for those knobs are deleted, not just unused.
4. Every `presence-inv` knob in the slice has its truth table in the commit
   message (env unset ⇒ field value ⇒ behaviour; env set ⇒ …).
5. `git diff` shows no change to any `Default` value versus §6 / the manifest.

### 7.0 The migration pattern, worked end to end

Every slice from S2 on is this same five-step edit. Do it once by hand from this
example before doing it 179 times.

**Knob: `INFR_FLASH_MIN_ROWS`** — Vulkan, int-with-default, read inside a hot
lowering path.

**Step 1 — find the read.** `grep -rn 'INFR_FLASH_MIN_ROWS' crates/*/src` →
`crates/infr-vulkan/src/adapter.rs:2756`:

```rust
// BEFORE — inside the per-op lowering, executed for every attention op
let flash_min_rows: usize = std::env::var("INFR_FLASH_MIN_ROWS")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(24);
let flash_geom = (rows >= 64 || (rows >= flash_min_rows && kv_len >= 8192)) && …;
```

**Step 2 — add the field with the SAME default** in
`infr-core/src/config/mod.rs`:

```rust
pub struct VulkanCfg {
    // …
    /// `INFR_FLASH_MIN_ROWS`: row floor for the flash-attention geometry gate.
    pub flash_min_rows: usize,
}
impl Default for VulkanCfg {
    fn default() -> Self { Self { /* … */ flash_min_rows: 24 } }
}
```

**Step 3 — add the parse (and ONLY the parse, R5)** in
`infr-core/src/config/env.rs`:

```rust
// env.rs owns the string→Option<T> step. No default, no clamp, no policy here.
p.kernels.vulkan.flash_min_rows = get("INFR_FLASH_MIN_ROWS").and_then(|v| v.parse().ok());
```

where `get: &dyn Fn(&str) -> Option<String>` is the injected reader (§8.8) —
`ConfigLayer::env()` passes `|k| std::env::var(k).ok()`, tests pass a `HashMap`.
`None` from an unparseable value reproduces today's `.and_then(parse).ok()`
behaviour exactly: a bad value falls back to the default rather than erroring.
(Contrast `INFR_SG`, §6.1, where a bad value MUST error.)

**Step 4 — rewrite the read site as a borrow, hoisted out of the hot path** (R6,
§10.9). `Recorder` already holds a backend reference, so:

```rust
// AFTER
let flash_min_rows = self.cfg.kernels.vulkan.flash_min_rows;
let flash_geom = (rows >= 64 || (rows >= flash_min_rows && kv_len >= 8192)) && …;
```

If the enclosing function has no `&self` with a config, thread `&VulkanCfg` down
one level — do NOT add a global (R4). If you cannot reach it in one level, that
is a **blocked site**: stop and record it in §10.10.

**Step 5 — convert the test.** Any test driving this knob via `EnvGuard`:

```rust
// BEFORE
let mut g = infr_core::test_env::EnvGuard::new();
g.set("INFR_FLASH_MIN_ROWS", "8");
let be = VulkanBackend::new()?;

// AFTER — no lock, no restore, no ordering hazard, runs in parallel
let cfg = Config { kernels: KernelCfg { vulkan: VulkanCfg { flash_min_rows: 8,
    ..Default::default() }, ..Default::default() }, ..Default::default() };
let be = VulkanBackend::new_with(Arc::new(cfg))?;
```

**Step 6 — flip `migrated: true`** in `manifest.rs` and re-run the §R3 greps.

**The `presence-inv` variant of step 3/4.** For `INFR_NO_GEMM_WARP` the field is
POSITIVE and the env layer inverts:

```rust
// mod.rs
pub gemm_warp: bool,             // Default: true
// env.rs — set to Some(false) ONLY when the var is present; never Some(true)
if get("INFR_NO_GEMM_WARP").is_some() { p.kernels.vulkan.gemm_warp = Some(false); }
// adapter.rs:960 — BEFORE: `… && std::env::var("INFR_NO_GEMM_WARP").is_err()`
//                   AFTER: `… && cfg.gemm_warp`
```

Truth table for the commit message (write one of these for EVERY `presence-inv`
knob):

| `INFR_NO_GEMM_WARP` | env layer emits | `cfg.gemm_warp` | warp GEMM |
| ------------------- | --------------- | --------------- | --------- |
| unset               | `None`          | `true`          | ON        |
| `""` (set, empty)   | `Some(false)`   | `false`         | OFF       |
| `"0"`               | `Some(false)`   | `false`         | OFF       |
| `"1"`               | `Some(false)`   | `false`         | OFF       |

Note row 3: for an `is_err()` knob, `INFR_NO_GEMM_WARP=0` turns the feature
**off**, because only presence matters. That is today's behaviour and R1 says
keep it. Contrast the four `budget::env_flag` knobs (§6.12 last row) where `"0"`
means off.

### Carried into later slices from S1

- **The transitional bridge lives in `crates/infr-cli/src/main.rs`**:
  `publish_transitional_env(cfg, specified)` re-publishes ten knobs (`INFR_DEV`,
  `INFR_CTX`, `INFR_UBATCH`, `INFR_TEMP`, `INFR_TOP_K`, `INFR_TOP_P`,
  `INFR_SEED`, `INFR_MAX_NEW`, `INFR_NO_THINK`, `INFR_IGNORE_EOS`) from the
  resolved config, and ONLY for paths a layer actually specified — that "only if
  specified" rule is what keeps the model-default sampling fallback intact. **S8
  deletes it**, once every deep reader takes its value from `Config`.
  `publish_thread_count` (`RAYON_NUM_THREADS`) is permanent and stays.
- **`set_default_sampling_env` (same file) still reads and writes env.**
  Converting it needs the "specified" partial threaded into
  `cmd_run`/`cmd_serve`/`cmd_multi`; it is part of the bridge and dies with it
  in S8.
- **`SeamModel::load` kept its signature** — S4 is where `infr-llama` grows its
  `Arc<Config>`, and where the CLI's remaining seam-side plumbing lands.
- **`Config::load` now runs for every subcommand**, so the five loud keys
  (`INFR_SG`, `INFR_SUBMIT_DISPATCHES`, the three device lists) now fail at
  startup rather than at backend construction — including on subcommands that
  never build a backend (`infr pull`). Intended; note it if a user reports it.

### Carried into later slices from S2

- **`Config::load_from_env()` bridges**: `VulkanBackend::new_selected`
  (`crates/infr-vulkan/src/lib.rs`) and `RocmBackend::new`
  (`crates/infr-rocm/src/backend.rs`) each construct a config from the
  environment ONCE and store it as `cfg: Arc<Config>`. **S5a** and **S6**
  replace those with `new_with(cfg)` taking the caller's config and delete the
  `load_from_env()` call. `infr-llama` needs no bridge — it reads `vk.cfg()` off
  the backend it already holds, which S4/S5 swap for the threaded `Arc<Config>`.
- **One R7 exception outstanding**: `crates/infr-llama/tests/cpu_backend.rs`
  still drives `INFR_MOE_SMALL_M` through `EnvGuard` (annotated in place),
  because `SeamModel` opens its own backends and takes no config until S4/S5.
  Convert it there.
- `budget.rs`'s `SpillNouns.env` and the `KV_SPILL`/`WEIGHT_SPILL` consts keep
  the `INFR_*` spellings as user-visible BANNER TEXT, not reads. Leave them.

### Carried into later slices from S4

- **`sampling.*` is NOT migrated yet** (6 keys). The seam reads them from
  `Config`, but the CLI's model-recommendation bridge
  (`apply_model_sampling_defaults` + `set_default_sampling_env`) and the DG
  bench arm still read `INFR_TEMP`/`TOP_K`/`TOP_P`/`SEED`/`IGNORE_EOS`/
  `MAX_NEW`. **S8** replaces the "was it specified?" env probe with the
  `specified` `PartialConfig` threaded from `main()`, then flips these keys.
  `INFR_NO_THINK` belongs to `infr-chat` (**S7**).
- **`prof.prof`** stays unmigrated until `infr-vulkan`'s recorder moves
  (**S5**).
- **`kernels.vulkan.{delta_strided,no_replay,gpu_pos}`** are two-crate knobs;
  the `infr-llama` half moved, **S5** takes the adapter half.
- **`SeamModel::load`'s own `load_from_env()`** remains (documented on the
  field) — **S8** deletes it once the CLI hands its config in everywhere.
- **`INFR_MOE_SMALL_M` in `cpu_backend.rs` still uses `EnvGuard`**: the seam
  opens Vulkan backends via `VulkanBackend::new()`, which builds its own config
  until **S5a**. Close it there.
- **Metal paths edited but not compile-verified**
  (`cargo check -p infr-llama --target x86_64-apple-darwin` fails on a native
  C++ dep here): the two `metal_decode_chain_*` tests and
  `metal_llama_replay_matches_static` were RESTRUCTURED (they used to flip an
  env var after the model loaded, which the config makes a no-op). macOS CI is
  the gate; read those closely.

### S5b — `infr-vulkan` hot-path tier knobs (the rest of S5)

S5a landed the construction half. S5b is everything read per-op or per-dispatch:
the `INFR_FLASH_*` family, `INFR_NO_GEMM_WARP`, the mmv/mrow tiers, the
`OnceLock`-memoized GEMV group (`GemvKnobs::resolve`), the atomic-memoized BDA
chunk caps (`cap_from_env`), `prof.prof`, and the Vulkan halves of the two-crate
knobs `delta_strided` / `no_replay` / `gpu_pos` whose `infr-llama` halves moved
in S4.

- `Recorder` borrows `&VulkanCfg` from the backend that created it (R6: no
  clone, no re-resolve per dispatch). The two memoized families must be HOISTED
  into the owning struct, not turned into per-call `getenv`s (§10.6).
- `INFR_PAGER_STATS` stays unflipped until **S6** (ROCm's pager still reads it).
- Closes the `EnvGuard` uses for `INFR_SEAM_NO_REPLAY` and `INFR_I8_COOPMAT`.
- Perf is the gate here, not just the goldens: interleaved decode AND prefill
  pairs vs the parent commit, warmed up (§9's thermal trap).

### S6 — `infr-metal` (26 sites / 20 keys) and `infr-rocm` (17 sites / 13 keys)

Metal is compile-checked only (§6.6). ROCm is fully runnable on the dev box and
is 3x bigger than an earlier draft claimed — budget for it accordingly.

The `FusionCfg.disable_env` change (§6.5 last note) lands here if S5 did not
already do it; coordinate so it happens exactly once.

**Exit:** `rocm_seam` 9/9 and
`cargo test -p infr-rocm --features rocm -- --include-ignored` 34/34 green; the
Metal polarity table is in the commit message.

### S7 — `infr-server`, `infr-chat`, `infr-gguf`, CLI presentation knobs

`INFR_API_KEY`, `INFR_MAX_TOKENS_CAP` (§6.9), `INFR_NO_THINK`,
`INFR_DEBUG_CHAT`, `INFR_DIFFUSION_VISUAL` → a clap flag (§6.10).

### S8 — remove the transitional bridges

Delete the CLI's env re-publication from S1. Assert R3 holds: all three greps in
R3 return only `infr-core/src/config/env.rs` (plus the §6.10 exclusions). Add
that assertion as a CI check or a `#[test]` that shells out, so it cannot rot.

### S9 — delete `infr_core::test_env`

Every remaining `EnvGuard` use should be gone by now; delete the module and the
guard uses. If any test still needs it, that test names a knob that was not
migrated — fix that instead.

### S10 — documentation

`README.md`: replace the env-var tables with a config-file reference + the
precedence rules. Add `docs/config.md` (user-facing) and a commented
`infr.example.toml`. Mark this plan `LANDED` and link the commits, matching how
`docs/backend-unification-plan.md` records its landed candidates.

## 8. Test obligations

The scaffold's own acceptance tests (precedence, polarity, manifest drift,
`--set` typos, bespoke-flag-beats-`--set`) landed with S0 in
`crates/infr-core/src/config/tests.rs`; they are not repeated here.

**Every later slice owes, per knob it migrates:** one test that sets the knob
through a `Config` (never through the environment) and asserts the behaviour it
gates. If a knob has no observable behaviour to assert, say so explicitly in the
commit message rather than skipping silently. A slice that migrates a knob whose
test still drives it via `EnvGuard` is incomplete (R7).

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
   alternate kernel is also correct — only the bench will. Note also that
   `INFR_NO_*=0` still means OFF (§7.0 truth table): presence, not value.
2. **`INFR_NO_MMV` vs `INFR_MMV_DECODE`.** `INFR_NO_MMV` is `presence-inv`,
   `INFR_MMV_DECODE` is `presence`; they interact at `adapter.rs:409` as
   `mmv_decode && mmv`. Read that site before migrating either. An earlier draft
   of §6.5 had `INFR_NO_MMV` recorded as `presence` — it is not.
3. **`INFR_MMV_MW` is tri-state** (`unset` = vendor default, any non-`"0"` value
   = force on, `"0"` = force off). It must map to `Option<bool>`, not `bool`.
4. **`INFR_FLASH_BM` is compared to the literal `"32"`**, not parsed as an int
   (`recorder.rs:4463`). Likewise `INFR_SG` (`"16"`/`"32"`), `INFR_MTP` (`"1"`),
   `INFR_METAL_PROFILE` (`"2"`/`"3"`), `INFR_ROCM_WMMA_TILE` (`"1x1"`/`"2x1"`/
   `"2x2"`).
5. **Empty string counts as "set"** for `is_ok()`/`is_some()` knobs but as "off"
   for `budget::flag_from` (`budget.rs:122-123`: neither `""` nor `"0"`) and for
   `INFR_API_KEY` (`.filter(|k| !k.is_empty())`). Preserve each site's grammar;
   do not unify them in this campaign.
6. **Do not memoize — and note three families ALREADY do.** No `OnceLock` around
   a config read (that is what broke `INFR_PAGER_STATS`). `gemv_knobs()`
   (`recorder.rs:177`), `cap_from_env` (`recorder.rs:679`, `AtomicU64`) and
   `spin_limit()` (`cpu/pool.rs:75`) memoize today. Migrating them REMOVES a
   memo, which means the value is now read per-call unless you hoist it to a
   struct field — do the hoist, or you have added a hot-path cost.
7. **`Sampler::from_env` has a doc contract**: unset ⇒ greedy, so library
   callers and goldens stay deterministic. `SamplingCfg::default()` must be
   `temp: 0.0`, `top_k: 20`, `top_p: 0.95`, `seed: None`.
8. **There are TWO `INFR_DEV` parsers and they are NOT shared.**
   `parse_dev_spec` lives in `crates/infr-cli/src/main.rs:102` and produces a
   `Backend` enum; `resolve_infr_dev_index` lives in
   `crates/infr-vulkan/src/lib.rs` (called at `:1043`) and produces a
   physical-device index, tolerating `metal`/`cpu`/empty. `infr-vulkan` does not
   and cannot depend on `infr-cli`. Do NOT try to unify them in this campaign —
   carry the raw string in `device.dev` and leave both parsers where they are
   (§6.12).
9. **`w_off`-style hot paths**: `adapter.rs` reads knobs inside per-op lowering.
   Hoist the read to the enclosing struct at construction where the value cannot
   change mid-run (§7.0 step 4).
10. **Blocked sites**: if a read site has no owner in scope, record it below
    with file:line and why — do not invent a global.

    _(none yet — append here as slices find them)_

11. **Do not "improve" a knob while migrating it.** `INFR_DN_CHUNK_SCAN` reads
    with `.is_err()` despite its positive name; `INFR_NO_GEMV_REG` silently wins
    over `INFR_GEMV_VARIANT`; `INFR_METAL_NODELTA` is read both ways. All of
    these are R1-frozen. Fix them in a follow-up plan, not here.

## 11. Decisions of record

All eight are DECIDED; they are binding on every remaining slice.

1. **File format** — TOML; lookup `--config <PATH>` → `./infr.toml` →
   `$XDG_CONFIG_HOME/infr/config.toml` (else `~/.config/infr/config.toml`);
   first existing file wins, no cross-file merging.
2. **Delivery** — thread `Arc<Config>`. A global/`OnceLock` is rejected, also as
   a shortcut for an awkward site: that is a blocked site (§10.10).
3. **`--set`** — ships, ADDITIVE to every existing bespoke flag (all keep their
   names and semantics). Same field from both ⇒ the bespoke flag wins and a
   warning names the field; two `--set`s for one path ⇒ error.
4. **Execution** — one slice per opus subagent; lead reviews, fixes, merges,
   pushes, and prunes this document after each stage so it holds only PENDING
   work.
5. **Unknown TOML key** — warn to stderr and ignore (an older binary must read a
   newer file). Malformed TOML, or a wrong-typed value for a KNOWN key, is still
   a hard error. `--set` with an unknown path is a hard error with a
   did-you-mean.
6. **`--set` grammar** — the CONFIG path (`kernels.vulkan.flash_splits=2`),
   identical to the TOML key path. Env NAMES are not accepted (not 1:1 with
   fields).
7. **Diagnostics from the file layer** — allowed; when a `prof.*`/`debug.*`
   field is non-default AND came from the FILE layer, print one startup line
   naming the file and the fields.
8. **KV dtype presence** — `KvCfg` keeps the parsed dtype AND a `*_specified`
   flag, so `INFR_KV_TYPE_K=nonsense` still suppresses auto-q8 and still falls
   through to f16, exactly as today.
