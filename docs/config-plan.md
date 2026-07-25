# config-plan.md — replace the `INFR_*` env gates with a layered `Config`

**Status: PLAN ONLY. No code has been written for this yet.** Every section
below is prescriptive: follow it literally. Where a decision is still open it is
marked **[DECIDE]** and must be answered by the repo owner before the slice that
needs it starts.

**Facts in this document were verified against `6573fb3`.** Line numbers drift;
every read site is quoted with enough source text to be re-found by string
search. The knob inventory (§6) is NOT hand-maintained truth — §6.0 gives the
commands that re-derive it, and S0 checks a generated manifest into the tree so
the compiler, not this file, is the authority.

## 1. The problem

Runtime behaviour is currently steered by **179 distinct `INFR_*` keys** read
across **206 literal `std::env::var` / `env::var_os` call sites** plus ~30 more
reads that go through a helper taking the key name as a `&str` parameter (§6.0).
They are spread over `infr-vulkan` (64 keys), `infr-llama` (38), `infr-metal`
(20), `infr-cli` (13), `infr-rocm` (13), `infr-cpu` (6), `infr-core`,
`infr-server`, `infr-chat`, `infr-gguf`, `infr-prof-rt`. The full inventory is
§6.

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

### 6.0 Re-derive this inventory before you trust it

These tables are a starting point, not an oracle. Run this first; if the count
differs from 179 the tree has moved and the tables below are stale:

```bash
# 1. every INFR_* literal in shipped code (this is the S0 manifest seed)
grep -rhoE '"INFR_[A-Z_0-9]*"' crates/*/src crates/*/build.rs | tr -d '"' | sort -u \
  | grep -vE '_TEST_|^INFR_(TP|EP|CPU|METAL)$'
# → 179 keys at 6573fb3

# 2. the subset a naive grep finds (do NOT stop here — see §1.6)
grep -rn 'env::var\(_os\)\?("INFR_' crates/*/src crates/*/build.rs
# → 206 sites / 153 keys

# 3. the ~30 reads a helper hides
grep -rn 'env_flag(\|env_mib(\|overflow_vram_reserve(\|EnvRows {\|disable_env:\|parse_device_list(\|cap_from_env(\|GemvKnobs::resolve' crates/*/src
```

Excluded by the filter above and deliberately so:

- `INFR_BUDGET_TEST_*`, `INFR_TIER_TEST_*`, `INFR_TEST_ENV_GUARD_*` — fixtures
  inside `#[cfg(test)]` modules that exist only to prove a wrapper reads its own
  variable.
- `INFR_TP` / `INFR_EP` — not keys. They are `label` strings passed to
  `parse_device_spec(spec, min, label)` in `seam/mod.rs`'s unit tests for error
  messages. The real keys are `INFR_TENSOR_PARALLEL` / `INFR_EXPERT_PARALLEL`
  (§6.11).
- `INFR_CPU` / `INFR_METAL` — **dead**. Removed as backend selectors; the only
  remaining references are `cli/main.rs`'s
  `legacy_metal_cpu_flags_are_no_longer_read` regression test and doc comments.
  Nothing reads them. Do not add them to `Config`; do not remove the test.

### 6.1 `device` — `DeviceCfg`

| Env                      | Grammar                                | Config path                               | Default                                                       | Read sites                                                                               |
| ------------------------ | -------------------------------------- | ----------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `INFR_DEV`               | string, TWO parsers — see §6.12        | `device.dev: Option<String>`              | `None`                                                        | `cli/main.rs:91` (→ `parse_dev_spec`); `vulkan/lib.rs:1043` (→ `resolve_infr_dev_index`) |
| `INFR_CTX`               | size (`parse_size`)                    | `device.ctx: Option<SizeSpec>`            | `None`                                                        | `cli/main.rs:3481`; `llama/chat/mod.rs:164`; `llama/chat/diffusion.rs:140`               |
| `INFR_UBATCH`            | int `>0`, ALSO presence — see §6.12    | `device.ubatch: Option<usize>`            | `1024` / iGPU-adaptive                                        | value: `llama/seam/mod.rs:577` (`ubatch_rows`); presence: `mod.rs:1254,1292,1354`        |
| `INFR_UBATCH_PARALLEL`   | int `>0`, default 256                  | `device.ubatch_parallel: usize`           | `256`                                                         | `llama/seam/mod.rs:613` (`ubatch_rows_parallel`)                                         |
| `INFR_SUBMIT_DISPATCHES` | int; unparseable = HARD ERROR; `0`=off | `device.submit_dispatches: Option<usize>` | `initial_submit_dispatch_cap(integrated)`                     | `vulkan/lib.rs:1980`                                                                     |
| `INFR_SG`                | exactly `"16"` or `"32"`; else ERROR   | `device.subgroup_pref: Option<u32>`       | vendor-derived (16 on Intel with `subgroup_min<=16`, else 32) | `vulkan/lib.rs:1852`                                                                     |
| `RAYON_NUM_THREADS`      | int (NOT `INFR_*`)                     | `device.threads: Option<usize>`           | `None`                                                        | set by cli; read by rayon                                                                |

`RAYON_NUM_THREADS` is third-party: keep publishing it as an env var from the
CLI (rayon has no other input), but source the value from `cfg.device.threads`.

**`INFR_UBATCH` default is not `None`.** `ubatch_rows()` falls back to the
placement-pin, then to `default_ubatch_rows()` = 1024 on a discrete device / on
CPU+Metal, and `infr_core::integrated_ubatch_rows(cu)` on an integrated GPU
(`seam/mod.rs:577-600`). `Option<usize>` in the config means "user pinned it";
the fallback chain stays where it is (R5).

**`INFR_SG` and `INFR_SUBMIT_DISPATCHES` reject bad values loudly today.** The
env layer must keep erroring, not silently fall back — that is a `Result` out of
`ConfigLayer::env()`, not an `unwrap_or`.

### 6.2 `sampling` — `SamplingCfg`

| Env               | Grammar                       | Config path                  | Default             | Read sites                                                                                                                 |
| ----------------- | ----------------------------- | ---------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `INFR_TEMP`       | f32, default 0.0              | `sampling.temp: f32`         | `0.0` (greedy)      | `llama/sampling.rs:44` (`from_env`); presence-probed at `cli/main.rs:3359`, printed at `:3447`                             |
| `INFR_TOP_K`      | usize, default 20             | `sampling.top_k: usize`      | `20`                | `llama/sampling.rs:45`; `cli/main.rs:3362,3448`                                                                            |
| `INFR_TOP_P`      | f32, default 0.95             | `sampling.top_p: f32`        | `0.95`              | `llama/sampling.rs:46`; `cli/main.rs:3365,3449`                                                                            |
| `INFR_SEED`       | u64, TWO defaults — see §6.12 | `sampling.seed: Option<u64>` | `None` ⇒ wall-clock | `llama/sampling.rs:383` (`seed_rng`, default = nanos); `cli/main.rs:2306` and `llama/chat/diffusion.rs:268` (default `42`) |
| `INFR_MAX_NEW`    | usize, default 2048           | `sampling.max_new: usize`    | `2048`              | `cli/main.rs:1657`                                                                                                         |
| `INFR_IGNORE_EOS` | presence                      | `sampling.ignore_eos: bool`  | `false`             | `cli/main.rs:2301`; `llama/seam/runner.rs:3938`; `llama/seam/model.rs:1593`; `llama/mtp/mod.rs:2380`                       |
| `INFR_NO_THINK`   | set AND `!= "0"`              | `sampling.no_think: bool`    | `false`             | `chat/template.rs:220`                                                                                                     |

`Sampler::from_env()` (llama/sampling.rs) becomes
`Sampler::from_cfg(&SamplingCfg)`. Its per-request override path
(`Sampler::resolve(req)`) is unchanged — `RequestCtx` still wins over config,
exactly as it wins over env today (§5.1).

**`INFR_SEED` has no single default** — see §6.12. Model it as `Option<u64>` and
keep BOTH fallbacks at their call sites (R5): `seed_rng()` keeps its wall-clock
branch, the two `unwrap_or(42)` sites keep the 42. A `sampling.seed: u64 = 42`
would make every `infr run` deterministic and is a behaviour change.

`cli/main.rs:3359-3366` reads `INFR_TEMP`/`TOP_K`/`TOP_P` with `.is_err()` to
decide whether to publish the flag value — that whole block disappears in S1
(the flags fill `ConfigOverrides` instead), so it needs no config field.

### 6.3 `kv` — `KvCfg`

| Env                           | Grammar                      | Config path                           | Default                            | Read sites                                                                          |
| ----------------------------- | ---------------------------- | ------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------- |
| `INFR_KV_TYPE_K`              | dtype name; ALSO presence    | `kv.type_k: Option<DType>`            | `None`                             | value: `llama/seam/runner.rs:451,476` (`parse_kv_fmt`); presence: `seam/mod.rs:725` |
| `INFR_KV_TYPE_V`              | dtype name; ALSO presence    | `kv.type_v: Option<DType>`            | `None`                             | value: `llama/seam/runner.rs:451,477`; presence: `seam/mod.rs:726`                  |
| `INFR_KV_Q8`                  | presence                     | `kv.force_q8: bool`                   | `false`                            | `seam/mod.rs:727`; `seam/model.rs:341`; `seam/runner.rs:464`                        |
| `INFR_KV_SLOTS`               | usize, default 4             | `kv.slots: usize`                     | `4`                                | `llama/seam/model.rs:200`                                                           |
| `INFR_NO_KV_RING`             | presence-inv                 | `kv.ring: bool`                       | `true`                             | `llama/seam/mod.rs:828`                                                             |
| `INFR_KV_INLINE`              | presence                     | `kv.inline_decode: bool`              | `false`                            | `vulkan/adapter.rs:2746`                                                            |
| `INFR_KV_COOPMAT_BDA`         | presence                     | `kv.coopmat_bda: bool`                | `false`                            | `vulkan/adapter.rs:2883`                                                            |
| `INFR_KV_OVERFLOW`            | flag (`budget::flag_from`)   | `kv.overflow: bool`                   | `false`                            | `llama/seam/mod.rs:720`; `vulkan/lib.rs:587`; `rocm/backend.rs:44`                  |
| `INFR_KV_OVERFLOW_VRAM_MB`    | MiB (`budget::mib_from`)     | `kv.overflow_vram_mb: Option<u64>`    | `None`                             | `vulkan/lib.rs:609`; `rocm/backend.rs:72`                                           |
| `INFR_KV_OVERFLOW_RESERVE_MB` | MiB (`budget::reserve_from`) | `kv.overflow_reserve_mb: Option<u64>` | `None` ⇒ `max(12% of VRAM, 2 GiB)` | `rocm/backend.rs:37`                                                                |

The K/V dtype parser already exists twice (`seam/runner.rs::parse_kv_fmt`,
`budget::parse_kv_dtype`); the config layer uses
`infr_core::budget::parse_kv_dtype` (`budget.rs:51`) — do not write a third.
Note `parse_kv_fmt` is a **gated** parse: the requested dtype is silently
downgraded to f16 unless the backend/alignment gates pass (`runner.rs:453-473`).
That gating is policy and stays at the call site (R5); the config carries only
the requested `DType`.

`flag_from` grammar (`budget.rs:122`): `Some(v)` with `v` neither `""` nor `"0"`
⇒ on. This is NOT the same as `is_ok()` — see §10.5.

### 6.4 `paging` — `PagingCfg`

| Env                                     | Grammar                        | Config path                                       | Default                         | Read sites                                                               |
| --------------------------------------- | ------------------------------ | ------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------ |
| `INFR_CACHE`                            | size (`parse_size`)            | `paging.cache: Option<SizeSpec>`                  | `None`                          | `llama/seam/mod.rs:907`                                                  |
| `INFR_PAGER_RING`                       | size, `>0`                     | `paging.ring: Option<SizeSpec>`                   | `None` ⇒ budget-fraction policy | `core/pager.rs:276` (`ring_bytes_policy`; pure half = `ring_bytes_from`) |
| `INFR_PAGER_STATS`                      | presence                       | `paging.stats: bool`                              | `false`                         | `vulkan/pager.rs:489`; `rocm/pager.rs:225`                               |
| `INFR_ROCM_EXPERT_BUDGET`               | size (`parse_size`)            | `paging.rocm_expert_budget: Option<SizeSpec>`     | `None`                          | `llama/seam/mod.rs:173`                                                  |
| `INFR_ROCM_WEIGHT_PREFETCH_SLOTS`       | usize, default 4, floored at 2 | `paging.rocm_prefetch_slots: Option<usize>`       | `4` (`DEFAULT_N_SLOTS`)         | `rocm/weight_pager.rs:92`                                                |
| `INFR_ROCM_WEIGHT_PREFETCH_MAX_BANK_MB` | usize MiB, default 256         | `paging.rocm_prefetch_max_bank_mb: Option<usize>` | `256` (`DEFAULT_MAX_BANK_MB`)   | `rocm/weight_pager.rs:81`                                                |
| `INFR_ROCM_WEIGHT_PREFETCH_OFF`         | presence                       | `paging.rocm_prefetch_off: bool`                  | `false`                         | `rocm/weight_pager.rs:156`                                               |
| `INFR_ROCM_WEIGHT_PREFETCH_STATS`       | presence                       | `paging.rocm_prefetch_stats: bool`                | `false`                         | `rocm/weight_pager.rs:210`                                               |
| `INFR_ROCM_WEIGHT_OVERFLOW`             | flag (`budget::flag_from`)     | `paging.rocm_weight_overflow: bool`               | `false`                         | `rocm/backend.rs:84`                                                     |
| `INFR_ROCM_WEIGHT_VRAM_MB`              | MiB (`budget::mib_from`)       | `paging.rocm_weight_vram_mb: Option<u64>`         | `None`                          | `rocm/backend.rs:94`                                                     |
| `INFR_ROCM_WEIGHT_OVERFLOW_RESERVE_MB`  | MiB (`budget::reserve_from`)   | `paging.rocm_weight_reserve_mb: Option<u64>`      | `None` ⇒ `max(12%, 2 GiB)`      | `rocm/backend.rs:108`                                                    |
| `INFR_ROCM_PAGER_NOOVERLAP`             | presence                       | `paging.rocm_no_overlap: bool`                    | `false`                         | `rocm/pager.rs:376`                                                      |

The two prefetch defaults are **not `None`** — an earlier draft of this table
said they were. `max_bank_bytes()` returns `DEFAULT_MAX_BANK_MB * 1024 * 1024`
when unset and `n_slots()` returns `DEFAULT_N_SLOTS.max(2)`
(`weight_pager.rs:69,74`).

### 6.5 `kernels.vulkan` — `VulkanCfg` (the biggest group)

Coopmat / tier selection (all read once in `VulkanBackend::new`, §5.2):

| Env                 | Grammar      | Config path          | Default | Site              |
| ------------------- | ------------ | -------------------- | ------- | ----------------- |
| `INFR_NO_COOPMAT`   | presence-inv | `coopmat: bool`      | `true`  | `lib.rs:1585`     |
| `INFR_CM_8X8`       | presence     | `coopmat_8x8: bool`  | `false` | `lib.rs:1520`     |
| `INFR_BF16_COOPMAT` | presence     | `bf16_coopmat: bool` | `false` | `adapter.rs:1558` |
| `INFR_F8_COOPMAT`   | presence     | `f8_coopmat: bool`   | `false` | `adapter.rs:1535` |
| `INFR_F8_PREPACK`   | presence     | `f8_prepack: bool`   | `false` | `adapter.rs:1587` |
| `INFR_I8_COOPMAT`   | presence     | `i8_coopmat: bool`   | `false` | `adapter.rs:1538` |
| `INFR_I8_ROW_SCALE` | presence     | `i8_row_scale: bool` | `false` | `adapter.rs:1618` |
| `INFR_NO_F16`       | presence-inv | `f16: bool`          | `true`  | `lib.rs:1583`     |
| `INFR_NO_I8DOT`     | presence-inv | `i8_dot: bool`       | `true`  | `lib.rs:1615`     |

GEMM / GEMV tiers:

| Env                                   | Grammar           | Config path                   | Default                 | Sites                                               |
| ------------------------------------- | ----------------- | ----------------------------- | ----------------------- | --------------------------------------------------- |
| `INFR_NO_GEMM_WARP`                   | presence-inv      | `gemm_warp: bool`             | `true`                  | `recorder.rs:2018`; `adapter.rs:960,1503,1725,4993` |
| `INFR_GEMM_WIDE_TILE`                 | presence          | `gemm_wide_tile: bool`        | `false`                 | `recorder.rs:2165`                                  |
| `INFR_NO_SMALL_BM`                    | presence-inv      | `small_bm: bool`              | `true`                  | `recorder.rs:2172`                                  |
| `INFR_NO_BM16`                        | presence-inv      | `bm16: bool`                  | `true`                  | `recorder.rs:2180`                                  |
| `INFR_NO_MMQ`                         | presence-inv      | `mmq: bool`                   | `true`                  | `adapter.rs:1667,4997`                              |
| `INFR_NO_MMQ_FALLBACK`                | presence-inv      | `mmq_fallback: bool`          | `true`                  | `adapter.rs:1471`                                   |
| `INFR_NO_MMV`                         | **presence-inv**  | `mmv: bool`                   | `true`                  | `adapter.rs:409,1341`                               |
| `INFR_MMV_DECODE`                     | presence          | `mmv_decode: bool`            | `false`                 | `adapter.rs:409`                                    |
| `INFR_NO_MMV_M4` / `INFR_NO_MMV_O4`   | presence-inv      | `mmv_m4` / `mmv_o4: bool`     | `true`                  | `recorder.rs:3386` / `:3385`                        |
| `INFR_MMV_MW`                         | `"0"`/other/unset | `mmv_mw: Option<bool>`        | `None` (vendor default) | `adapter.rs:509`                                    |
| `INFR_MMV_MW_WARPS`                   | int               | `mmv_mw_warps: Option<usize>` | `None`                  | `adapter.rs:740`                                    |
| `INFR_NO_MROW` / `INFR_NO_MROW16`     | presence-inv      | `mrow` / `mrow16: bool`       | `true`                  | `adapter.rs:1360` / `:1356`                         |
| `INFR_NO_F32_MROW` / `INFR_NO_F32_V4` | presence-inv      | `f32_mrow` / `f32_v4: bool`   | `true`                  | `recorder.rs:1769` / `:1770`                        |
| `INFR_MOE_SMALL_M`                    | int, clamp 0..=64 | `moe_small_m: usize`          | `8` (`tier::EnvRows`)   | `adapter.rs:159` (`MOE_SMALL_M`)                    |
| `INFR_CANVAS_CHUNK_N`                 | int, clamp 1..    | `canvas_chunk_n: usize`       | `3` (`tier::EnvRows`)   | `adapter.rs:905` (`CANVAS_CHUNK_N`)                 |

**`INFR_NO_MMV` is `presence-inv`, not `presence`** — an earlier draft had this
backwards. Both sites are `std::env::var("INFR_NO_MMV").is_err()`. The field is
`mmv: bool` defaulting to `true`. Getting this wrong disables the mmv tier for
everyone. `adapter.rs:409` is
`env::var("INFR_MMV_DECODE").is_ok() && env::var("INFR_NO_MMV").is_err()` — i.e.
`cfg.mmv_decode && cfg.mmv`.

Attention:

| Env                                      | Grammar                 | Config path                   | Default                            | Sites                        |
| ---------------------------------------- | ----------------------- | ----------------------------- | ---------------------------------- | ---------------------------- |
| `INFR_FLASH_SPLITS`                      | int                     | `flash_splits: Option<usize>` | `None`                             | `recorder.rs:4437,4689`      |
| `INFR_FLASH_BM`                          | exact string `"32"`     | `flash_bm32: bool`            | `false`                            | `recorder.rs:4463`           |
| `INFR_FLASH_MIN_ROWS`                    | usize, default 24       | `flash_min_rows: usize`       | `24`                               | `adapter.rs:2756`            |
| `INFR_FLASH_STAGE`                       | presence                | `flash_stage: bool`           | `false`                            | `adapter.rs:2872`            |
| `INFR_FLASH_DEQUANT`                     | presence                | `flash_dequant: bool`         | `false`                            | `adapter.rs:2788`            |
| `INFR_NO_FLASH_WARP`                     | presence-inv            | `flash_warp: bool`            | `true`                             | `recorder.rs:4484`           |
| `INFR_NO_NC_FA`                          | presence-inv            | `nc_fa: bool`                 | `true`                             | `adapter.rs:2934`            |
| `INFR_NO_QK_WARP` / `INFR_NO_PV_WARP`    | presence-inv            | `qk_warp` / `pv_warp: bool`   | `true`                             | `recorder.rs:4283` / `:4344` |
| `INFR_PV_SPLITS`                         | int                     | `pv_splits: Option<usize>`    | `None`                             | `recorder.rs:4325`           |
| `INFR_NO_ATTN_HD`                        | presence                | `no_attn_hd_spec: bool`       | `false` (de-memoized in `b9069a3`) | `gemm.rs:2865`               |
| `INFR_NO_MROWS_ATTN` / `INFR_MROWS_ATTN` | presence-inv / presence | `mrows_attn: Option<bool>`    | `None` ⇒ heuristic                 | `adapter.rs:2957` / `:2958`  |

**The mrows-attn pair is not a symmetric tri-state.** The site is
`… && env::var("INFR_NO_MROWS_ATTN").is_err() && ((rows >= 12 && kv_len >= 8192) || env::var("INFR_MROWS_ATTN").is_ok())`.
So: `NO_` set ⇒ off unconditionally (wins); `MROWS_ATTN` set ⇒ bypasses the
`rows`/`kv_len` heuristic; neither ⇒ heuristic decides. `Option<bool>` models it
only if `Some(false)` wins over `Some(true)` when both env vars are set. Encode
that in the env layer and test it.

DeltaNet + misc:

| Env                                     | Grammar      | Config path                   | Default | Sites                                                         |
| --------------------------------------- | ------------ | ----------------------------- | ------- | ------------------------------------------------------------- |
| `INFR_DN_CHUNK_SCAN`                    | presence-inv | `dn_chunk_scan: bool`         | `true`  | `adapter.rs:3447`                                             |
| `INFR_NO_DN_CHUNK` / `INFR_NO_DN_SPLIT` | presence-inv | `dn_chunk` / `dn_split: bool` | `true`  | `adapter.rs:3426` / `:3447,3487`                              |
| `INFR_DELTA_STRIDED`                    | presence     | `delta_strided: bool`         | `false` | `adapter.rs:3529`; ALSO `llama/seam/runner.rs:1911`           |
| `INFR_NO_PUSH_DESC`                     | presence-inv | `push_desc: bool`             | `true`  | `lib.rs:1357`                                                 |
| `INFR_NO_PIPELINE_CACHE`                | presence-inv | `pipeline_cache_disk: bool`   | `true`  | `pcache.rs:153`                                               |
| `INFR_NO_VRAM_GUARD`                    | presence     | `no_vram_guard: bool`         | `false` | `lib.rs:2297`                                                 |
| `INFR_NO_MOE_SM_POOL`                   | presence     | `no_moe_sm_pool: bool`        | `false` | `adapter.rs:4115`                                             |
| `INFR_SEAM_NO_REPLAY`                   | presence     | `no_replay: bool`             | `false` | `adapter.rs:175`; ALSO `llama/seam/runner.rs:3865` (`is_err`) |
| `INFR_NO_GPU_POS`                       | presence-inv | `gpu_pos: bool`               | `true`  | `adapter.rs:4473`; ALSO `llama/seam/runner.rs:3961`           |
| `INFR_NO_FUSE_ADD`                      | presence-inv | `fuse_add: bool`              | `true`  | `adapter.rs:869` → `core/fusion.rs:83`                        |

`INFR_DN_CHUNK_SCAN` is spelled POSITIVELY but read with `.is_err()` — setting
it **disables** the chunked scan. This is the one knob whose name lies about its
polarity; do not "fix" the name (R2).

`INFR_NO_FUSE_ADD` is not read in `adapter.rs` at all — it is a `&'static str`
in `fusion_cfg()`'s `FusionCfg.disable_env`, consumed by
`infr_core::fusion::env_disabled` (`fusion.rs:82-83`, `var_os(name).is_some()`).
Migrating it means changing `FusionCfg`'s field from
`disable_env: Option<&'static str>` to a resolved `enabled: bool`, which touches
all three backends at once (Vulkan + the two ROCm hatches, §6.7) — do it as one
change in whichever of S5/S6 lands first, and note it in the other's commit.

### 6.5b `kernels.vulkan` — the GEMV family (11 keys, currently `OnceLock`-memoized)

`recorder.rs:143-179` already has the exact shape this plan wants:
`GemvKnobs::resolve(get: impl Fn(&str) -> Option<String>)` is a **pure**
resolver over an injected reader, and `gemv_knobs()` is the impure `OnceLock`
wrapper that feeds it `std::env::var`. Migration = delete the wrapper, call
`resolve` from the config layer, store `GemvKnobs` in `VulkanCfg`. This is the
cheapest whole group in the campaign and should be done FIRST inside S5b as the
pattern-setter.

| Env                   | Grammar                      | Config path (`kernels.vulkan.gemv.*`) | Default       |
| --------------------- | ---------------------------- | ------------------------------------- | ------------- |
| `INFR_NO_GEMV_RM`     | presence                     | `no_rm: bool`                         | `false`       |
| `INFR_GEMV_RM`        | u32, default 2               | `rm: u32`                             | `2`           |
| `INFR_GEMV_RM_MAXOUT` | usize, default `usize::MAX`  | `rm_maxout: usize`                    | `usize::MAX`  |
| `INFR_GEMV_RM_MINOUT` | usize, default 2048          | `rm_minout: usize`                    | `2048`        |
| `INFR_NO_GEMV_SG`     | presence                     | `no_sg: bool`                         | `false`       |
| `INFR_NO_GEMV_ID_SG`  | presence                     | `no_id_sg: bool`                      | `false`       |
| `INFR_GEMV_SG_MINOUT` | usize, default 2048          | `sg_minout: usize`                    | `2048`        |
| `INFR_GEMV_SG_MAXOUT` | usize, default 8192          | `sg_maxout: usize`                    | `8192`        |
| `INFR_GEMV_SG_NR`     | u32, default 2               | `sg_nr: u32`                          | `2`           |
| `INFR_NO_GEMV_REG`    | presence; overrides the next | (folds into `variant`)                | —             |
| `INFR_GEMV_VARIANT`   | string                       | `variant: Option<String>`             | `Some("reg")` |

`variant` is computed, not read:
`if NO_GEMV_REG set { None } else { GEMV_VARIANT.or(Some("reg")) }`
(`recorder.rs:156-160`). Keep that expression verbatim; the config field holds
the RESULT, and the two env keys both feed it.

`recorder.rs:9423` (`gemv_knobs_resolve_matches_env_reads`) already unit-tests
`GemvKnobs::resolve` against a `HashMap` of synthetic env states — that test
survives the migration unchanged and is the template for §8.8.

### 6.5c `kernels.vulkan` — BDA chunk caps (2 keys, atomic-memoized)

| Env                    | Grammar                         | Config path                    | Default              |
| ---------------------- | ------------------------------- | ------------------------------ | -------------------- |
| `INFR_BDA_CHUNK_ELEMS` | u64, trimmed, `>= 2` else unset | `bda_chunk_elems: Option<u64>` | `BDA_CHUNK_UNIT_MAX` |
| `INFR_BDA_CHUNK_BYTES` | u64, trimmed, `>= 2` else unset | `bda_chunk_bytes: Option<u64>` | `BDA_CHUNK_UNIT_MAX` |

`recorder.rs:679-694` (`cap_from_env`) seeds an `AtomicU64` once per process; a
value below 2 is treated as unset. Note `bda_chunk_elem_cap()` /
`bda_chunk_byte_cap()` already have a `#[cfg(test)]` thread-local override
(`TEST_CHUNK_CAP`, `recorder.rs:715-725`) — once the knob is on `Config`, DELETE
that override and the `cfg(test)` branch; the parity test builds a `Config`
instead. That deletion is the exit criterion for this pair.

### 6.6 `kernels.metal` — `MetalCfg`

15 `presence-inv` disable-switches. Config fields are POSITIVE (`f16_cmm: bool`,
default `true`); the env layer inverts:

`INFR_METAL_NO_F16_NATIVE` (`exec.rs:2967`), `INFR_METAL_NO_F32_NATIVE`
(`:2969`), `INFR_METAL_NO_BF16_NATIVE` (`:2971`), `INFR_METAL_NO_F16_CMM`
(`:2978`), `INFR_METAL_NO_BF16_CMM` (`:2987`), `INFR_METAL_NO_F32_CMM`
(`:2996`), `INFR_METAL_NO_F16_RT` (`:3019`), `INFR_METAL_NO_BF16_RT` (`:3021`),
`INFR_METAL_NO_F32_RT` (`:3024`), `INFR_METAL_NO_KQUANT_NATIVE` (`:2127`),
`INFR_METAL_NO_Q5K_RT` (`:2634`), `INFR_METAL_NO_RMSNORM_VEC4` (`:2310`),
`INFR_METAL_NO_CONV1D_PAR` (`:4711`), `INFR_METAL_NO_DN_GATE_PREP` (`:4825`),
`INFR_METAL_NO_DN_NORM_PREP` (`:4828`).

Plus five that are NOT plain disable-switches:

| Env                     | Grammar                                                                         | Config path                          | Default |
| ----------------------- | ------------------------------------------------------------------------------- | ------------------------------------ | ------- |
| `INFR_METAL_LMHEAD_MRV` | presence ⇒ LIFTS the mrv `out_f` ceiling; read as `.is_err()` at `exec.rs:2678` | `metal.lmhead_mrv_uncapped: bool`    | `false` |
| `INFR_METAL_NODELTA`    | BOTH grammars — see §6.12                                                       | `metal.deltanet: bool`               | `true`  |
| `INFR_METAL_NOMOE`      | BOTH grammars — see §6.12                                                       | `metal.moe: bool`                    | `true`  |
| `INFR_METAL_PROFILE`    | presence ⇒ on; exact `"2"` ⇒ `prof_ops`; exact `"3"` ⇒ counter set              | `prof.metal_profile: Option<String>` | `None`  |
| `INFR_METAL_PROF_DEBUG` | presence                                                                        | `prof.metal_prof_debug: bool`        | `false` |

`INFR_METAL_PROFILE` is **not** an integer level: `lib.rs:184` is `is_ok()`,
`lib.rs:185` is `== Ok("2")`, `lib.rs:149` is `== Ok("3")`.
`INFR_METAL_PROFILE=1` enables profiling but neither op-profiling nor counters.
Model it as `Option<String>` and derive the three booleans in the accessor (R5),
or the levels will not compose the way they do today.

**Metal cannot be run on the dev box.** Its slice is compile-checked with
`cargo check -p infr-metal --all-targets --target x86_64-apple-darwin` and
validated by the macOS CI job. Do not guess at behaviour: keep the polarity
table in the commit message.

### 6.7 `kernels.rocm` / `kernels.cpu`

ROCm has **13 keys**, not 4 (an earlier draft undercounted badly). The paging
and overflow half lives in §6.4; the kernel half is:

| Env                      | Grammar                           | Config path                              | Default | Sites                                    |
| ------------------------ | --------------------------------- | ---------------------------------------- | ------- | ---------------------------------------- |
| `INFR_ROCM_WMMA_TILE`    | exact `"1x1"`/`"2x1"`/`"2x2"`     | `kernels.rocm.wmma_tile: Option<String>` | `None`  | `rocm/exec.rs:133`                       |
| `INFR_ROCM_NO_WMMA`      | presence                          | `kernels.rocm.no_wmma: bool`             | `false` | `rocm/exec.rs:182,1266`                  |
| `INFR_ROCM_NO_I8`        | presence                          | `kernels.rocm.i8: bool` (inverted)       | `true`  | `rocm/exec.rs:104,183,293,1267`          |
| `INFR_ROCM_NO_PIPE`      | presence-inv                      | `kernels.rocm.pipe: bool`                | `true`  | `rocm/exec.rs:193`                       |
| `INFR_ROCM_COOP`         | presence (OPT-IN)                 | `kernels.rocm.coop: bool`                | `false` | `rocm/exec.rs:236`                       |
| `INFR_ROCM_COOP_TILE`    | string, default `"128x64"`        | `kernels.rocm.coop_tile: Option<String>` | `None`  | `rocm/exec.rs:238`                       |
| `INFR_ROCM_BLAS`         | presence (OPT-IN rocBLAS prefill) | `kernels.rocm.blas: bool`                | `false` | `rocm/backend.rs:480`                    |
| `INFR_ROCM_NO_FUSE_ADD`  | presence-inv, via `FusionCfg`     | `kernels.rocm.fuse_add: bool`            | `true`  | `rocm/exec.rs:947` → `core/fusion.rs:83` |
| `INFR_ROCM_NO_FUSE_NORM` | presence-inv, via `FusionCfg`     | `kernels.rocm.fuse_norm: bool`           | `true`  | `rocm/exec.rs:951` → `core/fusion.rs:83` |

CPU:

| Env                    | Grammar                | Config path                    | Default | Sites                       |
| ---------------------- | ---------------------- | ------------------------------ | ------- | --------------------------- |
| `INFR_CPU_SPIN`        | u32, default `1 << 15` | `kernels.cpu.spin: u32`        | `32768` | `cpu/pool.rs:77` (MEMOIZED) |
| `INFR_CPU_NO_SPINPOOL` | set AND `!= "0"`       | `kernels.cpu.spinpool: bool`   | `true`  | `cpu/pool.rs:167`           |
| `INFR_CPU_REPACK_MB`   | usize, default 4096    | `kernels.cpu.repack_mb: usize` | `4096`  | `cpu/lib.rs:244` AND `:262` |

`INFR_CPU_SPIN` is behind `SPIN_LIMIT.get_or_init` — it is memoized today and is
therefore unsettable from a second test in the same process. Migrating it
removes the `OnceLock`; `spin_limit()` becomes a field read.

`CpuBackend::reference()` (`cpu/lib.rs:223`, landed `a1aed9e`) becomes
`kernels.cpu.reference: bool` in the same slice — it is already a private struct
field (`cpu/lib.rs:202`), `false` under `new()`/`Default`, set to `true` only by
the `reference()` constructor. **This is the model the rest of the campaign
should follow**: a knob that is a typed field on the owning struct, chosen by
the caller, with no env and no ambient state anywhere.

### 6.8 `spec` — `SpecCfg` (MTP / speculative decode)

| Env                       | Grammar                | Config path                   | Default | Site                                     |
| ------------------------- | ---------------------- | ----------------------------- | ------- | ---------------------------------------- |
| `INFR_MTP`                | **exact string `"1"`** | `spec.mtp: bool`              | `false` | `llama/mtp/mod.rs:106`                   |
| `INFR_MTP_TIME`           | presence               | `prof.mtp_time: bool`         | `false` | `mtp/mod.rs:2352`; `seam/runner.rs:3477` |
| `INFR_NO_MTP_CKPT`        | presence-inv           | `spec.mtp_ckpt: bool`         | `true`  | `mtp/mod.rs:2361`                        |
| `INFR_NO_MTP_REPRIME`     | **presence-inv**       | `spec.mtp_reprime: bool`      | `true`  | `mtp/mod.rs:2378`                        |
| `INFR_NO_MTP_DRAFT_CHAIN` | presence-inv           | `spec.mtp_draft_chain: bool`  | `true`  | `mtp/mod.rs:2507`                        |
| `INFR_SPEC_DRAFT`         | string path            | `spec.draft: Option<PathBuf>` | `None`  | `cli/main.rs:1421`                       |
| `INFR_SPEC_K`             | usize, default 6       | `spec.k: usize`               | `6`     | `cli/main.rs:1427`                       |
| `INFR_SPEC_DEBUG`         | presence               | `spec.debug: bool`            | `false` | `seam/model.rs:1660`                     |
| `INFR_DECODE_CHAIN`       | usize, default 8       | `spec.decode_chain: usize`    | `8`     | `seam/runner.rs:3944`                    |
| `INFR_NO_GPU_DRAFT_PROB`  | presence-inv           | `spec.gpu_draft_prob: bool`   | `true`  | `mtp/mod.rs:1749`                        |
| `INFR_NO_GPU_MTP_ACCEPT`  | presence-inv           | `spec.gpu_mtp_accept: bool`   | `true`  | `seam/runner.rs:3488`                    |
| `INFR_NO_GPU_ARGMAX`      | presence-inv           | `spec.gpu_argmax: bool`       | `true`  | `seam/runner.rs:1141,3487`               |
| `INFR_NO_GPU_SAMPLE`      | presence-inv           | `spec.gpu_sample: bool`       | `true`  | `seam/runner.rs:1153`                    |
| `INFR_NO_GPU_EMBED`       | presence-inv           | `spec.gpu_embed: bool`        | `true`  | `seam/runner.rs:406`                     |

Two corrections against an earlier draft: `INFR_NO_MTP_REPRIME` is
`presence-inv` (`mtp_ckpt && env::var("INFR_NO_MTP_REPRIME").is_err()`), NOT
`presence`; and `INFR_MTP` is not a free string — `mtp/mod.rs:106` is
`if std::env::var("INFR_MTP").ok().as_deref() != Some("1") { return … }`, so
`INFR_MTP=true` does nothing today. `spec.mtp: bool` with the env layer doing
`== Some("1")` is the behaviour-preserving mapping.

`spec.mtp_reprime` is ANDed with `spec.mtp_ckpt` at the site; keep the AND
there, not in the config (R5).

### 6.9 `prof` / `debug` / `serve`

`prof` (all presence unless noted): `INFR_PROF` (`vulkan/recorder.rs:981`,
`llama/seam/runner.rs:3637`), `INFR_PROF2` (`recorder.rs:952`),
`INFR_PROF2_SHAPES` (`recorder.rs:953`, ANDed with `prof2`), `INFR_PROF_DEC`
(`runner.rs:4206,4329`), `INFR_PROF_OPS` (`cpu/lib.rs:533`), `INFR_PROF_PF`
(`runner.rs:3819`), `INFR_PROFILE_OUT` (string path, `prof-rt/lib.rs:441`),
`INFR_VRAM_LOG` (`vulkan/lib.rs:982`), `INFR_MTP_TIME`, `INFR_DIFFUSION_TIME`
(`runner.rs:3019`), `INFR_EB_TRACE` (`llama/diffusion.rs:314`),
`INFR_METAL_PROFILE`, `INFR_METAL_PROF_DEBUG`.

`debug` (all presence): `INFR_DEBUG_BDA_CHUNK` (`recorder.rs:1523,3646`),
`INFR_DEBUG_COOPMAT` (`vulkan/lib.rs:1494,1638`), `INFR_DEBUG_WIDE_DISPATCH`
(`recorder.rs:1244`), `INFR_DEBUG_CHAT` (`chat/template.rs:231,237`),
`INFR_MOE_COUNTS_DEBUG` (`cpu/lib.rs:2081`), `INFR_MOE_COUNTS_DUMP`
(`cpu/lib.rs:2139`), `INFR_POISON_UNINIT` (`vulkan/lib.rs:3161`),
`INFR_NOBARRIER` (`recorder.rs:979`), `INFR_FULLBARRIER` (`recorder.rs:980`).

**Two knobs an earlier draft filed under `debug` are graph-shape knobs and are
`presence-inv`, not `presence`** — they must default to ENABLED:

| Env                     | Grammar      | Config path                   | Default | Site                                                         |
| ----------------------- | ------------ | ----------------------------- | ------- | ------------------------------------------------------------ |
| `INFR_NO_QKV_FUSE`      | presence-inv | `kernels.qkv_fuse: bool`      | `true`  | `llama/seam/runner.rs:50`                                    |
| `INFR_NO_GATED_RMSNORM` | presence-inv | `kernels.gated_rmsnorm: bool` | `true`  | `llama/seam/runner.rs:391` (ANDed with `caps.gated_rmsnorm`) |

They live in `infr-llama`, not `infr-vulkan`, so they migrate in **S4**, not S5.

`serve`:

| Env                   | Grammar                        | Config path                     | Default                              | Site                 |
| --------------------- | ------------------------------ | ------------------------------- | ------------------------------------ | -------------------- |
| `INFR_API_KEY`        | string, EMPTY = unset          | `serve.api_key: Option<String>` | `None`                               | `server/lib.rs:1192` |
| `INFR_MAX_TOKENS_CAP` | u32, must be `>0` else default | `serve.max_tokens_cap: u32`     | `131_072` (`DEFAULT_MAX_TOKENS_CAP`) | `server/lib.rs:1159` |

`INFR_API_KEY` uses `.filter(|k| !k.is_empty())` — an empty string means NO
auth, which is the opposite of the `is_ok()` presence grammar. Do not unify
them. `max_tokens_cap()` is deliberately read per-request (the doc comment says
so); after migration it reads `cfg.serve.max_tokens_cap`, still per-request,
still cheap.

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

### S0 — scaffold (no behaviour change, no migration)

1. Create `crates/infr-core/src/config/` per §4 with: `Config` + all section
   structs, all fields, `impl Default` reproducing today's defaults,
   `PartialConfig` + `merge`, `ConfigLayer::{file, env, cli}`, `Config::load`.
2. Generate `manifest.rs` with the §6.0 command; annotate every entry with
   section, field, grammar, default, `migrated: false`.
3. `env.rs` reads ALL keys from the manifest, but nothing consumes the result
   yet. `env.rs` takes an injected `get: &dyn Fn(&str) -> Option<String>` from
   day one (§8.8) — retrofitting that later is wasted work.
4. Add the `toml` crate to `infr-core`'s dependencies (workspace-pinned).
5. Tests in `config/tests.rs`: §8's precedence tests, plus a test asserting
   `Config::default() == Config::load_from_layers(&[])`.
6. **Do not touch any read site.** After S0 the tree behaves identically and
   every env read is still where it was.

**Exit:** `cargo test -p infr-core` green; §8.1–8.8 all present and passing;
`manifest.rs` key count matches the §6.0 command's output; zero lines changed
outside `crates/infr-core/`.

### S1 — CLI produces a `Config`

1. `infr-cli/src/main.rs`: add `--config <PATH>`; build `Config` in `main()`
   from file+env+flags; stop `set_var`-ing in `DeviceOpts::resolve` /
   `SamplingOpts::resolve` (the block at `main.rs:197-291`, plus the standalone
   writes at `:1441` (`INFR_TEMP=0`), `:1799` (`INFR_IGNORE_EOS=1`) and
   `:3360-3366`) — fill `ConfigOverrides` instead. Keep publishing
   `RAYON_NUM_THREADS` (§6.1).
2. Thread `Arc<Config>` into the commands (`run`/`serve`/`bench`/`chat`) and
   into `SeamModel::load`.
3. **Until later slices land, the deep readers still read env** — so in S1 ONLY,
   after building the config, the CLI re-publishes the knobs it used to publish
   (`INFR_DEV`, `INFR_CTX`, `INFR_UBATCH`, `INFR_TEMP`, `INFR_TOP_K`,
   `INFR_TOP_P`, `INFR_SEED`, `INFR_MAX_NEW`, `INFR_NO_THINK`,
   `INFR_IGNORE_EOS`) so behaviour is unchanged. Delete that re-publication in
   S8. Note this re-publication is what `cmd_bench` relies on at `main.rs:1799`.
4. Convert `cli/main.rs`'s `mod tests` off its hand-rolled `ENV_LOCK` (R7).
5. Add `--set <config.path>=<value>` — DECIDED, ship it (§11 [DECIDE-3]).
   ADDITIVE to the existing flags: every current flag keeps its name and
   semantics, `--set` only reaches the knobs that have no dedicated flag. A
   bespoke flag and a `--set` targeting the SAME field ⇒ the bespoke flag wins
   and a warning names the field; two `--set`s for the same path ⇒ error. Path
   grammar is still **[DECIDE-6]** (config path vs env name).

**Exit:** `grep -n 'set_var' crates/infr-cli/src/main.rs` shows only the S1
re-publication block and `RAYON_NUM_THREADS`; `infr run` / `infr bench` /
`infr serve` produce identical output to the parent commit on a fixed seed.

### S2 — `infr-core`'s own knobs

`tier::EnvRows` (2 knobs: `INFR_MOE_SMALL_M`, `INFR_CANVAS_CHUNK_N` — both
DECLARED in `infr-vulkan/src/adapter.rs` but RESOLVED by `infr-core`),
`budget::{env_flag, env_mib, overflow_vram_reserve}` (6 keys, §6.3/§6.4),
`pager::ring_bytes_policy` (1 key). These already have pure `*_from`/`resolve`
halves (landed `b9069a3`), so the slice is: delete the env-reading wrapper, take
the value from `&Config` at the call site. Smallest possible first migration —
use it to establish the pattern the other slices copy.

**Exit:** `budget.rs`, `tier.rs` and `pager.rs` contain zero `std::env::var`;
their `INFR_*_TEST_*` fixtures and the guarded "reads its variable" tests are
deleted with the wrappers.

### S3 — `infr-cpu`

7 read sites / 6 keys (§6.7 + `INFR_PROF_OPS`, `INFR_MOE_COUNTS_DEBUG`,
`INFR_MOE_COUNTS_DUMP`). `CpuBackend::new_with(cfg)`; fold `reference: bool`
into `kernels.cpu`; delete the `SPIN_LIMIT` `OnceLock`. Rewrite
`crates/infr-cpu/tests/decode_parity.rs` to build a `Config` instead of calling
`CpuBackend::reference()`.

**Exit:** `grep -rn 'INFR_' crates/infr-cpu/src` returns nothing.

### S4 — `infr-llama` seam (51 sites / 38 keys)

`kv`, `spec`, `multi` (§6.11), `device.ubatch`, `paging.cache`, `sampling`, and
the two graph-shape knobs from §6.9 (`INFR_NO_QKV_FUSE`,
`INFR_NO_GATED_RMSNORM`). `SeamModel` and `DenseSession<B, X>` gain
`Arc<Config>`. `Sampler::from_env` → `Sampler::from_cfg`; `Sampler::resolve`'s
signature and precedence are UNCHANGED (§5.1). This is the slice that unblocks
deleting most `EnvGuard` uses in `crates/infr-llama/tests/cpu_backend.rs`.

Split into S4a (`seam/mod.rs` — placement, kv gates, multi-GPU) and S4b
(`seam/runner.rs` + `mtp/` — the decode loop) if the diff exceeds ~800 lines.

**Exit:** `crates/infr-llama/tests/cpu_backend.rs` has no `EnvGuard` use for any
`kv.*`/`spec.*`/`device.ubatch` knob; `cpu_golden` hashes unmoved.

### S5 — `infr-vulkan` (74 sites / 64 keys, split into S5a/S5b)

S5a: `VulkanBackend::new_with` + the construction-time knobs — the six
`Capabilities` maskers (§5.2), `INFR_VRAM_LOG`, `INFR_NO_VRAM_GUARD`,
`INFR_SUBMIT_DISPATCHES`, `INFR_POISON_UNINIT`, `INFR_NO_PIPELINE_CACHE`,
`INFR_DEBUG_COOPMAT`.

S5b: `Recorder`/`adapter.rs`/`gemm.rs` kernel-tier knobs — the hot paths. Do the
GEMV family first (§6.5b — it is already a pure resolver, so it is the cheapest
and sets the pattern), then the BDA chunk caps (§6.5c, delete the `cfg(test)`
override), then the tier tables. Borrow `&VulkanCfg` from the backend; R6
applies (no clones per dispatch).

**Exit for S5b specifically:** an interleaved decode+prefill bench (§9) versus
the parent commit shows no regression outside noise. This is the ONLY slice
where a wrong polarity is invisible to the tests (§10.1), so the bench is not
optional.

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

## 8. Required tests

In `config/tests.rs` (these are the acceptance criteria for S0):

1. `default_config_matches_documented_defaults` — every field's `Default` equals
   the value in `manifest.rs` (which §6 documents).
2. `env_overrides_file` — file says `flash_splits = 2`, env says `4` ⇒ `4`.
3. `cli_overrides_env` — env says `4`, CLI says `8` ⇒ `8`.
4. `absent_layer_does_not_clobber` — a file setting only `[kv]` leaves
   `[kernels.vulkan]` at its env/default value.
5. `unknown_toml_key_warns_and_is_ignored` — typo detection WITHOUT a hard
   failure, per the decided **[DECIDE-5]**; the sibling case
   `unknown_set_path_is_an_error` covers `--set`. Original note: do not write
   this test until that decision is answered.
6. `bad_value_is_an_error_not_a_silent_default` — `ctx = "banana"` fails to
   load. **But only for the keys that error TODAY** (`INFR_SG`,
   `INFR_SUBMIT_DISPATCHES`, the three `multi` device lists, and any `SizeSpec`
   field). Every other key currently swallows a bad value via
   `.and_then(parse).ok().unwrap_or(default)`, and R1 says keep it. The test
   must be table-driven over both classes, with the class recorded in
   `manifest.rs`.
7. `presence_inverted_knobs_have_the_right_polarity` — table-driven over every
   `presence-inv` entry in `manifest.rs`: env set to `""`, `"0"` and `"1"` ⇒
   field `false` in all three; env unset ⇒ field `true`. The `"0"` case is the
   one that catches a wrong-grammar migration (§7.0).
8. `env_layer_reads_every_key` — iterate `manifest::KEYS`, set each one through
   the INJECTED reader (a `HashMap<String, String>`, never the real
   environment), assert the corresponding field changed. This is what stops a
   knob being silently dropped during migration. `recorder.rs:9425`
   (`GemvKnobs::resolve` against a `HashMap`) is the working precedent — copy
   its shape.
9. `manifest_matches_the_tree` — a test that re-runs the §6.0 grep (via
   `include_str!` of a checked-in list, or `std::process::Command` gated behind
   `#[ignore]`) and fails when a new `INFR_*` literal appears in `crates/*/src`
   without a manifest entry. Without this, the next feature branch silently
   re-introduces an ungoverned knob.
10. `dotted_path_setter_rejects_unknown_paths` — REQUIRED (`--set` ships, §11
    [DECIDE-3]): `--set kernels.vulkan.flash_splt=2` must error with a
    suggestion, not be ignored. Implement by matching against `manifest::KEYS`'
    field paths — the same table, so it cannot drift from the TOML schema.
    **[DECIDE-6]**: does `--set` take the CONFIG path
    (`kernels.vulkan.flash_splits`) or the ENV name (`INFR_FLASH_SPLITS`)? They
    are not 1:1 — `INFR_NO_GEMM_WARP` maps to `gemm_warp=false`, and
    `INFR_NO_GEMV_REG` + `INFR_GEMV_VARIANT` both map to one `variant` field.

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

## 11. Open decisions

- **[DECIDE-1] — DECIDED (repo owner, 2026-07-26): YES.** TOML, with the 3-step
  lookup `--config <PATH>` → `./infr.toml` → `~/.config/infr/config.toml`, first
  existing file wins, no merging across files. §4 is now normative, not a
  proposal.
- **[DECIDE-2] — DECIDED (repo owner, 2026-07-26): thread `Arc<Config>`.** The
  `OnceLock` + test-override alternative is REJECTED — do not reintroduce it in
  any slice, and do not use it as a shortcut for a site that is awkward to
  thread (that is a blocked site, §10.10). §5 is normative.
- **[DECIDE-3] — DECIDED (repo owner, 2026-07-26): ship `--set`, AND keep every
  bespoke flag.** `--set <config.path>=<value>` is ADDITIVE: every flag that
  exists today (`--dev`, `--ctx`, `--ubatch`, `--threads`, `--temp`, `--top-k`,
  `--top-p`, `--seed`, `--max-new`, `--no-think`, …) stays exactly as it is,
  with its current name and semantics. `--set` exists so the ~150 knobs that
  have no dedicated flag are reachable without inventing ~150 flags. Precedence
  WITHIN the CLI layer when both target the same field: **the bespoke flag
  wins** over `--set`, and passing both must print a warning naming the field,
  so a user who typed `--ctx 32k --set device.ctx=8k` is told which one applied.
  Two `--set`s for the same path is an error, not a silent last-wins.
- **[DECIDE-4] — DECIDED (repo owner, 2026-07-26): delegate slices to opus
  subagents**, one slice at a time; lead reviews + fixes + merges + pushes each,
  and PRUNES this document after each stage so it holds only PENDING work.
- **[DECIDE-5] — DECIDED (lead, 2026-07-26, owner may override): WARN, do not
  fail.** An unknown key in the TOML file prints one
  `[infr] config: unknown key \`kernels.vulkan.flash_splt\`
  (ignored)`line to stderr and is skipped, so an older binary can read a newer file and removing a knob is not a breaking change. Typo protection comes from the message, not from a hard failure.`--set`
  is the OPPOSITE: an unknown path there IS a hard error (§8.10), because it was
  typed on the command line for this run and silently ignoring it would give a
  wrong result with no second chance to notice. A malformed FILE (invalid TOML
  syntax, or a value of the wrong type for a known key) stays a hard error.
- **[DECIDE-6] — DECIDED (lead, 2026-07-26, owner may override): `--set` takes
  the CONFIG path**, identical to the TOML key path
  (`--set kernels.vulkan.flash_splits=2`), so there is ONE grammar to learn and
  `--set` ⇔ file entries are copy-pasteable in both directions. Env NAMES are
  not accepted as `--set` paths (they are not 1:1 with fields — `INFR_NO_*`
  inverts, `INFR_MMV_MW` is tri-state). An unknown path errors with a
  did-you-mean suggestion computed against `manifest::KEYS`.
- **[DECIDE-7] — DECIDED (lead, 2026-07-26, owner may override): YES, the file
  may set the diagnostic knobs** — they are ordinary fields and carving out an
  exception costs more than it saves. Mitigation for the "why is my server
  printing timings" failure mode: when any `prof.*` or `debug.*` field is
  non-default AND its value came from the FILE layer (not env, not CLI), print
  one line at startup naming the file and the fields it enabled.
- **[DECIDE-8] — DECIDED (lead, 2026-07-26, owner may override): preserve
  today's behaviour exactly (R1).** `KvCfg` keeps BOTH the parsed dtype
  (`type_k: Option<DType>`) and whether the knob was SPECIFIED at all
  (`type_k_specified: bool`, set by any layer that supplied a value, parseable
  or not). `kv_env_unset()`'s successor tests `type_k_specified`, so
  `INFR_KV_TYPE_K=nonsense` keeps suppressing auto-q8 and still falls through to
  f16 for the dtype — exactly as today. Do not "fix" this asymmetry in this
  campaign; it is a behaviour change and belongs in its own commit.
