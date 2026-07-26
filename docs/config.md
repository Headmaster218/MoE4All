# Configuring infr

infr has one configuration value, resolved once at startup from four layers and
then passed explicitly to the backends and sessions. Every knob is a typed field
on `infr_core::config::Config`; nothing reads the environment behind your back.

A commented starting point lives at [`infr.example.toml`](../infr.example.toml).

## Precedence

Four layers. **Later wins**, and a layer only overrides a field it actually
specifies — leaving `temp` out of your config file does not reset it to the
default, it just leaves the question to the layers below.

| Layer                  | Source                                 | Notes                                                |
| ---------------------- | -------------------------------------- | ---------------------------------------------------- |
| 1. Defaults (lowest)   | `impl Default for Config`              | The shipped behaviour.                               |
| 2. Config file         | TOML (see below)                       | Absent file = no-op. Malformed file = error.         |
| 3. Environment         | `INFR_*`                               | Same names as always — nothing was renamed.          |
| 4. CLI flags (highest) | `--dev`, `--ctx`, `--temp`, …, `--set` | A dedicated flag beats a `--set` for the same field. |

Two consequences worth knowing:

- **Every documented `INFR_*` variable still works.** The campaign that
  introduced `Config` changed _where the value goes_, not what you type. Your
  existing `INFR_PROF2=1 infr bench …` scripts are unaffected.
- **A flag beats an inherited environment variable**, because the CLI layer sits
  above the env layer. `INFR_CTX=32k infr run … --ctx 8192` runs at 8192.

## The config file

### Lookup

The **first existing** file wins. There is no merging across files.

1. `--config <PATH>` — and it is an **error** if that path does not exist.
2. `./infr.toml`
3. `$XDG_CONFIG_HOME/infr/config.toml`, else `~/.config/infr/config.toml`

Finding no file at all is a no-op, never an error.

### Format

TOML. The section path is the struct path, so a key's full name is exactly what
`--set` takes: `[kernels.vulkan] flash_splits = 2` is
`--set kernels.vulkan.flash_splits=2`.

The file speaks the **positive** field names. Where the environment has an
`INFR_NO_*` disable-switch, the config has the thing being enabled, defaulting
to `true`:

```toml
[device]
dev = "Vulkan1"      # same grammar as --dev / INFR_DEV
ctx = "32k"          # the shared size grammar: 8192 / 256k / 50%

[kv]
type_k = "q8_0"
type_v = "q8_0"

[paging]
cache = "8g"         # force the paged expert cache with an 8 GiB budget

[kernels.vulkan]
flash_splits = 2
gemm_warp = false    # NOT `no_gemm_warp` — INFR_NO_GEMM_WARP's field, inverted

[multi]
pipeline = [0, 1]    # or ["Vulkan0", "Vulkan1"]

[serve]
max_tokens_cap = 8192
```

Value grammars: booleans accept `true`/`false` (and `1`/`0`, `yes`/`no`,
`on`/`off` from `--set`); sizes take the shared `8192` / `256k` / `50%` grammar;
device lists take an array of indices or `VulkanN` strings; an `Option` field is
cleared with `""` or `"none"`.

### Unknown keys warn; wrong types fail

An unrecognized key — or a whole unrecognized section — is **warned about on
stderr and ignored**, with a did-you-mean:

```
[infr] config: unknown key `bogus` (ignored)
[infr] config: unknown key `kernels.vulkan.flash_split` (ignored) — did you mean `kernels.vulkan.flash_splits`?
```

That is deliberate: a config file written for a newer infr must not hard-fail on
an older binary, and removing a knob must not break everyone who has it in their
file. Typo protection comes from the warning line.

A **known** key given a value that does not parse into its type is a hard error:

```
Error: config `device.ctx`: expected a size/count like 8192, 256k or 50% (got "banana")
```

Note the asymmetry with the environment, which is frozen at today's behaviour:
`ctx = "banana"` in a file fails to load, while `INFR_CTX=banana` is silently
ignored and falls back. Five environment keys do reject a bad value loudly, at
startup, on every subcommand: `INFR_SG`, `INFR_SUBMIT_DISPATCHES`,
`INFR_PIPELINE`, `INFR_TENSOR_PARALLEL`, `INFR_EXPERT_PARALLEL`.

### Diagnostics announce themselves

If the **file** turns on a `prof.*` or `debug.*` knob, infr prints one line at
startup naming the file and the fields:

```
[infr] config: /home/you/.config/infr/config.toml enabled diagnostics: prof.prof2
```

So "why is my server printing timings" is answerable from the command line, even
when the cause is a global config file you forgot about.

## `--set`

Most of the 177 knobs have no dedicated flag. `--set <config.path>=<value>`
reaches all of them, using the same path grammar as the TOML file:

```bash
infr bench "$M" -p 512 -n 0 --set kernels.vulkan.flash_splits=2
infr run "$M" "hi" --set kv.type_k=q8_0 --set kv.type_v=q8_0
```

- The path is the **config path**, never the `INFR_*` name. They are not 1:1:
  `INFR_NO_GEMM_WARP` is `kernels.vulkan.gemm_warp=false`, `INFR_NO_GEMV_REG`
  and `INFR_GEMV_VARIANT` are both `kernels.vulkan.gemv.variant`, and
  `INFR_MMV_MW` is tri-state.
- An **unknown** path is a hard error with a did-you-mean (unlike the file layer
  — you typed it for this run, so silently ignoring it would give you a wrong
  answer with no second chance to notice):
  ```
  Error: unknown config path `kernels.vulkan.flash_split` — did you mean `kernels.vulkan.flash_splits`?
  ```
- The **same path twice** is an error, not a silent last-wins:
  ```
  Error: `--set kv.slots=` given more than once
  ```
- `--set` is **additive** to the dedicated flags, and loses to them. Passing
  both prints a warning naming the field:
  ```
  $ infr run "$M" "hi" --ctx 4096 --set device.ctx=8192
  [infr] config: `--set device.ctx=8192` ignored — the dedicated flag for `device.ctx` wins
  ```

### The dedicated flags

`--config` and `--set` are global. The rest are on `run` / `serve` (device flags
also on `bench`); `infr <cmd> --help` is the authority.

| Flag                     | Config path         | Env             |
| ------------------------ | ------------------- | --------------- |
| `--dev`                  | `device.dev`        | `INFR_DEV`      |
| `--ctx`                  | `device.ctx`        | `INFR_CTX`      |
| `-u` / `--ubatch`        | `device.ubatch`     | `INFR_UBATCH`   |
| `-t` / `--threads`       | `device.threads`    | —               |
| `--temp`                 | `sampling.temp`     | `INFR_TEMP`     |
| `--top-k`                | `sampling.top_k`    | `INFR_TOP_K`    |
| `--top-p`                | `sampling.top_p`    | `INFR_TOP_P`    |
| `--seed`                 | `sampling.seed`     | `INFR_SEED`     |
| `--max-new`              | `sampling.max_new`  | `INFR_MAX_NEW`  |
| `--no-think` / `--think` | `sampling.no_think` | `INFR_NO_THINK` |

`device.threads` has no `INFR_*` twin — it is published as `RAYON_NUM_THREADS`,
because rayon's global pool has no other input.

## What is tunable

The authority is
[`crates/infr-core/src/config/manifest.rs`](../crates/infr-core/src/config/manifest.rs):
every `INFR_*` key, the config path it lands on, and its value grammar, in one
table that the tests check against the tree. `Config::all_paths()` is the full
path list. What follows is the shape plus the knobs people actually reach for.

**`[device]`** — which GPU (`dev`), how much context (`ctx`), the prefill
micro-batch (`ubatch`, `ubatch_parallel`), CPU `threads`, and two low-level
device knobs (`submit_dispatches` for the iGPU submit splitter, `subgroup_pref`
to force subgroup 16 or 32).

**`[sampling]`** — `temp` (0 = greedy), `top_k`, `top_p`, `seed`, `max_new`,
`ignore_eos`, `no_think`. **Provenance matters here**: `infr run` / `infr serve`
fill `temp` / `top_k` / `top_p` / `max_new` from the model's own recommended
sampling (an arch-family table plus any `generation_config.json` beside the
model) only for knobs that **no layer specified**. Putting `temp` in your config
file pins it and suppresses that fallback. The `Config` defaults, which library
callers get, are greedy: `temp = 0.0`, `top_k = 20`, `top_p = 0.95`,
`max_new = 2048`. On `serve` these are the server defaults — a per-request
OpenAI `temperature`/`top_p` still overrides them.

**`[kv]`** — cache element format (`type_k` / `type_v`, plus the legacy
`force_q8` alias), prefix-cache `slots`, the sliding-window `ring`, and the
host-overflow trio (`overflow`, `overflow_vram_mb`, `overflow_reserve_mb`) that
spills KV to system RAM when VRAM runs out.

**`[paging]`** — the MoE expert cache and dense layer streaming: `cache` sizes
the paged VRAM budget (and forces paging even when the weights would have fit),
`ring` overrides the upload staging ring, `stats` prints per-pool
hit/miss/eviction counts. The `rocm_*` half of the section is the ROCm backend's
weight pager (prefetch slots, bank size, overflow budgets).

**`[kernels]`** — two backend-independent graph-shape gates (`qkv_fuse`,
`gated_rmsnorm`) plus one sub-section per backend. Everything under them is a
kernel **tier** override: the engine picks the best tier the device supports,
and these exist to force one off when bisecting a correctness or perf problem.

- **`[kernels.vulkan]`** (the biggest section — 64 of the 177 keys): capability
  masks (`coopmat`, `f16`, `i8_dot`, `coopmat_8x8`, `i8_coopmat`), GEMM/GEMV
  tiers (`gemm_warp`, `mmq`, `mmv`, `mrow`, `moe_small_m`, the
  `[kernels.vulkan.gemv]` sub-table), attention (`flash_warp`, `flash_splits`,
  `flash_min_rows`, `pv_splits`), DeltaNet (`dn_chunk_scan`, `dn_chunk`,
  `dn_split`), and the plumbing switches (`push_desc`, `pipeline_cache_disk`,
  `no_replay`, `no_vram_guard`, the BDA chunk caps). Note `f16 = false` also
  disables coopmat regardless of `coopmat`, matching `INFR_NO_F16`'s effect
  today.
- **`[kernels.metal]`** — the Apple backend's native/CMM/RT kernel families per
  dtype, plus `deltanet` and `moe`.
- **`[kernels.rocm]`** — `wmma_tile` (`1x1`/`2x1`/`2x2`), `no_wmma`, the int8
  (`i8`) and software-pipelined (`pipe`) kernels, opt-in `coop` and `blas`
  prefill, the two fusion gates, and `module_cache` (persist the hiprtc-compiled
  HIP module to `~/.cache/infr`; no `INFR_*` twin, like
  `kernels.cpu.reference`).
- **`[kernels.cpu]`** — `spin` (spin-pool idle ceiling), `spinpool`,
  `repack_mb`, and `reference` (the bit-reference kernel path, which has no
  `INFR_*` twin and never had one).

**`[spec]`** — MTP / speculative decode: `mtp`, `k`, `decode_chain`, `draft` (a
draft-model path), the GPU-side sampling steps (`gpu_argmax`, `gpu_sample`,
`gpu_embed`, …). MTP is currently parked; see the README.

**`[multi]`** — multi-GPU splits: `pipeline`, `tensor_parallel` (both need ≥ 2
devices) and `expert_parallel` (≥ 1); too few devices is a hard error. The three
`*_p2p` flags choose GPU-to-GPU transport (`true`, the default) over staging
through host RAM.

**`[prof]`** — `prof`, `prof2` (per-dispatch GPU timestamps), `prof2_shapes`,
`prof_dec`, `prof_ops`, `prof_pf`, `vram_log`, `mtp_time`, `diffusion_time`,
`profile_out` (a JSON report path), and the two Metal profiling knobs.

**`[serve]`** — `api_key` (bearer token; an **empty** value means no auth) and
`max_tokens_cap`. Per-request sampling is not here — it stays on the request.

**`[debug]`** — poison/barrier/dump switches: `coopmat` (print the enumerated
and chosen coopmat shapes — useful on Intel Arc), `bda_chunk`, `wide_dispatch`,
`chat`, `moe_counts`, `moe_counts_dump`, `poison_uninit`, `no_barrier`,
`full_barrier`.

## The `INFR_*` keys that are deliberately NOT config

Four keys keep reading the environment directly, for reasons a runtime `Config`
cannot fix:

| Key                        | Why                                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `INFR_PROFILE`             | **Build-time** input: read by `build.rs` in core/cpu/gguf/llama/vulkan to set `cfg(infr_profile)`. No runtime value exists when it is read. |
| `INFR_TEST_GGUF`           | Test fixture: points `infr-gguf`'s tests at a `.gguf` on disk.                                                                              |
| `INFR_TEST_MODEL`          | Test fixture: overrides the HF-cache lookup for the model-backed tests.                                                                     |
| `INFR_LLAMA_DIFFUSION_CLI` | Dev fixture: points `infr compare` at a `llama-diffusion-cli` binary on disk.                                                               |

Two related notes. `INFR_DIFFUSION_VISUAL` is also not a `Config` field — since
it only steers CLI presentation it became the plain flag
`infr run --diffusion-visual`, whose clap `env =` fallback keeps the old
spelling working. And `INFR_CPU` / `INFR_METAL` are **dead**: they were removed
as backend selectors with no aliases. Use `--dev cpu` / `--dev metal`, or
`INFR_DEV=cpu` / `INFR_DEV=metal`.

## Library callers

`Config` is a value, not a global. Build one and hand it out:

```rust
use infr_core::config::{Config, ConfigOverrides};
use std::sync::Arc;

// The full four-layer resolve (file + env + the CLI overrides you pass).
let cfg = Arc::new(Config::load(&ConfigOverrides::default())?);

// Or construct exactly what you want — no environment, no file, no ordering hazard.
let cfg = Arc::new(Config { kv: infr_core::config::KvCfg { slots: 8, ..Default::default() },
                            ..Default::default() });
```

Backends take it at construction (`VulkanBackend::new_with(cfg)`,
`RocmBackend::new_with(device_id, cfg)`, `CpuBackend::new_with(cfg)`), and
`Config::load_from_env()` is the defaults-plus-environment fold for a caller
that wants the environment honoured but no config file.

The migration history — the layer machinery, the per-knob polarity tables, and
the slice-by-slice record — is in [`config-plan.md`](config-plan.md).
