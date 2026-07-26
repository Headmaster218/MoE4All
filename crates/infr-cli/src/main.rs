//! `infr` CLI — `pull` / `run` / `serve`, all over the same engine + backend.
//! See docs/plan.md "Product surface".

use clap::{Parser, Subcommand};
use infr_core::config::{Config, ConfigOverrides, PartialConfig};
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "infr",
    version,
    about = "Pure-Rust, Vulkan-first LLM inference engine"
)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,

    /// Config file (TOML) — skips the default lookup, and is an ERROR if the path does not exist.
    /// Unset: the first existing of `./infr.toml`, `$XDG_CONFIG_HOME/infr/config.toml`,
    /// `~/.config/infr/config.toml` (an absent file is a no-op). Precedence, later wins:
    /// defaults < file < `INFR_*` env < flags.
    #[arg(long, value_name = "PATH", global = true)]
    config: Option<PathBuf>,

    /// Set one config field: `--set kernels.vulkan.flash_splits=2`. Repeatable. The path is the
    /// CONFIG path (the same one the TOML file uses), never the `INFR_*` name — they are not 1:1.
    /// ADDITIVE to the dedicated flags: where both name the same field the dedicated flag wins and
    /// says so; the same path twice, or an unknown path, is an error.
    #[arg(long = "set", value_name = "PATH=VALUE", global = true)]
    set: Vec<String>,

    /// Print shell completions to stdout and exit (packaging helper).
    #[arg(long, value_enum, value_name = "SHELL", hide = true)]
    completions: Option<CompletionShell>,

    /// Print the man page (troff) to stdout and exit (packaging helper).
    #[arg(long, hide = true)]
    man: bool,
}

/// Shells `--completions` can generate for: clap_complete's five core shells plus nushell
/// (separate generator crate). Mirrors gpur's packaging-helper flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CompletionShell {
    Bash,
    Zsh,
    Fish,
    Powershell,
    Elvish,
    Nushell,
}

impl CompletionShell {
    fn generate(self, cmd: &mut clap::Command) {
        use clap_complete::Shell;
        let out = &mut std::io::stdout();
        match self {
            CompletionShell::Bash => clap_complete::generate(Shell::Bash, cmd, "infr", out),
            CompletionShell::Zsh => clap_complete::generate(Shell::Zsh, cmd, "infr", out),
            CompletionShell::Fish => clap_complete::generate(Shell::Fish, cmd, "infr", out),
            CompletionShell::Powershell => {
                clap_complete::generate(Shell::PowerShell, cmd, "infr", out)
            }
            CompletionShell::Elvish => clap_complete::generate(Shell::Elvish, cmd, "infr", out),
            CompletionShell::Nushell => {
                clap_complete::generate(clap_complete_nushell::Nushell, cmd, "infr", out)
            }
        }
    }
}

/// The backend selected for a forward. Produced by the ONE decision function [`resolve_backend`]
/// (from `--dev` + whatever `device.dev` the other layers carry) and by the ONE reader
/// [`selected_backend`] (from the resolved [`Config`] alone) — so `--dev`,
/// [`DeviceOpts::overrides`] and every command's backend funnel (`run`/`serve`/`bench`) can never
/// disagree on the pick. Since S8 nothing publishes `INFR_DEV` back into the environment; the CLI
/// layer of the `Config` is the whole delivery mechanism.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Backend {
    /// A Vulkan GPU. `Some("Vulkan1")` pins a device (carried as `device.dev`); `None` =
    /// the default "first discrete GPU, else device 0".
    Vulkan(Option<String>),
    /// The Apple GPU (`device.dev = "metal"` / `INFR_DEV=metal`).
    Metal,
    /// The CPU reference backend (`device.dev = "cpu"` / `INFR_DEV=cpu`).
    Cpu,
    /// AMD GPU through the ROCm/HIP stack (`INFR_DEV=rocm` or `INFR_DEV=rocmN`).
    /// `Some("rocm1")` pins a device index; `None` = device 0. Note the index is glued to the
    /// name — [`parse_rocm_device`] strips the `rocm` prefix and parses the rest, so a separator
    /// (`rocm:1`) parses as device 0.
    Rocm(Option<String>),
}

/// The INHERITED half of the device decision, so the backend DECISION
/// ([`resolve_backend`]/[`selected_backend`]) is a pure, unit-testable function of its inputs rather
/// than a scatter of ad-hoc `std::env::var` reads with drifting precedence.
///
/// `device.dev` is the SINGLE device-selection knob (`INFR_DEV`, or `[device] dev` in the config
/// file, same grammar as `--dev`); since S1 it reaches this struct through the resolved [`Config`]
/// rather than through a direct `std::env::var`. The old `INFR_METAL=1`/`INFR_CPU=1` flags were
/// removed cleanly (no aliases) — use `INFR_DEV=metal`/`cpu`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct BackendEnv {
    /// The device spec (`VulkanN`/`metal`/`cpu`/`rocm[N]`), if non-empty.
    dev: Option<String>,
}

impl BackendEnv {
    /// The device spec the resolved config carries. The empty-string filter is this consumer's
    /// policy and stays here (the config layer keeps the raw spec, because `infr-vulkan`'s
    /// `resolve_infr_dev_index` tolerates an empty value and this parser does not).
    fn from_cfg(cfg: &Config) -> Self {
        Self {
            dev: cfg.device.dev.clone().filter(|s| !s.trim().is_empty()),
        }
    }
}

/// Parse a device spec (from `--dev` OR `INFR_DEV`) — the ONE grammar: a Vulkan GPU
/// (`Vulkan0`/`Vulkan1`/…), `metal`, or `cpu`, case-insensitive. The original casing is preserved
/// for the Vulkan spec (the deep reader re-parses it). Errors carry only the "expected forms" tail;
/// the caller wraps it with which source (`--dev` vs `INFR_DEV`) was bad.
fn parse_dev_spec(d: &str) -> anyhow::Result<Backend> {
    let lower = d.trim().to_ascii_lowercase();
    if lower.starts_with("vulkan") {
        Ok(Backend::Vulkan(Some(d.trim().to_string())))
    } else if lower == "metal" {
        Ok(Backend::Metal)
    } else if lower == "cpu" {
        Ok(Backend::Cpu)
    } else if lower == "rocm" || lower.starts_with("rocm") && lower.len() > 4 {
        Ok(Backend::Rocm(Some(d.trim().to_string())))
    } else {
        anyhow::bail!("expected a Vulkan GPU like `Vulkan0`/`Vulkan1`, `metal`, `cpu`, or `rocm`");
    }
}

/// The SINGLE backend decision, shared by `--dev` resolution and the per-command readers.
///
/// Precedence (mirrors how `--ctx`/`-u`/`-t` relate to their envs): the `--dev` flag > the
/// `INFR_DEV` env (SAME grammar/parser as `--dev`) > the default (`Vulkan(None)` = first discrete
/// GPU, else device 0). A garbage `--dev`/`INFR_DEV` errors early. The legacy `INFR_METAL`/`INFR_CPU`
/// flags were removed cleanly — they are no longer read; `INFR_DEV=metal`/`cpu` replaces them.
fn resolve_backend(dev: Option<&str>, env: BackendEnv) -> anyhow::Result<Backend> {
    // 1. an explicit `--dev` flag wins outright.
    if let Some(d) = dev {
        return parse_dev_spec(d).with_context(|| format!("invalid --dev `{d}`"));
    }
    // 2. `INFR_DEV` — the device-selection env, same grammar as `--dev`.
    if let Some(d) = env.dev.as_deref() {
        return parse_dev_spec(d).with_context(|| format!("invalid INFR_DEV `{d}`"));
    }
    // 3. default: the first discrete Vulkan GPU.
    Ok(Backend::Vulkan(None))
}

/// The backend the resolved [`Config`] selects — the ONE reader precedence for `run`/`serve`/
/// `bench`'s backend funnels. Equivalent to `resolve_backend(None, BackendEnv::from_cfg(cfg))`: an
/// explicit `--dev` already won inside the config fold (the CLI layer beats the env layer), so the
/// `None` arg is right, but a garbage `INFR_DEV`/`[device] dev` DOES surface here.
fn selected_backend(cfg: &Config) -> anyhow::Result<Backend> {
    resolve_backend(None, BackendEnv::from_cfg(cfg))
}

/// The `device.dev` spelling for a parsed [`Backend`] — the inverse of [`parse_dev_spec`], and the
/// value `--dev` contributes to the CLI config layer. Since S5a that layer is the ONLY route to
/// `infr-vulkan`'s device pick: it takes `cfg.device.dev` off the `Arc<Config>` it is constructed
/// with, so the S1 bridge no longer re-publishes `INFR_DEV`.
fn dev_spec_of(backend: &Backend) -> String {
    match backend {
        Backend::Vulkan(Some(d)) | Backend::Rocm(Some(d)) => d.clone(),
        // Unreachable from `parse_dev_spec` (it always carries the spec) — spelled out so a future
        // parser change cannot silently publish the wrong device.
        Backend::Vulkan(None) => "vulkan".to_string(),
        Backend::Rocm(None) => "rocm".to_string(),
        Backend::Metal => "metal".to_string(),
        Backend::Cpu => "cpu".to_string(),
    }
}

/// Total-order comparison of two f64 ratios for the sweep's worst-first ranking. `total_cmp` never
/// panics (unlike `partial_cmp().unwrap()`) and puts any NaN at the `Greater` end, so a malformed
/// subprocess-JSON value can't abort the whole sweep summary. Ascending order = worst ratio first.
fn nan_safe_ratio_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    a.total_cmp(&b)
}

/// The shared device/config flags for `run`/`serve`/`bench`. Every field is a knob the engine
/// resolves from the layered [`Config`], so this struct is a FRONT-END:
/// [`overrides`](DeviceOpts::overrides) turns the flags into the CLI config layer (they used to be
/// laundered through the process environment with `std::env::set_var`).
///
/// The unified `--dev` matches llama.cpp: a Vulkan GPU (`Vulkan0`/`Vulkan1`/…), `metal`, or `cpu`,
/// case-insensitive. This is the fix for `--dev metal`/`--dev cpu` being silent no-ops on
/// `run`/`serve` (they only honoured `INFR_METAL`/`INFR_CPU` before).
#[derive(clap::Args)]
struct DeviceOpts {
    /// Device for the forward: a Vulkan GPU (`Vulkan0`/`Vulkan1`/…), `metal` (Apple GPU), or `cpu`
    /// (reference backend). Case-insensitive; matches llama.cpp's --dev. Unset = the first discrete
    /// Vulkan GPU, else device 0.
    #[arg(long)]
    dev: Option<String>,
    /// Context window in tokens (`8192`, `256k`, or `50%` of the free-VRAM KV capacity). Config
    /// `device.ctx` / INFR_CTX. Default: the model's trained context, clamped to VRAM.
    #[arg(long, value_name = "TOKENS")]
    ctx: Option<String>,
    /// Physical batch = tokens per forward = the prefill chunk (matches llama-bench -ub). Config
    /// `device.ubatch` / INFR_UBATCH. Unset = the engine's adaptive chunk policy.
    #[arg(long, visible_alias = "ub", short = 'u', value_name = "N")]
    ubatch: Option<usize>,
    /// CPU threads (matches llama-bench -t). Config `device.threads`; published as
    /// RAYON_NUM_THREADS (rayon has no other input). Unset = all cores.
    #[arg(long, short = 't', value_name = "N")]
    threads: Option<usize>,
}

impl DeviceOpts {
    /// Fold the flags into the CLI config layer. Called once, up front, before the model loads (so
    /// the resolved `Config` — and the `RAYON_NUM_THREADS` publication that rides on it — lands
    /// before any parallel work spins the rayon pool up or the session seam builds a backend).
    ///
    /// PRECEDENCE, unchanged: an explicit `--dev` OVERRIDES an inherited `INFR_DEV` because the CLI
    /// layer beats the env layer in the fold; an UNSET `--dev` specifies nothing, so the inherited
    /// `INFR_DEV` (or `[device] dev`) survives, and with neither the default is "first discrete GPU,
    /// else device 0". `--dev`/`INFR_DEV` share the ONE parser ([`parse_dev_spec`],
    /// `vulkan*`/`metal`/`cpu`/`rocm*`, case-insensitive) and a garbage `--dev` still fails fast
    /// here, before anything is loaded. The deprecated `INFR_METAL`/`INFR_CPU` are still not
    /// written and still not read.
    fn overrides(&self, layer: &mut PartialConfig) -> anyhow::Result<()> {
        if let Some(d) = self.dev.as_deref() {
            let backend = parse_dev_spec(d).with_context(|| format!("invalid --dev `{d}`"))?;
            layer.device.dev = Some(Some(dev_spec_of(&backend)));
        }
        // `--ctx` shares the size grammar (`8192`, `256k`, `50%`) with INFR_CTX; a typo fails fast
        // here (the check `cmd_serve` used to own).
        if let Some(c) = &self.ctx {
            let spec = infr_core::parse_size(c)
                .ok_or_else(|| anyhow!("invalid --ctx `{c}` (expected e.g. 8192, 256k, or 50%)"))?;
            layer.device.ctx = Some(Some(spec));
        }
        // `-u 0` stays "adaptive": the value is carried verbatim and the seam's `ubatch_rows`
        // filters `v > 0`, exactly as it did when this was published as INFR_UBATCH=0. Likewise
        // RAYON_NUM_THREADS=0 falls back to all cores in rayon.
        //
        // `ubatch_specified` is the OTHER half of the knob (§6.12, and `seam::ubatch_rows`'s doc):
        // passing `-u` at all pins the chunk height, which is what disables the dense placement
        // sweeps — including for `-u 0`, where the value itself is unusable. That used to be
        // carried by the mere PRESENCE of the published `INFR_UBATCH`; the flag layer has to say
        // so explicitly now.
        if let Some(u) = self.ubatch {
            layer.device.ubatch = Some(Some(u));
            layer.device.ubatch_specified = Some(true);
        }
        if let Some(t) = self.threads {
            layer.device.threads = Some(Some(t));
        }
        Ok(())
    }
}

/// The shared SAMPLING flags for `run`/`serve` — the sibling of [`DeviceOpts`]. Same front-end
/// pattern: every field is a `sampling.*` config knob (`INFR_TEMP`/`INFR_TOP_K`/`INFR_TOP_P`/
/// `INFR_SEED`/`INFR_MAX_NEW`/`INFR_NO_THINK`) that the decode loop resolves, so
/// [`overrides`](SamplingOpts::overrides) fills the CLI config layer instead of writing the envs.
///
/// NOT on `bench`: bench pins greedy so its numbers are deterministic, so exposing sampling there
/// would be misleading. On `serve` these are the SERVER DEFAULTS — a per-request OpenAI field
/// (`temperature`/`top_p`/…) still overrides them via `GenParams::from_request`.
///
/// The layer is resolved BEFORE `apply_model_sampling_defaults` (which only fills a knob no layer
/// specified), so a flag the user passed wins and the rest fall back to the chat defaults.
#[derive(clap::Args)]
struct SamplingOpts {
    /// Sampling temperature (0 = greedy/argmax). Config `sampling.temp` / INFR_TEMP. Default: the
    /// model's recommended value (arch-family table + any generation_config.json beside the model)
    /// — e.g. 0.6 for Qwen3, 1.0 for Gemma; 0.6 fallback. The startup banner prints the effective
    /// value.
    #[arg(long, value_name = "T")]
    temp: Option<f32>,
    /// Top-k: keep only the k most-likely tokens (0 = keep all). Config `sampling.top_k` /
    /// INFR_TOP_K. Default: model-recommended (e.g. 20 for Qwen3, 64 for Gemma, off for Llama).
    #[arg(long = "top-k", value_name = "K")]
    top_k: Option<usize>,
    /// Top-p (nucleus): keep the smallest set whose probability mass ≥ p. Config `sampling.top_p` /
    /// INFR_TOP_P. Default: model-recommended (e.g. 0.95 for Qwen3/Gemma, 0.9 for Llama).
    #[arg(long = "top-p", value_name = "P")]
    top_p: Option<f32>,
    /// RNG seed for reproducible sampling. Config `sampling.seed` / INFR_SEED. Unset = seeded from
    /// the clock.
    #[arg(long, value_name = "N")]
    seed: Option<u64>,
    /// Max tokens to generate per reply. Config `sampling.max_new` / INFR_MAX_NEW. (No short flag:
    /// `-n` is taken by bench's `--n-gen` and serve's `--parallel`.)
    #[arg(long = "max-new", value_name = "N")]
    max_new: Option<usize>,
    /// Force reasoning OFF for models that emit `<think>` (config `sampling.no_think` /
    /// INFR_NO_THINK=1). Thinking is on by default where the template supports it.
    #[arg(long = "no-think", conflicts_with = "think")]
    no_think: bool,
    /// Force reasoning ON, overriding an inherited INFR_NO_THINK (`sampling.no_think = false`).
    #[arg(long)]
    think: bool,
}

impl SamplingOpts {
    /// Fold the passed flags into the CLI config layer. Only a flag the user actually set specifies
    /// anything, so an unset knob is left for the file/env layers, `apply_model_sampling_defaults`,
    /// or the reader default — in that order.
    fn overrides(&self, layer: &mut PartialConfig) {
        if let Some(t) = self.temp {
            layer.sampling.temp = Some(t);
        }
        if let Some(k) = self.top_k {
            layer.sampling.top_k = Some(k);
        }
        if let Some(p) = self.top_p {
            layer.sampling.top_p = Some(p);
        }
        if let Some(s) = self.seed {
            layer.sampling.seed = Some(Some(s));
        }
        if let Some(m) = self.max_new {
            layer.sampling.max_new = Some(m);
        }
        // `--no-think`/`--think` are mutually exclusive (clap `conflicts_with`); each maps to the
        // existing INFR_NO_THINK grammar (`=1` off, `=0`/unset on — see infr-chat template.rs),
        // which the bridge re-publishes verbatim.
        if self.no_think {
            layer.sampling.no_think = Some(true);
        } else if self.think {
            layer.sampling.no_think = Some(false);
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Download + cache a model (`org/repo[:quant]` from HuggingFace, or a path to a `.gguf`).
    Pull { model: String },
    /// List the Vulkan physical devices infr can see — index, name, type, VRAM — marking the
    /// device the default (no `--dev`) path binds. The index is the `--dev VulkanN` / `INFR_DEV`
    /// handle. Reports each device's external-memory extensions (GPU↔GPU / dma-buf feasibility).
    Devices,
    /// Interactive terminal chat (auto-pulls if missing).
    Run {
        model: String,
        /// Optional one-shot message (otherwise drop into a REPL).
        message: Option<String>,
        /// Live per-denoise-step canvas view (diffusion-gemma models only). Bare `--diffusion-visual`
        /// (or `=1`) draws it when stdout is a tty; `=force` draws it anyway, for scripted
        /// verification against piped stdout. CLI presentation only — NOT a `Config` field
        /// (docs/config-plan.md §6.10), so it never reaches the forward.
        #[arg(
            long = "diffusion-visual",
            value_name = "MODE",
            num_args = 0..=1,
            default_missing_value = "1",
            default_value = "",
            env = "INFR_DIFFUSION_VISUAL",
            hide_env = true
        )]
        diffusion_visual: String,
        #[command(flatten)]
        device: DeviceOpts,
        #[command(flatten)]
        sampling: SamplingOpts,
    },
    /// Start the OpenAI-compatible HTTP API (auto-pulls if missing).
    Serve {
        model: String,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Concurrent generation slots (llama-server's `-np`). N requests generate at once, each
        /// with its own KV cache, taking turns on the GPU at token granularity; the (N+1)'th queues.
        ///
        /// Each slot owns a full KV cache, so the DEFAULT per-slot context is the VRAM-fit window
        /// divided by N: the N slots together stay inside the same VRAM budget one slot is held to,
        /// and raising `-np` can never OOM a box that `-np 1` fit. The visible cost is a smaller
        /// per-request window. Pass `--ctx` to pin the per-slot window instead (then `N * ctx` must
        /// fit, and the Vulkan budget guard will say so if it doesn't).
        /// `--np` is accepted as an alias (llama-server spells this `-np`; clap shorts are a single
        /// character, so `-n` is the short form and `--np` the familiar long one).
        ///
        /// Unset defaults to 4 slots on the Vulkan seam. `Option` (rather than a `default_value_t`)
        /// so the CPU/Metal/diffusion serialised backends can tell an EXPLICIT `--parallel` (worth
        /// a "no multi-slot engine, ignored" note) from the default (silent).
        #[arg(long = "parallel", visible_alias = "np", short = 'n', value_name = "N")]
        parallel: Option<usize>,
        /// `--ctx` here is the PER-SLOT window (divided from the VRAM fit by `--parallel` when
        /// unset); see the shared `DeviceOpts` flags below.
        #[command(flatten)]
        device: DeviceOpts,
        /// Sampling flags are the SERVER DEFAULTS — a per-request OpenAI field still overrides them.
        #[command(flatten)]
        sampling: SamplingOpts,
    },
    /// Host SEVERAL models at once, each pinned to a physical GPU, on ONE OpenAI-compatible server
    /// (routed by model name). Data-parallel multi-device serving: `infr multi qwen@Vulkan0
    /// gemma@Vulkan1` runs the two models concurrently on the two GPUs — a request naming a model is
    /// dispatched to the generator on that model's device. Each spec is `MODEL[@VulkanN]` (a `.gguf`
    /// path or an `org/repo[:quant]` HF ref, optionally with a device suffix); omit `@VulkanN` to
    /// round-robin the specs across the enumerated Vulkan devices. Vulkan seam only (the concurrent
    /// engine) — `INFR_DEV=cpu`/`INFR_DEV=metal`/diffusion-gemma models aren't hosted here.
    Multi {
        /// Model specs `MODEL[@VulkanN]`, one per hosted model. At least one; devices without a
        /// suffix are assigned round-robin over the enumerated GPUs.
        #[arg(num_args = 1.., required = true, value_name = "MODEL[@VulkanN]")]
        models: Vec<String>,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Concurrent generation slots PER MODEL (each slot owns a full KV cache on that model's
        /// device). Defaults to 1: the iGPU shares system RAM and the demo hosts small models, so a
        /// modest per-model default keeps two models on two devices well inside memory. The
        /// cross-model concurrency (one request per model at once, on different GPUs) is what proves
        /// the device pool; raise this to add within-model concurrency.
        #[arg(
            long = "parallel",
            visible_alias = "np",
            short = 'n',
            default_value_t = 1,
            value_name = "N"
        )]
        parallel: usize,
        /// Per-slot context window (shared size grammar `8192`/`256k`/`50%`); applies to every hosted
        /// model. Unset = each model's VRAM-fit default on its own device.
        #[arg(long, value_name = "CTX")]
        ctx: Option<String>,
    },
    /// Benchmark prefill/decode tok/s — same interface as llama.cpp's `llama-bench` (-p/-n/-d/-r),
    /// so the two are directly comparable. Prefill (pp) when -n 0; decode (tg) when -p 0.
    Bench {
        model: String,
        /// Prompt tokens to process (prefill). pp throughput = n_prompt / time.
        #[arg(short = 'p', long = "n-prompt", default_value_t = 512)]
        n_prompt: usize,
        /// Tokens to generate (decode). tg throughput = n_gen / time. Set -p 0 to measure decode.
        #[arg(short = 'n', long = "n-gen", default_value_t = 0)]
        n_gen: usize,
        /// Context depth pre-filled (untimed) before measuring — matches llama-bench -d.
        #[arg(short = 'd', long = "n-depth", default_value_t = 0)]
        depth: usize,
        /// Combined prompt+gen turn `P,G` (matches llama-bench -pg): time ingesting P tokens THEN
        /// generating G; throughput = (P+G)/time. Models one coding-agent turn (read a file/tool
        /// result, then emit a reply). Overrides -p/-n when set.
        #[arg(long = "pg")]
        pg: Option<String>,
        /// Repetitions (reported value is the average).
        #[arg(short = 'r', long, default_value_t = 3)]
        reps: usize,
        /// GPU layers (matches llama-bench -ngl): >0 = Vulkan GPU forward; 0 = run on the CPU
        /// reference backend (no GPU), so `infr bench -ngl 0` is directly comparable to llama.cpp CPU.
        #[arg(long = "n-gpu-layers", visible_alias = "ngl", default_value_t = 999)]
        ngl: usize,
        /// Emit `[{"avg_ts": X}]` (same shape as `llama-bench -o json`) for scripted comparison.
        #[arg(long)]
        json: bool,
        /// Shared device/config flags: `--dev` (Vulkan/metal/cpu), `-u`/`--ubatch`, `-t`/`--threads`
        /// (`--ctx` is accepted too; it just sets INFR_CTX).
        #[command(flatten)]
        device: DeviceOpts,
    },
    /// Compare infr vs llama.cpp on coding-agent-shaped workloads (long context, replies at depth,
    /// whole turns). Shells out to `infr bench` and `llama-bench` with matching flags, same model+GPU.
    Compare {
        /// Model(s): one for the deep coding-agent scenarios, several with --sweep.
        #[arg(num_args = 1.., required = true)]
        models: Vec<String>,
        /// Survey mode: for EVERY model given, measure pp512 / tg128 / tg@depth on both tools and
        /// print a gap matrix + the worst ratios — the recurring "where are we behind llama.cpp"
        /// sweep. Without it, the single model gets the deep coding-agent scenarios below.
        #[arg(long)]
        sweep: bool,
        /// Decode depth for the sweep's at-depth metric.
        #[arg(long, default_value_t = 4096)]
        sweep_depth: usize,
        /// GPU device for both tools (matches llama-bench --dev; override if device order differs).
        #[arg(long, default_value = "Vulkan0")]
        dev: String,
        /// GPU layers for both tools (matches llama-bench -ngl): >0 = GPU; 0 = CPU comparison
        /// (infr CPU reference backend vs llama.cpp CPU). 0 lets `infr compare -ngl 0` bench CPU directly.
        #[arg(long = "n-gpu-layers", visible_alias = "ngl", default_value_t = 999)]
        ngl: usize,
        /// CPU threads for the -ngl 0 path on both tools (matches llama-bench -t). 0 = all cores.
        #[arg(short = 't', long, default_value_t = 0)]
        threads: usize,
        /// Repetitions per measurement (reported value is the average).
        #[arg(short = 'r', long, default_value_t = 3)]
        reps: usize,
        /// Pin the ubatch (per-forward chunk) on both tools. 0 = each tool's own default.
        #[arg(short = 'u', long = "ubatch-size", default_value_t = 0)]
        ubatch: usize,
        /// Session depths / prefill sizes (coding-agent scale). Stay within the model's context.
        #[arg(long, value_delimiter = ',', default_values_t = [8000usize, 16000, 32000])]
        ctx: Vec<usize>,
        /// Reply length for the decode-at-depth scenario.
        #[arg(long, default_value_t = 256)]
        gen: usize,
        /// Session turns as `P,G` (ingest P tokens, generate G). Repeat the flag for several shapes.
        #[arg(long = "turn", default_values_t = ["2048,256".to_string(), "8192,512".to_string()])]
        turns: Vec<String>,
        /// Path to the llama-bench binary.
        #[arg(long, default_value = "llama-bench")]
        llama_bench: String,
    },
}

// ---------------------------------------------------------------------------
// Signals — the graceful GPU shutdown path (see `infr_core::shutdown`)
// ---------------------------------------------------------------------------

/// `SIGINT`/`SIGTERM` handler. **Async-signal-safe by construction**: the first signal does nothing
/// but a lock-free atomic store ([`infr_core::shutdown::request_shutdown`]) — no allocation, no
/// locking, no Rust `println!` (which takes the stdout lock and would deadlock against an
/// interrupted `print!`). The engine's poll sites see the latch at their next submit boundary,
/// drain the GPU work already in flight, unwind, and let the backend's `Drop` destroy the device.
///
/// The SECOND signal is the user saying they have given up waiting, and is honoured immediately:
/// `write(2)` and `_exit(2)` are both on POSIX's async-signal-safe list (unlike `exit`, which runs
/// atexit handlers and flushes streams from a signal context). It is a real risk, so it says so.
#[cfg(unix)]
extern "C" fn on_signal(signo: libc::c_int) {
    if infr_core::shutdown::request_shutdown(signo) {
        return; // first signal: latch and let the engine wind down at its next submit boundary
    }
    const MSG: &[u8] = b"\ninfr: second signal - exiting NOW without draining the GPU. \
        If a submit was in flight, the device may stay wedged until reboot.\n";
    // SAFETY: `write` and `_exit` are async-signal-safe; MSG is a 'static byte string.
    unsafe {
        libc::write(2, MSG.as_ptr().cast(), MSG.len());
        libc::_exit(128 + signo);
    }
}

/// Install [`on_signal`] for `SIGINT` and `SIGTERM`, once, before anything touches the GPU.
///
/// Chosen over `tokio::signal` (which is in the tree via tokio's `full` feature) because three of
/// the four subcommands — `run`, `bench`, `compare` — are plain synchronous code with no runtime to
/// hang a signal future off, and `serve` builds its runtime only after the model is loaded (uploads
/// = submits: a signal during LOAD must already be safe). `libc` is a direct dep now but adds
/// nothing to the lockfile — it was already there under tokio and indicatif/console.
///
/// No `SA_RESTART`: an interrupted blocking `read(2)` at the chat prompt then returns `EINTR`,
/// which is what lets [`read_line_interruptible`] notice a Ctrl-C at an idle REPL instead of
/// sitting on the read until the user presses Enter. Everything else in the process that can see
/// an interrupted syscall already retries it (Rust's `io` retries `ErrorKind::Interrupted`, libdrm's
/// ioctl wrapper loops on `EINTR`).
#[cfg(unix)]
fn install_signal_handlers() {
    // SAFETY: a zeroed `sigaction` with a valid handler pointer and an empty mask is exactly what
    // POSIX asks for; the handler itself is async-signal-safe (see `on_signal`).
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_signal as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
    }
}

#[cfg(not(unix))]
fn install_signal_handlers() {}

/// The conventional exit status for a signal-terminated process, `128 + signo` (130 for `SIGINT`,
/// 143 for `SIGTERM`) — what a shell reports for a process killed by that signal, so scripts and
/// `timeout` see what they expect.
///
/// Called from `main` AFTER the subcommand has returned, i.e. after the engine has unwound and the
/// backend's `Drop` has destroyed the Vulkan device. `process::exit` here cannot strand a submit —
/// that is the whole point of doing it here and nowhere else.
fn exit_if_signalled() {
    use std::io::Write;
    let Some(signo) = infr_core::shutdown::shutdown_signal() else {
        return;
    };
    // Partial output the user already saw is theirs to keep — flush it.
    std::io::stdout().flush().ok();
    eprintln!("\ninfr: interrupted — GPU work drained, device released.");
    std::io::stderr().flush().ok();
    std::process::exit(128 + signo);
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    install_signal_handlers();

    let cli = Cli::parse();

    // Packaging helpers (hidden): emit completions / man page and exit. The man page includes
    // every subcommand and flag straight from the clap definitions, so it can't drift.
    if let Some(shell) = cli.completions {
        use clap::CommandFactory;
        shell.generate(&mut Cli::command());
        return Ok(());
    }
    if cli.man {
        use clap::CommandFactory;
        clap_mangen::Man::new(Cli::command()).render(&mut std::io::stdout())?;
        return Ok(());
    }

    let Some(cmd) = cli.cmd else {
        // arg_required_else_help covers the bare invocation; a flag-only call lands here.
        use clap::CommandFactory;
        Cli::command().print_help()?;
        return Ok(());
    };

    // THE configuration, resolved exactly once: defaults < config file < `INFR_*` < flags/`--set`
    // (docs/config-plan.md §2). It is a VALUE, threaded into the commands as an `Arc` — there is
    // deliberately no global (R4).
    let overrides = ConfigOverrides {
        config_path: cli.config.clone(),
        sets: cli.set.clone(),
        flags: cli_flag_layer(&cmd)?,
    };
    let cfg = Arc::new(Config::load(&overrides)?);
    publish_thread_count(&cfg);
    publish_profile_out(&cfg);
    // WHICH paths a layer actually specified — the one thing a resolved `Config` cannot answer,
    // and the input the model-recommended sampling defaults need (they may only fill a knob nobody
    // named). S1 answered it by re-publishing the values into the environment and probing it back;
    // S8 hands the partial down instead. Nothing writes `INFR_*` any more.
    let specified = specified_by_the_layers(&overrides)?;

    // The subcommand runs to completion (or to its abort) and EVERYTHING it owns — model, backend,
    // Vulkan device — is dropped as it returns. Only then does `exit_if_signalled` turn a latched
    // SIGINT/SIGTERM into 130/143, and it does so in preference to the abort error the unwinding
    // produced (an aborted forward reports `aborted: shutdown requested`, which is noise once we
    // are already saying "interrupted" with the right status).
    let res = dispatch(cmd, &cfg, &specified);
    exit_if_signalled();
    res
}

/// The CLI layer's BESPOKE half: what this invocation's dedicated flags specify (`--set` is folded
/// in by `infr-core`, and loses to these — §11 [DECIDE-3]).
///
/// Built before dispatch so a bad `--dev`/`--ctx` fails fast, before anything is downloaded or
/// loaded, exactly as `DeviceOpts::resolve` did.
fn cli_flag_layer(cmd: &Cmd) -> anyhow::Result<PartialConfig> {
    let mut layer = PartialConfig::default();
    match cmd {
        Cmd::Run {
            device, sampling, ..
        }
        | Cmd::Serve {
            device, sampling, ..
        } => {
            device.overrides(&mut layer)?;
            sampling.overrides(&mut layer);
        }
        Cmd::Bench { device, .. } => {
            device.overrides(&mut layer)?;
            // Benchmarks decode a FIXED token count (llama-bench semantics): never stop at EOS — a
            // model that emits EOS instantly on the dummy context would otherwise report fictional
            // tok/s. `cmd_bench` used to `set_var("INFR_IGNORE_EOS", "1")` itself; as a flag-level
            // override it wins over the file and the environment just as that overwrite did.
            layer.sampling.ignore_eos = Some(true);
        }
        Cmd::Pull { .. } | Cmd::Devices | Cmd::Multi { .. } | Cmd::Compare { .. } => {}
    }
    Ok(layer)
}

fn dispatch(cmd: Cmd, cfg: &Arc<Config>, specified: &PartialConfig) -> anyhow::Result<()> {
    match cmd {
        Cmd::Pull { model } => cmd_pull(&model),
        Cmd::Devices => cmd_devices(cfg),
        Cmd::Run {
            model,
            message,
            diffusion_visual,
            ..
        } => cmd_run(
            &model,
            message.as_deref(),
            &diffusion_visual,
            cfg,
            specified,
        ),
        Cmd::Serve {
            model,
            addr,
            parallel,
            ..
        } => cmd_serve(&model, &addr, parallel, cfg, specified),
        Cmd::Multi {
            models,
            addr,
            parallel,
            ctx,
        } => cmd_multi(&models, &addr, parallel, ctx.as_deref(), cfg, specified),
        Cmd::Bench {
            model,
            n_prompt,
            n_gen,
            depth,
            pg,
            reps,
            ngl,
            json,
            ..
        } => cmd_bench(&model, n_prompt, n_gen, depth, pg, reps, ngl, json, cfg),
        Cmd::Compare {
            models,
            sweep,
            sweep_depth,
            dev,
            ngl,
            threads,
            reps,
            ubatch,
            ctx,
            gen,
            turns,
            llama_bench,
        } => {
            if sweep {
                cmd_compare_sweep(
                    &models,
                    sweep_depth,
                    &dev,
                    ngl,
                    threads,
                    reps,
                    ubatch,
                    &llama_bench,
                    cfg,
                )
            } else {
                if models.len() > 1 {
                    anyhow::bail!(
                        "compare without --sweep takes ONE model (got {}); pass --sweep for the                          multi-model survey",
                        models.len()
                    );
                }
                cmd_compare(
                    &models[0],
                    &dev,
                    ngl,
                    threads,
                    reps,
                    ubatch,
                    &ctx,
                    gen,
                    &turns,
                    &llama_bench,
                    cfg,
                )
            }
        }
    }
}

use anyhow::{anyhow, Context};
use std::path::{Path, PathBuf};

/// Which paths some LAYER actually specified — the question a resolved [`Config`] cannot answer,
/// because it is all values and no provenance (`sampling.temp == 0.0` cannot tell `--temp 0` from
/// "nobody said anything", and the difference decides whether the model's recommended sampling
/// applies).
///
/// Re-derives the three layers through their PURE halves (re-reading the config file, which is why
/// this is not on any hot path), so nothing is printed twice: `Config::load` already emitted the
/// file's unknown-key warnings, the diagnostics announcement and any `--set`-shadowed-by-a-flag
/// warning.
///
/// PERMANENT since S8. It used to feed the transitional env bridge; now it is threaded straight
/// into the commands that need provenance — [`apply_model_sampling_defaults`] (the model's
/// recommended sampler may only fill a knob NO layer named) and `cmd_run`'s `max_new` default
/// (diffusion-gemma wants a different fallback from the autoregressive one). Nothing publishes
/// `INFR_*` any more.
fn specified_by_the_layers(overrides: &ConfigOverrides) -> anyhow::Result<PartialConfig> {
    use infr_core::config::{cli as cfg_cli, file as cfg_file, ConfigLayer};
    let mut layers = PartialConfig::default();
    if let Some(path) = cfg_file::discover(overrides.config_path.as_deref())? {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("config file {}", path.display()))?;
        layers.merge(cfg_file::parse_str(&text, &path)?.0);
    }
    layers.merge(ConfigLayer::env()?);
    layers.merge(cfg_cli::parse_reporting(overrides)?.0);
    Ok(layers)
}

/// `-t`/`--threads` → `RAYON_NUM_THREADS`.
///
/// PERMANENT, unlike the bridge below (docs/config-plan.md §6.1): rayon's global pool has no input
/// other than the process environment, and `device.threads` is not an `INFR_*` knob — it only ever
/// comes from the flag or the config file, so publishing it is the whole delivery mechanism, not a
/// stand-in for one. Must run before any parallel work spins the pool up, hence: in `main`.
fn publish_thread_count(cfg: &Config) {
    if let Some(t) = cfg.device.threads {
        std::env::set_var("RAYON_NUM_THREADS", t.to_string());
    }
}

/// `prof.profile_out` → the profiler runtime's report destination (S7).
///
/// PERMANENT, like [`publish_thread_count`] and unlike the S1 bridge: `infr-prof-rt` writes its
/// JSON from a C `atexit` hook that has no receiver and no config to borrow, and it cannot depend
/// on `infr-core` to fetch one (the dependency runs the other way). So the resolved value is PUSHED
/// into the reporter here, once, in `main` — a value handed across an API boundary, not laundered
/// through the environment. `None` leaves the reporter on stderr only, exactly as an unset
/// `INFR_PROFILE_OUT` did.
fn publish_profile_out(cfg: &Config) {
    infr_prof_rt::set_profile_out(cfg.prof.profile_out.as_deref().and_then(|p| p.to_str()));
}

/// Is `path` a usable LOCAL model: an existing regular FILE whose extension is `.gguf`
/// (case-insensitive)? A directory, a wrong-extension file, or a missing path is NOT local — the
/// gate that keeps `resolve` from treating a cwd entry that merely collides with an `org/repo` ref
/// as the model, and from sending a typo'd local path to a confusing network pull. Split out so the
/// classification is unit-testable without the network.
fn is_local_gguf(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// Resolve a model arg to (gguf_path, optional tokenizer_json_path).
/// Accept a path to a `.gguf` FILE or an `org/repo[:quant]` HuggingFace ref resolved via infr-hub.
/// The tokenizer is the `tokenizer.json` beside the GGUF if present, else `None` → derived from the
/// GGUF's embedded vocab (HF Hub blobs are content-addressed with no sidecar).
fn resolve(model: &str) -> anyhow::Result<(PathBuf, Option<PathBuf>)> {
    let p = Path::new(model);
    let gguf = if is_local_gguf(p) {
        PathBuf::from(model)
    } else {
        // Not an existing `.gguf` file → treat as an HF ref. A ref parse failure here means the arg
        // is neither: give a message that names both cases (a typo'd local path, or a directory /
        // non-`.gguf` file that happens to exist) instead of a bare parse error dressed as a
        // network pull.
        match infr_hub::ModelRef::parse(model) {
            Ok(r) => infr_hub::ensure(&r).map_err(|e| anyhow!("{e}"))?,
            Err(e) if p.exists() => {
                anyhow::bail!(
                    "`{model}` exists but is not a `.gguf` file — pass a path to a `.gguf`, or an \
                     `org/repo[:quant]` HuggingFace ref ({e})"
                );
            }
            Err(e) => {
                anyhow::bail!(
                    "`{model}` is not a `.gguf` file and not a valid `org/repo[:quant]` ref ({e})"
                );
            }
        }
    };
    let tok = gguf
        .parent()
        .map(|d| d.join("tokenizer.json"))
        .filter(|p| p.exists());
    Ok((gguf, tok))
}

/// `infr devices` — enumerate the Vulkan physical devices (index/name/type/VRAM), mark the default
/// pick, and report external-memory extensions so the multi-GPU campaign can see the P2P surface.
fn cmd_devices(cfg: &Config) -> anyhow::Result<()> {
    let devs = infr_vulkan::VulkanBackend::enumerate_devices(cfg).map_err(|e| anyhow!("{e}"))?;
    if devs.is_empty() {
        println!("no Vulkan physical devices found");
        return Ok(());
    }
    println!(
        "{} Vulkan device(s) (select with `--dev VulkanN` / `INFR_DEV=VulkanN`):\n",
        devs.len()
    );
    for d in &devs {
        let mark = if d.is_default_pick {
            "  <- default"
        } else {
            ""
        };
        // Bytes → GiB with one decimal (matches the backend's fmt_bytes feel; local to avoid a dep).
        let vram = format!("{:.1} GiB", d.vram_bytes as f64 / (1u64 << 30) as f64);
        println!(
            "  Vulkan{}: {} [{}, {} device-local]{}",
            d.index, d.name, d.device_type, vram, mark
        );
        let mut ext = Vec::new();
        if d.external_memory {
            ext.push("external_memory");
        }
        if d.external_memory_fd {
            ext.push("external_memory_fd");
        }
        if d.external_memory_dma_buf {
            ext.push("external_memory_dma_buf");
        }
        println!(
            "           external-memory: {}",
            if ext.is_empty() {
                "none".to_string()
            } else {
                ext.join(", ")
            }
        );
    }
    Ok(())
}

fn cmd_pull(model: &str) -> anyhow::Result<()> {
    let r = infr_hub::ModelRef::parse(model).map_err(|e| anyhow!("{e}"))?;
    // `pull` checks HF for the repo's latest commit and updates a stale cache (run/serve stay
    // cache-first via `ensure`). Offline → falls back to the cached copy.
    let path = infr_hub::ensure_latest(&r).map_err(|e| anyhow!("{e}"))?;
    println!("{}", path.display());
    Ok(())
}

/// Streams generated text through THE shared reasoning splitter (`infr_engine::ChatStream` — the
/// same one `infr serve` emits OpenAI deltas from), dimming the reasoning so it's visually
/// distinct from the answer. Because run and serve consume one splitter, every thinking model is
/// exposed identically on both surfaces — a new reasoning format lands in `infr-chat` once. On a
/// non-TTY the raw model text passes through untouched (markers preserved for piped output).
struct ThinkRender {
    split: infr_engine::ChatStream,
    in_think: bool,
    tty: bool,
}
impl ThinkRender {
    fn new() -> Self {
        use std::io::IsTerminal;
        Self {
            split: infr_engine::ChatStream::new(false),
            in_think: false,
            tty: std::io::stdout().is_terminal(),
        }
    }
    fn feed(&mut self, delta: &str) {
        use std::io::Write;
        if !self.tty {
            print!("{delta}");
            let _ = std::io::stdout().flush();
            return;
        }
        let in_think = &mut self.in_think;
        self.split.push(delta, &mut |d| Self::render(d, in_think));
        let _ = std::io::stdout().flush();
    }
    fn finish(&mut self) {
        use std::io::Write;
        if self.tty {
            let in_think = &mut self.in_think;
            self.split.finish(&mut |d| Self::render(d, in_think));
            if *in_think {
                print!("[0m");
                self.in_think = false;
            }
        }
        println!();
        let _ = std::io::stdout().flush();
    }
    /// Style transitions ride the delta KIND: entering Reasoning dims+italicizes, entering
    /// Content resets. The splitter already stripped the markers.
    fn render(d: infr_engine::Delta, in_think: &mut bool) {
        match d {
            infr_engine::Delta::Reasoning(t) => {
                if !*in_think {
                    print!("[2;3m");
                    *in_think = true;
                }
                print!("{t}");
            }
            infr_engine::Delta::Content(t) => {
                if *in_think {
                    // Close the dim reasoning with a blank line: `<think>` models emit their own
                    // whitespace around `</think>`, but channel-format models (diffusion-gemma,
                    // E2B) end reasoning at a bare `<channel|>` — without this the answer renders
                    // GLUED to the last thinking line ("…clearly.The capital of France is Paris.").
                    print!("[0m\n\n");
                    *in_think = false;
                    // The splitter strips markers, not whitespace: a think-model's own newline
                    // after `</think>` shouldn't stack a third blank line on top of ours.
                    print!("{}", t.trim_start_matches('\n'));
                    return;
                }
                print!("{t}");
            }
            infr_engine::Delta::ToolCall { .. } => {}
        }
    }
}

/// Build the per-backend generation primitive (`ChatModel`) for run AND serve, from the backend the
/// resolved config selects ([`selected_backend`]) and the cheap diffusion-gemma arch peek the caller
/// already did. ONE funnel so `cmd_run` and `cmd_serve` can never disagree on how a backend is
/// constructed (the DG→Metal→CPU→Vulkan selection used to be copy-pasted in both). The backend
/// banner goes to stderr on both surfaces.
fn build_chat_model(
    gguf: &Path,
    tok: Option<&Path>,
    is_dg: bool,
    cfg: &Arc<Config>,
) -> anyhow::Result<Box<dyn infr_llama::chat::ChatModel + Send>> {
    let backend = selected_backend(cfg)?;
    if is_dg {
        // diffusion-gemma (Phase 3/D, docs/diffusion-gemma.md): the entropy-bound block-diffusion
        // loop over a persistent session — Vulkan by default, CPU under INFR_DEV=cpu, Metal under
        // INFR_DEV=metal (the non-macOS build still compiles the Metal arm; `DiffusionGemmaChat::generate`
        // errors clearly at runtime there).
        eprintln!(
            "[{} — diffusion-gemma entropy-bound block decode]",
            match backend {
                Backend::Cpu => "cpu backend",
                Backend::Metal => "metal backend",
                Backend::Rocm(_) => "rocm backend",
                Backend::Vulkan(_) => "vulkan seam",
            }
        );
        let loaded = infr_llama::SeamModel::load_with(gguf, tok, cfg.clone())?;
        return Ok(match backend {
            Backend::Cpu => Box::new(infr_llama::chat::DiffusionGemmaChat::new_cpu(loaded)),
            Backend::Metal => Box::new(infr_llama::chat::DiffusionGemmaChat::new_metal(loaded)),
            Backend::Rocm(_) => {
                // Phase 0: diffusion-gemma on ROCm is not yet implemented.
                anyhow::bail!(
                    "ROCm diffusion-gemma not yet implemented — \
                     see docs/rocm-plan.md Phase 2"
                );
            }
            Backend::Vulkan(_) => Box::new(infr_llama::chat::DiffusionGemmaChat::new(loaded)),
        });
    }
    match &backend {
        Backend::Rocm(rocm_spec) => {
            let dev_idx = rocm_spec.as_deref().map(parse_rocm_device).unwrap_or(0);
            eprintln!(
                "[rocm backend — dense/MoE forward on AMD GPU via ROCm/HIP, persistent KV session]"
            );
            Ok(Box::new(infr_llama::chat::RocmSeamChat::new(
                infr_llama::SeamModel::load_with(gguf, tok, cfg.clone())?,
                dev_idx,
            )?))
        }
        Backend::Metal => {
            eprintln!(
                "[metal backend — dense/MoE forward on Apple GPU via the agnostic compute graph, persistent KV session]"
            );
            #[cfg(target_os = "macos")]
            {
                metal_chat_model(gguf, tok, cfg)
            }
            #[cfg(not(target_os = "macos"))]
            {
                Ok(Box::new(infr_llama::chat::CpuDenseChat::new_metal(
                    infr_llama::SeamModel::load_with(gguf, tok, cfg.clone())?,
                )))
            }
        }
        Backend::Cpu => {
            eprintln!(
                "[cpu backend — dense/MoE forward on CPU via the agnostic compute graph, no GPU]"
            );
            Ok(Box::new(infr_llama::chat::CpuDenseChat::new(
                infr_llama::SeamModel::load_with(gguf, tok, cfg.clone())?,
            )))
        }
        // The default: dense/MoE on the VULKAN agnostic seam — persistent multi-slot KV sessions
        // (per-turn suffix-only prefill), record-once decode replay, MoE expert auto-fit. qwen35
        // (Qwen3.5) lands here too (same seam, `Config::from_gguf` + `MixerW::DeltaNet`), as does
        // llama4 Scout (its Q2_K bank runs on the paged expert cache).
        Backend::Vulkan(_) => {
            eprintln!(
                "[vulkan seam — dense/MoE on the agnostic compute graph, persistent KV session]"
            );
            Ok(Box::new(infr_llama::chat::DenseSeamChat::new(
                infr_llama::SeamModel::load_with(gguf, tok, cfg.clone())?,
            )))
        }
    }
}

fn cmd_run(
    model: &str,
    message: Option<&str>,
    diffusion_visual: &str,
    cfg: &Arc<Config>,
    specified: &PartialConfig,
) -> anyhow::Result<()> {
    use std::io::Write;
    let (gguf, tok) = resolve(model)?;
    // diffusion-gemma (block text-diffusion, Phase 3 — docs/diffusion-gemma.md): a cheap arch peek
    // (no full SeamModel load) so the default token budget below and the ChatModel selection further
    // down can both branch on it. -n/max_new drives `blocks = ceil(n_predict / canvas_length)`
    // (256-token canvas) rather than autoregressive tokens, so the AR default of 2048 would run 8
    // whole blocks for a "Hi" reply; 1024 (4 blocks) is the same order of magnitude as a normal
    // chat reply and still overridable via INFR_MAX_NEW.
    let is_dg = infr_llama::diffusion::is_diffusion_gemma(&gguf);
    // Generation ceiling per reply (a turn also caps to remaining context). High enough for long
    // answers (lists/stories); override with `--max-new` / `sampling.max_new` / INFR_MAX_NEW.
    //
    // The knob has TWO defaults on this path — 2048 autoregressive, 1024 for diffusion-gemma — and
    // `SamplingCfg::default()` can only carry one, so the fallback stays HERE (R5) and `specified`
    // decides which side to take. That is exactly what the S1 bridge encoded by publishing
    // `INFR_MAX_NEW` only when a layer had named it.
    let max_new = if specified.is_path_set("sampling.max_new") {
        cfg.sampling.max_new
    } else if is_dg {
        1024
    } else {
        2048
    };

    // Multi-GPU PIPELINE (layer-split): `INFR_PIPELINE=Vulkan0,Vulkan1` splits ONE model's layers
    // across the listed devices (weights + KV per-layer resident on their device, the residual
    // handed across the boundary). One-shot (a single message), dense models only — a capacity
    // path (run a model too big for one device), byte-identical to single-device. Unset ⇒ the
    // normal persistent-session chat below (byte-for-byte unchanged).
    if let Some(devices) = infr_llama::seam::pipeline_devices(cfg).map(<[usize]>::to_vec) {
        let Some(msg) = message else {
            anyhow::bail!(
                "INFR_PIPELINE runs one-shot: pass a message, e.g. \
                 INFR_PIPELINE=Vulkan0,Vulkan1 infr run {model} \"your prompt\""
            );
        };
        if is_dg || !matches!(selected_backend(cfg)?, Backend::Vulkan(_)) {
            anyhow::bail!("INFR_PIPELINE is a Vulkan dense path — not compatible with INFR_DEV=cpu / INFR_DEV=metal / INFR_DEV=rocm / diffusion-gemma");
        }
        let cfg = apply_model_sampling_defaults(cfg, specified, &gguf);
        let loaded = infr_llama::SeamModel::load_with(&gguf, tok.as_deref(), cfg)?;
        let rendered = loaded.render_chat(msg)?;
        let out = loaded.generate_dense_vulkan_pipeline(&devices, &rendered, max_new)?;
        println!("{out}");
        return Ok(());
    }

    // Multi-GPU TENSOR PARALLELISM: `INFR_TENSOR_PARALLEL=Vulkan0,Vulkan1` SHARDS each layer's weight
    // matrices across the listed devices (column-parallel q/k/v/gate/up, row-parallel o/down) with a
    // P2P all-reduce per attention + per FFN — the single-stream decode speedup (splits the weight
    // GEMV). One-shot, dense models only; output equals single-device to reduction-order tolerance.
    // Unset ⇒ the normal persistent-session chat below (byte-for-byte unchanged).
    if let Some(devices) = infr_llama::seam::tensor_parallel_devices(cfg).map(<[usize]>::to_vec) {
        let Some(msg) = message else {
            anyhow::bail!(
                "INFR_TENSOR_PARALLEL runs one-shot: pass a message, e.g. \
                 INFR_TENSOR_PARALLEL=Vulkan0,Vulkan1 infr run {model} \"your prompt\""
            );
        };
        if is_dg || !matches!(selected_backend(cfg)?, Backend::Vulkan(_)) {
            anyhow::bail!("INFR_TENSOR_PARALLEL is a Vulkan dense path — not compatible with INFR_DEV=cpu / INFR_DEV=metal / INFR_DEV=rocm / diffusion-gemma");
        }
        let cfg = apply_model_sampling_defaults(cfg, specified, &gguf);
        let loaded = infr_llama::SeamModel::load_with(&gguf, tok.as_deref(), cfg)?;
        let rendered = loaded.render_chat(msg)?;
        let out = loaded.generate_dense_vulkan_tp(&devices, &rendered, max_new)?;
        println!("{out}");
        return Ok(());
    }

    // Multi-GPU EXPERT PARALLELISM (MoE): `INFR_EXPERT_PARALLEL=Vulkan0,Vulkan1` SPLITS the model's
    // experts across the listed devices (rank r owns experts [r·E/W, (r+1)·E/W)); the router +
    // attention run replicated on every rank, each computes only its band's experts, and one P2P
    // all-reduce per MoE layer combines the partial expert outputs — a capacity split + parallel
    // expert compute. One-shot, MoE models only; output equals single-device to reduction-order
    // tolerance. Unset ⇒ the normal persistent-session chat below (byte-for-byte unchanged).
    if let Some(devices) = infr_llama::seam::expert_parallel_devices(cfg).map(<[usize]>::to_vec) {
        let Some(msg) = message else {
            anyhow::bail!(
                "INFR_EXPERT_PARALLEL runs one-shot: pass a message, e.g. \
                 INFR_EXPERT_PARALLEL=Vulkan0,Vulkan1 infr run {model} \"your prompt\""
            );
        };
        if is_dg || !matches!(selected_backend(cfg)?, Backend::Vulkan(_)) {
            anyhow::bail!("INFR_EXPERT_PARALLEL is a Vulkan MoE path — not compatible with INFR_DEV=cpu / INFR_DEV=metal / INFR_DEV=rocm / diffusion-gemma");
        }
        let cfg = apply_model_sampling_defaults(cfg, specified, &gguf);
        let loaded = infr_llama::SeamModel::load_with(&gguf, tok.as_deref(), cfg)?;
        let rendered = loaded.render_chat(msg)?;
        let out = loaded.generate_moe_vulkan_ep(&devices, &rendered, max_new)?;
        println!("{out}");
        return Ok(());
    }

    // Build the per-backend generation primitive (`ChatModel`) — the ONE `build_chat_model` funnel
    // shared with `cmd_serve` (DG→Metal→CPU→Vulkan by the env-selected backend) — then wrap it in
    // the ONE shared `Chat` (infr_llama::model) that owns history + `<think>`-stripping and drives
    // the single REPL below. Every backend does history-based multi-turn; qwen35 (Qwen3.5) rides
    // the SAME standard `ChatModel` structs (issue #30). Model-aware chat sampling first (arch-family
    // table + generation_config sibling; a user `--temp`/`--top-k`/`--top-p` already in the env wins).
    let cfg = apply_model_sampling_defaults(cfg, specified, &gguf);
    let mut model = build_chat_model(&gguf, tok.as_deref(), is_dg, &cfg)?;
    // Compile the lazily-built pipelines NOW (like `serve` does before its first request) so the
    // first turn's reported prefill rate measures prefill, not one-time pipeline builds — a cold
    // diffusion-gemma prefill measured 26 t/s vs 1424 t/s warm, all compile.
    model.warmup()?;
    let mut chat = infr_llama::chat::Chat::new(model);
    // Live denoise canvas view (diffusion-gemma only — see `DiffusionVisual`'s doc); `None` when
    // unset/not-DG/not-a-tty leaves `run_chat_turn` on the exact pre-existing `chat.turn` path.
    let mut visual = if is_dg {
        DiffusionVisual::new(&gguf, diffusion_visual, &cfg)?
    } else {
        None
    };

    // One-shot (a message) or an interactive multi-turn REPL (every backend now supports it).
    if let Some(m) = message {
        run_chat_turn(&mut chat, m, max_new, visual.as_mut())?;
        return Ok(());
    }
    loop {
        match chat.repl_status() {
            Some(s) => print!("\n[{s}] > "),
            None => print!("\n> "),
        }
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if read_line_interruptible(&mut line)? == 0 {
            break; // EOF, or a signal arrived while we sat at the prompt
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "exit" | "quit" | ":q" | ":quit") {
            break;
        }
        if let Err(e) = run_chat_turn(&mut chat, line, max_new, visual.as_mut()) {
            // A shutdown-aborted turn reports the abort as an error; don't print it as a failure,
            // just leave the REPL (main then exits 130/143).
            if !infr_core::shutdown::shutdown_requested() {
                eprintln!("error: {e}");
            }
        }
        if infr_core::shutdown::shutdown_requested() {
            break;
        }
    }
    Ok(())
}

/// `read_line` that gives up when a signal has latched the shutdown.
///
/// `Stdin::read_line` cannot: `BufRead::read_until` swallows `EINTR` and re-issues the `read`, so a
/// Ctrl-C at an IDLE chat prompt would leave the latch set but the process parked on the terminal
/// until the user pressed Enter. Reading fd 0 directly lets the `EINTR` (the handler installs
/// without `SA_RESTART`) come back to us, where we check the latch and return EOF.
///
/// Byte-at-a-time so no input past the newline is ever consumed (nothing else in the process reads
/// stdin, but over-reading a pipe would silently eat the next prompt). One syscall per typed
/// character is free at human speed.
///
/// Returns bytes read (0 = EOF or shutdown), like `read_line`.
fn read_line_interruptible(line: &mut String) -> anyhow::Result<usize> {
    #[cfg(not(unix))]
    {
        return Ok(std::io::stdin().read_line(line)?);
    }
    #[cfg(unix)]
    {
        // Bytes, not chars: a multi-byte UTF-8 codepoint (any non-ASCII prompt) arrives one byte
        // per read and is only a `char` once it is whole.
        let mut buf: Vec<u8> = Vec::new();
        loop {
            if infr_core::shutdown::shutdown_requested() {
                return Ok(0);
            }
            let mut b = 0u8;
            // SAFETY: a 1-byte read into a live stack byte.
            let r = unsafe { libc::read(0, std::ptr::addr_of_mut!(b).cast(), 1) };
            match r {
                0 => break, // EOF
                1 => {
                    buf.push(b);
                    if b == b'\n' {
                        break;
                    }
                }
                _ => {
                    let e = std::io::Error::last_os_error();
                    if e.kind() == std::io::ErrorKind::Interrupted {
                        continue; // signal: the latch check at the top of the loop decides
                    }
                    return Err(e.into());
                }
            }
        }
        line.push_str(&String::from_utf8_lossy(&buf));
        Ok(buf.len())
    }
}

/// Run one chat turn through the shared [`Chat`]: stream pieces via the `<think>` renderer, then
/// print the prefill/decode stats line. `visual: Some` (diffusion-gemma, `INFR_DIFFUSION_VISUAL`
/// set, stdout a tty) drives the turn through `turn_with_step_hook` instead, so the live canvas
/// view redraws in a reserved terminal region while a block denoises; `None` is the exact
/// pre-existing `chat.turn` call, byte-for-byte.
///
/// `on_piece` now fires once PER FINISHED BLOCK (`DiffusionGemmaChat::generate_impl` streams each
/// block's committed text through the shared detok as soon as `diffusion_generate`'s `on_block`
/// hands it over — see that fn's doc), not once for the whole reply at turn end. So each call
/// erases whatever live region is currently reserved (`DiffusionVisual::end`, a no-op if none is —
/// e.g. before the very first block), which lets the just-finished block's text print normally
/// and PERMANENTLY via `render.feed` right where the region's cursor was left, scrolling up
/// naturally like any other transcript text. `DiffusionVisual::step` then lazily reserves a FRESH
/// region below that printed text the next time a block starts denoising — so the live-updating
/// region only ever shows the CURRENT in-progress block, never a finished one.
fn run_chat_turn(
    chat: &mut infr_llama::chat::Chat,
    message: &str,
    max_new: usize,
    visual: Option<&mut DiffusionVisual>,
) -> anyhow::Result<()> {
    let mut render = ThinkRender::new();
    let stats = match visual {
        Some(v) => {
            // Both closures below only need transient access (never concurrently — generation is
            // single-threaded and `on_step`/`on_piece` never interleave), so a `RefCell` lets them
            // share `v` without two simultaneous `&mut` captures.
            let visual = std::cell::RefCell::new(v);
            // Buffer the streamed pieces instead of feeding ThinkRender per block: the
            // template-aware rendering (think/channel-marker handling) only formats correctly
            // over the COMPLETE response — per-block feeding split markers across pieces and
            // broke the formatting (user report). The live region is the in-progress display;
            // the one-shot render below is the permanent transcript.
            let buffered = std::cell::RefCell::new(String::new());
            let mut on_piece = |p: &str| {
                buffered.borrow_mut().push_str(p);
            };
            let mut on_step =
                |view: infr_llama::diffusion::StepView| visual.borrow_mut().step(view);
            let result =
                chat.turn_with_step_hook(message, max_new, &mut on_piece, Some(&mut on_step));
            // Erase the live region first, then render the whole response through the
            // template-aware renderer exactly once — correct formatting, no duplication.
            visual.borrow_mut().end();
            render.feed(&buffered.borrow());
            result?
        }
        None => chat.turn(message, max_new, &mut |p| render.feed(p))?,
    };
    render.finish();
    let rate = |n: usize, s: f64| if s > 0.0 { n as f64 / s } else { 0.0 };
    eprintln!(
        "[prefill {} tok @ {:.0} tok/s ({:.0} ms) | decode {} tok @ {:.1} tok/s]",
        stats.n_prompt,
        rate(stats.n_prompt, stats.prompt_secs),
        stats.prompt_secs * 1000.0,
        stats.n_gen,
        rate(stats.n_gen, stats.decode_secs),
    );
    Ok(())
}

/// Fixed-size scratch region (rows) the live view redraws in place — capped rather than sized to
/// the real terminal so there's no ioctl/terminal-size dependency: worst case on a narrower
/// terminal a long line wraps, which only affects this scratch region's own redraw math, never the
/// surrounding transcript.
const DG_VISUAL_ROWS: usize = 16;
/// Default/fallback column cap per rendered line (item-truncated — see [`DiffusionVisual::cols`]),
/// same conservative-width rationale as [`DG_VISUAL_ROWS`]. A parseable `$COLUMNS` narrower than
/// this clamps `cols` down further (audit finding 2: without that clamp, a narrower real terminal
/// WRAPS a `DG_VISUAL_COLS`-wide line, so the region occupies more than `DG_VISUAL_ROWS` actual
/// terminal rows and the fixed cursor-up math in `step`/`end` desyncs — frames smear into each
/// other instead of overwriting cleanly).
const DG_VISUAL_COLS: usize = 120;

use std::io::{IsTerminal, Write as _};

/// `INFR_DIFFUSION_VISUAL` live denoise canvas view for `infr run` (diffusion-gemma only — see
/// `docs/diffusion-gemma.md`, ports the UX idea of the oracle's `--diffusion-visual` without
/// depending on it): per step, decode the block's CURRENT canvas fresh with a throwaway tokenizer
/// ([`OaiRenderer::decode_ids`] — cheap, ≤ canvas_len tokens, no GPU work), render accepted
/// (committed) runs as normal text and not-yet-accepted (still renoising — this sampler has no
/// literal mask token, see `crate::diffusion`'s module doc) runs as a dim `·` placeholder, and
/// redraw a fixed-height region in place (cursor-up + erase, DEC synchronized-update framing) so
/// the terminal never scrolls mid-block.
///
/// The region is reserved LAZILY (the first `step` call after construction, or after the previous
/// region was erased by `end`) and erased every time a block finishes (`run_chat_turn`'s
/// `on_piece`, once per block now — see `DiffusionGemmaChat::generate_impl`'s doc) so that block's
/// permanent transcript text prints cleanly, scrolling up naturally; the NEXT block's first `step`
/// then reserves a fresh region below it. So at any moment at most one region is on screen, and it
/// only ever shows the block currently denoising.
struct DiffusionVisual {
    oai: infr_llama::chat::OaiRenderer,
    /// Rows the previous frame's redraw advanced past the region's top — 0 before the first frame
    /// (mirrors the oracle's `cb_data.vis_prev_rows`).
    prev_rows: usize,
    /// Whether a region is currently reserved on screen. `step` reserves one lazily when this is
    /// `false`; `end` erases the region and clears this back to `false` (a no-op if already
    /// clear — `run_chat_turn` calls `end` unconditionally on every `on_piece` and once more as a
    /// post-turn safety net).
    active: bool,
    /// Column cap for one rendered line, in DISPLAY ITEMS (audit finding 1: never raw chars/bytes
    /// of a pre-rendered ANSI string) — `min(DG_VISUAL_COLS, $COLUMNS - 1)` when `COLUMNS` parses
    /// to something usable, else `DG_VISUAL_COLS`. Best-effort: infr has no ioctl-based terminal
    /// size probe, and `$COLUMNS` is a shell-maintained convention, not guaranteed exported to a
    /// child process — this only ever narrows the conservative default, never widens past it.
    cols: usize,
}

impl DiffusionVisual {
    /// `None` unless `mode` is `"1"` (stdout must be a tty) or `"force"` (bypasses the tty check —
    /// for scripted verification against piped stdout, e.g. `... | tail -20`).
    ///
    /// `mode` is `infr run --diffusion-visual`'s value. This is a CLI PRESENTATION knob, not a
    /// `Config` field (§6.10): nothing below `main` can observe it, and it steers a terminal
    /// drawing routine rather than the forward. The `INFR_DIFFUSION_VISUAL` spelling survives as
    /// clap's `env` fallback for the flag, so scripts that set it keep working (R2) and the value
    /// still enters through the flag layer, not a `std::env::var` in the middle of `cmd_run`.
    fn new(gguf: &Path, mode: &str, cfg: &Arc<Config>) -> anyhow::Result<Option<Self>> {
        let force = mode == "force";
        if !force && mode != "1" {
            return Ok(None);
        }
        if !force && !std::io::stdout().is_terminal() {
            return Ok(None);
        }
        let cols = std::env::var("COLUMNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .and_then(|c| c.checked_sub(1))
            .filter(|&c| c > 0)
            .map(|c| DG_VISUAL_COLS.min(c))
            .unwrap_or(DG_VISUAL_COLS);
        Ok(Some(Self {
            oai: infr_llama::chat::OaiRenderer::open(gguf, cfg.clone())?,
            prev_rows: 0,
            active: false,
            cols,
        }))
    }

    /// Reserve the region: hide the cursor, print `DG_VISUAL_ROWS` blank lines (scrolls once if
    /// already at the bottom, exactly like a normal print would — scrollback stays intact), then
    /// park the cursor back at the region's top.
    fn begin(&mut self) {
        self.prev_rows = 0;
        print!("\x1b[?25l{}", "\n".repeat(DG_VISUAL_ROWS));
        print!("\x1b[{DG_VISUAL_ROWS}A");
        std::io::stdout().flush().ok();
        self.active = true;
    }

    /// Redraw the region for one denoise step — lazily reserving it first (`begin`) if the last
    /// finished block's `on_piece` erased it (or this is the turn's very first step).
    fn step(&mut self, view: infr_llama::diffusion::StepView) {
        if !self.active {
            self.begin();
        }
        // One flat (display_char, is_dim) sequence: accepted runs decode to real text, sanitized
        // per audit finding 3 (`\r` dropped — never jumps the cursor mid-line; `\t` -> a single
        // space, never a literal tab); not-yet-accepted runs render as a dim `·` placeholder.
        // Kept as tagged chars, NOT pre-rendered ANSI text, so the truncation/line-splitting below
        // always operates on DISPLAY items and never raw escape bytes (audit finding 1).
        let mut items: Vec<(char, bool)> = Vec::new();
        let mut i = 0;
        while i < view.canvas.len() {
            if view.accepted[i] {
                let start = i;
                while i < view.canvas.len() && view.accepted[i] {
                    i += 1;
                }
                if let Ok(s) = self.oai.decode_ids(&view.canvas[start..i]) {
                    for c in s.chars() {
                        match c {
                            '\r' => {}                        // dropped: never jump the cursor
                            '\t' => items.push((' ', false)), // one space, never a literal tab
                            _ => items.push((c, false)),
                        }
                    }
                }
            } else {
                let start = i;
                while i < view.canvas.len() && !view.accepted[i] {
                    i += 1;
                }
                for _ in start..i {
                    items.push(('·', true));
                }
            }
        }

        // Split into display lines on '\n' only (audit finding 3's other half — a literal '\n' in
        // decoded text is the sole line-break signal), each capped at `self.cols` ITEMS (audit
        // findings 1/2), then rendered with dim runs balanced one `\x1b[2m`/`\x1b[0m` pair at a
        // time (`render_dim_line`) — never a bare escape fragment, never a leaked dim mode.
        let mut lines: Vec<Vec<(char, bool)>> = vec![Vec::new()];
        for &(c, dim) in &items {
            if c == '\n' {
                lines.push(Vec::new());
            } else {
                lines.last_mut().unwrap().push((c, dim));
            }
        }
        // No header line (user feedback: the "[diffusion] block N step S/M" rows read as
        // noise) — the resolving canvas itself is the progress indicator.
        let _ = (view.block, view.step, view.max_steps, view.committed_before);
        let mut rows: Vec<String> = Vec::new();
        for line in &lines {
            let cut = line.len().min(self.cols);
            rows.push(render_dim_line(&line[..cut]));
        }
        let keep = rows.len().min(DG_VISUAL_ROWS);
        let shown = &rows[rows.len() - keep..];

        // ?7l: auto-wrap OFF for the frame — a too-long line truncates at the margin instead
        // of wrapping to a second physical row, which would desync the fixed-row cursor math
        // (COLUMNS is a shell variable most shells don't export, so the item clamp can't know
        // the real width). Restored (?7h) at frame end.
        let mut frame = String::from("\x1b[?2026h\x1b[?7l"); // begin synchronized update
        if self.prev_rows > 0 {
            frame.push_str(&format!("\x1b[{}A", self.prev_rows));
        }
        frame.push('\r');
        for r in 0..DG_VISUAL_ROWS {
            if let Some(line) = shown.get(r) {
                frame.push_str(line);
            }
            frame.push_str("\x1b[K"); // erase to end of line — clears a shorter previous frame
            if r + 1 < DG_VISUAL_ROWS {
                frame.push('\n');
            }
        }
        frame.push_str("\x1b[?7h\x1b[?2026l"); // auto-wrap back on, end synchronized update
        self.prev_rows = DG_VISUAL_ROWS - 1;
        print!("{frame}");
        std::io::stdout().flush().ok();
    }

    /// Erase the region and restore the cursor, if one is currently reserved (a no-op otherwise —
    /// safe to call unconditionally, e.g. after every finished block AND once more at turn end).
    fn end(&mut self) {
        if !self.active {
            return;
        }
        let mut frame = String::new();
        if self.prev_rows > 0 {
            frame.push_str(&format!("\x1b[{}A", self.prev_rows));
        }
        frame.push_str("\r\x1b[J\x1b[?25h"); // erase from cursor to end of screen, show cursor
        print!("{frame}");
        std::io::stdout().flush().ok();
        self.prev_rows = 0;
        self.active = false;
    }
}

/// Render one line's `(display_char, is_dim)` items, wrapping each maximal DIM RUN in exactly one
/// `\x1b[2m`/`\x1b[0m` pair — the fix for audit finding 1: the old code truncated a pre-rendered
/// `"\x1b[2m·\x1b[0m"` string by raw `char` count, which could cut an escape sequence in half
/// (printing a literal `[2m` fragment) and/or drop the trailing reset, leaking dim mode into
/// whatever printed next. Operating on tagged items instead of ANSI bytes makes that class of bug
/// unrepresentable: every emitted `\x1b[2m` here is always paired with an `\x1b[0m` on the same
/// line, and truncation (by the caller, before this runs) only ever drops whole items.
fn render_dim_line(items: &[(char, bool)]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < items.len() {
        if items[i].1 {
            out.push_str("\x1b[2m");
            while i < items.len() && items[i].1 {
                out.push(items[i].0);
                i += 1;
            }
            out.push_str("\x1b[0m");
        } else {
            while i < items.len() && !items[i].1 {
                out.push(items[i].0);
                i += 1;
            }
        }
    }
    out
}

/// The Metal dense/MoE [`ChatModel`] for run AND serve: the persistent-session seam chat, or —
/// with `INFR_SPEC_DRAFT=<gguf>` — speculative decoding (a small same-tokenizer draft proposes up
/// to `INFR_SPEC_K` tokens per round, default 6; one batched target forward verifies; greedy-only,
/// pays off for ~8B-class targets — issue #16). One selection funnel so run and serve can never
/// disagree on how the Metal model is built.
#[cfg(target_os = "macos")]
fn metal_chat_model(
    gguf: &Path,
    tok: Option<&Path>,
    cfg: &Arc<Config>,
) -> anyhow::Result<Box<dyn infr_llama::chat::ChatModel + Send>> {
    if let Some(draft_path) = cfg.spec.draft.clone() {
        // Spec decode is GREEDY-ONLY: the draft/verify equality check assumes argmax, so a
        // non-zero temperature would reject every correctly-drafted token. S1 forced that by
        // writing `INFR_TEMP=0` into the process (`publish_transitional_greedy`); S8 pins it on
        // the VALUE the two models are loaded with, which is where the sampler has read it since
        // S4 — the env write had already stopped being observed by anything.
        let cfg = &{
            let mut greedy = (**cfg).clone();
            greedy.sampling.temp = 0.0;
            Arc::new(greedy)
        };
        // Target and draft get the SAME config — a process loading both hands them one value
        // (`docs/config-plan.md` §5.1), which is exactly what the shared environment did.
        let target = infr_llama::SeamModel::load_with(gguf, tok, cfg.clone())?;
        let draft = infr_llama::SeamModel::load_with(&draft_path, None, cfg.clone())?;
        // Upper bound on the draft length; the driver adapts the actual k per round to recent
        // acceptance (verify cost scales with rows on this hardware, so over-drafting
        // low-acceptance text costs real time).
        let k = cfg.spec.k;
        let (tb, db) = (
            std::fs::metadata(gguf).map(|m| m.len()).unwrap_or(0),
            std::fs::metadata(&draft_path).map(|m| m.len()).unwrap_or(0),
        );
        if db * 4 > tb {
            eprintln!(
                "[spec] warning: draft is more than 1/4 the target's size — \
                 speculation only pays when the target is much larger (see #16)"
            );
        }
        eprintln!("[metal spec — target + {k}-token draft verify, greedy (sampling.temp=0)]");
        Ok(Box::new(infr_llama::chat::SpecMetalChat::new(
            target, draft, k,
        )))
    } else {
        Ok(Box::new(infr_llama::chat::MetalSeamChat::new(
            infr_llama::SeamModel::load_with(gguf, tok, cfg.clone())?,
        )))
    }
}

/// Serve adapter for the seam-backed [`ChatModel`]s (qwen35 on any backend, dense/MoE on the
/// Vulkan seam or the CPU/Metal reference): renders the FULL OpenAI conversation — including tool
/// specs and prior tool calls/results — through the model's own chat template
/// (`infr_chat::render_chat_oai`, model-independent), generates through the SAME `ChatModel`
/// primitive `infr run`/`bench` drive (persistent session ⇒ per-request suffix-only prefill), and
/// streams through the same [`ChatStream`] splitter (reasoning/content/auto-parsed tool calls).
/// Grammar-FORCED tool_choice builds an llguidance constraint and generates through
/// `generate_constrained` (llama.cpp-parity reliability); auto/none stream through the parser.
/// SERIALISED: the backend is a `&mut`-only `ChatModel` with a single KV session, so concurrent
/// requests take turns behind a `Mutex`. Used for CPU / Metal / diffusion-gemma, which have no
/// multi-slot engine — `cmd_serve` pins those to `--parallel 1` so the queueing is explicit rather
/// than a silent serialisation of a server the user asked to parallelise.
struct SeamGenerator {
    model: std::sync::Mutex<Box<dyn infr_llama::chat::ChatModel + Send>>,
    renderer: infr_llama::chat::OaiRenderer,
}

impl SeamGenerator {
    fn new(
        gguf_path: &Path,
        model: Box<dyn infr_llama::chat::ChatModel + Send>,
        cfg: Arc<Config>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            model: std::sync::Mutex::new(model),
            renderer: infr_llama::chat::OaiRenderer::open(gguf_path, cfg)?,
        })
    }
}

/// CONCURRENT: N KV slots on the Vulkan seam, round-robin on the GPU at token granularity — see
/// [`infr_llama::parallel::ParallelSeam`]. The `infr serve --parallel N` default path.
struct ParallelGenerator {
    engine: infr_llama::parallel::ParallelSeam,
    renderer: infr_llama::chat::OaiRenderer,
}

impl ParallelGenerator {
    fn new(
        gguf_path: &Path,
        engine: infr_llama::parallel::ParallelSeam,
        cfg: Arc<Config>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            engine,
            renderer: infr_llama::chat::OaiRenderer::open(gguf_path, cfg)?,
        })
    }
}

/// The two things a serve backend must do, so the OpenAI wire logic (rendering, forced tool calls,
/// stop sequences, reasoning/content splitting) lives in ONE place — [`run_chat`] — instead of being
/// duplicated per backend.
trait GenBackend: Send + Sync {
    fn renderer(&self) -> &infr_llama::chat::OaiRenderer;

    /// The reply budget for a request that names none: `sampling.max_new` (`INFR_MAX_NEW`,
    /// default 2048) off the config this backend's renderer was opened with. A trait method
    /// rather than a free function because the config is per-backend state — and the default
    /// body serves both generators, since both hold the same renderer (S8; `run_chat` used to
    /// read the `INFR_MAX_NEW` variable from the process environment here).
    fn max_new_default(&self) -> usize {
        self.renderer().engine_cfg().sampling.max_new
    }

    /// This sequence's own state: sampling config, abort latch, and (when the engine is
    /// multi-slot) its turn on the GPU baton.
    fn request_ctx(
        &self,
        sampling: infr_llama::sampling::RequestSampling,
    ) -> infr_llama::sampling::RequestCtx;

    fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut infr_llama::grammar::Constraint>,
        req: &infr_llama::sampling::RequestCtx,
        on_piece: &mut dyn FnMut(&str),
    ) -> anyhow::Result<infr_llama::GenStats>;

    /// Drop the serialised session's persistent KV so the NEXT `generate` re-prefills from scratch.
    /// The forced-tool fallback calls this before its unconstrained retry so the retry can never
    /// inherit the constrained attempt's primed `<tool_call>` state. Default no-op: the multi-slot
    /// engine hands each request a fresh best-prefix slot, so there is nothing serialised to reset.
    fn reset(&self) {}
}

impl GenBackend for SeamGenerator {
    fn renderer(&self) -> &infr_llama::chat::OaiRenderer {
        &self.renderer
    }

    fn request_ctx(
        &self,
        sampling: infr_llama::sampling::RequestSampling,
    ) -> infr_llama::sampling::RequestCtx {
        // No gate: this backend is serialised by the Mutex below, so there is never more than one
        // sequence on the GPU to take turns with.
        infr_llama::sampling::RequestCtx::new(sampling)
    }

    fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut infr_llama::grammar::Constraint>,
        req: &infr_llama::sampling::RequestCtx,
        on_piece: &mut dyn FnMut(&str),
    ) -> anyhow::Result<infr_llama::GenStats> {
        let mut model = self.model.lock().expect("serve generator poisoned");
        match constraint {
            Some(c) => model.generate_constrained(prompt, max_new, c, Some(req), on_piece),
            None => model.generate(prompt, max_new, Some(req), on_piece),
        }
    }

    fn reset(&self) {
        self.model
            .lock()
            .expect("serve generator poisoned")
            .reset_kv();
    }
}

impl GenBackend for ParallelGenerator {
    fn renderer(&self) -> &infr_llama::chat::OaiRenderer {
        &self.renderer
    }

    fn request_ctx(
        &self,
        sampling: infr_llama::sampling::RequestSampling,
    ) -> infr_llama::sampling::RequestCtx {
        self.engine.request_ctx(sampling)
    }

    fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut infr_llama::grammar::Constraint>,
        req: &infr_llama::sampling::RequestCtx,
        on_piece: &mut dyn FnMut(&str),
    ) -> anyhow::Result<infr_llama::GenStats> {
        self.engine
            .generate(prompt, max_new, constraint, req, |p| on_piece(p))
    }
}

/// Map the validated request params onto the seam's per-request sampling scope. An ABSENT request
/// field stays `None`/neutral, so the decode loop keeps inheriting the CLI's `INFR_TEMP` /
/// `INFR_TOP_K` / `INFR_TOP_P` defaults for it — only an EXPLICIT field overrides.
fn request_sampling(p: &infr_server::GenParams) -> infr_llama::sampling::RequestSampling {
    infr_llama::sampling::RequestSampling {
        temp: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        seed: p.seed,
        presence_penalty: p.presence_penalty.unwrap_or(0.0),
        frequency_penalty: p.frequency_penalty.unwrap_or(0.0),
        repeat_penalty: p.repeat_penalty.unwrap_or(1.0),
        ..Default::default()
    }
}

impl infr_server::ChatGenerator for SeamGenerator {
    fn chat(
        &self,
        messages: &[infr_engine::ChatMessage],
        tools: Option<&serde_json::Value>,
        tool_choice: Option<&str>,
        params: &infr_server::GenParams,
        cancel: &std::sync::atomic::AtomicBool,
        on_delta: &mut dyn FnMut(infr_engine::Delta),
    ) -> anyhow::Result<infr_server::ChatOutcome> {
        run_chat(self, messages, tools, tool_choice, params, cancel, on_delta)
    }
}

impl infr_server::ChatGenerator for ParallelGenerator {
    fn chat(
        &self,
        messages: &[infr_engine::ChatMessage],
        tools: Option<&serde_json::Value>,
        tool_choice: Option<&str>,
        params: &infr_server::GenParams,
        cancel: &std::sync::atomic::AtomicBool,
        on_delta: &mut dyn FnMut(infr_engine::Delta),
    ) -> anyhow::Result<infr_server::ChatOutcome> {
        run_chat(self, messages, tools, tool_choice, params, cancel, on_delta)
    }
}

/// The OpenAI chat body, shared by every serve backend: render the conversation through the model's
/// own template, honour a FORCED tool_choice with an llguidance constraint, then stream the reply
/// through the reasoning/content/tool-call splitter with stop sequences applied to the raw text.
///
/// Backend-agnostic by construction — all the per-sequence state (sampling, RNG seed, penalties,
/// stop matcher, abort latch) is created HERE, per call, and handed to the backend explicitly. Two
/// concurrent calls therefore share nothing: that is what makes request A's `temperature` unable to
/// leak into request B (the old thread-local could not offer that guarantee once one thread stepped
/// several sequences).
fn run_chat(
    be: &dyn GenBackend,
    messages: &[infr_engine::ChatMessage],
    tools: Option<&serde_json::Value>,
    tool_choice: Option<&str>,
    params: &infr_server::GenParams,
    cancel: &std::sync::atomic::AtomicBool,
    on_delta: &mut dyn FnMut(infr_engine::Delta),
) -> anyhow::Result<infr_server::ChatOutcome> {
    {
        // `tools` arrives already parsed (a borrowed Value) — no Value→string→Value round-trip.
        let prompt = be.renderer().render(messages, tools)?;
        // The request's `max_tokens`/`max_completion_tokens` wins; `sampling.max_new`
        // (INFR_MAX_NEW, default 2048) is the server-side default for requests that don't set one.
        let max_new = params
            .max_tokens
            .map(|v| v as usize)
            .unwrap_or_else(|| be.max_new_default());
        // THIS sequence's own sampling (temperature/top_p/top_k/seed/penalties) + abort latch + GPU
        // turn. Owned by this call — not installed anywhere ambient — so it cannot be observed by,
        // or leak into, any other in-flight request.
        let req = be.request_ctx(request_sampling(params));
        // Forced tool_choice ("required"/named): grammar-constrain the call body (the same
        // llguidance machinery as the bespoke path — grammar::constrained_step runs inside the
        // seam decode). Prime the assistant turn with the <tool_call> opener and parse the
        // constrained JSON; on any failure fall back to unconstrained (mirrors LlamaGenerator).
        if let Some(mut constraint) = be.renderer().tool_constraint(tools, tool_choice)? {
            let primed = format!("{prompt}<tool_call>\n");
            let mut body = String::new();
            let mut tokens = (0u32, 0u32);
            let emitted = match be.generate(
                &primed,
                max_new,
                Some(&mut constraint),
                &req,
                &mut |p: &str| {
                    body.push_str(p);
                    // Client disconnected (server latched `cancel`): stop this constrained decode.
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        req.abort();
                    }
                },
            ) {
                Ok(gstats) => {
                    tokens = (gstats.n_prompt as u32, gstats.n_gen as u32);
                    let body = body.trim().trim_end_matches("</tool_call>").trim();
                    match serde_json::from_str::<serde_json::Value>(body) {
                        Ok(val) => val.get("name").and_then(|v| v.as_str()).map(|name| {
                            let arguments = val
                                .get("arguments")
                                .cloned()
                                .unwrap_or(serde_json::json!({}));
                            on_delta(infr_engine::Delta::ToolCall {
                                name: name.to_string(),
                                arguments: serde_json::to_string(&arguments)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            });
                        }),
                        Err(_) => None,
                    }
                    .is_some()
                }
                Err(e) => {
                    eprintln!("[tools] seam grammar-constrained generation failed ({e})");
                    false
                }
            };
            if emitted {
                return Ok(infr_server::ChatOutcome {
                    finish: infr_server::Finish::ToolCalls,
                    prompt_tokens: tokens.0,
                    completion_tokens: tokens.1,
                });
            }
            // The constrained attempt advanced the serialised session past the primed `<tool_call>`
            // opener + the (unparseable) body. Reset it so the unconstrained retry below re-prefills
            // the clean `prompt` from scratch, instead of inheriting that divergent KV state and
            // relying on the seam's common-prefix rewind to unwind it. No-op on the multi-slot
            // engine (each request already gets a fresh best-prefix slot).
            be.reset();
            eprintln!(
                "[tools] forced tool call produced no parseable call; falling back to unconstrained"
            );
        }
        let mut stream = infr_engine::ChatStream::new(tool_choice != Some("none"));
        // Stop sequences run on the RAW decoded text, BEFORE the reasoning/tool splitter — so a
        // stop string that spans two tokens still fires, and a partial stop prefix is held back
        // instead of being streamed out (see `StopMatcher`). A hit latches an abort on THIS
        // sequence's own `RequestCtx`, which its decode loop polls after the current token. The
        // matcher and the latch are both per-call locals, so N concurrent requests each stop on
        // their OWN stop strings and nobody else's.
        let mut stops = infr_server::StopMatcher::new(params.stop.clone());
        let stats = {
            let od = &mut *on_delta;
            // Template-prefilled thinking (the PROMPT ends with the `<think>` opener): inject a
            // synthetic opener so the splitter emits the head as Reasoning deltas, mirroring
            // `Chat::turn` — run and serve expose thinking identically.
            if infr_engine::prompt_prefills_think(&prompt) {
                stream.push("<think>", &mut *od);
            }
            let stats = be.generate(&prompt, max_new, None, &req, &mut |piece: &str| {
                let safe = stops.push(piece);
                if !safe.is_empty() {
                    stream.push(&safe, &mut *od);
                }
                // A stop sequence hit OR a client disconnect (the server latched `cancel`) both
                // abort THIS sequence's decode at its next poll, freeing the GPU slot promptly.
                if stops.hit() || cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    req.abort();
                }
            })?;
            // No stop fired: whatever the matcher was holding back was never a stop prefix.
            let tail = stops.flush();
            if !tail.is_empty() {
                stream.push(&tail, &mut *od);
            }
            stats
        };
        stream.finish(on_delta);
        let finish = if stops.hit() {
            infr_server::Finish::Stop
        } else if stats.n_gen >= max_new {
            // The budget was exhausted (EOS would have broken the loop earlier).
            infr_server::Finish::Length
        } else {
            infr_server::Finish::Stop
        };
        Ok(infr_server::ChatOutcome {
            finish,
            prompt_tokens: stats.n_prompt as u32,
            completion_tokens: stats.n_gen as u32,
        })
    }
}

/// Benchmark prefill (pp) or decode (tg) tok/s with the same -p/-n/-d/-r interface as
/// `llama-bench`, so `infr bench` and `llama-bench` are directly comparable. Dummy tokens (timing
/// is data-independent), `prefill_chunk` policy for the prefill batching (the engine's real path).
#[allow(clippy::too_many_arguments)]
fn cmd_bench(
    model: &str,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<String>,
    reps: usize,
    ngl: usize,
    json: bool,
    cfg: &Arc<Config>,
) -> anyhow::Result<()> {
    // `--dev`/`-u`/`-t` reached the engine before the model loads: `--dev` through this `cfg`
    // (which `VulkanBackend::new_with` reads its device pick off since S5a), `-u` through
    // INFR_UBATCH, `-t` through RAYON_NUM_THREADS. So `--dev metal`/`--dev cpu` route here exactly
    // like a raw `INFR_DEV=metal`/`INFR_DEV=cpu` invocation.
    // Bench also pins `sampling.ignore_eos` (see `cli_flag_layer`): benchmarks decode a FIXED token
    // count (llama-bench semantics) and never stop at EOS — a model that emits EOS instantly on the
    // dummy context would otherwise report fictional tok/s.
    // -pg "P,G": a coding-agent turn (ingest P then generate G); throughput = (P+G)/time.
    let pg = pg
        .map(|s| -> anyhow::Result<(usize, usize)> {
            let (p, g) = s.split_once(',').context("--pg expects `P,G`")?;
            Ok((p.trim().parse()?, g.trim().parse()?))
        })
        .transpose()?;
    let (gguf, tok) = resolve(model)?;
    // ONE backend reader shared with `cmd_run`/`cmd_serve` ([`selected_backend`], METAL > CPU >
    // Vulkan) — bench used to read CPU before METAL, disagreeing with run/serve. `-ngl 0` is bench's
    // own extra "force the CPU reference backend" gate (llama-bench semantics), so it wins first.
    let backend = if ngl == 0 {
        Backend::Cpu
    } else {
        selected_backend(cfg)?
    };
    // diffusion-gemma (Phase 4/D, docs/diffusion-gemma.md): llama-bench has no diffusion mode, so
    // `infr bench` measures infr's OWN decode shape (block prefill + canvas denoise, see
    // `cmd_bench_diffusion_gemma`'s doc) instead of routing through the AR pp/tg arms below.
    if infr_llama::diffusion::is_diffusion_gemma(&gguf) {
        let metal = matches!(backend, Backend::Metal);
        let cpu = matches!(backend, Backend::Cpu);
        return cmd_bench_diffusion_gemma(
            &gguf,
            tok.as_deref(),
            n_prompt,
            n_gen,
            depth,
            pg,
            reps,
            cpu,
            metal,
            json,
            cfg,
        );
    }
    // qwen35 (Qwen3.5) benches through the STANDARD arms below (`cmd_bench_cpu` / the seam's
    // `bench_vulkan` / `cmd_bench_metal`) — `SeamModel::load` drives it through the unified runner
    // (`Config::from_gguf` + `MixerW::DeltaNet`), reusing the exact same pp/tg/depth methodology
    // every other arch gets (no more qwen35-only bench arm or depth-accounting artifacts).
    // -ngl 0 (or `--dev cpu` → INFR_DEV=cpu): run on the CPU reference backend (no GPU), comparable to
    // `llama-bench -ngl 0`. llama4 benches through the standard Vulkan arm below like every other
    // model now (the paged expert cache) — only these force it onto this CPU arm.
    if matches!(backend, Backend::Cpu) {
        return cmd_bench_cpu(
            &gguf,
            tok.as_deref(),
            n_prompt,
            n_gen,
            depth,
            pg,
            reps,
            json,
            cfg,
        );
    }
    // Metal (set by `--dev metal` or `INFR_DEV=metal`): bench the dense forward on the Metal backend
    // through the agnostic seam — same pp/tg/pg + depth methodology as the CPU arm, directly
    // comparable to `llama-bench` on the Metal build.
    if matches!(backend, Backend::Metal) {
        return cmd_bench_metal(
            &gguf,
            tok.as_deref(),
            n_prompt,
            n_gen,
            depth,
            pg,
            reps,
            json,
            cfg,
        );
    }
    // ROCm (set by `--dev rocm[:N]` / `INFR_DEV=rocm`): bench the dense/MoE forward on the AMD
    // GPU through the ROCm seam — same pp/tg/pg + depth methodology as the Vulkan/Metal/CPU arms,
    // directly comparable to `llama-bench` on a HIP build. Device index parsed from the spec.
    if let Backend::Rocm(rocm_spec) = &backend {
        let dev_idx = rocm_spec.as_deref().map(parse_rocm_device).unwrap_or(0);
        let model = infr_llama::SeamModel::load_with(&gguf, tok.as_deref(), cfg.clone())?;
        let samples = model.bench_rocm(n_prompt, n_gen, depth, pg, reps, dev_idx)?;
        let label = if let Some((p, g)) = pg {
            format!("pg{p}+{g}")
        } else if n_gen > 0 {
            format!("tg{n_gen}")
        } else {
            format!("pp{n_prompt}")
        };
        print_bench_avg(&samples, &label, depth, "", reps, json);
        return Ok(());
    }
    // `--dev VulkanN` is `device.dev` on this `cfg`, which the seam hands to
    // `VulkanBackend::new_with` when it picks the physical device; the prefill chunk
    // (`-u`/`device.ubatch`) rides the same config. Nothing to set here — straight to the seam.
    let model = infr_llama::SeamModel::load_with(&gguf, tok.as_deref(), cfg.clone())?;
    let samples = model.bench_vulkan(n_prompt, n_gen, depth, pg, reps)?;
    let label = if let Some((p, g)) = pg {
        format!("pg{p}+{g}")
    } else if n_gen > 0 {
        format!("tg{n_gen}")
    } else {
        format!("pp{n_prompt}")
    };
    // MTP arm (issue #33, phase 4 — perf-bottleneck visibility): a model that ships an MTP head
    // gets its self-speculative decode measured ADDITIONALLY alongside the baseline tg above (not
    // instead — `36 t/s vs 100 t/s baseline` is the whole point of this pass being visible in one
    // run). Only when there's a decode length to measure (`-n > 0`) and no `--pg` (that's a
    // combined ingest+reply shape the MTP arm doesn't have an equivalent for). Models without a
    // head take the exact path above unchanged — `mtp` stays `None`, `print_bench_avg_mtp` falls
    // straight through to the pre-existing `print_bench_avg` line.
    let mtp = if infr_llama::mtp::mtp_enabled()
        && pg.is_none()
        && n_gen > 0
        && model.config().n_layer_nextn > 0
    {
        Some(bench_mtp_tg(&model, n_prompt, depth, n_gen, reps, cfg)?)
    } else {
        None
    };
    print_bench_avg_mtp(&samples, &label, depth, "", reps, json, mtp.as_ref());
    Ok(())
}

/// The MTP-spec-decode measurement `infr bench` reports alongside the baseline tg rate (issue #33,
/// phase 4) — for models whose GGUF ships an MTP head. Runs
/// [`infr_llama::mtp::generate_mtp_spec_vulkan_timed`] on a synthetic text prompt sized to match
/// `-p`/`-d` (the same "repeat a fixed sentence to ~N tokens" convention used below —
/// the MTP driver takes a rendered PROMPT, not raw token ids, unlike `bench_vulkan`'s dummy-id
/// arm), once per rep (no persistent MTP session yet — `docs/mtp.md`'s Phase 3 doc on
/// `generate_mtp_spec_vulkan`'s per-call fresh trunk+head — so each rep re-pays the full weight
/// upload; keep `-r` small for this arm), and aggregates the per-cycle draft/verify/catchup wall
/// time into phase shares + the accept rate (alpha) via [`infr_llama::mtp::MtpTiming`].
struct MtpBenchStats {
    n_gen: usize,
    ts: f64,
    alpha: f64,
    draft_pct: f64,
    verify_pct: f64,
    catchup_pct: f64,
}

/// The one sentence (~10 tokens/rep) every MTP measurement decodes from, on BOTH engines
/// ([`bench_mtp_tg`] for infr, `ModelBench::MTP_PROMPT` → llama-cli `-no-cnv` for the oracle) —
/// α is content-sensitive, so cross-engine mtp ratios are only meaningful on shared
/// un-templated content.
const MTP_SENTENCE: &str = "The quick brown fox jumps over the lazy dog. ";

fn bench_mtp_tg(
    model: &infr_llama::SeamModel,
    n_prompt: usize,
    depth: usize,
    n_gen: usize,
    reps: usize,
    cfg: &Config,
) -> anyhow::Result<MtpBenchStats> {
    let head = infr_llama::mtp::load_mtp_head(model.gguf(), model.config())?;
    let want = (n_prompt + depth).max(1);
    let prompt = MTP_SENTENCE.repeat(want.div_ceil(10));
    let mut samples = Vec::with_capacity(reps.max(1));
    let mut timing = infr_llama::mtp::MtpTiming::default();
    // Backend selection mirrors the rest of the CLI: a Metal pick (`--dev metal` / `INFR_DEV=metal`)
    // routes the MTP driver onto the Apple-GPU trunk+head (issue #39 — measuring Metal's accept rate
    // needs the Metal timed path, not Vulkan's), everything else stays on Vulkan. Timed fns share a
    // signature.
    let use_metal = matches!(selected_backend(cfg)?, Backend::Metal);
    for _ in 0..reps.max(1) {
        let (stats, t) = if use_metal {
            #[cfg(target_os = "macos")]
            {
                infr_llama::mtp::generate_mtp_spec_metal_timed(
                    model,
                    &head,
                    &prompt,
                    n_gen,
                    |_| {},
                )?
            }
            #[cfg(not(target_os = "macos"))]
            {
                anyhow::bail!("the Metal MTP bench (INFR_DEV=metal) requires macOS");
            }
        } else {
            infr_llama::mtp::generate_mtp_spec_vulkan_timed(model, &head, &prompt, n_gen, |_| {})?
        };
        samples.push(stats.n_gen as f64 / stats.decode_secs.max(1e-9));
        timing.add(&t);
    }
    let ts = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
    let (draft_pct, verify_pct, catchup_pct) = timing.phase_shares();
    Ok(MtpBenchStats {
        n_gen,
        ts,
        alpha: timing.alpha(),
        draft_pct,
        verify_pct,
        catchup_pct,
    })
}

/// [`print_bench_avg`] plus the optional MTP segment: `mtp` is `None` for every model without a
/// head (or `-pg`/`-n 0` bench calls), in which case this is BYTE-IDENTICAL to the pre-MTP output
/// (falls straight through to `print_bench_avg`) — the "no new flags, unchanged output" guarantee
/// for non-MTP models.
fn print_bench_avg_mtp(
    samples: &[f64],
    label: &str,
    depth: usize,
    // Backend tag passed through to the no-MTP fallback (" [metal]" / " [cpu]" / "") — PR #42
    // routed cmd_bench_metal here and the hardcoded "" silently dropped its " [metal]" tag.
    suffix: &str,
    reps: usize,
    json: bool,
    mtp: Option<&MtpBenchStats>,
) {
    let Some(m) = mtp else {
        print_bench_avg(samples, label, depth, suffix, reps, json);
        return;
    };
    let avg = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
    let ratio = m.ts / avg.max(1e-9);
    if json {
        println!(
            "[{{\"avg_ts\": {avg:.2}, \"mtp_ts\": {:.2}, \"mtp_ratio\": {ratio:.4}, \"alpha\": {:.4}, \"draft_pct\": {:.1}, \"verify_pct\": {:.1}, \"catchup_pct\": {:.1}}}]",
            m.ts, m.alpha, m.draft_pct, m.verify_pct, m.catchup_pct
        );
        return;
    }
    let d = if depth > 0 {
        format!(" @ d{depth}")
    } else {
        String::new()
    };
    println!(
        "{label}{d}: {avg:.1} t/s | mtp{}: {:.1} t/s ({ratio:.2}x, alpha={:.2}, draft {:.0}% verify {:.0}% catchup {:.0}%)  ({reps} reps)",
        m.n_gen, m.ts, m.alpha, m.draft_pct, m.verify_pct, m.catchup_pct
    );
}

/// Shared bench-result reporter: average the per-rep t/s samples and print either the JSON shape
/// (`[{"avg_ts": ..}]`, llama-bench-comparable) or `label[ @ dN][tag]: X t/s (N reps)`. One
/// implementation for the dense-CPU / qwen35 bench tails (they previously each had a copy).
fn print_bench_avg(samples: &[f64], label: &str, depth: usize, tag: &str, reps: usize, json: bool) {
    let avg = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
    if json {
        println!("[{{\"avg_ts\": {avg:.2}}}]");
    } else {
        let d = if depth > 0 {
            format!(" @ d{depth}")
        } else {
            String::new()
        };
        println!("{label}{d}{tag}: {avg:.1} t/s  ({reps} reps)");
    }
}

#[cfg(any(target_os = "macos", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetalBenchMetric {
    Prompt,
    Decode,
    Turn,
}

#[cfg(any(target_os = "macos", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MetalBenchShape {
    max_ctx: usize,
    depth_warm_prompt: Option<usize>,
    measured_prompt: usize,
    measured_gen: usize,
    tokens: usize,
    metric: MetalBenchMetric,
}

#[cfg(any(target_os = "macos", test))]
fn metal_bench_shape(
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<(usize, usize)>,
) -> MetalBenchShape {
    let (p_eff, g_eff) = pg.unwrap_or((n_prompt, n_gen));
    let depth_warm_prompt = (depth > 0).then_some(depth + 1);
    let (measured_prompt, measured_gen, tokens, metric) = if let Some((p, g)) = pg {
        (depth + p, g, p + g, MetalBenchMetric::Turn)
    } else if n_gen > 0 {
        (depth + 1, n_gen, n_gen, MetalBenchMetric::Decode)
    } else {
        // The runner batches all but the final prompt token. Include that frontier token so the
        // timed graph contains exactly `n_prompt` rows.
        (depth + n_prompt + 1, 0, n_prompt, MetalBenchMetric::Prompt)
    };
    MetalBenchShape {
        max_ctx: depth + p_eff.max(1) + g_eff + 16,
        depth_warm_prompt,
        measured_prompt,
        measured_gen,
        tokens,
        metric,
    }
}

/// Metal twin of [`cmd_bench_cpu`]: same pp/tg/pg + depth methodology on the Apple-GPU seam
/// backend (`SeamModel::bench_metal`). On non-macOS this arm is unreachable (the backend crate
/// compiles to nothing), so the whole body is cfg-gated.
#[allow(clippy::too_many_arguments)]
fn cmd_bench_metal(
    gguf: &Path,
    tok: Option<&Path>,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<(usize, usize)>,
    reps: usize,
    json: bool,
    cfg: &Arc<Config>,
) -> anyhow::Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (gguf, tok, n_prompt, n_gen, depth, pg, reps, json, cfg);
        anyhow::bail!("metal bench requires macOS");
    }
    #[cfg(target_os = "macos")]
    {
        let model = infr_llama::SeamModel::load_with(gguf, tok, cfg.clone())?;
        let shape = metal_bench_shape(n_prompt, n_gen, depth, pg);
        // ONE session for warmup + every rep: backend, uploaded weights, compiled pipelines and
        // the dequant/repack weight caches persist (each rep still measures a full prefill —
        // the depth warm resets the materialized tokens). A fresh backend per rep re-paid those
        // one-time costs inside the measurement.
        let mut sess = model.metal_session(shape.max_ctx)?;
        // One untimed warmup: page-cache the mmap + build the weight caches + compile pipelines.
        let _ = model.bench_metal(&mut sess, 8, 2, true, false);
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let reset_measured = if let Some(warm_prompt) = shape.depth_warm_prompt {
                let _ = model.bench_metal(&mut sess, warm_prompt, 0, true, false)?;
                false
            } else {
                true
            };
            let s = model.bench_metal(
                &mut sess,
                shape.measured_prompt,
                shape.measured_gen,
                reset_measured,
                true,
            )?;
            let secs = match shape.metric {
                MetalBenchMetric::Prompt => s.prompt_secs,
                MetalBenchMetric::Decode => s.decode_secs,
                MetalBenchMetric::Turn => s.prompt_secs + s.decode_secs,
            };
            let ts = shape.tokens as f64 / secs.max(1e-9);
            samples.push(ts);
        }
        let label = if let Some((p, g)) = pg {
            format!("pg{p}+{g}")
        } else if shape.metric == MetalBenchMetric::Decode {
            format!("tg{n_gen}")
        } else {
            format!("pp{n_prompt}")
        };
        // MTP arm (issue #33/#39): twin of the Vulkan path's arm — a model shipping an MTP head
        // gets its self-speculative Metal decode (accept rate + phase split) measured alongside the
        // baseline tg above. `bench_mtp_tg` reads the resolved backend (Metal here, via INFR_DEV) to
        // route onto the Metal timed driver. Models without a head keep `mtp = None` → byte-identical.
        let mtp = if infr_llama::mtp::mtp_enabled()
            && pg.is_none()
            && n_gen > 0
            && model.config().n_layer_nextn > 0
        {
            Some(bench_mtp_tg(&model, n_prompt, depth, n_gen, reps, cfg)?)
        } else {
            None
        };
        print_bench_avg_mtp(
            &samples,
            &label,
            depth,
            " [metal]",
            reps,
            json,
            mtp.as_ref(),
        );
        Ok(())
    }
}

/// CPU-backend bench (`infr bench -ngl 0`): the GPU bench's pp/tg/pg metrics on the agnostic CPU
/// reference path, using `SeamModel`'s token-level timing — directly comparable to `llama-bench -ngl 0`.
#[allow(clippy::too_many_arguments)]
fn cmd_bench_cpu(
    gguf: &Path,
    tok: Option<&Path>,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<(usize, usize)>,
    reps: usize,
    json: bool,
    cfg: &Arc<Config>,
) -> anyhow::Result<()> {
    let model = infr_llama::SeamModel::load_with(gguf, tok, cfg.clone())?;
    let measure_tg = pg.is_none() && n_gen > 0;
    // One untimed warmup (page-cache the mmap'd weights) before the timed reps.
    let _ = model.bench(depth.max(1), if measure_tg || pg.is_some() { 1 } else { 0 });
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let ts = if let Some((p, g)) = pg {
            let s = model.bench(p, g)?; // pg: prefill p + decode g, throughput (p+g)/total
            (p + g) as f64 / (s.prompt_secs + s.decode_secs)
        } else if measure_tg {
            let s = model.bench(depth.max(1), n_gen)?; // tg@depth: decode at `depth` context
            n_gen as f64 / s.decode_secs
        } else {
            let s = model.bench(n_prompt, 0)?; // pp: prefill only
            n_prompt as f64 / s.prompt_secs
        };
        samples.push(ts);
    }
    let label = if let Some((p, g)) = pg {
        format!("pg{p}+{g}")
    } else if measure_tg {
        format!("tg{n_gen}")
    } else {
        format!("pp{n_prompt}")
    };
    print_bench_avg(&samples, &label, depth, " [cpu]", reps, json);
    Ok(())
}

/// Aggregated result of one diffusion-gemma decode measurement (Phase 4/E,
/// `docs/diffusion-gemma.md`): the same numbers `cmd_bench_diffusion_gemma` prints, factored out of
/// [`dg_bench_run`] so the compare arm (`ModelBench::dg_infr`) can read them directly instead of
/// scraping this command's own stdout. `pp_ts`/`gen_ts`/`parallel_ts`/`steps` are already averaged
/// over `reps`; `last_np`/`last_ng` are the LAST rep's actual prompt/gen token counts (matches
/// every other bench arm's "label from the last rep" convention).
struct DgBenchResult {
    pp_ts: f64,
    /// n_gen / decode_secs — end-to-end, THIS run's own step count. NOT apples-to-apples across
    /// implementations (entropy-bound step counts are content/impl-sensitive) — see the compare
    /// arm's dg-step/dg-e2e comment for why `parallel_ts` is the metric that ratio is built on.
    gen_ts: f64,
    /// canvas_length * steps / decode_secs — the in-step-parallel rate: the number that reflects
    /// actual forward-pass work regardless of how many steps the entropy-bound sampler took.
    /// Directly comparable to llama.cpp's own `llama-diffusion-cli` "in-step parallel N tok/s".
    parallel_ts: f64,
    steps: f64,
    last_np: usize,
    last_ng: usize,
}

/// Core diffusion-gemma decode loop (Phase 4, `docs/diffusion-gemma.md`): drives
/// `crate::diffusion::diffusion_generate` directly over a persistent
/// `DiffusionGemmaCpuSession`/`DiffusionGemmaVulkanSession` — the SAME primitive
/// `DiffusionGemmaChat::generate` (run/serve) uses — rather than going through the generic
/// `ChatModel::generate`, because bench needs the step/block counts `GenStats` alone doesn't carry.
/// Used by BOTH `cmd_bench_diffusion_gemma` (prints the result) and `ModelBench::dg_infr` (Phase
/// 4/E, the `compare`/`compare --sweep` DG arm — reads the result directly instead of shelling out
/// to `infr bench` and parsing its printed text, which is what every OTHER arch's compare arm does
/// via `infr bench --json`; DG's bench has no `--json` support for the per-step fields).
///
/// `-p P`/`-d D` behave like every other bench arm: `D` extra prompt tokens are prefilled UNTIMED
/// first, then `P` more (an exact-prefix extension of the same real-text token sequence — see the
/// in-body comment on why NOT the AR arms' `i % 100` dummy ids) through the SAME session — its
/// reuse forwards only the new `P` suffix, so `prompt_secs` times exactly that, matching the
/// `-d`/`-p` split every other arch's bench reports. `-n N` then times the block-diffusion decode of
/// N tokens (`ceil(N / canvas_length)` whole canvas blocks — a `-n 64` request still denoises a
/// full `canvas_length`-token canvas per step, so besides the naive "committed tokens / decode
/// secs" rate this also reports the oracle's own "in-step parallel" rate
/// (`canvas_length * steps / decode_secs`, see the reference runs captured in
/// `docs/diffusion-gemma.md`) — the number that actually reflects how much forward-pass work ran.
/// `-n 0` measures prefill only (pp, like every other arch's `-n 0`).
///
/// `--pg` has no diffusion-shaped meaning (a denoise block isn't an ingest-then-reply AR turn) —
/// `cmd_bench_diffusion_gemma` errors clearly on it rather than silently mis-measuring; this
/// function itself has no `pg` parameter at all. `-t`/`-u` keep their generic meaning (rayon
/// threads for the entropy-bound sampler's per-position reduction; the shared prefill chunk size —
/// DG's causal prefill rides the exact same `generate_dense_backend` chunked loop as every other
/// arch) with no DG-specific wiring needed.
#[allow(clippy::too_many_arguments)]
fn dg_bench_run(
    gguf: &Path,
    tok: Option<&Path>,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    reps: usize,
    // `Some(text)`: chat-template-render + encode THIS prompt instead of the synthetic repeated
    // sentence, and ignore `n_prompt` (requires depth == 0). The compare arm passes its shared
    // natural prompt so BOTH sides' entropy-bound loops converge on the same (templated) content —
    // with the synthetic prompt infr always rode the 48-step cap while the oracle converged in ~23
    // (steps are content-sensitive, so e2e was double-penalized by prompt choice alone). See the
    // `prompt_ids` binding below for why this is templated, not raw-encoded.
    prompt_text: Option<&str>,
    cpu: bool,
    // Phase D: the Metal DG session (macOS only — see the `one_rep!` dispatch below). `cpu` wins
    // if both are set (matches `cmd_run`/`cmd_serve`'s precedence).
    metal: bool,
    ecfg: &Arc<Config>,
) -> anyhow::Result<DgBenchResult> {
    use infr_llama::diffusion::{diffusion_generate, EbConfig};
    let model = infr_llama::SeamModel::load_with(gguf, tok, ecfg.clone())?;
    let cfg = model.config();
    let canvas_len = cfg.canvas_length.max(1);
    let vocab = cfg.vocab;
    let eb = EbConfig::from_config(cfg);
    // `sampling.ignore_eos` (INFR_IGNORE_EOS — `cmd_bench` always pins it through the CLI flag
    // layer, see `cli_flag_layer`): a fixed generation budget should never early-stop on an EOS
    // id, matching every AR bench arm's semantics for the same knob.
    let eos_ids: Vec<u32> = if ecfg.sampling.ignore_eos {
        Vec::new()
    } else {
        cfg.eos_ids.clone()
    };
    // `sampling.seed`; `42` is THIS path's fallback (the knob has two defaults in the tree and
    // both stay at their sites — §6.12).
    let seed: u64 = ecfg.sampling.seed.unwrap_or(42);

    if prompt_text.is_some() && depth > 0 {
        anyhow::bail!("dg_bench_run: prompt_text override is depth-0 only (compare's DG arm)");
    }
    // Real-prompt mode (compare's DG arm): the reference fork's OWN CLI applies the model's chat
    // template by default before it ever decodes `-p`'s text (`common_params::enable_chat_template`
    // defaults `true`; `diffusion-cli.cpp`'s `apply_template`/`common_chat_templates_apply`, called
    // unless `--no-chat-template` is passed — which the compare arm's `llama_diffusion` invocation
    // does NOT pass). So the oracle's ~23-step convergence is measured on a rendered
    // `<start_of_turn>user ... <start_of_turn>model` turn, never on the bare sentence. Encoding the
    // bare sentence here (as this used to) fed our side a badly out-of-distribution prompt for an
    // instruction-tuned model: entropy never settled — `INFR_EB_TRACE=1` showed it oscillating
    // (not monotonically falling) with acceptance stuck near 1/256 for dozens of steps — so infr
    // rode the full step cap on EVERY block while the fork, decoding the SAME text but templated,
    // converged normally. Render the identical single-user-turn template `cmd_run`/
    // `DiffusionGemmaChat` already use for a real chat reply, so both tools decode the model's
    // actual instruction-tuned input shape instead of two different prompts that merely share text.
    let prompt_ids = prompt_text
        .map(|text| model.encode(&model.render_chat_messages(&[("user", text)])?))
        .transpose()?;
    let p_eff = match &prompt_ids {
        Some(ids) => ids.len().max(1),
        None => n_prompt.max(1), // DG always needs >=1 prompt token (the canvas's causal prefix)
    };
    let blocks_wanted = if n_gen > 0 {
        n_gen.div_ceil(canvas_len).max(1)
    } else {
        1
    };
    let max_ctx = depth + p_eff + blocks_wanted * canvas_len + 64;

    // Unlike the AR bench arms' `i % 100` dummy ids (matmul timing there truly IS
    // content-independent), DG's entropy-bound sampler converges (steps run, trim point) on the
    // ACTUAL logits — an out-of-distribution raw-id prompt made the model collapse the whole
    // canvas to one repeated token in 2 steps (`trim_canvas` then cut it to 0 committed tokens),
    // which measures a degenerate path instead of the shape `-n` asks for. Encode a real repeated
    // sentence instead (same "fixed synthetic prompt" convention `bench_mtp_tg` uses) and slice
    // it: `dummy(depth)` is then an exact prefix of `dummy(depth + p_eff)` by construction, so the
    // untimed depth warm + timed suffix prefill still gets the session's prefix-diff reuse.
    let full_ids = match prompt_ids {
        Some(ids) => ids,
        None => {
            let long_text =
                "The quick brown fox jumps over the lazy dog. ".repeat((depth + p_eff) / 4 + 8);
            model.encode(&long_text)?
        }
    };
    if full_ids.len() < depth + p_eff {
        anyhow::bail!(
            "internal: synthetic bench prompt too short ({} ids for depth {depth} + p {p_eff})",
            full_ids.len()
        );
    }
    let dummy = |n: usize| -> Vec<u32> { full_ids[..n.max(1)].to_vec() };

    let mut pps = Vec::with_capacity(reps.max(1));
    let mut gens = Vec::with_capacity(reps.max(1));
    let mut steps_v = Vec::with_capacity(reps.max(1));
    let mut parallel_v = Vec::with_capacity(reps.max(1));
    let (mut last_np, mut last_ng) = (0usize, 0usize);
    // Untimed pipeline-warmup tokens: NOT a prefix of the timed sequence (reversed), so the
    // timed prefill still re-prefills from scratch after the prefix-diff — the warmup only
    // pre-compiles the lazily-built pipelines, exactly like the AR bench arms' untimed warmup
    // (a cold DG prefill measured 26 t/s vs 1424 t/s warm — all one-time compile).
    let warm_ids: Vec<u32> = full_ids[..8.min(full_ids.len())]
        .iter()
        .rev()
        .copied()
        .collect();

    for _ in 0..reps.max(1) {
        macro_rules! one_rep {
            ($sess:expr) => {{
                let mut sess = $sess;
                sess.prefill(&model, &warm_ids)?; // untimed pipeline warmup (see warm_ids)
                if depth > 0 {
                    sess.prefill(&model, &dummy(depth))?; // untimed depth warm
                }
                if n_gen == 0 {
                    // pp-only: no canvas denoise, just the timed prefill (matches every other
                    // arch's `-n 0` meaning).
                    let t0 = std::time::Instant::now();
                    sess.prefill(&model, &dummy(depth + p_eff))?;
                    let secs = t0.elapsed().as_secs_f64();
                    pps.push(p_eff as f64 / secs.max(1e-9));
                    last_np = p_eff;
                } else {
                    let result = diffusion_generate(
                        &mut sess,
                        &model,
                        &dummy(depth + p_eff),
                        canvas_len,
                        vocab,
                        &eos_ids,
                        &eb,
                        n_gen,
                        seed,
                        max_ctx,
                        None,
                        None,
                    )?;
                    pps.push(p_eff as f64 / result.stats.prompt_secs.max(1e-9));
                    gens.push(result.stats.n_gen as f64 / result.stats.decode_secs.max(1e-9));
                    parallel_v.push(
                        (canvas_len * result.steps) as f64 / result.stats.decode_secs.max(1e-9),
                    );
                    steps_v.push(result.steps as f64);
                    last_np = p_eff;
                    last_ng = result.stats.n_gen;
                }
            }};
        }
        if cpu {
            one_rep!(model.diffusion_gemma_cpu_session(max_ctx));
        } else if metal {
            // Phase D: Metal DG session, macOS only — the non-macOS build still compiles this
            // arm (errors clearly at runtime instead), matching every other INFR_METAL arm.
            #[cfg(target_os = "macos")]
            {
                one_rep!(model.diffusion_gemma_metal_session(max_ctx)?);
            }
            #[cfg(not(target_os = "macos"))]
            {
                anyhow::bail!("the Metal backend is only available on macOS");
            }
        } else {
            one_rep!(model.diffusion_gemma_vulkan_session(max_ctx)?);
        }
    }

    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    Ok(DgBenchResult {
        pp_ts: avg(&pps),
        gen_ts: avg(&gens),
        parallel_ts: avg(&parallel_v),
        steps: avg(&steps_v),
        last_np,
        last_ng,
    })
}

/// diffusion-gemma bench (Phase 4, `docs/diffusion-gemma.md`): llama-bench has no diffusion mode
/// (no llama.cpp comparison possible via `llama-bench` — this is infr-only reporting; `infr
/// compare`'s DG arm instead shells `llama-diffusion-cli` from the fork directly, see
/// `ModelBench::llama_diffusion`), so this drives [`dg_bench_run`] and formats its result. See
/// `dg_bench_run`'s doc comment for the -p/-n/-d semantics.
#[allow(clippy::too_many_arguments)]
fn cmd_bench_diffusion_gemma(
    gguf: &Path,
    tok: Option<&Path>,
    n_prompt: usize,
    n_gen: usize,
    depth: usize,
    pg: Option<(usize, usize)>,
    reps: usize,
    cpu: bool,
    metal: bool,
    json: bool,
    cfg: &Arc<Config>,
) -> anyhow::Result<()> {
    if pg.is_some() {
        anyhow::bail!(
            "diffusion-gemma bench has no --pg equivalent (a denoise block isn't an ingest-then-reply \
             AR turn); use separate -p/-n instead — `infr bench <model> -p P -n N`"
        );
    }
    let r = dg_bench_run(
        gguf, tok, n_prompt, n_gen, depth, reps, None, cpu, metal, cfg,
    )?;
    let tag = if cpu {
        " [cpu]"
    } else if metal {
        " [metal]"
    } else {
        ""
    };
    if json {
        let a = if n_gen > 0 { r.gen_ts } else { r.pp_ts };
        println!("[{{\"avg_ts\": {a:.2}}}]");
    } else if n_gen > 0 {
        println!(
            "pp{}: {:.1} t/s | gen{}: {:.1} t/s (end-to-end) | {:.1} steps | in-step parallel {:.1} t/s{tag}  ({reps} reps)",
            r.last_np, r.pp_ts, r.last_ng, r.gen_ts, r.steps, r.parallel_ts,
        );
    } else {
        println!("pp{}: {:.1} t/s{tag}  ({reps} reps)", r.last_np, r.pp_ts);
    }
    Ok(())
}

/// The recurring "where are we behind llama.cpp" survey: for EVERY model given, measure
/// pp512 (prefill), tg128 (decode) and tg64 at `--sweep-depth` on both tools, print the matrix
/// as it fills (a sweep is long — partial results beat silence), and finish with the ratios
/// ranked worst-first so the next perf target is the top of the list.
#[allow(clippy::too_many_arguments)]
fn cmd_compare_sweep(
    models: &[String],
    depth: usize,
    dev: &str,
    ngl: usize,
    threads: usize,
    reps: usize,
    ubatch: usize,
    llama_bench: &str,
    cfg: &Arc<Config>,
) -> anyhow::Result<()> {
    const METRICS: [&str; 4] = ["pp512", "tg128", "tg64@d", "pp4@d"];
    println!(
        "\n{:<22} {:<10} | {:>9} | {:>9} | {:>10}",
        "model", "metric", "infr", "llama.cpp", "infr/llama"
    );
    println!("{:-<23}{:-<11}+{:-<11}+{:-<11}+{:-<12}", "", "", "", "", "");
    // (model, metric, infr, llama, ratio) for the ranked summary.
    let mut rows: Vec<(String, String, f64, f64)> = Vec::new();
    for model in models {
        let short = model.rsplit('/').next().unwrap_or(model);
        // Cooldown between models to reduce thermal skew across the sweep.
        std::thread::sleep(std::time::Duration::from_secs(10));
        let mb = match ModelBench::new(
            model,
            dev,
            ngl,
            threads,
            reps,
            ubatch,
            llama_bench,
            cfg.clone(),
        ) {
            Ok(mb) => mb,
            Err(e) => {
                println!("{short:<22} {:<10} | resolve failed: {e}", "-");
                continue;
            }
        };
        // diffusion-gemma (Phase 4/E): entirely different measurement shape (no upstream
        // llama-bench support — see `ModelBench::is_dg`'s doc comment), so it prints its own two
        // rows and skips the standard pp/tg/mtp metrics below entirely.
        if mb.is_dg {
            print_dg_sweep_rows(&mb, short, &mut rows);
            continue;
        }
        let ds = depth.to_string();
        // pp4@d: the tiny-suffix-turn shape (m=2..8 Linears at session depth) — the multi-row
        // GEMV / spec-verify path; invisible in pp512/tg but it IS multi-turn serve TTFT.
        let runs: [(&str, Vec<&str>, (usize, usize)); 4] = [
            ("pp512", vec!["-p", "512", "-n", "0"], (512, 0)),
            ("tg128", vec!["-p", "0", "-n", "128"], (0, 128)),
            (METRICS[2], vec!["-p", "0", "-n", "64", "-d", &ds], (0, 64)),
            (METRICS[3], vec!["-p", "4", "-n", "0", "-d", &ds], (4, 0)),
        ];
        for (metric, args, (np, ng)) in runs {
            let metric_label = if metric.ends_with("@d") {
                format!("{metric}{depth}") // "tg64@d" -> "tg64@d4096"
            } else {
                metric.to_string()
            };
            // Cooldown between metric pairs (infr + llama.cpp) to reduce thermal coupling.
            std::thread::sleep(std::time::Duration::from_secs(10));
            // A cold/low-power GPU can flub Vulkan device init on the first subprocess launch
            // after a while-idle; one retry absorbs that without forcing a manual re-run of the
            // whole sweep for one bad cell.
            let mut iv = mb.infr(&args);
            if let Err(e) = &iv {
                eprintln!("infr bench failed ({short} {metric_label}), retrying once: {e:#}");
                std::thread::sleep(std::time::Duration::from_secs(3));
                iv = mb.infr(&args);
            }
            let mut lv = mb.llama(np, ng, &args);
            if lv.is_none() {
                eprintln!("llama-bench failed ({short} {metric_label}), retrying once");
                std::thread::sleep(std::time::Duration::from_secs(3));
                lv = mb.llama(np, ng, &args);
            }
            let is = iv.as_ref().map(|v| format!("{v:.0}")).unwrap_or_else(|e| {
                eprintln!("infr bench failed ({short} {metric_label}): {e:#}");
                "ERR".into()
            });
            let ls = lv.map(|v| format!("{v:.0}")).unwrap_or_else(|| "NA".into());
            let ratio = match (iv.as_ref().ok(), lv) {
                (Some(&i), Some(l)) if l > 0.0 => {
                    rows.push((short.to_string(), metric_label.clone(), i, l));
                    format!("{:.2}x", i / l)
                }
                _ => "-".into(),
            };
            println!("{short:<22} {metric_label:<10} | {is:>9} | {ls:>9} | {ratio:>10}");
        }
        // mtp column (issue #33, phase 4): infr-vs-llama at the SAME tg128 shape above, but with
        // both tools' MTP spec decode on — MTP-capable models only (the cell costs one llama-cli
        // run + one infr mtp run; every other model prints a blank row instantly, keeping the
        // sweep's runtime dominated by the 4 metrics above, not this column).
        let metric_label = "mtp128".to_string();
        // MTP is PARKED (`infr_llama::mtp::mtp_enabled`'s doc) — the column stays in the table shape
        // (a blank row, same as any model without a head) so the sweep archive's columns keep lining
        // up with older archives, but neither engine is measured.
        if infr_llama::mtp::mtp_enabled() && mb.has_mtp_head() {
            let iv = mb.infr_mtp(&["-p", "0", "-n", "128"]);
            let lv = mb.llama_cli_mtp(128);
            let is = iv.as_ref().map(|v| format!("{v:.0}")).unwrap_or_else(|e| {
                eprintln!("infr mtp bench failed ({short}): {e:#}");
                "ERR".into()
            });
            let ls = lv.map(|v| format!("{v:.0}")).unwrap_or_else(|| "NA".into());
            let ratio = match (iv.as_ref().ok(), lv) {
                (Some(&i), Some(l)) if l > 0.0 => {
                    rows.push((short.to_string(), metric_label.clone(), i, l));
                    format!("{:.2}x", i / l)
                }
                _ => "-".into(),
            };
            println!("{short:<22} {metric_label:<10} | {is:>9} | {ls:>9} | {ratio:>10}");
        } else {
            println!(
                "{short:<22} {metric_label:<10} | {:>9} | {:>9} | {:>10}",
                "-", "-", "-"
            );
        }
    }
    // Worst-first: the top of this list is the next perf target. NaN-safe (`total_cmp`) so a single
    // malformed subprocess-JSON value (→ a NaN ratio) sorts to the end instead of panicking the
    // `partial_cmp().unwrap()` and discarding the whole sweep's results.
    rows.sort_by(|a, b| nan_safe_ratio_cmp(a.2 / a.3, b.2 / b.3));
    println!("\nBIGGEST GAPS (infr/llama, worst first):");
    for (m, metric, i, l) in rows.iter().take(10) {
        println!("  {:>5.2}x  {m:<22} {metric:<10} ({i:.0} vs {l:.0})", i / l);
    }
    Ok(())
}

/// `llama-cli`'s MTP arm has no JSON output (unlike `llama-bench -o json`) — it prints
/// `[ Prompt: X t/s | Generation: Y t/s ]` at exit, so pull the LAST `Generation: <float> t/s`
/// occurrence out of its combined stdout+stderr by hand (no regex dependency for one float). The
/// LAST match matters: some llama.cpp builds print an interim perf line before the final summary.
fn parse_llama_cli_gen_rate(output: &str) -> Option<f64> {
    const NEEDLE: &str = "Generation:";
    let mut last = None;
    let mut rest = output;
    while let Some(idx) = rest.find(NEEDLE) {
        let after = rest[idx + NEEDLE.len()..].trim_start();
        let end = after
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(after.len());
        if let Ok(v) = after[..end].parse::<f64>() {
            last = Some(v);
        }
        rest = &rest[idx + NEEDLE.len()..];
    }
    last
}

fn compare_infr_dev(dev: &str) -> &str {
    if dev.eq_ignore_ascii_case("MTL0") {
        "metal"
    } else {
        dev
    }
}

/// Parse a ROCm device spec (`rocm0`, `rocm1`, …, or bare `rocm`) into a device index.
fn parse_rocm_device(spec: &str) -> u32 {
    let lower = spec.to_ascii_lowercase();
    lower
        .strip_prefix("rocm")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

/// One model's infr-vs-llama.cpp bench harness: resolves the shared model ref once and shells
/// out to `infr bench --json` / `llama-bench -o json` with MATCHED flags. Both the deep
/// coding-agent scenarios (`cmd_compare`) and the multi-model survey (`cmd_compare_sweep`) run
/// through this, so the two tools are always measured identically.
struct ModelBench {
    exe: PathBuf,
    model: String,
    /// The resolved local GGUF path — `llama-cli`'s MTP arm (issue #33, phase 4) shells the
    /// binary directly (no `llama-bench` JSON plumbing for spec decode: `llama-bench` doesn't run
    /// spec at all), so it needs a real `-m <path>` regardless of whether `model` was an `-hf` ref.
    gguf_path: PathBuf,
    /// Sidecar tokenizer.json beside the GGUF, if any — DG's compare arm (`dg_infr`) loads the
    /// model in-process (`dg_bench_run`) rather than shelling out, so it needs this the same way
    /// `cmd_bench`/`cmd_run` do (see `resolve`'s doc comment).
    tok_path: Option<PathBuf>,
    /// arch=diffusion-gemma (Phase 4/E, `docs/diffusion-gemma.md`): gates the DG rows in
    /// `cmd_compare`/`cmd_compare_sweep` — no upstream-merged `llama-bench` support exists for
    /// this arch, so it takes a completely different pair of measurement paths (`dg_infr` +
    /// `llama_diffusion`) instead of the standard `infr`/`llama` pp/tg arms.
    is_dg: bool,
    llama_model_args: Vec<String>,
    llama_bench: String,
    dev: String,
    ngl: usize,
    threads: usize,
    reps: usize,
    ubatch: usize,
    /// The resolved process configuration. Only the DG arm needs it — [`dg_infr`](Self::dg_infr)
    /// runs `dg_bench_run` IN-PROCESS (every other arm shells out to `infr bench`, which resolves
    /// its own), and since S8 that function takes `sampling.ignore_eos` / `sampling.seed` from a
    /// `Config` rather than from the environment.
    cfg: Arc<Config>,
}

impl ModelBench {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &str,
        dev: &str,
        ngl: usize,
        threads: usize,
        reps: usize,
        ubatch: usize,
        llama_bench: &str,
        cfg: Arc<Config>,
    ) -> anyhow::Result<Self> {
        let exe = std::env::current_exe().context("locating the infr binary")?;
        // infr and llama.cpp share the HF Hub cache and the same `org/repo:quant` ref grammar, so
        // hand BOTH tools the same reference: `infr bench` takes `model` verbatim, and llama-bench
        // gets the matching `-hf`/`--hf-file` (or `-m` for a local path). Pull once up front so
        // `--offline` holds.
        let (resolved, tok_path) = resolve(model)?;
        // diffusion-gemma (Phase 4/E, docs/diffusion-gemma.md): the arch's PR isn't merged into
        // mainline llama.cpp, so `llama-bench` can't run it — but the reference fork at
        // `~/Projects/mxaddict/llama.cpp-dg` builds `llama-diffusion-cli`, which IS a usable
        // oracle (see `ModelBench::llama_diffusion`/`llama_diffusion_cli_path`). `is_dg` routes
        // `cmd_compare`/`cmd_compare_sweep` to the DG-shaped measurement pair instead of bailing.
        let is_dg = infr_llama::diffusion::is_diffusion_gemma(&resolved);
        let llama_model_args: Vec<String> = match infr_hub::ModelRef::parse(model)? {
            infr_hub::ModelRef::Repo { repo, sel } => {
                let mut a = vec!["--offline".to_string()];
                match sel {
                    None => a.extend(["-hf".into(), repo]),
                    Some(s) if s.to_lowercase().ends_with(".gguf") => {
                        a.extend(["--hf-repo".into(), repo, "--hf-file".into(), s])
                    }
                    Some(s) => a.extend(["-hf".into(), format!("{repo}:{s}")]),
                }
                a
            }
            infr_hub::ModelRef::Path(_) => {
                vec!["-m".to_string(), resolved.to_string_lossy().into_owned()]
            }
        };
        Ok(Self {
            exe,
            model: model.to_string(),
            gguf_path: resolved,
            tok_path,
            is_dg,
            llama_model_args,
            llama_bench: llama_bench.to_string(),
            dev: dev.to_string(),
            ngl,
            threads,
            reps,
            ubatch,
            cfg,
        })
    }

    /// Whether this model's GGUF ships an MTP head (issue #33, phase 4) — gates the extra
    /// `mtp`/`MTP DECODE` measurements in `cmd_compare`/`cmd_compare_sweep` so non-MTP models pay
    /// nothing extra.
    fn has_mtp_head(&self) -> bool {
        infr_llama::mtp::has_mtp_head(&self.gguf_path)
    }

    /// Run `infr bench --json` and return the parsed row object (`[{"avg_ts": .., ..}]`'s first
    /// element) — shared by [`infr`](Self::infr) (`avg_ts`) and [`infr_mtp`](Self::infr_mtp)
    /// (`mtp_ts`), so both read one shelled-out call's worth of plumbing.
    fn infr_json(&self, args: &[&str]) -> anyhow::Result<serde_json::Value> {
        use std::process::Command;
        let mut c = Command::new(&self.exe);
        c.arg("bench")
            .arg(&self.model)
            .args(["-r", &self.reps.to_string()]);
        c.args([
            "--ngl",
            &self.ngl.to_string(),
            "--dev",
            compare_infr_dev(&self.dev),
        ]);
        if self.threads > 0 {
            c.args(["-t", &self.threads.to_string()]);
        }
        if self.ubatch > 0 {
            c.args(["-u", &self.ubatch.to_string()]);
        }
        c.args(args).arg("--json");
        let out = c.output().context("running `infr bench`")?;
        serde_json::from_slice(&out.stdout).with_context(|| {
            format!(
                "parsing infr bench output: {}",
                String::from_utf8_lossy(&out.stdout)
            )
        })
    }

    /// Run `infr bench` (this binary) and read its single-row [{"avg_ts":X}].
    fn infr(&self, args: &[&str]) -> anyhow::Result<f64> {
        self.infr_json(args)?[0]["avg_ts"]
            .as_f64()
            .context("infr bench: missing avg_ts")
    }

    /// Run `infr bench` and read the `mtp_ts` field `bench_mtp_tg`'s JSON output adds for
    /// MTP-capable models (issue #33, phase 4) — `args` should include `-n <N>` matching the
    /// baseline `infr` call this is compared against (the sweep's/compare's tg128 shape).
    fn infr_mtp(&self, args: &[&str]) -> anyhow::Result<f64> {
        self.infr_json(args)?[0]["mtp_ts"]
            .as_f64()
            .context("infr bench: missing mtp_ts (model has no MTP head, or -n was 0)")
    }

    /// Run `llama-bench -o json` and pick the row matching (n_prompt, n_gen): -pg adds extra rows.
    fn llama(&self, np: usize, ng: usize, args: &[&str]) -> Option<f64> {
        use std::process::Command;
        let mut c = Command::new(&self.llama_bench);
        c.args(&self.llama_model_args);
        c.args([
            "-ngl",
            &self.ngl.to_string(),
            "-dev",
            &self.dev,
            "-fa",
            "auto",
            "-r",
            &self.reps.to_string(),
            "-o",
            "json",
        ]);
        if self.threads > 0 {
            c.args(["-t", &self.threads.to_string()]);
        }
        if self.ubatch > 0 {
            c.args(["-ub", &self.ubatch.to_string()]);
        }
        c.args(args);
        let out = c.output().ok()?;
        let rows: serde_json::Value = match serde_json::from_slice(&out.stdout) {
            Ok(v) => v,
            Err(_) => {
                eprintln!(
                    "llama-bench failed (np={np} ng={ng}): {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                );
                return None;
            }
        };
        rows.as_array()?.iter().find_map(|r| {
            let p = r["n_prompt"].as_u64()? as usize;
            let g = r["n_gen"].as_u64()? as usize;
            (p == np && g == ng).then(|| r["avg_ts"].as_f64())?
        })
    }

    /// The fixed prompt both tools' MTP arm decodes from — the SAME raw sentence
    /// [`bench_mtp_tg`] feeds infr's side, passed to llama-cli with `-no-cnv` so neither side
    /// chat-templates it. MTP throughput is dominated by the accept rate α, and α is strongly
    /// content-sensitive (greedy drafting collapses on some regimes) — the two engines are only
    /// comparable decoding the SAME un-templated content. The previous asymmetry (infr: raw fox
    /// sentence, α≈0.51 on the 9B; llama-cli: chat-templated "capital of France", a friendlier
    /// regime) manufactured ~0.05-0.08 of phantom ratio gap in the sweep's mtp128 row.
    const MTP_PROMPT: &str = MTP_SENTENCE;

    /// `llama-cli` sits alongside `llama-bench` in the same install (`/usr/sbin` on this box) —
    /// derive its path from `--llama-bench` instead of adding a second CLI flag: replace the
    /// binary name when `llama_bench` names it explicitly, else assume it's on `PATH` like the
    /// `llama-bench` default is.
    fn llama_cli_path(&self) -> String {
        if self.llama_bench.contains("llama-bench") {
            self.llama_bench.replace("llama-bench", "llama-cli")
        } else {
            "llama-cli".to_string()
        }
    }

    /// Shell `llama-cli` directly with MTP spec decode on (`llama-bench` has no spec-decode mode
    /// at all — issue #33's phase 4 context) and parse the LAST `Generation: X t/s` line from its
    /// combined stdout+stderr (robust to which stream a given llama.cpp build writes the perf
    /// summary to). `-r` reps average, matching every other measurement this tool prints.
    fn llama_cli_mtp(&self, n_gen: usize) -> Option<f64> {
        use std::process::Command;
        let cli = self.llama_cli_path();
        let gguf = self.gguf_path.to_string_lossy().into_owned();
        let n = n_gen.to_string();
        let mut samples = Vec::with_capacity(self.reps.max(1));
        for _ in 0..self.reps.max(1) {
            let out = Command::new(&cli)
                .args(["-m", &gguf])
                .args(["-ngl", "99"])
                .args(["-p", Self::MTP_PROMPT])
                .args(["-n", &n])
                .args(["--temp", "0", "--single-turn", "-no-cnv"])
                .args(["--spec-type", "draft-mtp", "--spec-draft-n-max", "6"])
                .output()
                .ok()?;
            let combined = format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            match parse_llama_cli_gen_rate(&combined) {
                Some(v) => samples.push(v),
                None => {
                    eprintln!(
                        "llama-cli MTP run produced no parseable `Generation: X t/s` line: {}",
                        combined.trim()
                    );
                    return None;
                }
            }
        }
        Some(samples.iter().sum::<f64>() / samples.len().max(1) as f64)
    }

    /// infr's side of the DG compare arm (Phase 4/E): drives [`dg_bench_run`] — the SAME core
    /// `cmd_bench_diffusion_gemma` uses — in-process, at a fixed `-n <n_gen>` decode with a
    /// minimal 1-token prompt and no extra depth (this arm measures decode throughput only, not
    /// a coding-agent-shaped prefill scenario). `cpu` follows `ngl == 0`, matching every other
    /// arm's `-ngl 0` = CPU-reference-backend convention; Metal has no wiring here (`compare`
    /// doesn't expose a Metal device selector — out of scope for this arm).
    fn dg_infr(&self, n_gen: usize) -> anyhow::Result<DgBenchResult> {
        dg_bench_run(
            &self.gguf_path,
            self.tok_path.as_deref(),
            1, // ignored — the shared natural prompt below defines the prefix
            n_gen,
            0,
            self.reps,
            // The SAME prompt the fork oracle decodes from (`Self::DG_PROMPT`) — entropy-bound
            // step counts are content-sensitive, so e2e is only comparable on shared content.
            Some(Self::DG_PROMPT),
            self.ngl == 0,
            false,
            &self.cfg,
        )
    }

    /// Fixed prompt for [`llama_diffusion`](Self::llama_diffusion)'s oracle run — arbitrary but
    /// fixed content, matching `Self::MTP_PROMPT`'s convention of one shared literal both sides
    /// decode from (entropy-bound step counts are content-sensitive, see `dg_bench_run`'s comment
    /// on why `infr`'s own dummy prompt is a fixed real sentence rather than raw ids).
    const DG_PROMPT: &str = "Write a short story about a robot learning to paint.";

    /// The reference fork's own build directories — `~/Projects/mxaddict/llama.cpp-dg`, the only
    /// place `arch=diffusion-gemma` support actually exists (it isn't merged into mainline
    /// llama.cpp): `build-vulkan` (GPU, `cmake -DGGML_VULKAN=ON`) or `build` (CPU-only). `cpu`
    /// picks between them, mirroring `-ngl 0` vs `-ngl >0` on every other arm. Used both as
    /// [`llama_diffusion_cli_path`](Self::llama_diffusion_cli_path)'s last-resort candidate AND as
    /// [`llama_diffusion`](Self::llama_diffusion)'s fallback target (see its doc comment).
    fn fork_diffusion_cli_path(cpu: bool) -> PathBuf {
        let fork = PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("Projects/mxaddict/llama.cpp-dg");
        if cpu {
            fork.join("build/bin/llama-diffusion-cli")
        } else {
            fork.join("build-vulkan/bin/llama-diffusion-cli")
        }
    }

    /// Resolve `llama-diffusion-cli`'s binary path (Phase 4/E compare arm) — precedence, in order:
    ///   1. `INFR_LLAMA_DIFFUSION_CLI` env var (explicit override, e.g. a custom build location).
    ///   2. `llama-diffusion-cli` on `PATH` (a manual PATH walk — no extra `which` dependency for
    ///      one binary lookup).
    ///   3. [`fork_diffusion_cli_path`](Self::fork_diffusion_cli_path) — the reference fork's build.
    ///
    /// CAVEAT: mainline llama.cpp already ships a generic `llama-diffusion-cli` (LLaDA/Dream
    /// support), so tier 2 can resolve to a REAL binary that nonetheless doesn't know
    /// `arch=diffusion-gemma` (the fork's own unmerged addition) — it loads and errors "unknown
    /// model architecture" rather than failing to resolve at all, so this function alone can't
    /// detect the mismatch. [`llama_diffusion`](Self::llama_diffusion) is the one that actually
    /// runs the binary, so it catches that specific failure and falls through to tier 3 itself.
    fn llama_diffusion_cli_path(cpu: bool) -> PathBuf {
        if let Ok(p) = std::env::var("INFR_LLAMA_DIFFUSION_CLI") {
            return PathBuf::from(p);
        }
        if let Ok(path_var) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path_var) {
                let cand = dir.join("llama-diffusion-cli");
                if cand.is_file() {
                    return cand;
                }
            }
        }
        Self::fork_diffusion_cli_path(cpu)
    }

    /// Shell `llama-diffusion-cli` (see [`llama_diffusion_cli_path`](Self::llama_diffusion_cli_path)
    /// for the resolution precedence) at a fixed `-n <n_gen>` decode and parse its own
    /// throughput/in-step-parallel/step-count numbers straight out of combined stdout+stderr (this
    /// binary has no `-o json` mode). `-ngl 999 -dev <dev>` for the GPU compare, `-ngl 0` for CPU —
    /// mirrors [`Self::llama`]'s handling.
    ///
    /// FALLBACK: if the resolved binary loads but rejects the model with "unknown model
    /// architecture" (the tier-2 PATH caveat documented on `llama_diffusion_cli_path` — a mainline
    /// install's generic diffusion CLI lacking this arch), retry against
    /// [`fork_diffusion_cli_path`](Self::fork_diffusion_cli_path) directly instead of reporting a
    /// hard failure. Detected once per call and then stuck to for the remaining reps.
    fn llama_diffusion(&self, n_gen: usize) -> Option<DgLlamaResult> {
        use std::process::Command;
        let cpu = self.ngl == 0;
        let mut cli = Self::llama_diffusion_cli_path(cpu);
        if !cli.is_file() {
            eprintln!(
                "llama-diffusion-cli not found ({}): set INFR_LLAMA_DIFFUSION_CLI, put it on PATH, \
                 or build the fork at ~/Projects/mxaddict/llama.cpp-dg (cmake -DGGML_VULKAN=ON \
                 -B build-vulkan && cmake --build build-vulkan --target llama-diffusion-cli)",
                cli.display()
            );
            return None;
        }
        let n = n_gen.to_string();
        let run = |cli: &Path| -> Option<String> {
            let mut c = Command::new(cli);
            c.args(&self.llama_model_args);
            c.args(["-p", Self::DG_PROMPT, "-n", &n]);
            if cpu {
                c.args(["-ngl", "0"]);
            } else {
                c.args(["-ngl", "999", "-dev", &self.dev]);
            }
            let out = c.output().ok()?;
            Some(format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ))
        };
        let mut e2e = Vec::with_capacity(self.reps.max(1));
        let mut step = Vec::with_capacity(self.reps.max(1));
        let mut steps_v = Vec::with_capacity(self.reps.max(1));
        let mut fell_back = false;
        for _ in 0..self.reps.max(1) {
            let mut combined = run(&cli)?;
            if !fell_back && combined.contains("unknown model architecture") {
                let fork = Self::fork_diffusion_cli_path(cpu);
                eprintln!(
                    "{} doesn't support arch=diffusion-gemma (mainline install?) — falling back \
                     to the fork build at {}",
                    cli.display(),
                    fork.display()
                );
                cli = fork;
                fell_back = true;
                combined = run(&cli)?;
            }
            match parse_llama_diffusion_cli_output(&combined) {
                Some(r) => {
                    e2e.push(r.e2e_ts);
                    step.push(r.step_ts);
                    steps_v.push(r.steps);
                }
                None => {
                    eprintln!(
                        "llama-diffusion-cli produced no parseable throughput line: {}",
                        combined.trim()
                    );
                    return None;
                }
            }
        }
        let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        Some(DgLlamaResult {
            e2e_ts: avg(&e2e),
            step_ts: avg(&step),
            steps: avg(&steps_v),
        })
    }
}

/// llama.cpp's `llama-diffusion-cli` (the fork's DG oracle, Phase 4/E) prints its summary as two
/// plain-text lines (no `-o json` mode on this binary), e.g.:
///   `total time: 39559.95ms, time per step: 1582.40ms (25 steps over 1 blocks, entropy-bound)`
///   `throughput: 6.5 tok/s (256 tok in 39559.95ms), in-step parallel 162 tok/s (256-tok canvas x 25.0 steps/block)`
/// `e2e_ts` ("throughput:") and `step_ts` ("in-step parallel ... tok/s") both come off the SECOND
/// line; `steps` (total denoising steps run) comes off the FIRST line's "N steps over" — parsed
/// separately since it's the one number not repeated on the throughput line.
struct DgLlamaResult {
    e2e_ts: f64,
    step_ts: f64,
    steps: f64,
}

/// Pull a `float` out of `s` immediately after the first occurrence of `needle` (shared by
/// [`parse_llama_diffusion_cli_output`]'s two fields — same "walk forward from a fixed label"
/// approach as [`parse_llama_cli_gen_rate`], factored out here since this arm needs it twice).
fn parse_float_after(s: &str, needle: &str) -> Option<f64> {
    let idx = s.find(needle)?;
    let after = s[idx + needle.len()..].trim_start();
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(after.len());
    after[..end].parse::<f64>().ok()
}

/// Parse `llama-diffusion-cli`'s two-line summary — see [`DgLlamaResult`]'s doc comment for the
/// exact format. `steps` is scraped from " steps over " on the first line (walking BACKWARD to
/// the start of the number, the mirror image of [`parse_float_after`]'s forward walk, since the
/// number precedes the needle here); `e2e_ts`/`step_ts` come off the "throughput:"/"in-step
/// parallel " labels on the second line via `parse_float_after`.
fn parse_llama_diffusion_cli_output(output: &str) -> Option<DgLlamaResult> {
    let steps = {
        const NEEDLE: &str = " steps over ";
        let idx = output.find(NEEDLE)?;
        let before = &output[..idx];
        let start = before
            .rfind(|c: char| !(c.is_ascii_digit() || c == '.'))
            .map(|i| i + 1)
            .unwrap_or(0);
        before[start..].parse::<f64>().ok()?
    };
    let line = output.lines().find(|l| l.contains("in-step parallel"))?;
    let e2e_ts = parse_float_after(line, "throughput:")?;
    let step_ts = parse_float_after(line, "in-step parallel ")?;
    Some(DgLlamaResult {
        e2e_ts,
        step_ts,
        steps,
    })
}

/// diffusion-gemma's fixed decode length for the compare arm (Phase 4/E) — mirrors the oracle
/// invocation quoted in `docs/diffusion-gemma.md` (`-n 256`), so numbers seen here line up with
/// numbers already captured there.
const DG_N_GEN: usize = 256;

/// diffusion-gemma's compare arm (Phase 4/E, `docs/diffusion-gemma.md`): shared by
/// `cmd_compare`/`cmd_compare_sweep` at the point their old hard bail on `arch=diffusion-gemma`
/// used to sit (`ModelBench::new` no longer bails — see `is_dg`'s doc comment). Runs BOTH tools'
/// DG decode once at a fixed `-n 256` and returns the raw measurements; callers format their own
/// rows because `cmd_compare` prints a labeled table and `cmd_compare_sweep` needs `(model,
/// metric, infr, llama)` tuples for the ranked summary.
///
/// METRIC HONESTY: DG is entropy-bound — both implementations run a DIFFERENT NUMBER of denoise
/// steps for the same `-n 256` request (the early-stop point is content/implementation-sensitive:
/// see `dg_bench_run`'s comment on why the dummy prompt is a fixed real sentence, not raw ids), so
/// wall-clock end-to-end tok/s is NOT apples-to-apples between infr and llama.cpp — whichever tool
/// happens to trim its canvas earlier looks faster for free, independent of actual GPU work done.
/// The number that DOES reflect forward-pass work is per-step throughput
/// (`canvas_length * steps / decode_secs`): llama.cpp's own oracle already reports this as
/// "in-step parallel N tok/s", and infr's `dg_bench_run` computes the IDENTICAL formula
/// (`DgBenchResult::parallel_ts`), so `dg-step` is the metric this arm treats as the real
/// infr-vs-llama ratio (it's what feeds the sweep's "BIGGEST GAPS" ranking). `dg-e2e` is still
/// printed for visibility — informational only, each side's OWN step count folded into the row
/// text so the mismatch is visible instead of hidden.
fn dg_compare_measure(mb: &ModelBench) -> (anyhow::Result<DgBenchResult>, Option<DgLlamaResult>) {
    let infr = mb.dg_infr(DG_N_GEN);
    let llama = mb.llama_diffusion(DG_N_GEN);
    (infr, llama)
}

/// `cmd_compare_sweep`'s DG rows: `dg-step` (in-step-parallel ratio — the real gap metric, feeds
/// `rows`/the ranked summary) and `dg-e2e` (informational, own-step-count e2e, NOT fed into
/// `rows`). See [`dg_compare_measure`]'s doc comment for why the split.
fn print_dg_sweep_rows(mb: &ModelBench, short: &str, rows: &mut Vec<(String, String, f64, f64)>) {
    let (infr, llama) = dg_compare_measure(mb);
    let step_label = "dg-step".to_string();
    let is = infr
        .as_ref()
        .map(|r| format!("{:.0}", r.parallel_ts))
        .unwrap_or_else(|e| {
            eprintln!("infr DG bench failed ({short}): {e:#}");
            "ERR".into()
        });
    let ls = llama
        .as_ref()
        .map(|r| format!("{:.0}", r.step_ts))
        .unwrap_or_else(|| "NA".into());
    let ratio = match (infr.as_ref().ok(), llama.as_ref()) {
        (Some(i), Some(l)) if l.step_ts > 0.0 => {
            rows.push((
                short.to_string(),
                step_label.clone(),
                i.parallel_ts,
                l.step_ts,
            ));
            format!("{:.2}x", i.parallel_ts / l.step_ts)
        }
        _ => "-".into(),
    };
    println!("{short:<22} {step_label:<10} | {is:>9} | {ls:>9} | {ratio:>10}");

    let e2e_label = "dg-e2e".to_string();
    match (&infr, &llama) {
        (Ok(i), Some(l)) => println!(
            "{short:<22} {e2e_label:<10} | {:>6.1}t/s@{:.0}st | {:>6.1}t/s@{:.0}st | informational (steps differ)",
            i.gen_ts, i.steps, l.e2e_ts, l.steps
        ),
        _ => println!(
            "{short:<22} {e2e_label:<10} | {:>9} | {:>9} | {:>10}",
            "-", "-", "-"
        ),
    }
}

/// Compare infr vs llama.cpp on coding-agent-shaped workloads. Shells out to `infr bench` (this same
/// binary) and `llama-bench` with matching flags, so both run the SAME model + GPU under one driver.
/// Scenarios (the target workload — see memory infr-optimization-priority):
///   • CONTEXT LOAD — cold prefill of a repo/file dump (pp at each ctx size)
///   • REPLY @depth — decode a `gen`-token reply with a session already in context (tg @ depth)
///   • SESSION TURN — ingest P then generate G at session depth (pg, the realistic per-turn unit)
#[allow(clippy::too_many_arguments)]
fn cmd_compare(
    model: &str,
    dev: &str,
    ngl: usize,
    threads: usize,
    reps: usize,
    ubatch: usize,
    ctx: &[usize],
    gen: usize,
    turns: &[String],
    llama_bench: &str,
    cfg: &Arc<Config>,
) -> anyhow::Result<()> {
    let mb = ModelBench::new(
        model,
        dev,
        ngl,
        threads,
        reps,
        ubatch,
        llama_bench,
        cfg.clone(),
    )?;
    let infr_b = |args: &[&str]| mb.infr(args);
    let llama_b = |np: usize, ng: usize, args: &[&str]| mb.llama(np, ng, args);

    let model_name = Path::new(model)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model);
    let ub_s = if ubatch > 0 {
        ubatch.to_string()
    } else {
        "tool-default".into()
    };
    println!("\nmodel: {model_name}   reps: {reps}   ubatch: {ub_s}");

    let row = |label: String, i: anyhow::Result<f64>, l: Option<f64>| {
        let is = i
            .as_ref()
            .map(|v| format!("{v:.0}"))
            .unwrap_or_else(|_| "ERR".into());
        let ls = l.map(|v| format!("{v:.0}")).unwrap_or_else(|| "NA".into());
        let ratio = match (i.as_ref().ok(), l) {
            (Some(&iv), Some(lv)) if lv > 0.0 => format!("{:.2}x", iv / lv),
            _ => "-".into(),
        };
        println!("{label:<17} | {is:>10} | {ls:>10} | {ratio:>8}");
    };
    let hdr = |title: &str| {
        println!(
            "\n{title:<17} | {:>10} | {:>10} | {:>8}",
            "infr", "llama.cpp", "infr/llama"
        );
        println!("{:-<18}+{:-<12}+{:-<12}+{:-<10}", "", "", "", "");
    };

    // diffusion-gemma (Phase 4/E): entirely different measurement shape (no upstream llama-bench
    // support for this arch — see `ModelBench::is_dg`'s doc comment), so it skips every AR
    // scenario below (CONTEXT LOAD / REPLY @depth / MTP / SHORT TURN / TURN all assume a
    // prefill-then-decode AR turn shape DG doesn't have) and returns after its own two rows.
    if mb.is_dg {
        hdr("DIFFUSION-GEMMA DECODE"); // entropy-bound block-diffusion decode, fixed -n 256
        let (infr, llama) = dg_compare_measure(&mb);
        let infr_steps = infr.as_ref().ok().map(|r| r.steps);
        let infr_e2e = infr.as_ref().ok().map(|r| r.gen_ts);
        row(
            "dg-step".to_string(),
            infr.map(|r| r.parallel_ts),
            llama.as_ref().map(|r| r.step_ts),
        );
        let fmt_st = |ts: Option<f64>, st: Option<f64>| match (ts, st) {
            (Some(t), Some(s)) => format!("{t:.1} t/s @ {s:.0} steps"),
            _ => "-".into(),
        };
        println!(
            "  dg-e2e (informational — step counts differ, not apples-to-apples): infr {}   llama {}",
            fmt_st(infr_e2e, infr_steps),
            fmt_st(llama.as_ref().map(|r| r.e2e_ts), llama.as_ref().map(|r| r.steps)),
        );
        return Ok(());
    }

    hdr("CONTEXT LOAD"); // cold prefill of a repo/file dump
    for &n in ctx {
        let np = n.to_string();
        row(
            format!("pp{n}"),
            infr_b(&["-p", &np, "-n", "0"]),
            llama_b(n, 0, &["-p", &np, "-n", "0"]),
        );
    }

    hdr("REPLY @depth"); // decode a reply with a session already in context
    let g = gen.to_string();
    for &d in ctx {
        let ds = d.to_string();
        row(
            format!("tg{gen}@{d}"),
            infr_b(&["-p", "0", "-n", &g, "-d", &ds]),
            llama_b(0, gen, &["-p", "0", "-n", &g, "-d", &ds]),
        );
    }

    // MTP DECODE (issue #33, phase 4 — perf-bottleneck visibility for the self-speculative decode
    // path, currently NET NEGATIVE at 36 t/s vs a ~100 t/s baseline): four numbers — infr/llama at
    // baseline tg, infr/llama with MTP on — plus each tool's own mtp/base speedup, so the gap AND
    // its shape (is infr's speculation slower to begin with, or is its baseline just slower) are
    // both visible in one section. Absent entirely for a GGUF with no `nextn.*` head.
    if mb.has_mtp_head() {
        hdr("MTP DECODE"); // self-speculative decode: baseline tg vs MTP-spec tg, both tools
        const N: usize = 128;
        let ns = N.to_string();
        let infr_base = infr_b(&["-p", "0", "-n", &ns]);
        let infr_mtp = mb.infr_mtp(&["-p", "0", "-n", &ns]);
        let llama_base = llama_b(0, N, &["-p", "0", "-n", &ns]);
        let llama_mtp = mb.llama_cli_mtp(N);
        // Each tool's own speedup (mtp/base) — NOT an infr-vs-llama ratio, so computed here rather
        // than through `row`'s ratio column (which is always infr/llama at one shape).
        let infr_speedup = match (infr_base.as_ref(), infr_mtp.as_ref()) {
            (Ok(&b), Ok(&m)) if b > 0.0 => Some(m / b),
            _ => None,
        };
        let llama_speedup = match (llama_base, llama_mtp) {
            (Some(b), Some(m)) if b > 0.0 => Some(m / b),
            _ => None,
        };
        row(format!("tg{N}"), infr_base, llama_base); // infr-vs-llama @ baseline
        row(format!("mtp{N}"), infr_mtp, llama_mtp); // infr-vs-llama @ MTP
        let fmt_x = |v: Option<f64>| v.map(|v| format!("{v:.2}x")).unwrap_or_else(|| "-".into());
        println!(
            "  own-tool speedup (mtp/base): infr {}   llama {}",
            fmt_x(infr_speedup),
            fmt_x(llama_speedup),
        );
    }

    // SHORT TURN — the tiny suffix prefill (m = 2..8 Linears): what a brief follow-up costs in a
    // warm multi-turn session (KV prefix reused, only the new tokens forward). This is the
    // multi-row-GEMV / spec-verify shape; 16 sits past the mrow gate (the first GEMM point) as
    // the cliff-edge reference. Cold (@0) isolates the kernels; @depth adds the attention term.
    hdr("SHORT TURN"); // tiny suffix prefill on a warm session
    for &n in &[2usize, 4, 8, 16] {
        let np = n.to_string();
        row(
            format!("pp{n}"),
            infr_b(&["-p", &np, "-n", "0"]),
            llama_b(n, 0, &["-p", &np, "-n", "0"]),
        );
        for &d in ctx {
            let ds = d.to_string();
            row(
                format!("pp{n}@{d}"),
                infr_b(&["-p", &np, "-n", "0", "-d", &ds]),
                llama_b(n, 0, &["-p", &np, "-n", "0", "-d", &ds]),
            );
        }
    }

    for t in turns {
        let (p, gg) = t.split_once(',').context("--turn expects `P,G`")?;
        let (pn, gn): (usize, usize) = (p.trim().parse()?, gg.trim().parse()?);
        hdr(&format!("TURN {t}")); // ingest P then generate G at session depth
        for &d in ctx {
            let ds = d.to_string();
            row(
                format!("pg{t}@{d}"),
                infr_b(&["--pg", t, "-d", &ds]),
                llama_b(pn, gn, &["-p", "0", "-n", "0", "-pg", t, "-d", &ds]),
            );
        }
    }
    Ok(())
}

/// Architecture-family recommended sampling `(temp, top_k, top_p)` — the published per-family
/// defaults (`top_k = 0` = keep-all, i.e. top_k disabled). infr enables `<think>` by default, so
/// the Qwen3.x row uses its THINKING recommendation. An unknown arch falls back to the neutral
/// chat default the whole engine used before this table existed.
fn arch_sampling(arch: &str) -> (f32, usize, f32) {
    use infr_llama::arch::*;
    match arch {
        // Qwen3.x thinking rec (Qwen team): temp 0.6 / top_k 20 / top_p 0.95.
        QWEN3 | QWEN3_MOE | QWEN35 | QWEN35_MOE => (0.6, 20, 0.95),
        // Qwen2/2.5 rec: a touch hotter, tighter nucleus.
        QWEN2 => (0.7, 20, 0.8),
        // Gemma (Google): higher temp, wide top_k.
        GEMMA3 | GEMMA4 | DIFFUSION_GEMMA => (1.0, 64, 0.95),
        // Meta Llama 3.x / 4 default generation_config: temp 0.6, top_p 0.9, top_k off.
        LLAMA | LLAMA4 => (0.6, 0, 0.9),
        _ => (0.6, 20, 0.95),
    }
}

/// Per-field sampling override from a `generation_config.json` sitting BESIDE the model (the HF
/// convention — the model's OWN recommended generation params). Present for full-repo checkouts,
/// absent for the GGUF-only distributions `infr pull` fetches (then the [`arch_sampling`] table
/// drives). Each field is independent — a config that sets only `temperature` overrides just that.
fn generation_config_sampling(gguf: &std::path::Path) -> (Option<f32>, Option<usize>, Option<f32>) {
    let Some(path) = gguf.parent().map(|d| d.join("generation_config.json")) else {
        return (None, None, None);
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return (None, None, None);
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return (None, None, None);
    };
    (
        v.get("temperature")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32),
        v.get("top_k").and_then(|x| x.as_u64()).map(|x| x as usize),
        v.get("top_p").and_then(|x| x.as_f64()).map(|x| x as f32),
    )
}

/// The model-aware sampling defaults for `gguf`: the [`arch_sampling`] family table, with any
/// [`generation_config_sampling`] sibling override applied per field. Returns `(temp, top_k, top_p,
/// source_label)` — the label is for the CLI banner so the effective sampler is never a mystery.
fn model_sampling_defaults(gguf: &std::path::Path) -> (f32, usize, f32, String) {
    let arch = infr_llama::arch::arch_of(gguf);
    let (mut t, mut k, mut p) = arch
        .as_deref()
        .map(arch_sampling)
        .unwrap_or((0.6, 20, 0.95));
    let mut src = arch.clone().unwrap_or_else(|| "default".to_string());
    let (gt, gk, gp) = generation_config_sampling(gguf);
    if gt.is_some() || gk.is_some() || gp.is_some() {
        src = format!("{}+generation_config", arch.as_deref().unwrap_or("default"));
    }
    if let Some(x) = gt {
        t = x;
    }
    if let Some(x) = gk {
        k = x;
    }
    if let Some(x) = gp {
        p = x;
    }
    (t, k, p, src)
}

/// Fold the model's recommended sampling (see [`model_sampling_defaults`]) into the config the
/// models this command loads are handed, for the knobs no layer specified, and print a one-line
/// banner of the EFFECTIVE sampler. A `--temp`/`--top-k`/`--top-p` the user passed is already in
/// the resolved `cfg` (via `SamplingOpts::overrides`), so it wins and shows through here. Shared
/// by `run`, `serve` and `multi`.
///
/// The seam reads `sampling.*` off this returned `Config` (S4), so what this function returns IS
/// what `infr run` samples with — the library default is greedy, which is right for goldens and
/// wrong for a chat REPL (pure argmax makes a thinking model degenerate).
///
/// **Precedence, unchanged:** the model recommendation is the LOWEST source, below the config file
/// and below the environment — so it may only fill a knob NO layer specified. S1 answered "did a
/// layer specify it?" by re-publishing the resolved values into the process environment and then
/// probing `std::env::var(..).is_ok()`; S8 asks the `specified` [`PartialConfig`] threaded down
/// from `main()` directly, and nothing writes `INFR_*` any more.
///
/// One edge behaves better than it did: a value no layer could PARSE (`INFR_TEMP=banana`) used to
/// answer "specified" to the env probe — the variable was present — and so suppressed the model
/// recommendation, leaving the sampler on the library's greedy default. It is not in `specified`
/// (the env layer emits `None` for an unparseable `Float`, §7.0 step 3), so the recommendation now
/// applies, which is what an ignored bad value is supposed to mean.
fn apply_model_sampling_defaults(
    cfg: &Arc<Config>,
    specified: &PartialConfig,
    gguf: &std::path::Path,
) -> Arc<Config> {
    let (t, k, p, src) = model_sampling_defaults(gguf);
    let mut out = (**cfg).clone();
    if !specified.is_path_set("sampling.temp") {
        out.sampling.temp = t;
    }
    if !specified.is_path_set("sampling.top_k") {
        out.sampling.top_k = k;
    }
    if !specified.is_path_set("sampling.top_p") {
        out.sampling.top_p = p;
    }
    eprintln!(
        "[sampling: temp={} top_k={} top_p={} ({src}); --temp/--top-k/--top-p to override]",
        out.sampling.temp, out.sampling.top_k, out.sampling.top_p,
    );
    Arc::new(out)
}

fn cmd_serve(
    model: &str,
    addr: &str,
    parallel: Option<usize>,
    cfg: &Arc<Config>,
    specified: &PartialConfig,
) -> anyhow::Result<()> {
    let (gguf, tok) = resolve(model)?;
    let model_id = gguf
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let sockaddr: std::net::SocketAddr = addr.parse().context("invalid --addr")?;
    // `--parallel` was EXPLICITLY set iff `Some`; the concurrent-path slot count defaults to 4.
    let parallel_explicit = parallel;
    let parallel = parallel.unwrap_or(4).max(1);

    // `--ctx` is the PER-SLOT context. `DeviceOpts::overrides` already validated it into
    // `device.ctx` (shared size grammar `8192`/`256k`/`50%`); the ParallelSeam below takes it from
    // the resolved config and divides it across the slots. One grammar, one meaning
    // (`infr_core::parse_size`, which is also what the `INFR_CTX` env layer parses).
    let is_dg = infr_llama::diffusion::is_diffusion_gemma(&gguf);
    let is_vulkan = !is_dg && matches!(selected_backend(cfg)?, Backend::Vulkan(_));

    let cfg = &apply_model_sampling_defaults(cfg, specified, &gguf);

    // ── the CONCURRENT path: dense/MoE/qwen35 on the Vulkan seam ────────────────────────────────
    // N KV slots off ONE weight upload, round-robin on the GPU at token granularity. This is the
    // default `infr serve` engine.
    if is_vulkan {
        let loaded = infr_llama::SeamModel::load_with(&gguf, tok.as_deref(), cfg.clone())?;
        // `--ctx` (or INFR_CTX) is the PER-SLOT window: an explicit token count is used verbatim,
        // a `%` is a fraction of the whole device's KV capacity split across the slots, and unset
        // derives it from the VRAM fit divided by N. See `SeamModel::vulkan_slot_ctx`.
        let want_ctx = cfg.device.ctx;
        let t0 = std::time::Instant::now();
        let engine = infr_llama::parallel::ParallelSeam::new(loaded, parallel, want_ctx)?;
        eprintln!(
            "warmup: pipelines compiled in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        let n_slots = engine.n_slots();
        let max_ctx = engine.max_ctx();
        let generator: std::sync::Arc<dyn infr_server::ChatGenerator> =
            std::sync::Arc::new(ParallelGenerator::new(&gguf, engine, cfg.clone())?);
        let rt = tokio::runtime::Runtime::new()?;
        println!(
            "infr serve: {model_id} on http://{sockaddr}  (OpenAI /v1, {n_slots} slot{} x {max_ctx} ctx)",
            if n_slots == 1 { "" } else { "s" },
        );
        return rt.block_on(infr_server::serve(
            generator,
            model_id,
            sockaddr,
            n_slots,
            cfg.clone(),
        ));
    }

    // ── the SERIALISED path: CPU / Metal / ROCm / diffusion-gemma ────────────────────────────
    // These have no multi-slot engine (one `&mut` ChatModel, one KV session), so concurrent
    // requests would only queue behind a Mutex. Warn ONLY when the user EXPLICITLY asked for
    // parallelism (`--parallel N>1`) — the default (unset) must stay silent, since every
    // CPU/Metal/ROCm/diffusion serve would otherwise print a spurious "ignored" note.
    if matches!(parallel_explicit, Some(n) if n > 1) {
        eprintln!(
            "note: --parallel {} ignored on the CPU/Metal/ROCm/diffusion backends (no multi-slot \
             engine); serving 1 request at a time. The Vulkan seam is the concurrent engine.",
            parallel_explicit.unwrap()
        );
    }

    // Seam-backed serve — the ONE engine: the SAME ChatModel + multi-slot session `infr run` uses
    // (built through the SAME `build_chat_model` funnel), so serve gets per-request suffix-only
    // prefill and cross-conversation prefix seeding for free. INFR_DEV=cpu / INFR_DEV=metal select
    // the reference backends; Vulkan is the default. qwen35 shares the SAME funnel as every other arch
    // (issue #30). Metal also honours INFR_SPEC_DRAFT (draft-verify speculative decode) via
    // `metal_chat_model` inside the funnel.
    let mut m = build_chat_model(&gguf, tok.as_deref(), is_dg, cfg)?;
    {
        // Compile every lazily-built pipeline NOW (a tiny throwaway generation) so the first
        // request doesn't pay seconds of pipeline builds on top of its own prefill.
        let t0 = std::time::Instant::now();
        m.warmup()?;
        eprintln!(
            "warmup: pipelines compiled in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        let generator: std::sync::Arc<dyn infr_server::ChatGenerator> =
            std::sync::Arc::new(SeamGenerator::new(&gguf, m, cfg.clone())?);
        let rt = tokio::runtime::Runtime::new()?;
        println!("infr serve: {model_id} on http://{sockaddr}  (OpenAI /v1, agnostic seam)");
        rt.block_on(infr_server::serve(
            generator,
            model_id,
            sockaddr,
            1,
            cfg.clone(),
        ))
    }
}

/// Split a `MODEL[@VulkanN]` spec into (model_ref, Option<device_index>). The device suffix is the
/// text after the LAST `@` when it is `VulkanN` (case-insensitive) or bare digits; anything else is
/// treated as part of the model reference (so an `@` inside a path or HF ref is left alone). Returns
/// an error for an `@`-suffix that looks device-shaped but isn't a valid index.
fn parse_model_spec(spec: &str) -> anyhow::Result<(&str, Option<usize>)> {
    let Some(at) = spec.rfind('@') else {
        return Ok((spec, None));
    };
    let (head, tail) = (&spec[..at], &spec[at + 1..]);
    let lower = tail.to_ascii_lowercase();
    let digits = lower.strip_prefix("vulkan").unwrap_or(&lower);
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        let idx: usize = digits
            .parse()
            .map_err(|_| anyhow!("invalid device suffix `@{tail}` in `{spec}`"))?;
        Ok((head, Some(idx)))
    } else {
        // Not a device suffix (e.g. an `@` inside the ref) — keep the whole thing as the model.
        Ok((spec, None))
    }
}

/// `infr multi` — host several models at once, each pinned to a physical GPU, on ONE OpenAI server
/// routed by model name. Data-parallel multi-device serving (Slice 1 of the multi-GPU campaign):
/// each model is a self-contained concurrent-slot [`ParallelSeam`] on its OWN backend/device
/// (`new_on`), so nothing crosses devices; the server dispatches a request to the generator for the
/// model it names. Graceful shutdown drains EVERY device (the server aborts all in-flight requests,
/// then each generator — and its backend — drops as `serve_multi` returns).
#[allow(clippy::too_many_arguments)]
fn cmd_multi(
    specs: &[String],
    addr: &str,
    parallel: usize,
    ctx: Option<&str>,
    cfg: &Arc<Config>,
    specified: &PartialConfig,
) -> anyhow::Result<()> {
    let sockaddr: std::net::SocketAddr = addr.parse().context("invalid --addr")?;
    let parallel = parallel.max(1);

    // `infr multi` is the VULKAN concurrent engine only — the reference backends have no multi-slot
    // engine and no device pool to spread across. Refuse the reference-backend envs up front rather
    // than silently ignoring them.
    if !matches!(selected_backend(cfg)?, Backend::Vulkan(_)) {
        anyhow::bail!(
            "`infr multi` hosts models on the Vulkan device pool; drop INFR_DEV=cpu/INFR_DEV=metal \
             (the CPU/Metal reference backends have no multi-slot engine)"
        );
    }

    // `--ctx` shares the size grammar with the rest of the CLI; validate once, apply to every model.
    let want_ctx = match ctx {
        Some(c) => {
            let spec = infr_core::parse_size(c)
                .ok_or_else(|| anyhow!("invalid --ctx `{c}` (expected e.g. 8192, 256k, or 50%)"))?;
            Some(spec)
        }
        None => None,
    };

    // Enumerate the device pool once: for validation, round-robin assignment of specs that omit a
    // device, and the routing table's device names.
    let devices = infr_vulkan::VulkanBackend::enumerate_devices(cfg).map_err(|e| anyhow!("{e}"))?;
    if devices.is_empty() {
        anyhow::bail!("no Vulkan physical devices found (see `infr devices`)");
    }

    // Load each model on its assigned device. Sequential on purpose: two concurrent multi-GiB weight
    // uploads to two devices would spike total memory; one at a time is the safe order (the iGPU
    // shares system RAM). Each `ParallelSeam::new_on` prints its own device selection + warmup line.
    let mut entries: Vec<(
        String,
        std::sync::Arc<dyn infr_server::ChatGenerator>,
        usize,
    )> = Vec::with_capacity(specs.len());
    let mut routing: Vec<(String, usize, String)> = Vec::with_capacity(specs.len());
    let mut used_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    // The config EVERY hosted model is loaded with. Sampling defaults are process-wide here (see
    // the `i == 0` note below), so the first model's recommendation is folded in once and the same
    // value is handed to all of them — exactly what the shared environment used to do.
    let mut hosted_cfg = cfg.clone();

    for (i, spec) in specs.iter().enumerate() {
        let (model_ref, dev_opt) = parse_model_spec(spec)?;
        // Round-robin the pool for specs without an explicit device.
        let dev = dev_opt.unwrap_or(i % devices.len());
        if dev >= devices.len() {
            anyhow::bail!(
                "`{spec}` asks for Vulkan{dev} but this system has {} device(s) (see `infr devices`)",
                devices.len()
            );
        }
        let (gguf, tok) = resolve(model_ref)?;
        if infr_llama::diffusion::is_diffusion_gemma(&gguf) {
            anyhow::bail!(
                "`{model_ref}` is a diffusion-gemma model, which `infr multi` does not host \
                 (no multi-slot engine); serve it on its own with `infr serve`"
            );
        }
        // Sampling defaults are process-global; apply the FIRST model's so the banner is honest, and
        // note that all hosted models share them (a per-model override would need per-request fields).
        if i == 0 {
            hosted_cfg = apply_model_sampling_defaults(cfg, specified, &gguf);
        }

        // A stable, unique wire id: the file stem, disambiguated by device when two specs collide
        // (e.g. the same model hosted twice on two GPUs — the demo).
        let stem = gguf
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        let mut model_id = stem.clone();
        if !used_ids.insert(model_id.clone()) {
            model_id = format!("{stem}@Vulkan{dev}");
            // Extremely unlikely second collision (same model, same device, twice): number it.
            let mut n = 2;
            while !used_ids.insert(model_id.clone()) {
                model_id = format!("{stem}@Vulkan{dev}#{n}");
                n += 1;
            }
        }

        eprintln!(
            "[multi] loading {model_id} on Vulkan{dev} ({})",
            devices[dev].name
        );
        let loaded = infr_llama::SeamModel::load_with(&gguf, tok.as_deref(), hosted_cfg.clone())?;
        let engine =
            infr_llama::parallel::ParallelSeam::new_on(Some(dev), loaded, parallel, want_ctx)?;
        let n_slots = engine.n_slots();
        let dev_name = engine.device_name();
        let generator: std::sync::Arc<dyn infr_server::ChatGenerator> =
            std::sync::Arc::new(ParallelGenerator::new(&gguf, engine, hosted_cfg.clone())?);
        routing.push((model_id.clone(), dev, dev_name));
        entries.push((model_id, generator, n_slots));
    }

    // Print the model → device routing so the demo shows exactly which model landed on which GPU.
    println!(
        "\ninfr multi: {} model(s) on http://{sockaddr}  (OpenAI /v1)",
        entries.len()
    );
    println!("  routing (model -> device):");
    for (id, dev, name) in &routing {
        println!("    {id}  ->  Vulkan{dev} ({name})");
    }
    println!();

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(infr_server::serve_multi(entries, sockaddr, hosted_cfg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_llama::arch::*;

    /// A synthetic `INFR_*` environment, driven through the config env layer's INJECTED reader.
    ///
    /// This is what replaced the module's old `static ENV_LOCK: Mutex<()>` (docs/config-plan.md R7):
    /// the tests that used to mutate the real process environment — and therefore had to serialise
    /// against each other and restore afterwards — now build a `Config` from a `HashMap` and assert
    /// on the resolved value. Nothing here touches the process environment, so it all runs in
    /// parallel with everything else in the binary.
    fn env_layer(pairs: &[(&str, &str)]) -> PartialConfig {
        let map: std::collections::HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        infr_core::config::env::parse(&|k| map.get(k).cloned())
            .expect("the env layer rejected a synthetic environment")
    }

    /// The layers a real invocation folds, minus the config file: `Default` < env < flags/`--set`.
    fn resolve_cfg(env: &[(&str, &str)], flags: PartialConfig, sets: &[&str]) -> Config {
        let cli = infr_core::config::ConfigLayer::cli(&ConfigOverrides {
            sets: sets.iter().map(|s| (*s).to_string()).collect(),
            flags,
            ..Default::default()
        })
        .expect("the cli layer rejected a valid override");
        Config::load_from_layers(&[env_layer(env), cli])
    }

    fn device_opts(dev: Option<&str>) -> DeviceOpts {
        DeviceOpts {
            dev: dev.map(str::to_string),
            ctx: None,
            ubatch: None,
            threads: None,
        }
    }

    #[test]
    fn metal_prefill_bench_materializes_depth_and_times_exact_rows() {
        let shallow = metal_bench_shape(4, 0, 0, None);
        assert_eq!(shallow.depth_warm_prompt, None);
        assert_eq!(shallow.measured_prompt, 5);
        assert_eq!(shallow.measured_gen, 0);
        assert_eq!(shallow.tokens, 4);
        assert_eq!(shallow.metric, MetalBenchMetric::Prompt);

        let deep = metal_bench_shape(4, 0, 4096, None);
        assert_eq!(deep.depth_warm_prompt, Some(4097));
        assert_eq!(deep.measured_prompt, 4101);
        assert_eq!(deep.measured_gen, 0);
        assert_eq!(deep.tokens, 4);
        assert_eq!(deep.metric, MetalBenchMetric::Prompt);
    }

    #[test]
    fn metal_decode_and_turn_benches_preserve_depth_prefix() {
        let decode = metal_bench_shape(0, 128, 4096, None);
        assert_eq!(decode.depth_warm_prompt, Some(4097));
        assert_eq!(decode.measured_prompt, 4097);
        assert_eq!(decode.measured_gen, 128);
        assert_eq!(decode.tokens, 128);
        assert_eq!(decode.metric, MetalBenchMetric::Decode);

        let turn = metal_bench_shape(0, 0, 4096, Some((32, 16)));
        assert_eq!(turn.depth_warm_prompt, Some(4097));
        assert_eq!(turn.measured_prompt, 4128);
        assert_eq!(turn.measured_gen, 16);
        assert_eq!(turn.tokens, 48);
        assert_eq!(turn.metric, MetalBenchMetric::Turn);
    }

    #[test]
    fn model_spec_parses_device_suffix() {
        assert_eq!(
            parse_model_spec("qwen.gguf@Vulkan1").unwrap(),
            ("qwen.gguf", Some(1))
        );
        assert_eq!(
            parse_model_spec("qwen.gguf@vulkan0").unwrap(),
            ("qwen.gguf", Some(0))
        );
        // Bare-digit suffix also accepted.
        assert_eq!(
            parse_model_spec("qwen.gguf@2").unwrap(),
            ("qwen.gguf", Some(2))
        );
        // No suffix.
        assert_eq!(parse_model_spec("qwen.gguf").unwrap(), ("qwen.gguf", None));
        // An `@` that isn't device-shaped (e.g. a revision) is left as part of the ref.
        assert_eq!(
            parse_model_spec("org/repo@main").unwrap(),
            ("org/repo@main", None)
        );
    }

    // ── finding 1: unified backend decision (resolve_backend / selected_backend) ────────────────

    /// Build a `BackendEnv` from an `INFR_DEV` value, for pure decision tests (no process env).
    fn env(dev: Option<&str>) -> BackendEnv {
        BackendEnv {
            dev: dev.map(str::to_string),
        }
    }

    #[test]
    fn dev_flag_beats_infr_dev_env() {
        // The `--dev` flag wins over INFR_DEV: `--dev metal` under `INFR_DEV=cpu` resolves to Metal.
        assert_eq!(
            resolve_backend(Some("metal"), env(Some("cpu"))).unwrap(),
            Backend::Metal
        );
        assert_eq!(
            resolve_backend(Some("cpu"), env(Some("Vulkan1"))).unwrap(),
            Backend::Cpu
        );
        // Case-insensitive; the original casing is preserved for the Vulkan spec.
        assert_eq!(
            resolve_backend(Some("Vulkan1"), BackendEnv::default()).unwrap(),
            Backend::Vulkan(Some("Vulkan1".to_string()))
        );
        // A bogus --dev is rejected.
        assert!(resolve_backend(Some("gpu9"), BackendEnv::default()).is_err());
    }

    #[test]
    fn infr_dev_env_parses_same_grammar_as_dev_flag() {
        // No flag: INFR_DEV drives the pick, same grammar as `--dev`.
        assert_eq!(
            resolve_backend(None, env(Some("Vulkan1"))).unwrap(),
            Backend::Vulkan(Some("Vulkan1".to_string()))
        );
        assert_eq!(
            resolve_backend(None, env(Some("cpu"))).unwrap(),
            Backend::Cpu
        );
        assert_eq!(
            resolve_backend(None, env(Some("metal"))).unwrap(),
            Backend::Metal
        );
        // Garbage INFR_DEV errors early with a clear message (typo protection).
        let e = resolve_backend(None, env(Some("foo"))).unwrap_err();
        let msg = format!("{e:#}");
        assert!(msg.contains("INFR_DEV"), "message names the source: {msg}");
        assert!(msg.contains("foo"), "message echoes the bad value: {msg}");
        // No env at all → the default, first discrete Vulkan GPU.
        assert_eq!(
            resolve_backend(None, BackendEnv::default()).unwrap(),
            Backend::Vulkan(None)
        );
    }

    #[test]
    fn compare_translates_llama_metal_device_for_infr() {
        assert_eq!(compare_infr_dev("MTL0"), "metal");
        assert_eq!(compare_infr_dev("mtl0"), "metal");
        assert_eq!(compare_infr_dev("Vulkan1"), "Vulkan1");
        assert_eq!(compare_infr_dev("metal"), "metal");
        assert_eq!(compare_infr_dev("cpu"), "cpu");
    }

    #[test]
    fn legacy_metal_cpu_flags_are_no_longer_read() {
        // The old INFR_METAL=1 / INFR_CPU=1 flags were removed cleanly — with no INFR_DEV set they
        // do NOT select a backend (they are not even config knobs: `manifest::NOT_KNOBS`), so the
        // resolver falls through to the Vulkan default.
        let cfg = resolve_cfg(
            &[("INFR_CPU", "1"), ("INFR_METAL", "1")],
            PartialConfig::default(),
            &[],
        );
        assert_eq!(cfg.device.dev, None);
        assert_eq!(selected_backend(&cfg).unwrap(), Backend::Vulkan(None));
    }

    #[test]
    fn dev_flag_lands_on_the_config_and_beats_an_inherited_infr_dev() {
        // Inherit INFR_DEV=cpu AND a stale INFR_CPU (a dead env), then pass `--dev vulkan0`: the
        // CLI layer beats the env layer in the fold, so `device.dev` is the flag's spec and the
        // reader agrees. That spec is what the deep Vulkan device pick has to see.
        let mut flags = PartialConfig::default();
        device_opts(Some("vulkan0")).overrides(&mut flags).unwrap();
        let cfg = resolve_cfg(&[("INFR_DEV", "cpu"), ("INFR_CPU", "1")], flags, &[]);
        assert_eq!(cfg.device.dev.as_deref(), Some("vulkan0"));
        assert_eq!(
            selected_backend(&cfg).unwrap(),
            Backend::Vulkan(Some("vulkan0".to_string()))
        );

        // `--dev metal` resolves to the `metal` spec (never to the removed INFR_METAL flag).
        let mut flags = PartialConfig::default();
        device_opts(Some("metal")).overrides(&mut flags).unwrap();
        let cfg = resolve_cfg(&[("INFR_DEV", "Vulkan1")], flags, &[]);
        assert_eq!(cfg.device.dev.as_deref(), Some("metal"));
        assert_eq!(selected_backend(&cfg).unwrap(), Backend::Metal);
    }

    #[test]
    fn unset_dev_flag_preserves_the_inherited_infr_dev() {
        // The other half of the precedence `DeviceOpts` has always documented: an UNSET `--dev`
        // specifies nothing, so whatever the environment (or the config file) said survives.
        let mut flags = PartialConfig::default();
        device_opts(None).overrides(&mut flags).unwrap();
        assert!(!flags.is_path_set("device.dev"));
        let cfg = resolve_cfg(&[("INFR_DEV", "Vulkan1")], flags, &[]);
        assert_eq!(cfg.device.dev.as_deref(), Some("Vulkan1"));
        assert_eq!(
            selected_backend(&cfg).unwrap(),
            Backend::Vulkan(Some("Vulkan1".to_string()))
        );
        // …and with neither, the default pick.
        let cfg = resolve_cfg(&[], PartialConfig::default(), &[]);
        assert_eq!(selected_backend(&cfg).unwrap(), Backend::Vulkan(None));
    }

    #[test]
    fn device_flags_land_on_the_config_layer() {
        let opts = DeviceOpts {
            dev: None,
            ctx: Some("32k".to_string()),
            ubatch: Some(512),
            threads: Some(6),
        };
        let mut flags = PartialConfig::default();
        opts.overrides(&mut flags).unwrap();
        let cfg = resolve_cfg(&[], flags, &[]);
        assert_eq!(cfg.device.ctx, Some(infr_core::SizeSpec::Bytes(32 * 1024)));
        assert_eq!(cfg.device.ubatch, Some(512));
        assert!(cfg.device.ubatch_specified, "-u pins the chunk height");
        assert_eq!(cfg.device.threads, Some(6));

        // `-u 0` keeps its "adaptive" meaning: carried verbatim, filtered by the seam (`v > 0`) —
        // but it is still SPECIFIED, which is what keeps the dense placement sweeps off (§6.12).
        let mut flags = PartialConfig::default();
        DeviceOpts {
            dev: None,
            ctx: None,
            ubatch: Some(0),
            threads: None,
        }
        .overrides(&mut flags)
        .unwrap();
        let zero = resolve_cfg(&[], flags, &[]);
        assert_eq!(zero.device.ubatch, Some(0));
        assert!(
            zero.device.ubatch_specified,
            "`-u 0` must still count as a pin — the value is unusable, the intent is not"
        );

        // …and NO `-u` at all specifies neither half.
        let none = resolve_cfg(&[], PartialConfig::default(), &[]);
        assert_eq!(none.device.ubatch, None);
        assert!(!none.device.ubatch_specified);

        // A garbage `--ctx` still fails fast, with the same message.
        let e = DeviceOpts {
            dev: None,
            ctx: Some("banana".to_string()),
            ubatch: None,
            threads: None,
        }
        .overrides(&mut PartialConfig::default())
        .unwrap_err();
        assert!(format!("{e:#}").contains("invalid --ctx"), "{e:#}");
    }

    #[test]
    fn sampling_flags_land_on_the_config_layer_and_beat_the_env() {
        let opts = SamplingOpts {
            temp: Some(0.0),
            top_k: Some(40),
            top_p: Some(0.8),
            seed: Some(7),
            max_new: Some(64),
            no_think: false,
            think: false,
        };
        let mut flags = PartialConfig::default();
        opts.overrides(&mut flags);
        // An explicit `--temp 0` is NOT the same as an unspecified temp, even though 0.0 is also
        // the default: the layer records that it was ASKED for, which is what stops the
        // model-recommended default from filling it in.
        assert!(flags.is_path_set("sampling.temp"));
        let cfg = resolve_cfg(&[("INFR_TEMP", "0.6"), ("INFR_TOP_K", "20")], flags, &[]);
        assert_eq!(cfg.sampling.temp, 0.0);
        assert_eq!(cfg.sampling.top_k, 40);
        assert!((cfg.sampling.top_p - 0.8).abs() < 1e-6);
        assert_eq!(cfg.sampling.seed, Some(7));
        assert_eq!(cfg.sampling.max_new, 64);

        // `--no-think` / `--think` are the two INFR_NO_THINK polarities.
        let mut flags = PartialConfig::default();
        SamplingOpts {
            no_think: true,
            ..no_sampling_opts()
        }
        .overrides(&mut flags);
        assert!(resolve_cfg(&[], flags, &[]).sampling.no_think);
        let mut flags = PartialConfig::default();
        SamplingOpts {
            think: true,
            ..no_sampling_opts()
        }
        .overrides(&mut flags);
        assert!(
            !resolve_cfg(&[("INFR_NO_THINK", "1")], flags, &[])
                .sampling
                .no_think
        );
    }

    #[test]
    fn bench_pins_ignore_eos_through_the_cli_layer() {
        // What `cmd_bench`'s `set_var("INFR_IGNORE_EOS", "1")` used to do: a flag-level override,
        // so it wins over the environment exactly as that overwrite did.
        let layer = cli_flag_layer(&bench_cmd()).unwrap();
        assert!(layer.is_path_set("sampling.ignore_eos"));
        assert!(resolve_cfg(&[], layer, &[]).sampling.ignore_eos);
        // …and `run` does NOT pin it.
        let layer = cli_flag_layer(&Cmd::Run {
            model: "m".to_string(),
            message: None,
            diffusion_visual: String::new(),
            device: device_opts(None),
            sampling: no_sampling_opts(),
        })
        .unwrap();
        assert!(!layer.is_path_set("sampling.ignore_eos"));
    }

    #[test]
    fn set_reaches_a_knob_with_no_flag_and_loses_to_the_flag_that_has_one() {
        // `--set` is the escape hatch for the knobs with no dedicated flag…
        let cfg = resolve_cfg(
            &[],
            PartialConfig::default(),
            &["kernels.vulkan.flash_splits=2"],
        );
        assert_eq!(cfg.kernels.vulkan.flash_splits, Some(2));
        // …it beats the environment…
        let cfg = resolve_cfg(
            &[("INFR_TEMP", "0.6")],
            PartialConfig::default(),
            &["sampling.temp=0.2"],
        );
        assert!((cfg.sampling.temp - 0.2).abs() < 1e-6);
        // …and it LOSES to the dedicated flag for the same field (§11 [DECIDE-3]).
        let mut flags = PartialConfig::default();
        SamplingOpts {
            temp: Some(0.9),
            ..no_sampling_opts()
        }
        .overrides(&mut flags);
        let cfg = resolve_cfg(&[], flags, &["sampling.temp=0.2"]);
        assert!((cfg.sampling.temp - 0.9).abs() < 1e-6);
    }

    #[test]
    fn unknown_set_path_is_a_hard_error_with_a_suggestion() {
        let err = infr_core::config::ConfigLayer::cli(&ConfigOverrides {
            sets: vec!["nonsense.path=1".to_string()],
            ..Default::default()
        })
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown config path"), "{msg}");
        assert!(msg.contains("nonsense.path"), "{msg}");

        // A near miss suggests the real path.
        let err = infr_core::config::ConfigLayer::cli(&ConfigOverrides {
            sets: vec!["kv.slot=2".to_string()],
            ..Default::default()
        })
        .unwrap_err();
        assert!(format!("{err}").contains("kv.slots"), "{err}");
    }

    /// A `bench` invocation with everything at its clap default — the shape `cli_flag_layer` sees.
    fn bench_cmd() -> Cmd {
        Cmd::Bench {
            model: "m".to_string(),
            n_prompt: 512,
            n_gen: 0,
            depth: 0,
            pg: None,
            reps: 3,
            ngl: 999,
            json: false,
            device: device_opts(None),
        }
    }

    fn no_sampling_opts() -> SamplingOpts {
        SamplingOpts {
            temp: None,
            top_k: None,
            top_p: None,
            seed: None,
            max_new: None,
            no_think: false,
            think: false,
        }
    }

    // ── finding 3: local `.gguf` FILE vs HF ref classifier ──────────────────────────────────────
    #[test]
    fn is_local_gguf_requires_an_existing_gguf_file() {
        // Unique scratch dir (no tempfile dep); cleaned at the end.
        let dir = std::env::temp_dir().join(format!(
            "infr-cli-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let gguf = dir.join("model.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        assert!(is_local_gguf(&gguf), "an existing .gguf file is local");
        // Case-insensitive extension.
        let gguf_up = dir.join("model.GGUF");
        std::fs::write(&gguf_up, b"x").unwrap();
        assert!(is_local_gguf(&gguf_up));

        // A directory (even named `*.gguf`) is NOT local → falls through to the HF ref path.
        let dir_gguf = dir.join("weights.gguf");
        std::fs::create_dir_all(&dir_gguf).unwrap();
        assert!(!is_local_gguf(&dir_gguf));
        // A non-.gguf file is not local.
        let bin = dir.join("model.bin");
        std::fs::write(&bin, b"x").unwrap();
        assert!(!is_local_gguf(&bin));
        // A missing path is not local (a typo'd path → clearer error, not a network pull).
        assert!(!is_local_gguf(&dir.join("nope.gguf")));
        // An `org/repo` HF ref is obviously not a local file.
        assert!(!is_local_gguf(Path::new("qwen/Qwen3-8B")));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── finding 4: NaN-safe sweep ranking ───────────────────────────────────────────────────────
    #[test]
    fn nan_safe_ratio_cmp_does_not_panic_and_sinks_nan() {
        use std::cmp::Ordering;
        assert_eq!(nan_safe_ratio_cmp(0.5, 1.5), Ordering::Less);
        assert_eq!(nan_safe_ratio_cmp(2.0, 1.0), Ordering::Greater);
        // A malformed value (0.0/0.0 = NaN) must NOT panic the sort and must sort to the end.
        let mut v = [1.0f64, f64::NAN, 0.5, 2.0];
        v.sort_by(|a, b| nan_safe_ratio_cmp(*a, *b));
        assert_eq!(v[0], 0.5);
        assert_eq!(v[1], 1.0);
        assert_eq!(v[2], 2.0);
        assert!(v[3].is_nan(), "NaN sinks to the end instead of aborting");
    }

    #[test]
    fn arch_sampling_table_is_family_specific() {
        // Qwen3.x thinking rec; also the fallback for any unknown arch.
        assert_eq!(arch_sampling(QWEN3), (0.6, 20, 0.95));
        assert_eq!(arch_sampling(QWEN35_MOE), (0.6, 20, 0.95));
        assert_eq!(arch_sampling("some-future-arch"), (0.6, 20, 0.95));
        // Qwen2/2.5 runs hotter with a tighter nucleus.
        assert_eq!(arch_sampling(QWEN2), (0.7, 20, 0.8));
        // Gemma: high temp, wide top_k.
        assert_eq!(arch_sampling(GEMMA4), (1.0, 64, 0.95));
        // Llama: top_k off (0 = keep all), top_p 0.9.
        assert_eq!(arch_sampling(LLAMA), (0.6, 0, 0.9));
    }
}
