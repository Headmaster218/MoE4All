//! Graphical launch parameters: the knob tree, the three presets, and the engine child process.
//!
//! The knob tree comes from `infr config-schema`, which needs no GPU, no model and no config file —
//! that is what makes it possible to edit launch parameters while the engine is stopped. Anything
//! the startup wizard can set is listed here explicitly; the rest of the schema falls through to
//! the advanced section, so a knob added to the engine shows up without a console change.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::state::Console;

/// Config paths `scripts/infr-wizard.ps1` sets. Kept as an explicit list rather than a section
/// prefix: the wizard's "常用" screen is a curated subset, and `kernels.*` alone is 187 knobs.
pub const SCRIPT_KNOBS: &[&str] = &[
    "device.dev",
    "device.ctx",
    "device.ubatch",
    "device.threads",
    "device.auto_profile",
    "device.vram_budget",
    "device.vram_reserve",
    "device.ram_budget",
    "device.submit_dispatches",
    "kv.type_k",
    "kv.type_v",
    "kv.overflow",
    "kv.overflow_vram_mb",
    "kv.overflow_reserve_mb",
    "kv.paged",
    "kv.paged_page_tokens",
    "kv.paged_cache_max",
    "kv.paged_elastic",
    "kv.paged_yield_under_pressure",
    "kv.ram_cache_max",
    "kv.session_cache_dir",
    "kv.session_idle_secs",
    "kv.session_cache_max",
    "kv.session_cache_ttl_hours",
    "paging.cache",
    "paging.host_dma",
    "paging.ring",
    "paging.ring_slots",
    "paging.stats",
    "prof.pager_profile",
    "prof.stages",
    "prof.vram",
    "kernels.vulkan.cpu_miss_threads",
    "kernels.vulkan.cpu_miss_max",
    "kernels.vulkan.cpu_miss_push",
    "kernels.vulkan.cpu_miss_host_result",
    "kernels.vulkan.cpu_miss_token_park",
    "kernels.vulkan.cpu_miss_spin",
    "spec.mtp",
    "sampling.temp",
    "sampling.top_k",
    "sampling.top_p",
    "sampling.seed",
    "sampling.max_new",
    "sampling.no_think",
    "sampling.reasoning_effort",
    "serve.api_key",
    "vision.image_url_allow_private",
    "uncensor.vector",
];

/// The 全手动 next-level menu: memory geometry and diagnostics. These are the knobs the wizard only
/// reveals after the operator picks manual, because a wrong value here is an OOM, not a slow run.
pub const MANUAL_KNOBS: &[&str] = &[
    "device.vram_budget",
    "device.vram_reserve",
    "device.ram_budget",
    "device.mem_guard_headroom",
    "device.mem_guard_ram_headroom",
    "device.mem_guard_poll_ms",
    "device.ubatch",
    "device.ubatch_parallel",
    "device.submit_dispatches",
    "device.subgroup_pref",
    "device.threads",
    "paging.cache",
    "paging.host_dma",
    "paging.ring",
    "paging.ring_slots",
    "paging.stats",
    "kv.overflow",
    "kv.overflow_vram_mb",
    "kv.overflow_reserve_mb",
    "kv.paged_page_tokens",
    "kv.paged_cache_max",
    "kv.ram_cache_max",
    "prof.pager_profile",
    "prof.stages",
    "prof.vram",
];

/// Options that exist ONLY as CLI flags — they have no config path, so `--set` cannot express them.
/// Surfaced with a `cli:` prefix so the form treats them like any other field.
struct CliFlag {
    path: &'static str,
    flag: &'static str,
    ty: &'static str,
    default: &'static str,
    doc: &'static str,
}

const CLI_FLAGS: &[CliFlag] = &[
    CliFlag {
        path: "cli:model",
        flag: "",
        ty: "String",
        default: "",
        doc: "聊天模型：GGUF 路径或 org/repo[:quant]。留空则只服务嵌入/语音。",
    },
    CliFlag {
        path: "cli:config",
        flag: "--config",
        ty: "PathBuf",
        default: "",
        doc: "TOML 配置文件。留空按 ./infr.toml → $XDG_CONFIG_HOME/infr/config.toml → ~/.config/infr/config.toml 查找。",
    },
    CliFlag {
        path: "cli:addr",
        flag: "--addr",
        ty: "String",
        default: "127.0.0.1:8080",
        doc: "引擎监听地址。控制台会把转发目标同步改到这里。",
    },
    CliFlag {
        path: "cli:parallel",
        flag: "--parallel",
        ty: "usize",
        default: "",
        doc: "并发生成槽数（llama-server 的 -np）。留空 = 1 槽，单请求独占整个显存拟合窗口。",
    },
    CliFlag {
        path: "cli:mmproj",
        flag: "--mmproj",
        ty: "PathBuf",
        default: "",
        doc: "Qwen3.8 视觉投影 GGUF。填了才接受图片部分。",
    },
    CliFlag {
        path: "cli:embedding-model",
        flag: "--embedding-model",
        ty: "String",
        default: "",
        doc: "原生 GGUF 嵌入模型，挂在 /v1/embeddings。",
    },
    CliFlag {
        path: "cli:embedding-idle-timeout",
        flag: "--embedding-idle-timeout",
        ty: "u64",
        default: "300",
        doc: "嵌入权重空闲多少秒后卸载；0 =常驻到服务关闭。",
    },
    CliFlag {
        path: "cli:embedding-runner",
        flag: "--embedding-runner",
        ty: "PathBuf",
        default: "",
        doc: "改用 llama-server 兼容 worker 执行嵌入，而不是原生执行。",
    },
    CliFlag {
        path: "cli:transcription-model",
        flag: "--transcription-model",
        ty: "PathBuf",
        default: "",
        doc: "原生转写 GGUF，挂在 /v1/audio/transcriptions，首次请求时加载。",
    },
    CliFlag {
        path: "cli:transcription-dev",
        flag: "--transcription-dev",
        ty: "String",
        default: "cpu",
        doc: "转写后端，目前只支持 cpu。",
    },
    CliFlag {
        path: "cli:transcription-threads",
        flag: "--transcription-threads",
        ty: "usize",
        default: "4",
        doc: "转写 CPU 线程数。",
    },
    CliFlag {
        path: "cli:transcription-idle-timeout",
        flag: "--transcription-idle-timeout",
        ty: "u64",
        default: "300",
        doc: "转写权重空闲卸载秒数；0 =常驻。",
    },
    CliFlag {
        path: "cli:transcription-upload-dir",
        flag: "--transcription-upload-dir",
        ty: "PathBuf",
        default: "",
        doc: "受控上传目录。留空 = <安装目录>/data/audio-upload。",
    },
    CliFlag {
        path: "cli:speech-model",
        flag: "--speech-model",
        ty: "PathBuf",
        default: "",
        doc: "原生 Melo GGUF，挂在 /v1/audio/speech（需要 tts 构建）。",
    },
    CliFlag {
        path: "cli:speech-lexicon",
        flag: "--speech-lexicon",
        ty: "PathBuf",
        default: "",
        doc: "外部资产目录，需含 lexicon.txt / tokens.txt / config.json。",
    },
    CliFlag {
        path: "cli:speech-threads",
        flag: "--speech-threads",
        ty: "usize",
        default: "4",
        doc: "语音合成 CPU 线程数。",
    },
    CliFlag {
        path: "cli:speech-idle-timeout",
        flag: "--speech-idle-timeout",
        ty: "u64",
        default: "300",
        doc: "语音权重空闲卸载秒数；0 =常驻。",
    },
    CliFlag {
        path: "cli:speech-max-output-seconds",
        flag: "--speech-max-output-seconds",
        ty: "f32",
        default: "30",
        doc: "单段合成音频的硬上限秒数（1..=120）。",
    },
];

/// Read the knob tree from the engine binary. Cached: it is compile-time information, so one read
/// per console process is enough, and a stopped engine must still be configurable.
pub async fn schema(console: &Arc<Console>) -> anyhow::Result<serde_json::Value> {
    if let Some(cached) = console.schema.lock().unwrap().clone() {
        return Ok(cached);
    }
    let binary = console.binary.clone();
    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new(&binary).arg("config-schema").output()
    })
    .await??;
    if !output.status.success() {
        anyhow::bail!(
            "`{} config-schema` 失败：{}",
            console.binary.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let mut payload: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let fields = payload
        .get_mut("schema")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| anyhow::anyhow!("config-schema 输出里没有 schema 数组"))?;
    for flag in CLI_FLAGS {
        fields.push(serde_json::json!({
            "path": flag.path, "ty": flag.ty, "default": flag.default, "doc": flag.doc,
            "flag": flag.flag,
        }));
    }
    let script: Vec<&str> = SCRIPT_KNOBS
        .iter()
        .copied()
        .chain(CLI_FLAGS.iter().map(|f| f.path))
        .collect();
    payload["script_knobs"] = serde_json::json!(script);
    payload["manual_knobs"] = serde_json::json!(MANUAL_KNOBS);
    payload["binary"] = serde_json::json!(console.binary.display().to_string());
    payload["shutdown_file"] = serde_json::json!(shutdown_path(console).display().to_string());
    *console.schema.lock().unwrap() = Some(payload.clone());
    Ok(payload)
}

/// The console owns `serve.shutdown_file`: it is how "停止引擎" asks for a graceful exit instead of
/// killing the process mid-token.
fn shutdown_path(console: &Console) -> PathBuf {
    console
        .state_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("engine.shutdown")
}

#[derive(Debug, Deserialize)]
pub struct LaunchRequest {
    #[serde(default = "default_preset")]
    pub preset: String,
    /// path -> value, only the fields the operator touched.
    #[serde(default)]
    pub set: BTreeMap<String, String>,
    /// Risk acknowledgement, required when the launch names a refusal-direction vector. Enforced
    /// here as well as in the browser so the gate is not merely a dialog someone can skip.
    #[serde(default)]
    pub consent: bool,
}

fn default_preset() -> String {
    "conservative".into()
}

/// Turn the form into an argv. Split out from the spawn so the command line the UI previews is the
/// one that actually runs, and so it can be tested without a GPU. `known` is every path the engine
/// advertised — the curated lists are a subset of it, not the whole legal set.
pub fn build_args(
    req: &LaunchRequest,
    shutdown_file: &std::path::Path,
    known: &std::collections::HashSet<String>,
) -> anyhow::Result<Vec<String>> {
    if !matches!(req.preset.as_str(), "conservative" | "aggressive" | "manual") {
        anyhow::bail!("未知的配置方式：{}", req.preset);
    }
    let mut args = vec!["serve".to_string()];
    let mut model: Option<String> = None;
    let mut sets: Vec<(String, String)> = Vec::new();
    for (path, value) in &req.set {
        if value.chars().any(|c| c.is_control()) {
            anyhow::bail!("{path} 的值里有控制字符");
        }
        if path == "serve.shutdown_file" {
            // The console owns this one: it is how "停止引擎" asks for a graceful exit.
            continue;
        }
        if let Some(flag) = CLI_FLAGS.iter().find(|f| f.path == path) {
            if value.is_empty() || value == flag.default {
                continue;
            }
            if flag.flag.is_empty() {
                model = Some(value.clone());
            } else {
                args.push(flag.flag.to_string());
                args.push(value.clone());
            }
            continue;
        }
        // An unknown path is a HARD error in the engine; saying so here keeps the message in the
        // operator's language instead of a clap usage dump in the log view.
        if !known.contains(path) {
            anyhow::bail!("未知配置项：{path}");
        }
        sets.push((path.clone(), value.clone()));
    }
    if let Some(model) = model {
        args.push(model);
    }
    if req.preset != "manual" {
        args.push("--set".into());
        args.push(format!("device.auto_profile={}", req.preset));
    }
    for (path, value) in sets {
        if path == "device.auto_profile" && req.preset != "manual" {
            continue;
        }
        args.push("--set".into());
        args.push(format!("{path}={value}"));
    }
    args.push("--set".into());
    args.push(format!("serve.shutdown_file={}", shutdown_file.display()));
    Ok(args)
}

/// Every path the engine advertised, for argv validation.
pub fn schema_paths(payload: &serde_json::Value) -> std::collections::HashSet<String> {
    payload
        .get("schema")
        .and_then(|v| v.as_array())
        .map(|fields| {
            fields
                .iter()
                .filter_map(|f| f.get("path").and_then(|p| p.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

pub async fn start(console: &Arc<Console>, req: LaunchRequest) -> anyhow::Result<serde_json::Value> {
    if console.child.lock().unwrap().is_some() {
        anyhow::bail!("引擎已由控制台启动，先停止再改参数");
    }
    if req.set.get("uncensor.vector").is_some_and(|v| !v.is_empty()) && !req.consent {
        anyhow::bail!("开启去审查投影需要先勾选风险同意");
    }
    let payload = schema(console).await?;
    let known = schema_paths(&payload);
    let dir = console
        .state_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&dir)?;
    let shutdown_file = shutdown_path(console);
    let _ = std::fs::remove_file(&shutdown_file);
    let args = build_args(&req, &shutdown_file, &known)?;

    let mut command = tokio::process::Command::new(&console.binary);
    command
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(|e| anyhow::anyhow!("启动 {} 失败：{e}", console.binary.display()))?;
    let pid = child.id();

    if let Some(stdout) = child.stdout.take() {
        pump(console.clone(), stdout);
    }
    if let Some(stderr) = child.stderr.take() {
        pump(console.clone(), stderr);
    }

    // Retarget the data path at whatever address this launch asked for.
    let addr = req
        .set
        .get("cli:addr")
        .filter(|v| !v.is_empty())
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:8080".into());
    *console.upstream.lock().unwrap() = format!("http://{addr}");
    let key = req.set.get("serve.api_key").filter(|v| !v.is_empty()).cloned();
    *console.upstream_key.lock().unwrap() = key;
    *console.shutdown_file.lock().unwrap() = Some(shutdown_file);
    *console.child.lock().unwrap() = Some(child);
    if req.consent {
        // The operator just acknowledged the risk for this launch; the hot toggle must not ask again
        // for the same decision.
        console.uncensor.lock().unwrap().consented = true;
    }
    {
        let mut engine = console.engine.lock().unwrap();
        engine.managed = true;
        engine.running = true;
        engine.pid = pid;
    }
    console.log(format!("引擎已启动 pid={pid:?}：{} {}", console.binary.display(), args.join(" ")));
    Ok(serde_json::json!({ "pid": pid, "args": args }))
}

/// Engine stdout/stderr into the log view. Both streams are kept: the startup banner goes to one
/// and tracing to the other, and an operator debugging a boot failure needs both in order.
fn pump<R>(console: Arc<Console>, reader: R)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            console.engine_log_line(line);
        }
    });
}

/// Graceful stop: the engine polls `serve.shutdown_file` and exits on its own, finishing in-flight
/// work. Killing the child would drop requests mid-token and leave the KV session cache half-written.
pub async fn stop(console: &Arc<Console>) -> anyhow::Result<()> {
    let file = console
        .shutdown_file
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("引擎不是控制台启动的，无法从这里停止"))?;
    std::fs::write(&file, b"stop\n")?;
    console.log("已写入停止标志，等待引擎自行退出");
    Ok(())
}

/// Reap a finished child so the next launch is allowed and the pill stops saying "running".
pub fn reap(console: &Arc<Console>) {
    let mut slot = console.child.lock().unwrap();
    let Some(child) = slot.as_mut() else { return };
    match child.try_wait() {
        Ok(Some(status)) => {
            console.log(format!("引擎已退出：{status}"));
            *slot = None;
            drop(slot);
            console.cancel_streaming();
            let mut engine = console.engine.lock().unwrap();
            engine.running = false;
            engine.pid = None;
        }
        Ok(None) => {}
        Err(e) => {
            console.log(format!("等待引擎退出失败：{e}"));
            *slot = None;
        }
    }
}