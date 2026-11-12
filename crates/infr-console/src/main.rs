//! INFR Console — a standalone reverse proxy with a dashboard.
//!
//! It sits between the client and a running `infr serve`, so it can measure what the engine cannot
//! see from outside (time to first token, which of the three decode states a turn is in, which tool
//! is being called) and account per client API key. It depends on NO engine crate: everything it
//! knows about resources comes from the engine's `/internal/stats`, so it builds and runs on a box
//! with no GPU, no Vulkan SDK and no model.

mod launch;
mod proxy;
mod state;
#[cfg(test)]
mod tests;
mod ui;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use tokio::sync::broadcast;

use state::{Console, EngineView, Series, UncensorState};

#[derive(Parser)]
#[command(name = "infr-console", version, about = "INFR 推理控制台：反向代理 + 仪表盘 + 图形化启动参数")]
struct Cli {
    /// Address the console listens on. Point your client here, not at the engine.
    #[arg(long, default_value = "127.0.0.1:8090")]
    listen: String,
    /// The engine's OpenAI-compatible base URL.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    upstream: String,
    /// The ENGINE's own API key (serve.api_key / INFR_API_KEY). Never sent to a client.
    #[arg(long)]
    upstream_key: Option<String>,
    /// Engine binary used by the launch page. Defaults to `infr` beside this executable.
    #[arg(long)]
    binary: Option<PathBuf>,
    /// Where client API keys and their counters are persisted.
    #[arg(long)]
    state: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();
    let cli = Cli::parse();

    let binary = cli.binary.unwrap_or_else(default_binary);
    let state_path = cli.state.unwrap_or_else(default_state_path);
    let keys = load_keys(&state_path);
    let (events, _) = broadcast::channel(4096);
    let console = Arc::new(Console {
        upstream: Mutex::new(cli.upstream.trim_end_matches('/').to_string()),
        upstream_key: Mutex::new(cli.upstream_key),
        binary,
        state_path,
        http: reqwest::Client::builder()
            .pool_max_idle_per_host(8)
            .build()?,
        keys: Mutex::new(keys),
        rounds: Mutex::new(Default::default()),
        series: Mutex::new(Series::default()),
        uncensor: Mutex::new(UncensorState::default()),
        engine: Mutex::new(EngineView::default()),
        engine_log: Mutex::new(Default::default()),
        child: Mutex::new(None),
        shutdown_file: Mutex::new(None),
        schema: Mutex::new(None),
        conversations: Mutex::new(HashMap::new()),
        events,
        subscribers: AtomicUsize::new(0),
        next_round: AtomicU64::new(1),
        console_log: Mutex::new(Default::default()),
    });
    console.log(format!("控制台启动，上游 {}", console.upstream()));

    tokio::spawn(poll_engine(Arc::clone(&console)));

    let app = ui::router(Arc::clone(&console));

    let listener = tokio::net::TcpListener::bind(&cli.listen).await?;
    println!("INFR Console  http://{}", listener.local_addr()?);
    println!("上游引擎      {}", console.upstream());
    println!("引擎二进制    {}", console.binary.display());
    axum::serve(listener, app).await?;
    Ok(())
}

fn default_binary() -> PathBuf {
    let name = if cfg!(windows) { "infr.exe" } else { "infr" };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join(name);
            if beside.is_file() {
                return beside;
            }
        }
    }
    PathBuf::from(name)
}

fn default_state_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("infr-console")
        .join("keys.json")
}

fn load_keys(path: &std::path::Path) -> Vec<state::ApiKey> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

/// 1 Hz engine poll. Reads `/internal/stats`, which is non-consuming: it must not steal the terminal
/// reporter's statistics window or the dashboard's decode counter.
async fn poll_engine(console: Arc<Console>) {
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    // (slot ordinal, req_id) -> prefill tokens at the previous sample. The engine reports prefill
    // PROGRESS, not a prefill rate, so the rate is the delta over the poll interval.
    let mut previous: HashMap<(usize, u64), u64> = HashMap::new();
    let mut last = Instant::now();
    loop {
        tick.tick().await;
        launch::reap(&console);
        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64().max(0.001);
        last = now;

        let url = format!("{}/internal/stats", console.upstream());
        let mut request = console.http.get(&url).timeout(Duration::from_secs(3));
        if let Some(key) = console.upstream_key() {
            request = request.bearer_auth(key);
        }
        let stats = match request.send().await {
            Ok(resp) if resp.status().is_success() => resp.json::<serde_json::Value>().await.ok(),
            _ => None,
        };

        let mut decode = 0.0f64;
        let mut prefill = 0.0f64;
        if let Some(stats) = &stats {
            let mut seen = HashMap::new();
            for slot in stats.get("slots").and_then(|v| v.as_array()).into_iter().flatten() {
                let Some(req) = slot.get("request") else { continue };
                let ordinal = slot.get("ordinal").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let req_id = req.get("req_id").and_then(|v| v.as_u64()).unwrap_or(0);
                let speed = req.get("tokens_per_second").and_then(|v| v.as_f64()).unwrap_or(0.0);
                decode = decode.max(speed);
                let tokens = req.get("prefill_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                seen.insert((ordinal, req_id), tokens);
                let phase = req.get("phase").and_then(|v| v.as_str()).unwrap_or("");
                if matches!(phase, "Starting" | "Prefill") {
                    if let Some(before) = previous.get(&(ordinal, req_id)) {
                        prefill = prefill.max(tokens.saturating_sub(*before) as f64 / dt);
                    }
                }
            }
            previous = seen;
            let vector_loaded = stats
                .pointer("/uncensor/vector_loaded")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            console.uncensor.lock().unwrap().vector_loaded = vector_loaded;
        } else {
            previous.clear();
        }

        {
            let mut engine = console.engine.lock().unwrap();
            engine.reachable = stats.is_some();
            engine.stats = stats;
            if !engine.managed {
                engine.running = engine.reachable;
            }
        }
        console.series.lock().unwrap().push(state::now_ms(), decode, prefill);
    }
}