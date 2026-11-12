//! Shared console state: API keys, per-round records, the speed series and the engine handle.
//!
//! Everything here is derived from what the proxy sees on the wire plus the engine's
//! `/internal/stats` poll. Nothing is inferred from engine internals, so the console keeps working
//! against any OpenAI-compatible upstream — the resource tiles simply go absent.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Rounds kept for the history list. Older ones are dropped whole, frames and all.
const ROUND_CAP: usize = 500;
/// SSE frames kept per round. A long generation exceeds this; the record says so rather than
/// silently pretending the captured prefix is the whole reply.
pub(crate) const FRAME_CAP: usize = 4000;
/// 1 Hz samples retained for the speed chart (3 minutes).
const SERIES_CAP: usize = 180;
/// Engine stdout lines retained for the log view.
const LOG_CAP: usize = 400;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Streaming,
    Done,
    Error,
    Cancelled,
}

/// The three decode states the operator asked to tell apart, plus the two that precede them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Prefill,
    Thinking,
    ToolCall,
    Output,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub label: String,
    /// The token a CLIENT presents to the console. The upstream engine's own key is separate and
    /// never leaves the console.
    pub token: String,
    #[serde(default)]
    pub requests: u64,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub ttft_sum_ms: u64,
    #[serde(default)]
    pub ttft_count: u64,
    #[serde(default)]
    pub prefill_sum: f64,
    #[serde(default)]
    pub prefill_count: u64,
    #[serde(default)]
    pub decode_sum: f64,
    #[serde(default)]
    pub decode_count: u64,
}

/// One key's counters as the UI reads them. Averages are computed here so a key with no samples
/// renders as absent rather than as a plausible zero.
#[derive(Serialize)]
pub struct KeyView {
    pub id: String,
    pub label: String,
    pub token: String,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub ttft_avg_ms: Option<u64>,
    pub prefill_avg_tps: Option<f64>,
    pub decode_avg_tps: Option<f64>,
}

impl ApiKey {
    pub fn view(&self) -> KeyView {
        let avg = |sum: f64, n: u64| (n > 0).then(|| sum / n as f64);
        KeyView {
            id: self.id.clone(),
            label: self.label.clone(),
            token: self.token.clone(),
            requests: self.requests,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            cached_tokens: self.cached_tokens,
            ttft_avg_ms: (self.ttft_count > 0).then(|| self.ttft_sum_ms / self.ttft_count),
            prefill_avg_tps: avg(self.prefill_sum, self.prefill_count),
            decode_avg_tps: avg(self.decode_sum, self.decode_count),
        }
    }

    pub fn fold(&mut self, r: &Round) {
        self.requests += 1;
        self.prompt_tokens += r.prompt_tokens;
        self.completion_tokens += r.completion_tokens;
        self.cached_tokens += r.cached_prompt_tokens;
        if let Some(ttft) = r.ttft_ms {
            self.ttft_sum_ms += ttft;
            self.ttft_count += 1;
        }
        if let Some(v) = r.prefill_tps {
            self.prefill_sum += v;
            self.prefill_count += 1;
        }
        if let Some(v) = r.decode_tps {
            self.decode_sum += v;
            self.decode_count += 1;
        }
    }
}

/// Proxy-side prefix comparison against the previous turn of the same conversation. An ESTIMATE:
/// the console cannot tokenize, so it counts messages and characters. The engine's own
/// `cached_prompt_tokens` is the authoritative number and is shown beside it, never replaced by it.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ReuseEst {
    pub messages: usize,
    pub chars: usize,
    pub ratio: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Frame {
    pub seq: u64,
    pub data: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Round {
    pub id: u64,
    pub key_id: String,
    pub key_label: String,
    pub model: String,
    pub api: &'static str,
    pub started_at_ms: u64,
    pub ended_at_ms: Option<u64>,
    pub status: Status,
    pub phase: Phase,
    pub tools: Vec<String>,
    pub uncensor: bool,
    pub prompt_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub completion_tokens: u64,
    pub reuse_est: Option<ReuseEst>,
    pub ttft_ms: Option<u64>,
    /// Rate over the deltas seen SO FAR, so a running turn shows a speed instead of a blank.
    /// `finish` promotes it to `decode_tps`.
    pub live_tps: Option<f64>,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    pub duration_ms: Option<u64>,
    pub frames_truncated: bool,
    /// The request body with base64 image payloads elided — a screenshot turn would otherwise
    /// put tens of megabytes of pixels in every history payload.
    pub request: serde_json::Value,
    #[serde(skip)]
    pub frames: Vec<Frame>,
    #[serde(skip)]
    pub first_delta_ms: Option<u64>,
    #[serde(skip)]
    pub prefill_ms: Option<u64>,
}

impl Round {
    /// Record one decoded piece. `kind` decides the tri-state; a tool name is kept once.
    pub fn observe(&mut self, kind: DeltaKind, tool: Option<&str>, tokens: u64) {
        let now = now_ms();
        if self.first_delta_ms.is_none() {
            self.first_delta_ms = Some(now);
            self.ttft_ms = Some(now.saturating_sub(self.started_at_ms));
        }
        self.completion_tokens += tokens;
        // The first token is not part of the inter-token interval; counting it would inflate the
        // rate by one token over a window that never contained it.
        let span = now.saturating_sub(self.first_delta_ms.unwrap_or(now));
        let counted = self.completion_tokens.saturating_sub(1);
        if span > 0 && counted > 0 {
            self.live_tps = Some(counted as f64 * 1000.0 / span as f64);
        }
        self.phase = match kind {
            DeltaKind::Reasoning => Phase::Thinking,
            DeltaKind::Tool => Phase::ToolCall,
            DeltaKind::Content => Phase::Output,
        };
        if let Some(name) = tool {
            if !name.is_empty() && !self.tools.iter().any(|t| t == name) {
                self.tools.push(name.to_string());
            }
        }
    }

    pub fn finish(&mut self, status: Status) {
        let now = now_ms();
        self.ended_at_ms = Some(now);
        self.duration_ms = Some(now.saturating_sub(self.started_at_ms));
        self.status = status;
        self.decode_tps = self.live_tps;
        if let Some(prefill) = self.prefill_ms {
            let evaluated = self.prompt_tokens.saturating_sub(self.cached_prompt_tokens);
            if prefill > 0 && evaluated > 0 {
                self.prefill_tps = Some(evaluated as f64 * 1000.0 / prefill as f64);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaKind {
    Reasoning,
    Tool,
    Content,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Series {
    pub t: Vec<u64>,
    pub decode: Vec<f64>,
    pub prefill: Vec<f64>,
}

impl Series {
    pub fn push(&mut self, t: u64, decode: f64, prefill: f64) {
        self.t.push(t);
        self.decode.push(decode);
        self.prefill.push(prefill);
        if self.t.len() > SERIES_CAP {
            self.t.remove(0);
            self.decode.remove(0);
            self.prefill.remove(0);
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct UncensorState {
    pub default_on: bool,
    pub consented: bool,
    /// Whether the ENGINE loaded a projection vector. Without it the toggle is inert, and the UI
    /// must say so rather than let an operator believe they turned something on.
    pub vector_loaded: bool,
}

/// What the console knows about the engine process. `managed` distinguishes "we started it" from
/// "we attached to one somebody else started" — only the former may be stopped from the UI.
#[derive(Clone, Debug, Default, Serialize)]
pub struct EngineView {
    pub managed: bool,
    pub running: bool,
    pub reachable: bool,
    pub pid: Option<u32>,
    pub stats: Option<serde_json::Value>,
    pub log_tail: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FrameEvent {
    pub round: u64,
    pub seq: u64,
    pub data: String,
}

pub struct Console {
    /// Where the proxy forwards. Behind a mutex because launching the engine from the UI on a
    /// different `--addr` has to retarget the data path, not just the command line.
    pub upstream: Mutex<String>,
    pub upstream_key: Mutex<Option<String>>,
    pub binary: std::path::PathBuf,
    pub state_path: std::path::PathBuf,
    pub http: reqwest::Client,
    pub keys: Mutex<Vec<ApiKey>>,
    pub rounds: Mutex<VecDeque<Round>>,
    pub series: Mutex<Series>,
    pub uncensor: Mutex<UncensorState>,
    pub engine: Mutex<EngineView>,
    pub engine_log: Mutex<VecDeque<String>>,
    pub child: Mutex<Option<tokio::process::Child>>,
    pub shutdown_file: Mutex<Option<std::path::PathBuf>>,
    pub schema: Mutex<Option<serde_json::Value>>,
    pub conversations: Mutex<HashMap<String, Vec<String>>>,
    pub events: broadcast::Sender<Arc<FrameEvent>>,
    /// Live UI subscribers. Frame broadcasting is skipped entirely at zero, so a headless proxy
    /// pays nothing for a dashboard nobody is watching.
    pub subscribers: AtomicUsize,
    pub next_round: AtomicU64,
    pub console_log: Mutex<VecDeque<String>>,
}

impl Console {
    pub fn upstream(&self) -> String {
        self.upstream.lock().unwrap().clone()
    }

    pub fn upstream_key(&self) -> Option<String> {
        self.upstream_key.lock().unwrap().clone()
    }

    pub fn log(&self, line: impl Into<String>) {
        let line = format!("{} {}", clock_string(), line.into());
        let mut log = self.console_log.lock().unwrap();
        log.push_back(line);
        while log.len() > LOG_CAP {
            log.pop_front();
        }
    }

    pub fn engine_log_line(&self, line: String) {
        let mut log = self.engine_log.lock().unwrap();
        log.push_back(line);
        while log.len() > LOG_CAP {
            log.pop_front();
        }
    }

    pub fn broadcast_frame(&self, round: u64, seq: u64, data: String) {
        if self.subscribers.load(Ordering::Relaxed) == 0 {
            return;
        }
        let _ = self.events.send(Arc::new(FrameEvent { round, seq, data }));
    }

    /// Longest common message prefix with the previous turn of this conversation, in messages and
    /// characters. The conversation key is supplied by the caller (see `proxy::conversation_key`).
    pub fn estimate_reuse(&self, conv: &str, messages: &[String]) -> Option<ReuseEst> {
        let mut convs = self.conversations.lock().unwrap();
        let previous = convs.insert(conv.to_string(), messages.to_vec());
        let previous = previous?;
        let total: usize = messages.iter().map(|m| m.chars().count()).sum();
        if total == 0 {
            return None;
        }
        let mut matched = 0;
        let mut chars = 0;
        for (a, b) in previous.iter().zip(messages) {
            if a != b {
                break;
            }
            matched += 1;
            chars += b.chars().count();
        }
        Some(ReuseEst { messages: matched, chars, ratio: chars as f64 / total as f64 })
    }

    pub fn round_snapshot(&self, id: u64) -> Option<(Round, Vec<Frame>)> {
        let rounds = self.rounds.lock().unwrap();
        rounds.iter().find(|r| r.id == id).map(|r| (r.clone(), r.frames.clone()))
    }

    pub fn push_round(&self, round: Round) {
        let mut rounds = self.rounds.lock().unwrap();
        rounds.push_front(round);
        while rounds.len() > ROUND_CAP {
            rounds.pop_back();
        }
    }

    /// Mutate one round in place. Returns false when it has already aged out of the window.
    pub fn with_round(&self, id: u64, f: impl FnOnce(&mut Round)) -> bool {
        let mut rounds = self.rounds.lock().unwrap();
        match rounds.iter_mut().find(|r| r.id == id) {
            Some(r) => {
                f(r);
                true
            }
            None => false,
        }
    }

    pub fn fold_key(&self, id: u64) {
        let round = {
            let rounds = self.rounds.lock().unwrap();
            rounds.iter().find(|r| r.id == id).cloned()
        };
        let Some(round) = round else { return };
        let mut keys = self.keys.lock().unwrap();
        if let Some(key) = keys.iter_mut().find(|k| k.id == round.key_id) {
            key.fold(&round);
        }
    }

    pub fn save_keys(&self) -> anyhow::Result<()> {
        let keys = self.keys.lock().unwrap().clone();
        let json = serde_json::to_string_pretty(&keys)?;
        if let Some(parent) = self.state_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.state_path, json)?;
        Ok(())
    }

    /// Close out rounds whose stream will never end because the engine process is gone. A history
    /// row stuck on "输出中" forever would be a lie about what happened.
    pub fn cancel_streaming(&self) {
        let mut rounds = self.rounds.lock().unwrap();
        for round in rounds.iter_mut() {
            if round.status == Status::Streaming {
                round.finish(Status::Cancelled);
            }
        }
    }
}

fn clock_string() -> String {
    let secs = now_ms() / 1000;
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}")
}

/// A fresh client token. `getrandom` is the OS CSPRNG — a guessable console token would let anyone
/// on the network spend the operator's GPU.
pub fn new_token(prefix: &str) -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("OS CSPRNG unavailable");
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("{prefix}{hex}")
}

/// Elide base64 payloads (screenshots) from a request body before it is retained.
pub fn redact_request(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, v) in map.iter_mut() {
                if key == "data" || key == "b64_json" {
                    if let serde_json::Value::String(s) = v {
                        if s.len() > 256 {
                            *v = serde_json::Value::String(format!("<{} base64 bytes elided>", s.len()));
                            continue;
                        }
                    }
                }
                redact_request(v);
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(redact_request),
        serde_json::Value::String(s) if s.starts_with("data:") && s.len() > 512 => {
            *s = format!("<data URL, {} bytes elided>", s.len());
        }
        _ => {}
    }
}