//! Shared console state: API keys, per-round records, the speed series and the engine handle.
//!
//! Everything here is derived from what the proxy sees on the wire plus the engine's
//! `/internal/stats` poll. Nothing is inferred from engine internals, so the console keeps working
//! against any OpenAI-compatible upstream — the resource tiles simply go absent.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::auth::AuthState;
use crate::crypto::SecureStore;

/// Hard safety ceiling across every conversation. The per-conversation cap below is what
/// normally decides what stays; this only bounds the table if a client opens thousands of
/// distinct conversations.
const ROUND_CAP: usize = 1000;
/// Rounds kept for each conversation. Rounds past this are removed from the list outright —
/// not merely compacted — so an agent loop running for hundreds of turns cannot pile up.
const ROUNDS_PER_CONVERSATION: usize = 100;
/// SSE frames kept per round. A long generation exceeds this; the record says so rather than
/// silently pretending the captured prefix is the whole reply.
pub(crate) const FRAME_CAP: usize = 4000;
/// 1 Hz samples retained for the speed chart (3 minutes).
const SERIES_CAP: usize = 180;
/// Minimum decode window before publishing a live rate. A burst of SSE deltas can land in the same
/// millisecond, so the first inter-token windows are tiny and would flash hundreds of t/s for a model
/// that decodes at tens. The cumulative average is representative once this much time has elapsed.
const LIVE_TPS_MIN_SPAN_MS: u64 = 250;
/// Engine stdout lines retained for the log view.
const LOG_CAP: usize = 400;
/// Rounds that keep their full SSE frame log. Older, finished rounds are compacted into a few
/// synthetic frames carrying the assembled reply, so a long history cannot pin millions of
/// delta strings in memory. A currently streaming round is never compacted.
const FRAME_ROUND_CAP: usize = 48;
/// Distinct conversations kept in the prefix-reuse table. Beyond this the least-recently-written
/// conversation is evicted. The table only needs the previous turn to estimate reuse, so an
/// unbounded map (keyed by the full first message) was pure accumulation.
const CONVERSATION_CAP: usize = 256;
/// How many complete request messages are retained in each round's history record. Longer
/// conversations are replaced by a compact marker so memory does not grow quadratically with
/// context length (500 rounds each carrying the entire growing prompt).
const REQUEST_KEEP_MESSAGES: usize = 8;
/// Maximum characters of assembled reply text kept in a compacted frame. A pathological runaway
/// generation must not pin an unbounded string in memory either.
const COMPACT_REPLY_CAP: usize = 512 * 1024;

/// Fold a finished round's raw SSE deltas into one synthetic chat chunk. The sidebar's `assemble()`
/// reads `choices[].delta.{content,reasoning_content,tool_calls}`, so a single frame carrying the
/// concatenated text renders identically while releasing thousands of small frame strings.
fn compact_frames(frames: &[Frame]) -> Vec<Frame> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tools: Vec<(u64, String, String)> = Vec::new(); // (index, name, arguments)
    let mut seq = 0u64;

    for frame in frames {
        seq = seq.max(frame.seq);
        if frame.data == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&frame.data) else {
            continue;
        };

        // Chat Completions deltas: choices[].delta.{content,reasoning_content,tool_calls}.
        for choice in chunk
            .get("choices")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            let Some(delta) = choice.get("delta") else {
                continue;
            };
            if let Some(s) = delta.get("content").and_then(|v| v.as_str()) {
                append_capped(&mut content, s);
            }
            if let Some(s) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                append_capped(&mut reasoning, s);
            }
            for call in delta
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
            {
                let index = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let name = call
                    .pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let args = call
                    .pointer("/function/arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                push_tool(&mut tools, index, name, args);
            }
        }

        // Responses API events: fold their text/tool payloads into the same synthetic chat chunk
        // so `assemble()` renders an aged Responses round identically to a live one.
        match chunk.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "response.output_text.delta"
            | "response.reasoning_text.delta"
            | "response.reasoning_summary_text.delta" => {
                let s = chunk.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if ty_is_reasoning(&chunk) {
                    append_capped(&mut reasoning, s);
                } else {
                    append_capped(&mut content, s);
                }
            }
            "response.function_call_arguments.delta" => {
                let idx = chunk.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
                let index = stable_hash(idx) % 1024;
                push_tool(
                    &mut tools,
                    index,
                    "",
                    chunk.get("delta").and_then(|v| v.as_str()).unwrap_or(""),
                );
            }
            "response.output_item.done" => {
                if let Some(item) = chunk.get("item") {
                    if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let index = item
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(|s| stable_hash(s) % 1024)
                            .unwrap_or(0);
                        push_tool(&mut tools, index, name, "");
                    }
                }
            }
            _ => {}
        }
    }

    if content.is_empty() && reasoning.is_empty() && tools.is_empty() {
        return Vec::new();
    }
    let mut delta = serde_json::Map::new();
    if !content.is_empty() {
        delta.insert("content".into(), serde_json::Value::String(content));
    }
    if !reasoning.is_empty() {
        delta.insert(
            "reasoning_content".into(),
            serde_json::Value::String(reasoning),
        );
    }
    if !tools.is_empty() {
        tools.sort_by_key(|(i, _, _)| *i);
        let calls: Vec<serde_json::Value> = tools
            .into_iter()
            .map(|(index, name, args)| {
                serde_json::json!({
                    "index": index,
                    "function": { "name": name, "arguments": args },
                })
            })
            .collect();
        delta.insert("tool_calls".into(), serde_json::Value::Array(calls));
    }
    let chunk = serde_json::json!({ "choices": [{ "delta": delta }] });
    vec![Frame {
        seq: seq.saturating_add(1),
        data: chunk.to_string(),
    }]
}

fn ty_is_reasoning(chunk: &serde_json::Value) -> bool {
    matches!(
        chunk.get("type").and_then(|v| v.as_str()).unwrap_or(""),
        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta"
    )
}

fn push_tool(tools: &mut Vec<(u64, String, String)>, index: u64, name: &str, args: &str) {
    if let Some(slot) = tools.iter_mut().find(|(i, _, _)| *i == index) {
        slot.1.push_str(name);
        append_capped(&mut slot.2, args);
    } else {
        tools.push((index, name.to_string(), args.to_string()));
    }
}

fn append_capped(dst: &mut String, src: &str) {
    if dst.len() >= COMPACT_REPLY_CAP {
        return;
    }
    let room = COMPACT_REPLY_CAP - dst.len();
    if src.len() <= room {
        dst.push_str(src);
        return;
    }
    // Slice on a char boundary: a byte offset can land mid-codepoint for CJK text and panic.
    let mut end = room;
    while end > 0 && !src.is_char_boundary(end) {
        end -= 1;
    }
    dst.push_str(&src[..end]);
}

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

/// Compact stand-in for one message in the prefix-reuse table. Estimating reuse only needs to
/// compare successive messages for equality and sum their lengths; keeping the full text (a
/// growing system prompt plus the entire conversation history) was the reason memory climbed
/// with context size. A 64-bit digest is plenty for a best-effort estimate.
#[derive(Clone, Copy)]
struct MsgFingerprint {
    hash: u64,
    chars: usize,
}

fn stable_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Bounded prefix-reuse table. One fingerprint list per conversation, evicting the
/// least-recently-written conversation once [`CONVERSATION_CAP`] is reached. The map key is a
/// digest of the conversation anchor (which itself embeds the full first message), so nothing
/// here retains prompt text.
#[derive(Default)]
pub struct ConversationTable {
    entries: HashMap<u64, Vec<MsgFingerprint>>,
    /// Most-recently-written first; used only to pick the eviction victim.
    lru: VecDeque<u64>,
}

impl ConversationTable {
    fn estimate(&mut self, conv: &str, messages: &[String]) -> Option<ReuseEst> {
        let key = stable_hash(conv);
        let current: Vec<MsgFingerprint> = messages
            .iter()
            .map(|m| MsgFingerprint {
                hash: stable_hash(m),
                chars: m.chars().count(),
            })
            .collect();
        let previous = self.entries.insert(key, current.clone());

        // Refresh recency: drop any stale occurrence, then put this conversation at the front.
        self.lru.retain(|&k| k != key);
        self.lru.push_front(key);
        while self.entries.len() > CONVERSATION_CAP {
            match self.lru.pop_back() {
                Some(old) => {
                    self.entries.remove(&old);
                }
                None => break,
            }
        }

        let previous = previous?;
        let total: usize = current.iter().map(|f| f.chars).sum();
        if total == 0 {
            return None;
        }
        let mut matched = 0;
        let mut chars = 0;
        for (a, b) in previous.iter().zip(current.iter()) {
            if a.hash != b.hash {
                break;
            }
            matched += 1;
            chars += b.chars;
        }
        Some(ReuseEst {
            messages: matched,
            chars,
            ratio: chars as f64 / total as f64,
        })
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
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
    /// Stable session id (a compact hash of the conversation anchor) so the dashboard can group an
    /// agent's many turns into one conversation window instead of one window per request.
    pub conv: String,
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
    /// Set when the raw delta frames of an aged round were folded into a single assembled frame.
    #[serde(skip)]
    pub frames_compacted: bool,
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
        if span >= LIVE_TPS_MIN_SPAN_MS && counted > 0 {
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

/// The operator's saved launch settings: the preset and the knob values they touched. Persisted
/// beside the keys file so a console restart — or a browser reload — restores the form exactly as
/// it was left, instead of silently falling back to schema defaults. Only touched values are kept;
/// everything else is re-derived from the engine's own schema at launch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "default_settings_preset")]
    pub preset: String,
    #[serde(default)]
    pub set: BTreeMap<String, String>,
}

fn default_settings_preset() -> String {
    "conservative".into()
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            preset: default_settings_preset(),
            set: BTreeMap::new(),
        }
    }
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
    /// Encrypted local state store for keys, settings and login records.
    pub store: std::sync::Arc<SecureStore>,
    /// Login records and in-memory sessions. Never serialized directly.
    pub auth: Mutex<AuthState>,
    /// True when the listener terminates TLS, which adds `Secure` to session cookies and HSTS.
    pub tls: bool,
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
    pub conversations: Mutex<ConversationTable>,
    pub events: broadcast::Sender<Arc<FrameEvent>>,
    /// Pinged when a round finishes so the dashboard pushes a fresh snapshot at once instead of
    /// waiting for the next tick — a completed turn must read 空闲 immediately, not a second later.
    pub state_dirty: broadcast::Sender<()>,
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
        // Durable copy first: the in-memory ring below only keeps the last LOG_CAP lines, so a
        // crash that scrolls past them (or a restart) would otherwise lose the evidence entirely.
        self.append_engine_log_file(&line);
        let mut log = self.engine_log.lock().unwrap();
        log.push_back(line);
        while log.len() > LOG_CAP {
            log.pop_front();
        }
    }

    /// Engine log file beside the keys/settings file. Appended per line and rotated at a bounded
    /// size (one backup kept) so a long-running serve can't fill the disk, yet a TDR/device-lost
    /// crash stays readable after the fact instead of vanishing with the 400-line UI ring.
    pub fn engine_log_path(&self) -> std::path::PathBuf {
        self.state_path
            .parent()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("engine.log")
    }

    fn append_engine_log_file(&self, line: &str) {
        const ROTATE_BYTES: u64 = 8 * 1024 * 1024;
        let path = self.engine_log_path();
        if let Ok(meta) = std::fs::metadata(&path) {
            if meta.len() > ROTATE_BYTES {
                let _ = std::fs::rename(&path, path.with_extension("log.1"));
            }
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
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
        self.conversations.lock().unwrap().estimate(conv, messages)
    }

    pub fn round_snapshot(&self, id: u64) -> Option<(Round, Vec<Frame>)> {
        let rounds = self.rounds.lock().unwrap();
        rounds
            .iter()
            .find(|r| r.id == id)
            .map(|r| (r.clone(), r.frames.clone()))
    }

    pub fn push_round(&self, round: Round) {
        let mut rounds = self.rounds.lock().unwrap();
        let conv = round.conv.clone();
        rounds.push_front(round);
        while rounds.len() > ROUND_CAP {
            rounds.pop_back();
        }
        // Per-conversation cap: drop this conversation's oldest turns from the list entirely.
        // `rounds` is newest-first, so once a conv has more than the cap, delete every later
        // entry belonging to it. Other conversations must not be disturbed.
        let mut seen = 0usize;
        let mut stale = Vec::new();
        for (i, r) in rounds.iter().enumerate() {
            if r.conv != conv {
                continue;
            }
            seen += 1;
            if seen > ROUNDS_PER_CONVERSATION {
                stale.push(i);
            }
        }
        for i in stale.into_iter().rev() {
            rounds.remove(i);
        }
        // Newest rounds keep their raw delta stream so the 流式 JSON view stays useful. Finished
        // rounds past the window are folded into one synthetic frame: the assembled reply is kept,
        // the thousands of tiny delta strings are released.
        for r in rounds.iter_mut().skip(FRAME_ROUND_CAP) {
            if r.status != Status::Streaming && !r.frames_compacted && !r.frames.is_empty() {
                r.frames = compact_frames(&r.frames);
                r.frames_compacted = true;
                // Keep the honest "capture was cut short" signal: the synthetic frame still only
                // holds whatever prefix fit within FRAME_CAP at stream time.
            }
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
        let _ = self.state_dirty.send(());
    }

    pub fn save_keys(&self) -> anyhow::Result<()> {
        let keys = self.keys.lock().unwrap().clone();
        self.store.save("keys.json", &keys)
    }

    /// Launch settings live beside the keys file, in the same console state directory. A separate
    /// entry keeps the keys payload (which the proxy rewrites on every counter fold) from carrying
    /// the whole knob tree, and lets settings survive even when no key has been created yet.
    pub fn load_settings(&self) -> Settings {
        self.store.load("settings.json").unwrap_or_default()
    }

    pub fn save_settings(&self, settings: &Settings) -> anyhow::Result<()> {
        self.store.save("settings.json", settings)
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

/// Maximum characters kept of any single string in a retained request body. One pasted
/// transcript can be as large as a whole conversation on its own.
const REQUEST_STRING_CAP: usize = 8192;

/// Bound the request body retained for the sidebar's 请求 tab.
///
/// The dominant cost is not one request but 500 rounds each carrying the *entire* growing prompt:
/// an agent conversation's messages array grows every turn, so storing it whole per round is
/// quadratic in context length. Keep the opening message and the most recent turns intact and
/// collapse the middle into a one-line marker. Base64 is elided first via [`redact_request`].
pub fn compact_request(value: &mut serde_json::Value) {
    redact_request(value);
    if let Some(obj) = value.as_object_mut() {
        for field in ["messages", "input"] {
            let Some(items) = obj.get_mut(field).and_then(|v| v.as_array_mut()) else {
                continue;
            };
            let n = items.len();
            if n <= REQUEST_KEEP_MESSAGES {
                continue;
            }
            let tail_from = n - (REQUEST_KEEP_MESSAGES - 1);
            let omitted = tail_from - 1;
            let mut compact = Vec::with_capacity(REQUEST_KEEP_MESSAGES + 1);
            compact.push(items[0].clone());
            compact.push(serde_json::json!({
                "role": "system",
                "content": format!("<{omitted} 条历史消息已省略，以控制控制台内存>"),
            }));
            compact.extend(items[tail_from..].iter().cloned());
            *items = compact;
        }
    }
    cap_strings(value, REQUEST_STRING_CAP);
}

fn cap_strings(v: &mut serde_json::Value, max: usize) {
    match v {
        serde_json::Value::String(s) => {
            if s.chars().count() > max {
                let head: String = s.chars().take(max).collect();
                *s = format!("{head}… <过长已截断>");
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(|x| cap_strings(x, max)),
        serde_json::Value::Object(map) => map.values_mut().for_each(|x| cap_strings(x, max)),
        _ => {}
    }
}

/// Elide base64 payloads (screenshots) from a request body before it is retained.
pub fn redact_request(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, v) in map.iter_mut() {
                if key == "data" || key == "b64_json" {
                    if let serde_json::Value::String(s) = v {
                        if s.len() > 256 {
                            *v = serde_json::Value::String(format!(
                                "<{} base64 bytes elided>",
                                s.len()
                            ));
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
