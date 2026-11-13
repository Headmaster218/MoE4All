//! The reverse proxy. Sits in the data path, so it can measure what the engine cannot see from
//! outside: time to first token, which of the three decode states a turn is in, which tool is being
//! called, and how much of a conversation's prefix repeats the previous turn.
//!
//! Bytes are forwarded VERBATIM. The tap parses a copy; it never re-serializes, so a client sees
//! exactly the framing the engine produced.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::state::{now_ms, redact_request, Console, DeltaKind, Phase, Round, Status};

/// Headers that describe one hop and must not be copied across it.
const HOP_BY_HOP: [&str; 8] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

pub async fn handle(State(console): State<Arc<Console>>, req: Request<Body>) -> Response {
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();
    let method = req.method().clone();
    let headers = req.headers().clone();

    let body = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
        Ok(b) => b,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("body read: {e}")),
    };

    let key = match authenticate(&console, &headers) {
        Ok(k) => k,
        Err(resp) => return resp,
    };

    let tracked = matches!(path.as_str(), "/v1/chat/completions" | "/v1/responses");
    if !tracked {
        return forward_plain(&console, &method, &format!("{path}{query}"), &headers, body).await;
    }

    let mut json: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            // Not ours to interpret — hand it upstream and let the engine produce the error.
            console.log(format!("unparseable body on {path}: {e}"));
            return forward_plain(&console, &method, &format!("{path}{query}"), &headers, body).await;
        }
    };

    let (key_id, key_label) = key
        .map(|k| (k.id, k.label))
        .unwrap_or_else(|| ("_open".into(), "（无鉴权）".into()));

    let uncensor = apply_uncensor(&console, &mut json);
    let messages = message_strings(&json);
    let conv = conversation_key(&key_id, &messages);
    let reuse_est = console.estimate_reuse(&conv, &messages);

    let mut request_record = json.clone();
    redact_request(&mut request_record);

    let id = console.next_round.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let streaming = json.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let round = Round {
        id,
        key_id: key_id.clone(),
        key_label,
        model: json.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        api: if path == "/v1/responses" { "responses" } else { "chat" },
        started_at_ms: now_ms(),
        ended_at_ms: None,
        status: Status::Streaming,
        phase: Phase::Prefill,
        tools: Vec::new(),
        uncensor,
        prompt_tokens: 0,
        cached_prompt_tokens: 0,
        completion_tokens: 0,
        reuse_est,
        ttft_ms: None,
        live_tps: None,
        prefill_tps: None,
        decode_tps: None,
        duration_ms: None,
        frames_truncated: false,
        request: request_record,
        frames: Vec::new(),
        first_delta_ms: None,
        prefill_ms: None,
    };
    console.push_round(round);

    let outbound = serde_json::to_vec(&json).unwrap_or_else(|_| body.to_vec());
    let url = format!("{}{path}{query}", console.upstream());
    let mut builder = console.http.request(method.clone(), &url);
    for (name, value) in headers.iter() {
        let name = name.as_str();
        if HOP_BY_HOP.contains(&name) || name == "authorization" || name == "content-length" || name == "host" {
            continue;
        }
        builder = builder.header(name, value);
    }
    if let Some(key) = console.upstream_key() {
        builder = builder.bearer_auth(key);
    }
    let upstream = match builder.header("content-type", "application/json").body(outbound).send().await {
        Ok(r) => r,
        Err(e) => {
            console.with_round(id, |r| r.finish(Status::Error));
            console.fold_key(id);
            console.log(format!("upstream unreachable: {e}"));
            return error_response(StatusCode::BAD_GATEWAY, &format!("upstream: {e}"));
        }
    };

    let status = upstream.status();
    let resp_headers = upstream.headers().clone();

    if !status.is_success() {
        let bytes = upstream.bytes().await.unwrap_or_default();
        console.with_round(id, |r| r.finish(Status::Error));
        console.fold_key(id);
        let mut out = Response::builder().status(status);
        for (name, value) in resp_headers.iter() {
            if !HOP_BY_HOP.contains(&name.as_str()) && name.as_str() != "content-length" {
                out = out.header(name, value);
            }
        }
        return out.body(Body::from(bytes)).unwrap_or_else(|_| error_response(StatusCode::BAD_GATEWAY, "response build"));
    }

    if !streaming {
        let bytes = upstream.bytes().await.unwrap_or_default();
        absorb_whole(&console, id, &bytes);
        console.with_round(id, |r| r.finish(Status::Done));
        console.fold_key(id);
        return passthrough_response(status, &resp_headers, Body::from(bytes));
    }

    let tap = Tap::new(console, id, if path == "/v1/responses" { Api::Responses } else { Api::Chat });
    let stream = futures_util::stream::unfold(Some((upstream, tap)), |state| async move {
        let (mut resp, mut tap) = state?;
        match resp.chunk().await {
            Ok(Some(bytes)) => {
                tap.feed(&bytes);
                Some((Ok::<Bytes, std::io::Error>(bytes), Some((resp, tap))))
            }
            Ok(None) => {
                tap.finish(Status::Done);
                None
            }
            Err(e) => {
                tap.finish(Status::Error);
                Some((Err(std::io::Error::other(e)), None))
            }
        }
    });
    passthrough_response(status, &resp_headers, Body::from_stream(stream))
}

fn passthrough_response(status: StatusCode, headers: &HeaderMap, body: Body) -> Response {
    let mut out = Response::builder().status(status);
    for (name, value) in headers.iter() {
        let n = name.as_str();
        if HOP_BY_HOP.contains(&n) || n == "content-length" {
            continue;
        }
        out = out.header(name, value);
    }
    out.body(body).unwrap_or_else(|_| error_response(StatusCode::BAD_GATEWAY, "response build"))
}

/// Non-tracked routes: same key substitution, no record.
async fn forward_plain(
    console: &Arc<Console>,
    method: &axum::http::Method,
    path_and_query: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Response {
    let url = format!("{}{path_and_query}", console.upstream());
    let mut builder = console.http.request(method.clone(), &url);
    for (name, value) in headers.iter() {
        let n = name.as_str();
        if HOP_BY_HOP.contains(&n) || n == "authorization" || n == "content-length" || n == "host" {
            continue;
        }
        builder = builder.header(n, value);
    }
    if let Some(key) = console.upstream_key() {
        builder = builder.bearer_auth(key);
    }
    match builder.body(body.to_vec()).send().await {
        Ok(resp) => {
            let status = resp.status();
            let headers = resp.headers().clone();
            let bytes = resp.bytes().await.unwrap_or_default();
            passthrough_response(status, &headers, Body::from(bytes))
        }
        Err(e) => error_response(StatusCode::BAD_GATEWAY, &format!("upstream: {e}")),
    }
}

/// With no keys configured the console is an open relay to the upstream — the same posture as the
/// engine itself with `serve.api_key` unset. Once ANY key exists, every request must present one.
fn authenticate(
    console: &Arc<Console>,
    headers: &HeaderMap,
) -> Result<Option<crate::state::ApiKey>, Response> {
    let keys = console.keys.lock().unwrap();
    if keys.is_empty() {
        return Ok(None);
    }
    let presented = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
    match presented.and_then(|token| keys.iter().find(|k| k.token == token)) {
        Some(k) => Ok(Some(k.clone())),
        None => Err(error_response(StatusCode::UNAUTHORIZED, "unknown console API key")),
    }
}

/// The hot toggle. The engine's projection vector is baked at startup, but whether it is APPLIED is
/// a per-request field, so switching it needs no engine restart. An explicit client value wins.
///
/// Both directions are injected: `infr_uncensor::effective` falls back to TRUE once a vector is
/// loaded, so omitting the field would leave the projection on while the dashboard said "关".
pub(crate) fn apply_uncensor(console: &Arc<Console>, json: &mut serde_json::Value) -> bool {
    let state = console.uncensor.lock().unwrap().clone();
    let obj = match json.as_object_mut() {
        Some(o) => o,
        None => return false,
    };
    for field in ["uncensor", "experimental_speed_projection"] {
        if let Some(explicit) = obj.get(field).and_then(|v| v.as_bool()) {
            return explicit;
        }
    }
    if !state.vector_loaded {
        return false;
    }
    let on = state.default_on && state.consented;
    obj.insert("uncensor".into(), serde_json::Value::Bool(on));
    on
}

/// Flatten each message to one comparable string. Content may be a plain string or a parts array
/// (text + image_url); both are rendered so a screenshot turn still compares correctly.
fn message_strings(json: &serde_json::Value) -> Vec<String> {
    let items = json
        .get("messages")
        .and_then(|v| v.as_array())
        .or_else(|| json.get("input").and_then(|v| v.as_array()));
    let Some(items) = items else {
        return json
            .get("input")
            .and_then(|v| v.as_str())
            .map(|s| vec![s.to_string()])
            .unwrap_or_default();
    };
    items
        .iter()
        .map(|m| {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let content = match m.get("content") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            format!("{role}\u{1}{content}")
        })
        .collect()
}

/// Conversation identity for prefix comparison: the key plus the FIRST message. Two turns of one
/// chat share their opening system/user message; a different chat does not. Deliberately not a
/// hash of the whole prompt — that would make every turn its own conversation.
fn conversation_key(key_id: &str, messages: &[String]) -> String {
    let anchor = messages.first().map(|s| s.as_str()).unwrap_or("");
    format!("{key_id}\u{1}{anchor}")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Api {
    Chat,
    Responses,
}

/// Incremental SSE reader. Chunks split mid-event, so a line buffer carries the remainder.
struct Tap {
    console: Arc<Console>,
    round: u64,
    api: Api,
    pending: Vec<u8>,
    seq: u64,
    finished: bool,
}

impl Tap {
    fn new(console: Arc<Console>, round: u64, api: Api) -> Self {
        Self { console, round, api, pending: Vec::new(), seq: 0, finished: false }
    }

    fn feed(&mut self, bytes: &[u8]) {
        self.pending.extend_from_slice(bytes);
        while let Some(pos) = find_double_newline(&self.pending) {
            let event: Vec<u8> = self.pending.drain(..pos + 2).collect();
            self.handle_event(&event);
        }
    }

    fn handle_event(&mut self, event: &[u8]) {
        for line in event.split(|b| *b == b'\n') {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let Some(data) = line.strip_prefix(b"data:") else { continue };
            let data = data.strip_prefix(b" ").unwrap_or(data);
            let text = String::from_utf8_lossy(data).into_owned();
            self.seq += 1;
            let seq = self.seq;
            self.console.broadcast_frame(self.round, seq, text.clone());
            self.console.with_round(self.round, |r| {
                if r.frames.len() < crate::state::FRAME_CAP {
                    r.frames.push(crate::state::Frame { seq, data: text.clone() });
                } else {
                    r.frames_truncated = true;
                }
            });
            if text == "[DONE]" {
                continue;
            }
            let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
            self.absorb(&chunk);
        }
    }

    fn absorb(&mut self, chunk: &serde_json::Value) {
        match self.api {
            Api::Chat => self.absorb_chat(chunk),
            Api::Responses => self.absorb_responses(chunk),
        }
    }

    fn absorb_chat(&mut self, chunk: &serde_json::Value) {
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.apply_usage(usage);
        }
        for choice in chunk.get("choices").and_then(|v| v.as_array()).into_iter().flatten() {
            let delta = choice.get("delta").unwrap_or(&serde_json::Value::Null);
            if let Some(s) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                if !s.is_empty() {
                    self.mark(DeltaKind::Reasoning, None, 1);
                }
            }
            if let Some(s) = delta.get("content").and_then(|v| v.as_str()) {
                if !s.is_empty() {
                    self.mark(DeltaKind::Content, None, 1);
                }
            }
            for call in delta.get("tool_calls").and_then(|v| v.as_array()).into_iter().flatten() {
                let name = call.pointer("/function/name").and_then(|v| v.as_str());
                let args = call.pointer("/function/arguments").and_then(|v| v.as_str());
                if name.is_some_and(|n| !n.is_empty()) || args.is_some_and(|a| !a.is_empty()) {
                    self.mark(DeltaKind::Tool, name, 1);
                }
            }
        }
        // The engine's own final timings are authoritative for prompt accounting and prefill time.
        if let Some(t) = chunk.pointer("/choices/0/timings").or_else(|| chunk.get("timings")) {
            self.apply_timings(t);
        }
    }

    fn absorb_responses(&mut self, chunk: &serde_json::Value) {
        let kind = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match kind {
            "response.output_text.delta" => self.mark(DeltaKind::Content, None, 1),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                self.mark(DeltaKind::Reasoning, None, 1)
            }
            "response.function_call_arguments.delta" => self.mark(DeltaKind::Tool, None, 1),
            "response.output_item.added" | "response.output_item.done" => {
                if let Some(item) = chunk.get("item") {
                    if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                        let name = item.get("name").and_then(|v| v.as_str());
                        self.mark(DeltaKind::Tool, name, 0);
                    }
                }
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                if let Some(response) = chunk.get("response") {
                    if let Some(usage) = response.get("usage") {
                        self.apply_usage(usage);
                    }
                }
            }
            _ => {}
        }
    }

    fn mark(&self, kind: DeltaKind, tool: Option<&str>, tokens: u64) {
        self.console.with_round(self.round, |r| r.observe(kind, tool, tokens));
    }

    fn apply_usage(&self, usage: &serde_json::Value) {
        let prompt = usage
            .get("prompt_tokens")
            .or_else(|| usage.get("input_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cached = usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .or_else(|| usage.pointer("/input_tokens_details/cached_tokens"))
            .or_else(|| usage.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let completion = usage
            .get("completion_tokens")
            .or_else(|| usage.get("output_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        self.console.with_round(self.round, |r| {
            if prompt > 0 {
                r.prompt_tokens = prompt;
                r.cached_prompt_tokens = cached.min(prompt);
            }
            // Deltas already counted tokens one by one; usage is the authority when it is larger
            // (reasoning tokens some backends omit from deltas).
            r.completion_tokens = r.completion_tokens.max(completion);
        });
    }

    fn apply_timings(&self, t: &serde_json::Value) {
        let prompt_ms = t.get("prompt_ms").and_then(|v| v.as_f64());
        let cached = t.get("cached_n").and_then(|v| v.as_u64());
        let prompt_n = t.get("prompt_n").and_then(|v| v.as_u64());
        self.console.with_round(self.round, |r| {
            if let Some(ms) = prompt_ms {
                r.prefill_ms = Some(ms as u64);
            }
            if let Some(n) = prompt_n {
                r.prompt_tokens = n;
            }
            if let Some(n) = cached {
                r.cached_prompt_tokens = n.min(r.prompt_tokens);
            }
        });
    }

    fn finish(&mut self, status: Status) {
        if self.finished {
            return;
        }
        self.finished = true;
        if !self.pending.is_empty() {
            let tail = std::mem::take(&mut self.pending);
            self.handle_event(&tail);
        }
        self.console.with_round(self.round, |r| r.finish(status));
        self.console.fold_key(self.round);
    }
}

fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

/// Non-streaming reply: usage plus the engine's timings, same fields as the streaming path.
fn absorb_whole(console: &Arc<Console>, id: u64, bytes: &[u8]) {
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(bytes) else { return };
    let usage = json.get("usage").or_else(|| json.pointer("/response/usage"));
    let timings = json.pointer("/choices/0/timings").or_else(|| json.get("timings"));
    let mut tools = Vec::new();
    for choice in json.get("choices").and_then(|v| v.as_array()).into_iter().flatten() {
        for call in choice.pointer("/message/tool_calls").and_then(|v| v.as_array()).into_iter().flatten() {
            if let Some(name) = call.pointer("/function/name").and_then(|v| v.as_str()) {
                tools.push(name.to_string());
            }
        }
    }
    console.with_round(id, |r| {
        if let Some(u) = usage {
            let prompt = u.get("prompt_tokens").or_else(|| u.get("input_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);
            let cached = u
                .pointer("/prompt_tokens_details/cached_tokens")
                .or_else(|| u.pointer("/input_tokens_details/cached_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            r.prompt_tokens = prompt;
            r.cached_prompt_tokens = cached.min(prompt);
            r.completion_tokens = u
                .get("completion_tokens")
                .or_else(|| u.get("output_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
        }
        if let Some(t) = timings {
            if let Some(ms) = t.get("prompt_ms").and_then(|v| v.as_f64()) {
                r.prefill_ms = Some(ms as u64);
            }
            if let Some(n) = t.get("cached_n").and_then(|v| v.as_u64()) {
                r.cached_prompt_tokens = n.min(r.prompt_tokens);
            }
        }
        r.tools = tools;
        if !r.tools.is_empty() {
            r.phase = Phase::ToolCall;
        } else if r.completion_tokens > 0 {
            r.phase = Phase::Output;
        }
        // A non-streaming reply has no observable first-token instant. Leave TTFT absent rather
        // than report the whole call as if it were the wait for the first token.
    });
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({ "error": { "message": message, "type": "console_error" } });
    let mut resp = (status, axum::Json(body)).into_response();
    resp.headers_mut().insert("content-type", HeaderValue::from_static("application/json"));
    resp
}