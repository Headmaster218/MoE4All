//! OpenAI-compatible HTTP server (axum + SSE). Talks only to `infr-engine` — never the GPU.
//!
//! Reference for the wire mapping (streaming, `reasoning_content`, tool_calls): the working
//! shim at `~/Projects/scratch/dgemma-openai-server.py`. See docs/plan.md "server".
//!
//! Routes (`auth` = gated by `serve.api_key` when one is configured — see [`auth_gate`]):
//!   GET  /health                -> 200 OK                                              (open)
//!   GET  /v1/models             -> { object: "list", data: [{ id, object, owned_by }] } (auth)
//!   POST /v1/chat/completions   -> chat.completion | SSE chat.completion.chunk stream   (auth)
//!   POST /v1/embeddings         -> OpenAI-compatible normalized float embeddings         (auth)
//!
//! Two process-level limits bound one request's hold on a `--parallel` slot: `serve.max_tokens_cap`
//! (tokens — see [`clamp_max_tokens`]) and `serve.request_timeout_secs` (wall clock — see
//! [`request_timeout`]). Both are `serve.*` config, never read from the environment here.
//!
//! Delta mapping:
//!   `Delta::Reasoning`  -> `delta.reasoning_content`
//!   `Delta::Content`    -> `delta.content`
//!   `Delta::ToolCall`   -> `delta.tool_calls[]`  (finish_reason "tool_calls")

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use infr_core::config::Config;
use infr_engine::{ChatMessage, Delta, ToolCall};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use tokio::sync::Semaphore;

/// Why generation ended — the OpenAI `finish_reason`. The generator reports it; the handlers
/// serialize it (a tool call still overrides to [`Finish::ToolCalls`] at the wire layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// EOS, or a `stop` sequence fired.
    Stop,
    /// The `max_tokens` / `max_completion_tokens` budget was exhausted.
    Length,
    /// A tool call was emitted.
    ToolCalls,
}

impl Finish {
    pub fn as_str(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::ToolCalls => "tool_calls",
        }
    }
}

/// Generation backend the server drives — it never knows the model/GPU underneath. Implemented by
/// the CLI's per-arch adapters (`infr-cli`'s `ParallelGenerator` over `infr_llama::ParallelSeam` for
/// the Vulkan seam; `SeamGenerator` wraps any `infr_llama::ChatModel` for the rest, including
/// `DiffusionGemmaChat` — see `docs/diffusion-gemma.md`).
///
/// [`GenParams`] carries the request's PER-REQUEST sampling config (temperature/top_p/top_k/seed/
/// penalties/stop/max_tokens). Every field is an `Option` whose `None` means "inherit the process
/// default" — so a request that sends nothing generates EXACTLY as it did before this existed.
///
/// **`&self`, `Send + Sync`.** This is the whole concurrency contract: the server calls `chat` from
/// N request tasks at once and the generator is responsible for its own slot allocation and GPU
/// turn-taking. It used to be `&mut self` behind an `Arc<Mutex<_>>` the handlers held for an ENTIRE
/// generation, which meant request #2 waited for request #1 to finish — head-of-line blocking, no
/// parallelism. An implementation that genuinely cannot run concurrently (CPU / Metal / diffusion
/// today) keeps an internal `Mutex` and is served with `--parallel 1`, which is honest rather than
/// silently serialising a server the user asked to parallelise.
pub trait ChatGenerator: Send + Sync {
    /// Run one chat turn.
    ///
    /// * `tools` is the request's `tools` array as a borrowed [`serde_json::Value`] — passed by
    ///   reference so the generator parses the ONE it was already given instead of a
    ///   `Value`→string→`Value` round-trip (see audit finding 6).
    /// * `cancel` is a PER-REQUEST abort latch. The server sets it when the client disconnects (an
    ///   SSE `send` starts failing); the generator must poll it in its decode loop and stop promptly
    ///   so the GPU slot is freed rather than held to `max_tokens`. It is ORed with the process-wide
    ///   shutdown latch, never a replacement for it.
    /// * Returns a [`ChatOutcome`] carrying the finish reason AND the real prompt/completion token
    ///   counts so the handler can populate `usage` truthfully.
    fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
        tool_choice: Option<&str>,
        params: &GenParams,
        cancel: &AtomicBool,
        on_delta: &mut dyn FnMut(Delta),
    ) -> anyhow::Result<ChatOutcome>;
}

/// Embedding backend driven by `/v1/embeddings`. Like [`ChatGenerator`], the server knows
/// nothing about the model or device underneath it.
pub trait EmbeddingGenerator: Send + Sync {
    fn embed(&self, inputs: &[String]) -> anyhow::Result<EmbeddingOutcome>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingOutcome {
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: u32,
}

/// What one [`ChatGenerator::chat`] call produced: why it ended, plus the real token counts the
/// handler needs for a truthful `usage` block (`total = prompt + completion`). Counts are what the
/// generator actually tokenized/emitted — never a `content.len()/4` estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatOutcome {
    pub finish: Finish,
    /// Total prompt tokens presented to the model, including any prefix served from KV cache.
    pub prompt_tokens: u32,
    /// Prompt tokens served from a reusable prefix cache and not recomputed for this request.
    pub cached_prompt_tokens: u32,
    pub completion_tokens: u32,
}

// ---------------------------------------------------------------------------
// Request DTOs
// ---------------------------------------------------------------------------

/// Top-level chat completion request (OpenAI wire format).
///
/// Unknown fields are IGNORED (no `deny_unknown_fields`) — an OpenAI client sending `n`, `user`,
/// `logit_bias`, … must not 400. Known-but-invalid VALUES do 400, via [`GenParams::from_request`].
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessageDto>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// OpenAI `tool_choice`: `"auto"` | `"required"` | `"none"` | `{"type":"function","function":
    /// {"name":..}}`. Normalised to a string (the function name for a named choice) by [`tool_choice_str`].
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// OpenAI's rename of `max_tokens`. Preferred when both are present.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Not in the OpenAI schema; llama.cpp/vLLM/Ollama all accept it and so do we.
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// `"\n"` or `["\n", "END"]` (OpenAI: up to 4).
    #[serde(default)]
    pub stop: Option<serde_json::Value>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// llama.cpp extension (1.0 = off).
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
}

/// The validated, per-request generation config handed to [`ChatGenerator::chat`]. `None` fields
/// mean "the request didn't say" — the generator leaves the process default (`INFR_TEMP` /
/// `INFR_TOP_K` / `INFR_TOP_P` / `INFR_MAX_NEW`) in charge for exactly those.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
    /// Already normalised: empty strings dropped (an empty stop would fire on the first token).
    pub stop: Vec<String>,
}

/// An OpenAI-shaped 400: `{"error":{"message":..,"type":"invalid_request_error","param":..}}`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamError {
    pub param: &'static str,
    pub message: String,
}

impl GenParams {
    /// Validate + normalise the request's sampling fields. Out-of-range values are a 400, NEVER a
    /// silent clamp and never a panic (OpenAI's own ranges: temperature 0..2, top_p 0..1,
    /// presence/frequency -2..2, at most 4 stop sequences).
    pub fn from_request(req: &ChatRequest) -> Result<Self, ParamError> {
        let rng = |param: &'static str,
                   v: Option<f32>,
                   lo: f32,
                   hi: f32|
         -> Result<Option<f32>, ParamError> {
            match v {
                Some(x) if !x.is_finite() || x < lo || x > hi => Err(ParamError {
                    param,
                    message: format!("{param} must be between {lo} and {hi}, got {x}"),
                }),
                other => Ok(other),
            }
        };

        let top_k = match req.top_k {
            // 0 = "no top-k" in llama.cpp; negative is meaningless.
            Some(k) if k < 0 => {
                return Err(ParamError {
                    param: "top_k",
                    message: format!("top_k must be >= 0, got {k}"),
                })
            }
            Some(k) => Some(k as usize),
            None => None,
        };

        let stop = match &req.stop {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(a)) => {
                let mut v = Vec::with_capacity(a.len());
                for item in a {
                    match item.as_str() {
                        Some(s) => v.push(s.to_string()),
                        None => {
                            return Err(ParamError {
                                param: "stop",
                                message: "stop must be a string or an array of strings".into(),
                            })
                        }
                    }
                }
                v
            }
            Some(_) => {
                return Err(ParamError {
                    param: "stop",
                    message: "stop must be a string or an array of strings".into(),
                })
            }
        };
        if stop.len() > 4 {
            return Err(ParamError {
                param: "stop",
                message: format!("at most 4 stop sequences are supported, got {}", stop.len()),
            });
        }
        // An empty stop string would match at every position — drop it rather than dead-stop.
        let stop: Vec<String> = stop.into_iter().filter(|s| !s.is_empty()).collect();

        if let Some(p) = req.repeat_penalty {
            if !p.is_finite() || p <= 0.0 {
                return Err(ParamError {
                    param: "repeat_penalty",
                    message: format!("repeat_penalty must be > 0, got {p}"),
                });
            }
        }

        Ok(Self {
            // OpenAI renamed `max_tokens` -> `max_completion_tokens`; the new name wins.
            max_tokens: req.max_completion_tokens.or(req.max_tokens),
            temperature: rng("temperature", req.temperature, 0.0, 2.0)?,
            top_p: rng("top_p", req.top_p, 0.0, 1.0)?,
            top_k,
            seed: req.seed,
            presence_penalty: rng("presence_penalty", req.presence_penalty, -2.0, 2.0)?,
            frequency_penalty: rng("frequency_penalty", req.frequency_penalty, -2.0, 2.0)?,
            repeat_penalty: req.repeat_penalty,
            stop,
        })
    }
}

// ---------------------------------------------------------------------------
// Stop sequences
// ---------------------------------------------------------------------------

/// Incremental stop-sequence matcher over the DECODED text stream.
///
/// The hard part is that a stop string need not align with token boundaries: `"\n\n"` may arrive as
/// `"\n"` + `"\n"`, and `"END"` as `"E"` + `"ND"`. Two rules make that work:
///
/// 1. **Match on the accumulated tail**, not on the individual piece — so a split stop still fires.
/// 2. **Hold back** the longest suffix of the emitted text that is a PREFIX of some stop string.
///    Streaming clients must never see `"E"` from a token that turns out to begin `"END"`; if the
///    next piece completes the stop, that `"E"` was never ours to send. The hold-back is bounded by
///    `longest_stop - 1` bytes, so at most 3-4 bytes of latency in practice.
///
/// On a hit, the text BEFORE the stop string is emitted and the stop string itself is discarded
/// (OpenAI does not include it in the completion).
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    /// Text seen but not yet emitted (a possible stop prefix).
    held: String,
    hit: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
            hit: false,
        }
    }

    pub fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// A stop sequence has fired: generation must halt and no further text may be emitted.
    pub fn hit(&self) -> bool {
        self.hit
    }

    /// Feed one decoded piece; returns the text that is now SAFE to emit (possibly empty).
    pub fn push(&mut self, piece: &str) -> String {
        if self.hit {
            return String::new();
        }
        if self.stops.is_empty() {
            return piece.to_string();
        }
        self.held.push_str(piece);

        // 1. Full match anywhere in the held tail -> emit the head, drop the stop and the rest.
        if let Some(cut) = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min()
        {
            self.hit = true;
            let out = self.held[..cut].to_string();
            self.held.clear();
            return out;
        }

        // 2. No match: hold back the longest suffix that could still BECOME one.
        let hold = self.longest_partial_suffix();
        let split = self.held.len() - hold;
        let out = self.held[..split].to_string();
        self.held.drain(..split);
        out
    }

    /// End of generation with no stop hit: whatever is still held was never a stop, so emit it.
    pub fn flush(&mut self) -> String {
        if self.hit {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }

    /// Length (bytes) of the longest suffix of `held` that is a proper prefix of some stop string.
    /// Always lands on a char boundary: a suffix of `held` equal to `stop[..n]` starts at `stop`'s
    /// first byte, which is a UTF-8 lead byte.
    fn longest_partial_suffix(&self) -> usize {
        let max = self
            .stops
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .saturating_sub(1)
            .min(self.held.len());
        for n in (1..=max).rev() {
            let start = self.held.len() - n;
            if !self.held.is_char_boundary(start) {
                continue;
            }
            let tail = &self.held[start..];
            if self
                .stops
                .iter()
                .any(|s| s.as_bytes().starts_with(tail.as_bytes()))
            {
                return n;
            }
        }
        0
    }
}

/// Normalise OpenAI `tool_choice` to a string the generator understands: `"auto"`/`"required"`/
/// `"none"` pass through; a `{"type":"function","function":{"name":N}}` object becomes `N`.
///
/// The caller has already handled ABSENT/`null` (that is "no choice" → `Ok(None)` at the call site).
/// This function is only reached for a PRESENT value, so an object that lacks a usable
/// `function.name` — or any non-string/non-object shape — is a MALFORMED forced-tool request and is
/// a 400 ([`ParamError`]), NOT a silent downgrade to "auto" (audit finding 6).
fn tool_choice_str(v: &serde_json::Value) -> Result<Option<String>, ParamError> {
    match v {
        serde_json::Value::String(s) => Ok(Some(s.clone())),
        serde_json::Value::Object(_) => v
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .map(|s| Some(s.to_owned()))
            .ok_or_else(|| ParamError {
                param: "tool_choice",
                message: "tool_choice object must have a string `function.name`".into(),
            }),
        _ => Err(ParamError {
            param: "tool_choice",
            message: "tool_choice must be \"auto\"/\"required\"/\"none\" or a function object"
                .into(),
        }),
    }
}

/// The three `tool_choice` values that are POLICIES rather than tool names.
const TOOL_CHOICE_POLICIES: [&str; 3] = ["auto", "none", "required"];

/// Cross-check a normalised `tool_choice` against the request's `tools`, which
/// [`tool_choice_str`] cannot do on its own — it never sees `tools`.
///
/// Two holes this closes, both of which used to produce ordinary unconstrained assistant text
/// (backlog B22). `tool_constraint_for` returns `Ok(None)` the moment `tools` is absent, without
/// ever looking at `tool_choice`, and `run_chat` reads that `None` as "no constraint wanted":
///
/// - **A forced choice with no `tools`.** `"tool_choice":"required"` and no tools is a request the
///   server cannot honour — there is nothing to call — so it is a 400 rather than a silent
///   downgrade to free text. `"none"` is exempt: it asks for no tool call, which is satisfiable
///   with no tools, and `"auto"` means "your discretion".
/// - **A name that is not in `tools`.** A misspelled function name reached the generator as a
///   forced choice and then vanished. With `tools` present it is worse than nothing: the filter in
///   `tool_constraint_for` yields an empty array and `forced_tool_call_grammar` builds
///   `{"anyOf": []}`, a grammar that matches no output at all.
///
/// Same policy as the existing malformed-object 400s: a forced tool call the server cannot deliver
/// is an error, never a quiet fallback to "auto" (audit finding 6).
fn validate_tool_choice(
    tool_choice: Option<&str>,
    tools: Option<&serde_json::Value>,
) -> Result<(), ParamError> {
    let Some(choice) = tool_choice else {
        return Ok(());
    };
    if TOOL_CHOICE_POLICIES.contains(&choice) && choice != "required" {
        return Ok(());
    }
    // `tools` may be absent, `null`, or an empty array — all "no tools offered".
    let names: Vec<&str> = tools
        .and_then(|t| t.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|t| t.get("function")?.get("name")?.as_str())
                .collect()
        })
        .unwrap_or_default();
    if names.is_empty() {
        return Err(ParamError {
            param: "tool_choice",
            message: format!(
                "tool_choice {choice:?} requires a non-empty `tools` array; none was supplied"
            ),
        });
    }
    if choice == "required" || names.contains(&choice) {
        return Ok(());
    }
    Err(ParamError {
        param: "tool_choice",
        message: format!(
            "tool_choice {choice:?} names no tool in `tools` (have: {})",
            names.join(", ")
        ),
    })
}

/// A single chat message.  `content` may be a JSON string or a content-part array.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessageDto {
    pub role: String,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    /// Assistant's prior tool calls (OpenAI `[{id,type,function:{name,arguments}}]`), replayed on the
    /// next turn so the model sees its own calls.
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// Request/response DTOs — /v1/embeddings
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Response DTOs — /v1/models
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

// ---------------------------------------------------------------------------
// Response DTOs — /v1/chat/completions  (non-streaming)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OAIToolCall>>,
}

/// OpenAI-shaped tool call (used in both streaming and non-streaming responses).
#[derive(Debug, Clone, Serialize)]
pub struct OAIToolCall {
    pub index: usize,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: OAIFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct OAIFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: PromptTokensDetails,
}

impl UsageInfo {
    fn from_outcome(outcome: ChatOutcome) -> Self {
        Self {
            prompt_tokens: outcome.prompt_tokens,
            completion_tokens: outcome.completion_tokens,
            total_tokens: outcome
                .prompt_tokens
                .saturating_add(outcome.completion_tokens),
            prompt_tokens_details: PromptTokensDetails {
                cached_tokens: outcome.cached_prompt_tokens.min(outcome.prompt_tokens),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Response DTOs — /v1/chat/completions  (streaming chunks)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: DeltaPayload,
    pub finish_reason: Option<String>,
}

/// The `delta` field inside a streaming chunk.  Fields absent when `None`.
#[derive(Debug, Default, Serialize)]
pub struct DeltaPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OAIToolCall>>,
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

/// Per-request id: a process-monotonic counter, NOT a timestamp.
///
/// It is deliberately not derived from the wall clock. Two requests admitted in the same
/// millisecond (routine under `--parallel N`) would collide on a `now()`-based id, and the whole
/// value of the id is that the arrival line and the completion line for ONE request can be joined
/// in a log that has N of them interleaved. It is also NOT the wire `id` ([`make_id`]): that one is
/// a client-facing `chatcmpl-…` string, and pinning a log key to a wire format is how the log
/// breaks when the wire format moves.
static REQ_SEQ: AtomicU64 = AtomicU64::new(1);

fn next_req_id() -> u64 {
    REQ_SEQ.fetch_add(1, Ordering::Relaxed)
}

/// The server-wide counters behind the periodic throughput line, and the two gauges behind
/// `active`/`queued`.
///
/// **Everything here is an atomic and nothing here is a lock.** The only counter touched from
/// inside a generation is [`Self::bump_gen`], one relaxed `fetch_add` per emitted delta — the
/// decode loop must not acquire anything, because it is holding the GPU baton for every other
/// sequence behind it (the same reason the SSE channel is unbounded).
///
/// The four `interval_*` counters are DRAINED (swapped to zero) by each report, which is what makes
/// the reported numbers cover the interval rather than the process lifetime. There is deliberately
/// no cumulative total kept alongside them: a total nobody drains is the thing that silently turns
/// a rate into an average-since-boot.
#[derive(Debug, Default)]
struct ServeStats {
    /// Prompt tokens PREFILLED in this interval. Folded once per request, at completion — the real
    /// count from [`ChatOutcome`], which is not knowable before the generator has tokenized.
    interval_prompt_tokens: AtomicU64,
    /// Tokens GENERATED in this interval, live. Incremented per delta while a generation runs (so a
    /// long request shows up in the intervals it spans, not only in the one it ends in) and
    /// RECONCILED at completion against `ChatOutcome::completion_tokens`, which is authoritative:
    /// a delta is a text piece and is only approximately a token (a think-tag boundary can split
    /// or merge one). The correction is signed, hence `i64`.
    interval_gen_tokens: AtomicI64,
    /// Requests that COMPLETED in this interval (success or failure).
    interval_completed: AtomicU64,
    /// Requests that FAILED in this interval — a subset of `interval_completed`.
    interval_failed: AtomicU64,
    /// Gauge: requests generating right now (holding a slot permit).
    active: AtomicU64,
    /// Gauge: requests admitted by the handler but still waiting for a slot permit.
    queued: AtomicU64,
    /// Which interval is currently open. Incremented by every [`Self::drain`].
    ///
    /// A completion's correction is only meaningful against the deltas it corrects, and those
    /// deltas may already have been drained and REPORTED. Stamping each request's live count with
    /// the window it landed in is what lets [`Self::fold_completion`] tell "still correctable" from
    /// "already published" (backlog B24).
    window: AtomicU64,
}

impl ServeStats {
    /// One decoded piece. The ONE call on the hot path — a single relaxed add.
    fn bump_gen(&self, n: i64) {
        self.interval_gen_tokens.fetch_add(n, Ordering::Relaxed);
    }

    /// The interval currently accepting tokens.
    fn window(&self) -> u64 {
        self.window.load(Ordering::Relaxed)
    }

    /// Fold one finished request's exact tallies in, once.
    ///
    /// **The correction is clamped to what this request still has in the OPEN window.** Deltas are
    /// only an estimate of tokens (a think-tag boundary can split or merge one), so completion
    /// reconciles against `ChatOutcome`'s authoritative count. But a retraction cannot reach a
    /// window that has already been drained and logged, and subtracting it from the current one
    /// takes the tokens out of whatever OTHER request is generating now — which is how a −1 from
    /// request A turned request B's three real tokens into two (B24, proved).
    ///
    /// So a negative correction is applied only against `rec.deltas_in_window`, and only while
    /// `rec.window` is still open. Anything older stays as it was reported: an interval line is a
    /// statement about a window that has closed, and the honest move is to leave it alone rather
    /// than to bill the difference to somebody else. A POSITIVE correction is always applied — it
    /// is new information about tokens nobody has counted yet, the same shape as `prompt_tokens`
    /// arriving at completion.
    fn fold_completion(&self, rec: &ReqRecord) {
        self.interval_prompt_tokens.fetch_add(
            u64::from(rec.prompt_tokens.saturating_sub(rec.cached_prompt_tokens)),
            Ordering::Relaxed,
        );
        let correction = i64::from(rec.gen_tokens) - rec.deltas as i64;
        if correction > 0 {
            self.bump_gen(correction);
        } else if correction < 0 && rec.window == self.window() {
            // Retract at most what this request itself put into the window that is still open.
            let retractable = rec.deltas_in_window.min(i64::MAX as u64) as i64;
            self.bump_gen(-correction.abs().min(retractable));
        }
        self.interval_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Fold one request that ended in an error. Its partial deltas are already counted; there is no
    /// [`ChatOutcome`] to reconcile against, so nothing is corrected.
    fn fold_failure(&self) {
        self.interval_completed.fetch_add(1, Ordering::Relaxed);
        self.interval_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Take the interval counters (resetting them to zero) and sample the gauges. `elapsed` is the
    /// REAL time since the last drain, not the nominal period — a tick that ran late must not
    /// inflate the rate it reports.
    fn drain(&self, elapsed: Duration) -> StatsWindow {
        // Close the current window BEFORE taking the counters: from here on, a completion whose
        // deltas landed in the old window can no longer retract them (see `fold_completion`).
        self.window.fetch_add(1, Ordering::Relaxed);
        StatsWindow {
            elapsed,
            prompt_tokens: self.interval_prompt_tokens.swap(0, Ordering::Relaxed),
            gen_tokens: self.interval_gen_tokens.swap(0, Ordering::Relaxed).max(0) as u64,
            completed: self.interval_completed.swap(0, Ordering::Relaxed),
            failed: self.interval_failed.swap(0, Ordering::Relaxed),
            active: self.active.load(Ordering::Relaxed),
            queued: self.queued.load(Ordering::Relaxed),
            busy_slots: 0,
            total_slots: 0,
        }
    }
}

/// One interval's drained counters — the whole input to one periodic log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct StatsWindow {
    elapsed: Duration,
    prompt_tokens: u64,
    gen_tokens: u64,
    completed: u64,
    failed: u64,
    active: u64,
    queued: u64,
    /// Slot permits held across every hosted model — KV slot occupancy.
    busy_slots: u64,
    /// Slot permits in existence across every hosted model (the sum of `--parallel N`).
    total_slots: u64,
}

impl StatsWindow {
    /// Was there anything to report? An interval in which the server did nothing at all emits NO
    /// line — the periodic report is activity-only, deliberately not a heartbeat, so an idle server
    /// leaves a clean log. A request that is mid-generation counts as activity (`active > 0`) even
    /// when it has produced no token yet, so a long single request still reports every interval it
    /// spans.
    fn has_activity(&self) -> bool {
        self.prompt_tokens > 0
            || self.gen_tokens > 0
            || self.completed > 0
            || self.active > 0
            || self.queued > 0
    }

    /// Prompt tokens ingested per WALL second of the interval.
    ///
    /// This is the server's aggregate ingest throughput, NOT one model's prefill speed: the
    /// denominator is the interval, including the time the server spent decoding or idle. The
    /// per-request completion line carries the other number (that request's `prompt_tokens / TTFT`),
    /// and the two are supposed to differ — one answers "how much is this box doing", the other
    /// "how fast is this model".
    fn prefill_tps(&self) -> f64 {
        per_second(self.prompt_tokens, self.elapsed)
    }

    /// Tokens generated per WALL second of the interval — aggregate across every in-flight request.
    /// Same scope note as [`Self::prefill_tps`].
    fn decode_tps(&self) -> f64 {
        per_second(self.gen_tokens, self.elapsed)
    }
}

/// `n` per second over `d`. A zero (or absurdly small) window yields 0.0 rather than an infinity —
/// a log line reading `inf` is worse than one reading 0.
fn per_second(n: u64, d: Duration) -> f64 {
    let secs = d.as_secs_f64();
    if secs > 0.0 {
        n as f64 / secs
    } else {
        0.0
    }
}

/// RAII gauge for [`ServeStats::queued`]: a request waiting for a slot permit. Dropped the moment
/// the permit is acquired (or the acquire fails during shutdown).
///
/// A guard rather than a matching `fetch_sub` because there are three ways out of the wait — got
/// the permit, the semaphore closed, the task was dropped — and a gauge that leaks on one of them
/// reads as a permanently-queued request forever after.
struct QueuedGuard(Arc<ServeStats>);

impl QueuedGuard {
    fn new(stats: Arc<ServeStats>) -> Self {
        stats.queued.fetch_add(1, Ordering::Relaxed);
        Self(stats)
    }
}

impl Drop for QueuedGuard {
    fn drop(&mut self) {
        self.0.queued.fetch_sub(1, Ordering::Relaxed);
    }
}

/// RAII gauge for [`ServeStats::active`]: a request holding a slot permit and generating. Moved
/// INTO the blocking task alongside the permit, so it is released on exactly the same events the
/// permit is — including an unwinding panic inside the decode closure.
struct ActiveGuard(Arc<ServeStats>);

impl ActiveGuard {
    fn new(stats: Arc<ServeStats>) -> Self {
        stats.active.fetch_add(1, Ordering::Relaxed);
        Self(stats)
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The per-request tally kept INSIDE the generation, as plain locals in the blocking task. It is
/// folded into [`ServeStats`] once, at completion — the request's own timing needs no sharing, and
/// making it shared would put a second atomic (or worse) on the decode path for nothing.
#[derive(Debug)]
struct ReqTally {
    started: Instant,
    /// When the FIRST delta arrived: the boundary between prefill and decode. `None` means the
    /// generation produced nothing.
    first_delta: Option<Instant>,
    /// Text deltas seen (content + reasoning). Reconciled against the real token count at the end.
    deltas: u64,
    /// The [`ServeStats`] window `deltas_in_window` is counted against.
    window: u64,
    /// Deltas this request has contributed to `window` — i.e. how much of the still-open interval
    /// is this request's, and therefore the most a completion correction may retract from it (B24).
    deltas_in_window: u64,
}

impl ReqTally {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            first_delta: None,
            deltas: 0,
            window: 0,
            deltas_in_window: 0,
        }
    }

    /// Called from the generator's `on_delta` callback for a TEXT delta. A few increments, one
    /// relaxed atomic load and one relaxed atomic add — no lock, no allocation, nothing that can
    /// block the decode loop.
    ///
    /// The load is what keeps `deltas_in_window` honest across a drain: a reporter tick between two
    /// deltas moves the window, and this request's share of the new one restarts at zero. The load
    /// and the add are not atomic together, so a drain landing exactly between them can misplace a
    /// single token; that is one token on an interval line, and closing it would mean a lock on the
    /// decode path, which is the one thing this whole structure exists to avoid.
    fn on_text_delta(&mut self, stats: &ServeStats) {
        if self.first_delta.is_none() {
            self.first_delta = Some(Instant::now());
        }
        let w = stats.window();
        if w != self.window {
            self.window = w;
            self.deltas_in_window = 0;
        }
        self.deltas += 1;
        self.deltas_in_window += 1;
        stats.bump_gen(1);
    }

    /// Close the tally out against the generator's authoritative counts.
    fn finish(&self, outcome: ChatOutcome, finish: Finish) -> ReqRecord {
        let total = self.started.elapsed();
        // TTFT is the prefill boundary. With no delta at all (an empty completion) there is no
        // boundary to draw, so the whole request counts as prefill and decode time is zero.
        let prefill = self
            .first_delta
            .map_or(total, |t| t.saturating_duration_since(self.started));
        ReqRecord {
            prompt_tokens: outcome.prompt_tokens,
            cached_prompt_tokens: outcome.cached_prompt_tokens.min(outcome.prompt_tokens),
            gen_tokens: outcome.completion_tokens,
            deltas: self.deltas,
            window: self.window,
            deltas_in_window: self.deltas_in_window,
            prefill,
            decode: total.saturating_sub(prefill),
            total,
            finish,
        }
    }
}

/// One finished request, as the completion log line and [`ServeStats::fold_completion`] see it.
#[derive(Debug, Clone, Copy)]
struct ReqRecord {
    prompt_tokens: u32,
    cached_prompt_tokens: u32,
    gen_tokens: u32,
    deltas: u64,
    /// The stats window this request's last deltas landed in — see [`ServeStats::fold_completion`].
    window: u64,
    /// How many of `deltas` were counted into `window`.
    deltas_in_window: u64,
    /// Time to the first delta — the prefill.
    prefill: Duration,
    /// From the first delta to the end — the decode.
    decode: Duration,
    total: Duration,
    finish: Finish,
}

impl ReqRecord {
    /// THIS request's prefill speed: prompt tokens over its time-to-first-delta. A per-request
    /// number, unlike [`StatsWindow::prefill_tps`].
    fn prefill_tps(&self) -> f64 {
        per_second(
            u64::from(self.prompt_tokens.saturating_sub(self.cached_prompt_tokens)),
            self.prefill,
        )
    }

    /// THIS request's decode speed: generated tokens over the time after the first delta.
    fn decode_tps(&self) -> f64 {
        per_second(u64::from(self.gen_tokens), self.decode)
    }
}

/// Emit one request's completion line at INFO. The counterpart to the arrival line
/// ([`log_request_start`]), joined to it by `req`.
fn log_request_done(req_id: u64, model: &str, stream: bool, rec: &ReqRecord) {
    tracing::info!(
        req = req_id,
        model,
        stream,
        prompt_tokens = rec.prompt_tokens,
        gen_tokens = rec.gen_tokens,
        prefill_tps = format_args!("{:.1}", rec.prefill_tps()),
        decode_tps = format_args!("{:.1}", rec.decode_tps()),
        total_ms = format_args!("{:.0}", rec.total.as_secs_f64() * 1000.0),
        finish = rec.finish.as_str(),
        "request done"
    );
}

/// Emit one request's arrival line at INFO.
///
/// `prompt_chars` and not prompt TOKENS, and that is a real limitation rather than an oversight:
/// the tokenizer lives behind [`ChatGenerator`], so at arrival the server genuinely does not know
/// how many tokens the messages are. The true count is on the completion line, from
/// [`ChatOutcome`]; the char count is the arrival-time proxy for "how big is this".
///
/// **No prompt text, ever.** Counts only. Putting user prompt text into an operator's logs is a
/// privacy decision that has been made in the negative — do not add a preview here, not even
/// truncated, not even behind a flag.
fn log_request_start(
    req_id: u64,
    route: &'static str,
    model: &str,
    messages: usize,
    prompt_chars: usize,
    max_tokens: Option<u32>,
    stream: bool,
) {
    tracing::info!(
        req = req_id,
        route,
        model,
        messages,
        prompt_chars,
        max_tokens = ?max_tokens,
        stream,
        "request start"
    );
}

/// The configured period for the periodic throughput line, or `None` when it is switched OFF.
///
/// `serve.stats_interval_secs` (`INFR_SERVE_STATS_SECS`), where `0` means "no periodic line" —
/// the same "0 disables" grammar as [`request_timeout`], and for the same reason: an operator
/// piping this server's logs somewhere expensive needs a way to say no from the environment, over
/// a config file that said yes.
fn stats_interval(cfg: &Config) -> Option<Duration> {
    let secs = cfg.serve.stats_interval_secs;
    (secs > 0).then(|| Duration::from_secs(secs))
}

/// The periodic throughput reporter: drain the interval counters every `period` and, IF anything
/// happened, log one line.
///
/// **Shutdown.** It polls the same process-wide latch [`shutdown_latched`] does, at the same 50 ms
/// granularity, so a Ctrl-C ends it at the next poll rather than up to a full period later — and it
/// is a plain tokio task, which cannot hold the process open by itself: when `serve_state` returns
/// the runtime drops and the task goes with it. It emits one FINAL drain on the way out so the
/// tokens generated in the last partial interval are not silently dropped.
async fn stats_reporter(state: AppState, period: Duration) {
    const POLL: Duration = Duration::from_millis(50);
    let mut last = Instant::now();
    loop {
        tokio::time::sleep(POLL).await;
        let shutting_down = infr_core::shutdown::shutdown_requested();
        if !shutting_down && last.elapsed() < period {
            continue;
        }
        let mut window = state.stats.drain(last.elapsed());
        (window.busy_slots, window.total_slots) = state.slot_occupancy();
        last = Instant::now();
        if window.has_activity() {
            tracing::info!(
                interval_s = format_args!("{:.1}", window.elapsed.as_secs_f64()),
                prefill_tps = format_args!("{:.1}", window.prefill_tps()),
                decode_tps = format_args!("{:.1}", window.decode_tps()),
                gen_tokens = window.gen_tokens,
                prompt_tokens = window.prompt_tokens,
                completed = window.completed,
                failed = window.failed,
                active = window.active,
                queued = window.queued,
                kv_slots = format_args!("{}/{}", window.busy_slots, window.total_slots),
                "serve stats"
            );
        }
        if shutting_down {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

/// One hosted model: its wire id, its generator, and its OWN admission semaphore.
///
/// `engine` is an `Option<Arc<dyn ChatGenerator>>` so an entry can exist without a live model (the
/// headless test/`/v1/models` state). It is an `Arc`, NOT an `Arc<Mutex<_>>`: generation runs
/// concurrently and the generator synchronises itself (see [`ChatGenerator`]'s `&self` contract).
///
/// `slots` is PER-MODEL admission control — one permit per KV slot (`--parallel N`). A request
/// takes a permit for the whole of its generation and returns it at the end, so at most N generate
/// concurrently on THAT model and the N+1'th QUEUES (tokio's semaphore is FIFO) rather than being
/// rejected. Per-model (not global) is the point of the multi-model server: a model on the discrete
/// GPU and a model on the iGPU each admit their own N independently — one busy model never starves
/// another that lives on a different device.
#[derive(Clone)]
struct ModelEntry {
    id: Arc<str>,
    engine: Option<Arc<dyn ChatGenerator>>,
    slots: Arc<Semaphore>,
    /// How many permits `slots` was created with (`--parallel N`, floored at 1). `Semaphore` reports
    /// how many are AVAILABLE but not how many exist, and occupancy is the difference — so the
    /// capacity has to be remembered here or the periodic line cannot say `2/4`.
    capacity: usize,
}

#[derive(Clone)]
struct EmbeddingEntry {
    id: Arc<str>,
    engine: Arc<dyn EmbeddingGenerator>,
    slots: Arc<Semaphore>,
    capacity: usize,
}

impl EmbeddingEntry {
    fn new(id: &str, engine: Arc<dyn EmbeddingGenerator>, n_parallel: usize) -> Self {
        let capacity = n_parallel.max(1);
        Self {
            id: Arc::from(id),
            engine,
            slots: Arc::new(Semaphore::new(capacity)),
            capacity,
        }
    }
}

impl ModelEntry {
    fn new(id: &str, engine: Option<Arc<dyn ChatGenerator>>, n_parallel: usize) -> Self {
        let capacity = n_parallel.max(1);
        Self {
            id: Arc::from(id),
            engine,
            slots: Arc::new(Semaphore::new(capacity)),
            capacity,
        }
    }
}

/// Shared server state — a non-empty, ordered set of hosted [`ModelEntry`]s. A request is routed to
/// the entry whose `id` matches its `model` field; an unknown/empty `model` falls to the DEFAULT
/// (the first entry). The single-model server is just the one-entry case, so its behaviour — and the
/// hot path — is byte-identical to before this became multi-model.
#[derive(Clone)]
pub struct AppState {
    /// `models[0]` is the default chat route when chat is hosted. Embedding-only servers leave it
    /// empty and reject chat requests cleanly.
    models: Arc<Vec<ModelEntry>>,
    /// Independently routed embedding models. Their semaphores and engines do not share chat KV
    /// slots; future unified memory policy coordinates the resources below this API boundary.
    embeddings: Arc<Vec<EmbeddingEntry>>,
    /// The resolved process configuration (S7). The handler reads `serve.api_key` and
    /// `serve.max_tokens_cap` off it instead of the environment; it is an EXPLICIT constructor
    /// parameter on every entry point that can host a real model, so an embedder cannot silently
    /// end up with auth disabled by forgetting to pass one.
    cfg: Arc<Config>,
    /// Server-wide request/throughput counters (B10). Shared with the periodic reporter task
    /// [`stats_reporter`] that `serve_state` spawns; a state built for tests simply has no reporter
    /// draining it.
    stats: Arc<ServeStats>,
}

impl AppState {
    /// Wrap a single loaded generator for production use with `n_parallel` concurrent slots — the
    /// single-model server (the overwhelming common case). Equivalent to a one-entry
    /// [`multi`](Self::multi).
    pub fn new(
        generator: Arc<dyn ChatGenerator>,
        model_id: impl Into<String>,
        n_parallel: usize,
        cfg: Arc<Config>,
    ) -> Self {
        Self {
            models: Arc::new(vec![ModelEntry::new(
                &model_id.into(),
                Some(generator),
                n_parallel,
            )]),
            embeddings: Arc::new(Vec::new()),
            cfg,
            stats: Arc::default(),
        }
    }

    /// Host SEVERAL models at once, each with its OWN generator and per-model slot count. Each
    /// `(model_id, generator, n_parallel)` becomes a routable entry; the FIRST is the default route
    /// (used for a request with an unknown or empty `model`). This is the multi-device host: pass a
    /// generator pinned to each physical GPU and the server routes by model name — see `infr multi`.
    ///
    /// Panics if `entries` is empty (the state invariant is a non-empty model set); the CLI never
    /// calls it with none.
    pub fn multi(entries: Vec<(String, Arc<dyn ChatGenerator>, usize)>, cfg: Arc<Config>) -> Self {
        assert!(
            !entries.is_empty(),
            "AppState::multi needs at least one model"
        );
        let models = entries
            .into_iter()
            .map(|(id, gen, n)| ModelEntry::new(&id, Some(gen), n))
            .collect();
        Self {
            models: Arc::new(models),
            embeddings: Arc::new(Vec::new()),
            cfg,
            stats: Arc::default(),
        }
    }

    /// No-engine state — for /health, /v1/models, and serialisation tests. Takes the config too,
    /// so an auth/cap test drives `serve.*` through a `Config` value rather than the environment
    /// (R7).
    pub fn headless(model_id: impl Into<String>, cfg: Arc<Config>) -> Self {
        Self {
            models: Arc::new(vec![ModelEntry::new(&model_id.into(), None, 1)]),
            embeddings: Arc::new(Vec::new()),
            cfg,
            stats: Arc::default(),
        }
    }

    /// KV slot occupancy across every hosted model: `(busy, total)` permits. `busy` is what the
    /// semaphores are NOT handing out right now, which is exactly the set of generations in flight.
    fn slot_occupancy(&self) -> (u64, u64) {
        let chat = self.models.iter().fold((0, 0), |(busy, total), m| {
            let free = m.slots.available_permits().min(m.capacity);
            (busy + (m.capacity - free) as u64, total + m.capacity as u64)
        });
        self.embeddings.iter().fold(chat, |(busy, total), m| {
            let free = m.slots.available_permits().min(m.capacity);
            (busy + (m.capacity - free) as u64, total + m.capacity as u64)
        })
    }

    /// Route a request's `model` field to a hosted entry: an exact id match, else the default
    /// (first) entry — mirroring OpenAI servers, which never 404 on the model name. Returns a clone
    /// of the entry's `Arc` handles (cheap) so the caller owns them across the `spawn_blocking`.
    fn route(&self, requested: &str) -> ModelEntry {
        self.models
            .iter()
            .find(|m| &*m.id == requested)
            .unwrap_or(&self.models[0])
            .clone()
    }

    /// Attach one independently loaded embedding model to an existing chat state.
    pub fn with_embedding(
        mut self,
        generator: Arc<dyn EmbeddingGenerator>,
        model_id: impl Into<String>,
        n_parallel: usize,
    ) -> Self {
        let mut embeddings = (*self.embeddings).clone();
        embeddings.push(EmbeddingEntry::new(&model_id.into(), generator, n_parallel));
        self.embeddings = Arc::new(embeddings);
        self
    }

    /// State for a standalone embedding server.
    pub fn embedding(
        generator: Arc<dyn EmbeddingGenerator>,
        model_id: impl Into<String>,
        n_parallel: usize,
        cfg: Arc<Config>,
    ) -> Self {
        Self {
            models: Arc::new(Vec::new()),
            embeddings: Arc::new(vec![EmbeddingEntry::new(
                &model_id.into(),
                generator,
                n_parallel,
            )]),
            cfg,
            stats: Arc::default(),
        }
    }

    fn route_embedding(&self, requested: &str) -> Option<EmbeddingEntry> {
        let default = self.embeddings.first()?;
        Some(
            self.embeddings
                .iter()
                .find(|m| &*m.id == requested)
                .unwrap_or(default)
                .clone(),
        )
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum [`Router`].  Extracted so tests can call it with a [`AppState::headless`] state.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/embeddings", post(embeddings_handler))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

/// Start the OpenAI-compatible server bound to `addr`, serving `engine` under `model_id` with
/// `n_parallel` concurrent generation slots (`--parallel N`).
pub async fn serve(
    generator: Arc<dyn ChatGenerator>,
    model_id: String,
    addr: SocketAddr,
    n_parallel: usize,
    cfg: Arc<Config>,
) -> anyhow::Result<()> {
    serve_state(AppState::new(generator, model_id, n_parallel, cfg), addr).await
}

/// Start the server hosting SEVERAL models at once, each routed by its `model_id` and admitted with
/// its own `n_parallel` slot count (`entries[0]` is the default route). This is the multi-device
/// host (`infr multi`): each generator can be pinned to a different physical GPU, and the server
/// dispatches a request to the generator for the model it names. Graceful shutdown drains EVERY
/// model's in-flight requests before any backend drops (the axum layer stops accepting, then each
/// generation returns at its next abort poll — see [`shutdown_latched`]); when `serve_multi`
/// returns, the runtime drops and every backend's device is destroyed in turn.
pub async fn serve_multi(
    entries: Vec<(String, Arc<dyn ChatGenerator>, usize)>,
    addr: SocketAddr,
    cfg: Arc<Config>,
) -> anyhow::Result<()> {
    serve_state(AppState::multi(entries, cfg), addr).await
}

/// Start an embedding-only OpenAI-compatible server.
pub async fn serve_embedding(
    generator: Arc<dyn EmbeddingGenerator>,
    model_id: String,
    addr: SocketAddr,
    n_parallel: usize,
    cfg: Arc<Config>,
) -> anyhow::Result<()> {
    serve_state(
        AppState::embedding(generator, model_id, n_parallel, cfg),
        addr,
    )
    .await
}

/// Start one chat model and one embedding model in the same process/API endpoint.
#[allow(clippy::too_many_arguments)]
pub async fn serve_with_embedding(
    chat: Arc<dyn ChatGenerator>,
    chat_id: String,
    chat_parallel: usize,
    embedding: Arc<dyn EmbeddingGenerator>,
    embedding_id: String,
    embedding_parallel: usize,
    addr: SocketAddr,
    cfg: Arc<Config>,
) -> anyhow::Result<()> {
    let state = AppState::new(chat, chat_id, chat_parallel, cfg).with_embedding(
        embedding,
        embedding_id,
        embedding_parallel,
    );
    serve_state(state, addr).await
}

/// Bind + run the axum server over a fully-built [`AppState`] (single- or multi-model). The one
/// place the listener, the graceful-shutdown latch, and `axum::serve` live.
async fn serve_state(state: AppState, addr: SocketAddr) -> anyhow::Result<()> {
    let n_models = state.models.len() + state.embeddings.len();
    // The periodic throughput reporter (B10), unless `serve.stats_interval_secs` is 0. It reads the
    // SAME shutdown latch the drain path does, and it is aborted here as well: whichever way the
    // server ends, no task is left ticking.
    let reporter = stats_interval(&state.cfg)
        .map(|period| tokio::spawn(stats_reporter(state.clone(), period)));
    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, %n_models, "infr-server listening");
    let served = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_latched())
        .await;
    if let Some(h) = reporter {
        h.abort();
    }
    served?;
    Ok(())
}

/// Resolves once the process-wide shutdown latch is set (SIGINT/SIGTERM — see
/// [`infr_core::shutdown`]), which is `axum`'s cue to stop accepting connections and let the
/// in-flight ones finish. The requests themselves see the SAME latch through the decode loop's
/// abort poll, so each one stops issuing new GPU work at its next token/chunk boundary, drains what
/// it already submitted, and returns what it had — then `serve` returns, the runtime drops, the
/// engine drops, and the Vulkan device is destroyed properly. No `process::exit` anywhere on this
/// path: exiting under a live submit is the bug this whole mechanism exists to prevent.
///
/// A POLL (50 ms) rather than `tokio::signal::ctrl_c`, on purpose: the CLI already owns SIGINT and
/// SIGTERM via `sigaction`, and `tokio::signal` would install its OWN handler over the top of it —
/// last writer wins, and the loser's semantics (in this case, "drain the GPU") silently vanish. One
/// handler, one latch, everything downstream reads the latch.
async fn shutdown_latched() {
    while !infr_core::shutdown::shutdown_requested() {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    tracing::info!("shutdown requested — draining in-flight requests");
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Deliberately UNAUTHENTICATED, even when `serve.api_key` is configured. A load balancer, a
/// container orchestrator or an uptime probe has to reach this without holding the operator's
/// bearer token, and the response is a bare 200 with no body: it discloses nothing about the model
/// set, the config, or the machine. Every route that DOES disclose something is gated — see
/// [`auth_gate`].
async fn health_handler() -> StatusCode {
    StatusCode::OK
}

/// The hosted model list — GATED by the same bearer check as `/v1/chat/completions`.
///
/// This endpoint enumerates every model id the process is serving. With `serve.api_key` set the
/// operator has said the server is not open to whoever can reach the port, and an unauthenticated
/// caller must not be able to inventory what is hosted (which names to try, how many devices are
/// behind it, which private fine-tune is loaded) — that is reconnaissance, and it used to be free.
///
/// Returns a [`Response`] rather than `Json<ModelsResponse>` purely so the 401 can share the body
/// shape the chat handler returns; the 200 body is byte-identical to what it has always been.
/// With auth DISABLED (the default) [`auth_gate`] returns `None` and this is the same open endpoint
/// as before — the localhost experience does not change.
async fn models_handler(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    let mut data: Vec<ModelCard> = state
        .models
        .iter()
        .map(|m| ModelCard {
            id: m.id.to_string(),
            object: "model",
            owned_by: "local",
        })
        .collect();
    for model in state.embeddings.iter() {
        if !data
            .iter()
            .any(|card| card.id.as_str() == model.id.as_ref())
        {
            data.push(ModelCard {
                id: model.id.to_string(),
                object: "model",
                owned_by: "local",
            });
        }
    }
    Json(ModelsResponse {
        object: "list",
        data,
    })
    .into_response()
}

async fn embeddings_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<EmbeddingRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return param_error(None, e.body_text()),
    };
    if !matches!(req.encoding_format.as_deref(), None | Some("float")) {
        return param_error(
            Some("encoding_format"),
            "only encoding_format=\"float\" is supported".into(),
        );
    }
    let inputs = match req.input {
        EmbeddingInput::One(text) => vec![text],
        EmbeddingInput::Many(texts) if texts.is_empty() => {
            return param_error(Some("input"), "input array must not be empty".into())
        }
        EmbeddingInput::Many(texts) => texts,
    };
    let Some(entry) = state.route_embedding(&req.model) else {
        return json_error(StatusCode::NOT_FOUND, "no embedding model is loaded".into());
    };
    let model_id = entry.id.to_string();
    let queued = QueuedGuard::new(state.stats.clone());
    let Ok(permit) = entry.slots.clone().acquire_owned().await else {
        state.stats.fold_failure();
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        );
    };
    drop(queued);
    let active = ActiveGuard::new(state.stats.clone());
    let engine = entry.engine.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _active = active;
        engine.embed(&inputs)
    })
    .await
    .map_err(anyhow::Error::from)
    .and_then(|r| r);
    match result {
        Err(error) => {
            state.stats.fold_failure();
            json_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
        }
        Ok(outcome) => {
            state
                .stats
                .interval_prompt_tokens
                .fetch_add(u64::from(outcome.prompt_tokens), Ordering::Relaxed);
            state
                .stats
                .interval_completed
                .fetch_add(1, Ordering::Relaxed);
            Json(EmbeddingResponse {
                object: "list",
                data: outcome
                    .embeddings
                    .into_iter()
                    .enumerate()
                    .map(|(index, embedding)| EmbeddingData {
                        object: "embedding",
                        embedding,
                        index,
                    })
                    .collect(),
                model: model_id,
                usage: EmbeddingUsage {
                    prompt_tokens: outcome.prompt_tokens,
                    total_tokens: outcome.prompt_tokens,
                },
            })
            .into_response()
        }
    }
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    // Optional bearer auth, checked before any work, so an unauthenticated request cannot even
    // reach model routing / slot admission.
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    // Malformed JSON / wrong types: an OpenAI-shaped 400, not axum's default 422 text body.
    let Json(req) = match body {
        Ok(j) => j,
        Err(e) => return param_error(None, e.body_text()),
    };
    let mut params = match GenParams::from_request(&req) {
        Ok(p) => p,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    // Cap an absurd explicit budget so one request can't pin a slot forever. Unset stays unset.
    params.max_tokens = clamp_max_tokens(params.max_tokens, max_tokens_cap(&state.cfg));
    let messages: Vec<ChatMessage> = req.messages.iter().map(dto_to_engine).collect();
    // Pass the request's `tools` array THROUGH as a Value (moved into the blocking task) — no
    // Value→string→Value round-trip (audit finding 6).
    let tools: Option<serde_json::Value> = req.tools.clone();
    // A PRESENT-but-malformed forced tool_choice is a 400, not a silent downgrade to "auto".
    let tool_choice: Option<String> = match req.tool_choice.as_ref() {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match tool_choice_str(v) {
            Ok(tc) => tc,
            Err(e) => return param_error(Some(e.param), e.message),
        },
    };
    // …and a WELL-FORMED forced choice the server cannot honour is a 400 too: the constraint
    // builder ignores `tool_choice` entirely when `tools` is absent, so without this the request
    // would quietly generate ordinary text instead (B22).
    if let Err(e) = validate_tool_choice(tool_choice.as_deref(), tools.as_ref()) {
        return param_error(Some(e.param), e.message);
    }

    if state.models.is_empty() {
        return json_error(StatusCode::NOT_FOUND, "no chat model is loaded".into());
    }

    // Route to the hosted model named in the request (exact id), else the default (first) entry.
    // The response `model` field echoes the entry ACTUALLY served, not the raw request string, so a
    // client that omitted/mis-named the model sees which one answered.
    let entry = state.route(&req.model);
    let model_id = entry.id.to_string();
    let ctx = ReqCtx {
        id: next_req_id(),
        cid: make_id(),
        model_id,
        created: unix_ts(),
        // Resolved HERE, once, so both paths see the same policy and neither reaches for the config
        // from inside a blocking task. `None` (the default) = no deadline, exactly as before.
        deadline: request_timeout(&state.cfg),
        stream: req.stream,
        stats: state.stats.clone(),
    };
    log_request_start(
        ctx.id,
        "/v1/chat/completions",
        &ctx.model_id,
        messages.len(),
        messages.iter().map(|m| m.content.len()).sum(),
        params.max_tokens,
        ctx.stream,
    );

    if ctx.stream {
        streaming(entry, messages, tools, tool_choice, params, ctx).await
    } else {
        non_streaming(entry, messages, tools, tool_choice, params, ctx).await
    }
}

/// Everything about one in-flight request that is not its messages: identity (the log's `req` and
/// the wire's `chatcmpl-…`), the reply framing, the deadline policy, and the shared counters.
///
/// It exists because [`streaming`] and [`non_streaming`] took nine positional arguments and the
/// instrumentation wanted three more — at which point the next `String` inserted in the wrong slot
/// is a bug the compiler cannot see.
struct ReqCtx {
    /// Log-only, monotonic. See [`next_req_id`].
    id: u64,
    /// The client-facing completion id (`chatcmpl-…`).
    cid: String,
    /// The model that ACTUALLY answered, echoed on the wire.
    model_id: String,
    created: i64,
    deadline: Option<Duration>,
    stream: bool,
    stats: Arc<ServeStats>,
}

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

async fn non_streaming(
    entry: ModelEntry,
    messages: Vec<ChatMessage>,
    tools: Option<serde_json::Value>,
    tool_choice: Option<String>,
    params: GenParams,
    ctx: ReqCtx,
) -> Response {
    let ReqCtx {
        id: req_id,
        cid,
        model_id,
        created,
        deadline,
        stream,
        stats,
    } = ctx;
    // Wait for a free slot ON THIS MODEL. With `--parallel N`, the (N+1)'th concurrent request to
    // this model queues HERE — in the async runtime, holding no thread — and is admitted FIFO as
    // soon as one of that model's generations finishes. A different model's slots are independent.
    // The guard is what the periodic line's `queued` counts; it drops however the wait ends.
    let queued = QueuedGuard::new(stats.clone());
    let Ok(permit) = entry.slots.clone().acquire_owned().await else {
        stats.fold_failure();
        tracing::warn!(req = req_id, "rejected — server shutting down");
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        );
    };
    drop(queued);
    let active = ActiveGuard::new(stats.clone());
    let engine_arc = entry.engine.clone();
    let cid_blk = cid.clone();
    let stats_blk = stats.clone();

    // Per-request abort latch. It lives OUT here, not inside the closure, so the deadline below can
    // reach it: the generator polls it in its decode loop, and that poll is the only way to stop a
    // `spawn_blocking` task early. There is still no client-disconnect signal on this path (the
    // whole reply is buffered, so no send can fail), which is precisely why the deadline matters
    // most here — a client that hung up cannot be noticed, and burns its slot to completion.
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_blk = cancel.clone();

    let mut handle = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        // The permit is MOVED into the blocking task and dropped when it ends: a slot is held for
        // exactly the generation, and the next queued request is admitted the moment it frees. The
        // `active` gauge rides along with it, so the two can never disagree.
        let _permit = permit;
        let _active = active;
        let Some(engine) = engine_arc else {
            anyhow::bail!("no engine loaded");
        };

        let mut reasoning = String::new();
        let mut content = String::new();
        let mut tool_calls: Vec<OAIToolCall> = Vec::new();
        // Per-request tally: plain locals, folded into the shared counters once at the end.
        let mut tally = ReqTally::new();

        let outcome = engine
            .chat(
                &messages,
                tools.as_ref(),
                tool_choice.as_deref(),
                &params,
                &cancel_blk,
                &mut |delta| match delta {
                    Delta::Reasoning(t) => {
                        tally.on_text_delta(&stats_blk);
                        reasoning.push_str(&t);
                    }
                    Delta::Content(t) => {
                        tally.on_text_delta(&stats_blk);
                        content.push_str(&t);
                    }
                    Delta::ToolCall { name, arguments } => {
                        let idx = tool_calls.len();
                        tool_calls.push(OAIToolCall {
                            index: idx,
                            id: format!("call_{cid_blk}_{idx}"),
                            kind: "function",
                            function: OAIFunction { name, arguments },
                        });
                    }
                },
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok((reasoning, content, tool_calls, outcome, tally))
    });

    // The deadline, and the ONE thing about it that is easy to get wrong.
    //
    // `tokio::time::timeout(d, handle).await` returning `Err` and being propagated to the client is
    // NOT a fix: the blocking task is not cancellable, so it would keep decoding, keep holding its
    // `OwnedSemaphorePermit`, and the `--parallel` slot — the entire reason for having a deadline —
    // would stay occupied while we reported failure. So the timeout is used only to WAKE US: on
    // expiry we latch the abort flag the generator polls and then go back to awaiting the SAME
    // join. The task ends at its next token boundary, the permit drops, and we still have every
    // delta it produced before then, which is what the client gets.
    //
    // No watchdog TASK on this path (unlike [`streaming`]): the handler is already awaiting the
    // join, so the timer can live inside its own future. Nothing to spawn means nothing to tear
    // down — the `Timeout` future is dropped by the `.await` that resolves it, whichever way it
    // went, and a request that finishes in 5 ms under an hour-long deadline leaves nothing behind.
    let mut deadline_hit = false;
    let joined = match deadline {
        None => handle.await,
        Some(d) => match tokio::time::timeout(d, &mut handle).await {
            Ok(joined) => joined,
            Err(_elapsed) => {
                deadline_hit = true;
                cancel.store(true, Ordering::Relaxed);
                tracing::warn!(
                    req = req_id,
                    timeout_s = d.as_secs(),
                    "request deadline hit — returning the partial completion"
                );
                handle.await
            }
        },
    };
    let result = joined.map_err(anyhow::Error::from).and_then(|r| r);

    match result {
        Err(e) => {
            stats.fold_failure();
            tracing::warn!(req = req_id, error = %e, "request failed");
            json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        }
        Ok((reasoning, content, tool_calls, outcome, tally)) => {
            // A deadline hit is a TRUNCATION, not a failure: the client keeps the partial reply
            // (a 500 would throw away work it can use) and `finish_reason` says "length", which is
            // OpenAI's reason for a completion that ran out of budget.
            //
            // It has to be decided here because the generator cannot know. From inside the decode
            // loop, the latch we set is indistinguishable from any other abort, so it reports a
            // clean `Finish::Stop` — and "stop" tells the client the model finished its thought,
            // which would be a lie. Only the handler that armed the deadline knows it fired.
            //
            // A tool call still wins, as it does for every other reason: the call was emitted
            // whole, and a client that gets `tool_calls` can act on it.
            let finish = if !tool_calls.is_empty() {
                Finish::ToolCalls
            } else if deadline_hit {
                Finish::Length
            } else {
                outcome.finish
            };
            // Fold the request's tallies in ONCE, here, and log its completion line.
            let rec = tally.finish(outcome, finish);
            stats.fold_completion(&rec);
            log_request_done(req_id, &model_id, stream, &rec);
            Json(ChatCompletionResponse {
                id: cid,
                object: "chat.completion",
                created,
                model: model_id,
                choices: vec![CompletionChoice {
                    index: 0,
                    message: AssistantMessage {
                        role: "assistant",
                        content: if content.is_empty() {
                            None
                        } else {
                            Some(content.clone())
                        },
                        reasoning_content: if reasoning.is_empty() {
                            None
                        } else {
                            Some(reasoning)
                        },
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(tool_calls)
                        },
                    },
                    finish_reason: finish.as_str().into(),
                }],
                usage: UsageInfo::from_outcome(outcome),
            })
            .into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming path (SSE)
// ---------------------------------------------------------------------------

async fn streaming(
    entry: ModelEntry,
    messages: Vec<ChatMessage>,
    tools: Option<serde_json::Value>,
    tool_choice: Option<String>,
    params: GenParams,
    ctx: ReqCtx,
) -> Response {
    let ReqCtx {
        id: req_id,
        cid,
        model_id,
        created,
        deadline,
        stream,
        stats,
    } = ctx;
    // UNBOUNDED on purpose. The generator's `on_delta` callback is invoked from inside the decode
    // loop — which, under `--parallel N`, is holding the GPU baton. A bounded channel would make a
    // slow (or stalled) SSE client apply backpressure straight into that callback, so ONE
    // non-draining client would block the GPU step it is inside of and stall every OTHER sequence
    // behind it: precisely the head-of-line blocking this whole change exists to remove. Decoupling
    // the socket from the decode loop costs a queue whose depth is bounded anyway by `max_tokens`
    // (a few thousand short strings, worst case), which is the right trade.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();

    // Same per-model admission gate as the non-streaming path — the (N+1)'th concurrent stream to
    // this model queues here. Taken BEFORE the SSE response is returned, so a queued client simply
    // waits for its first byte rather than being handed an open-but-silent stream.
    let queued = QueuedGuard::new(stats.clone());
    let Ok(permit) = entry.slots.clone().acquire_owned().await else {
        stats.fold_failure();
        tracing::warn!(req = req_id, "rejected — server shutting down");
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        );
    };
    drop(queued);
    let active = ActiveGuard::new(stats.clone());
    let engine_arc = entry.engine.clone();
    // Clone sender + strings for use inside the on_delta callback closure.
    let tx_cb = tx.clone();
    let cid_cb = cid.clone();
    let model_cb = model_id.clone();
    let stats_cb = stats.clone();

    // Per-request abort latch: set when the client disconnects (an SSE `send` starts failing) and
    // polled by the generator's decode loop so it stops promptly and frees the GPU slot instead of
    // running to `max_tokens` into a dead socket (audit finding 2).
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_cb = cancel.clone();

    // Wall-clock deadline. Unlike the non-streaming path, nothing here awaits the generation's join
    // — the SSE response is returned as soon as the stream exists — so the timer needs its own
    // task, and that task needs an off switch. [`arm_deadline`] hands back a sender whose DROP is
    // that switch; it is moved into the blocking closure below, so the watchdog dies with the
    // generation (normally or by panic) instead of accumulating one sleeper per request.
    //
    // `deadline_hit` is separate from `cancel` because `cancel` is ALSO latched by a client
    // disconnect, and the finish chunk must not report a timeout as a hangup or vice versa.
    let deadline_hit = Arc::new(AtomicBool::new(false));
    let deadline_hit_cb = deadline_hit.clone();
    let done_tx = deadline.map(|d| arm_deadline(d, cancel.clone(), deadline_hit.clone()));

    tokio::task::spawn_blocking(move || {
        // Held for exactly this generation; freed for the next queued request on return. The
        // `active` gauge is released by the same return (or unwind).
        let _permit = permit;
        let _active = active;
        // Disarms the deadline watchdog when this task ends — see [`arm_deadline`]. `None` when no
        // deadline was configured, in which case there is no watchdog to disarm.
        let _done_tx = done_tx;
        // Closes the stream exactly once, however this closure ends. It emits `[DONE]` always, and
        // — unless `settled()` says a terminal frame already went out — reports the generation as a
        // failure. Both matter on an unwinding panic, which skips every arm below (B23).
        let mut done = DoneGuard {
            tx: tx.clone(),
            req_id,
            stats: stats.clone(),
            settled: false,
        };

        // First chunk: role delta (mirrors the Python shim's opening chunk).
        let _ = tx.send(Ok(sse_chunk(
            &cid,
            &model_id,
            created,
            DeltaPayload {
                role: Some("assistant".into()),
                ..Default::default()
            },
            None,
        )));

        let Some(engine) = engine_arc else {
            // `fail` sends the error frame, folds the statistic and logs; `DoneGuard` then closes
            // the stream with `[DONE]`.
            done.fail("no engine loaded");
            return;
        };

        let mut tc_index = 0usize;
        let mut saw_tool_call = false;
        // Per-request tally: plain locals inside the generation, folded in once below.
        let mut tally = ReqTally::new();

        let res = engine.chat(
            &messages,
            tools.as_ref(),
            tool_choice.as_deref(),
            &params,
            &cancel_cb,
            &mut |delta| {
                let payload = match delta {
                    Delta::Reasoning(t) => {
                        tally.on_text_delta(&stats_cb);
                        DeltaPayload {
                            reasoning_content: Some(t),
                            ..Default::default()
                        }
                    }
                    Delta::Content(t) => {
                        tally.on_text_delta(&stats_cb);
                        DeltaPayload {
                            content: Some(t),
                            ..Default::default()
                        }
                    }
                    Delta::ToolCall { name, arguments } => {
                        let tc = OAIToolCall {
                            index: tc_index,
                            id: format!("call_{cid_cb}_{tc_index}"),
                            kind: "function",
                            function: OAIFunction { name, arguments },
                        };
                        tc_index += 1;
                        saw_tool_call = true;
                        DeltaPayload {
                            tool_calls: Some(vec![tc]),
                            ..Default::default()
                        }
                    }
                };
                // A failed send means the receiver (the client's stream) is gone. Latch the abort so
                // the decode loop stops at its next poll and returns the slot.
                if tx_cb
                    .send(Ok(sse_chunk(&cid_cb, &model_cb, created, payload, None)))
                    .is_err()
                {
                    cancel_cb.store(true, Ordering::Relaxed);
                }
            },
        );

        match res {
            Ok(outcome) => {
                // Same honesty rule as the non-streaming path: the generator saw only an abort and
                // reports `Stop`, so the deadline has to relabel it "length" — the budget ran out,
                // the model did not finish. A tool call still wins.
                let finish = if saw_tool_call {
                    Finish::ToolCalls
                } else if deadline_hit_cb.load(Ordering::Relaxed) {
                    Finish::Length
                } else {
                    outcome.finish
                };
                // Finish chunk: empty delta + finish_reason.
                let _ = tx.send(Ok(sse_chunk(
                    &cid,
                    &model_id,
                    created,
                    DeltaPayload::default(),
                    Some(finish.as_str().into()),
                )));
                // Fold the request's tallies in ONCE, here, and log its completion line. The finish
                // chunk above IS this stream's terminal frame, so the guard must not also report a
                // failure when it drops.
                let rec = tally.finish(outcome, finish);
                stats.fold_completion(&rec);
                log_request_done(req_id, &model_id, stream, &rec);
                done.settled();
            }
            Err(e) => {
                // A mid-stream failure is NOT a clean `stop` — the error frame lets the client tell
                // this apart from success (matching the non-streaming 500). `[DONE]` still follows,
                // via `DoneGuard` (audit finding 1).
                done.fail(&e.to_string());
            }
        }
        // `DoneGuard` drops here (or on an unwinding panic). It sends `[DONE]`, and if nothing
        // above settled the stream it first reports the request as failed (B23).
    });

    // Bridge the mpsc receiver to an async Stream for axum's Sse.
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });

    Sse::new(stream).into_response()
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Closes the SSE stream exactly once when it drops, from the normal end of the generation OR from
/// an unwinding panic inside the decode closure.
///
/// Two jobs, and the second one exists because the first was not enough:
///
/// 1. `[DONE]` is ALWAYS the final frame. A strict SSE client blocks until it sees the sentinel, so
///    a swallowed panic must never leave it unsent (audit finding 1).
/// 2. A generation that never reached its terminal frame is reported AS A FAILURE — error frame,
///    `fold_failure`, and a `warn!` carrying the `req` id.
///
/// The streaming path discards its `spawn_blocking` join handle, so a panic in
/// `ChatGenerator::chat` unwinds past the `match res` and the `Err` arm never runs. Before this
/// guard owned the failure path, the wire showed the role chunk and then `[DONE]` with no error
/// frame and no terminal `finish_reason` — indistinguishable from a short success — while
/// `interval_failed` AND `interval_completed` both stayed zero, so the throughput line under-
/// reported the failure and the completion at once (backlog B23, proved).
///
/// [`Self::settled`] is what separates the two cases. Every path that produces its own terminal
/// frame calls it; anything that unwinds or returns early does not, and the guard speaks for it.
/// A flag rather than `std::thread::panicking()` because an early `return` is not a panic and still
/// leaves the stream unterminated.
struct DoneGuard {
    tx: tokio::sync::mpsc::UnboundedSender<Result<Event, Infallible>>,
    /// The request id, so the failure this guard reports joins its `request start` line.
    req_id: u64,
    stats: Arc<ServeStats>,
    /// Set by [`Self::settled`] once a terminal frame is out and the tallies are folded.
    settled: bool,
}

impl DoneGuard {
    /// The generation reached a terminal frame and folded its own tallies — the guard should send
    /// nothing but `[DONE]`.
    fn settled(&mut self) {
        self.settled = true;
    }

    /// Report this request as failed, exactly once: terminal error frame, `fold_failure`, `warn!`.
    ///
    /// The ONE place a streaming failure is recorded, so the wire frame, the statistic and the log
    /// line cannot drift apart or be emitted twice. Called explicitly by the paths that know why
    /// they failed, and by [`Drop`] for the paths that never got the chance.
    fn fail(&mut self, msg: &str) {
        if self.settled {
            return;
        }
        self.settled = true;
        // A failed send just means the client is already gone; the accounting still has to be
        // right, so the fold and the log are NOT conditional on it.
        let _ = self.tx.send(Ok(sse_error_event(msg)));
        self.stats.fold_failure();
        tracing::warn!(req = self.req_id, error = msg, "request failed");
    }
}

impl Drop for DoneGuard {
    fn drop(&mut self) {
        self.fail("internal error: generation ended without a terminal frame");
        let _ = self.tx.send(Ok(Event::default().data("[DONE]")));
    }
}

/// The OpenAI error envelope — `{"error": {"message": .., "type": ..}}` — that every failure this
/// server reports is wrapped in, whether it leaves as an HTTP body or as an SSE frame.
///
/// This is a WIRE format: clients (the OpenAI SDKs, `curl | jq .error.message`) match on the outer
/// key and on `type`, and two tests pin the shape. It was spelled out inline at three call sites,
/// two of which were character-for-character identical; the third ([`param_error`]) adds `param` and
/// `code` on top rather than being a different envelope. Building the common two fields here means
/// a renamed key cannot reach only some of the responses.
fn error_body(msg: &str, ty: &str) -> serde_json::Value {
    serde_json::json!({"error": {"message": msg, "type": ty}})
}

/// A terminal SSE error frame: `data: {"error":{...}}`. Distinguishable from a normal
/// `chat.completion.chunk` and from `[DONE]`, so a client can tell a mid-stream failure apart from a
/// clean completion (audit finding 1).
fn sse_error_event(msg: &str) -> Event {
    Event::default().data(error_body(msg, "server_error").to_string())
}

/// Serialize a delta payload into an SSE event carrying a `chat.completion.chunk`.
fn sse_chunk(
    cid: &str,
    model: &str,
    created: i64,
    delta: DeltaPayload,
    finish_reason: Option<String>,
) -> Event {
    let chunk = ChatCompletionChunk {
        id: cid.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
    };
    Event::default()
        .json_data(chunk)
        .expect("ChatCompletionChunk always serializes")
}

fn json_error(status: StatusCode, msg: String) -> Response {
    (status, Json(error_body(&msg, "server_error"))).into_response()
}

/// OpenAI-shaped 400 for a bad request parameter (`invalid_request_error`, with the offending
/// `param` named). NOT a clamp and NOT a panic — see [`GenParams::from_request`].
///
/// Two fields on top of the shared [`error_body`], and they are deliberately absent from the
/// server-error responses rather than emitted as nulls there: `param` only means something when a
/// specific request field is at fault, and OpenAI's own 5xx bodies carry neither. `code` is always
/// null — we mint no error codes — but it is present because SDKs read `error.code` on a 400.
/// Appending them keeps the key order (`message`, `type`, `param`, `code`) the responses have
/// always had; `serde_json`'s `preserve_order` makes that order observable on the wire.
fn param_error(param: Option<&str>, msg: String) -> Response {
    let mut body = error_body(&msg, "invalid_request_error");
    body["error"]["param"] = serde_json::json!(param);
    body["error"]["code"] = serde_json::Value::Null;
    (StatusCode::BAD_REQUEST, Json(body)).into_response()
}

/// Process-monotonic tie-breaker so two requests in the SAME millisecond (routine under
/// `--parallel N`) never mint the same completion `id` — which also keeps the derived
/// `call_{cid}_{idx}` tool-call ids unique (audit finding 4).
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn make_id() -> String {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let seq = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-{ms}-{seq}")
}

/// Default ceiling for `max_tokens`/`max_completion_tokens` when `INFR_MAX_TOKENS_CAP` is unset —
/// generous (128k) but finite, so one request cannot pin a slot for an absurd budget.
const DEFAULT_MAX_TOKENS_CAP: u32 = 131_072;

/// The configured `max_tokens` ceiling: `serve.max_tokens_cap` (`INFR_MAX_TOKENS_CAP`) if a
/// positive integer, else [`DEFAULT_MAX_TOKENS_CAP`]. Read per-request off the borrowed `Config`
/// the [`AppState`] owns (S7 — this used to be a per-request `std::env::var`).
///
/// The `> 0` guard stays HERE, at the accessor (R5): the env layer already drops a non-positive
/// `INFR_MAX_TOKENS_CAP`, but a config FILE can name the same field, and "must be a positive
/// integer, else the default" is this knob's grammar whatever layer supplied it.
fn max_tokens_cap(cfg: &Config) -> u32 {
    let cap = cfg.serve.max_tokens_cap;
    if cap > 0 {
        cap
    } else {
        DEFAULT_MAX_TOKENS_CAP
    }
}

/// Clamp an explicit `max_tokens` to `cap`. `None` (unset) passes through untouched — the generator
/// keeps applying its own `INFR_MAX_NEW` default, so behaviour is unchanged for requests that don't
/// ask for a budget. Only an absurdly large EXPLICIT value is capped (audit finding 5).
fn clamp_max_tokens(requested: Option<u32>, cap: u32) -> Option<u32> {
    requested.map(|v| v.min(cap))
}

/// The configured per-request wall-clock deadline, or `None` for "unbounded" — which is the
/// DEFAULT (`serve.request_timeout_secs` = 0) and today's behaviour.
///
/// `0` and unset are the same thing on purpose: the knob's whole grammar is "seconds, or 0 for no
/// deadline", so an operator can disarm a deadline a config file set by exporting
/// `INFR_REQUEST_TIMEOUT_SECS=0` — a `None`-only spelling could not express that from the
/// environment (see the env layer's note on why this knob is NOT `.filter(|v| v > 0)`).
///
/// Why a deadline exists at all: nothing else bounds how long one request may occupy a `--parallel`
/// slot. `max_tokens_cap` bounds TOKENS, and its 128k default is many hours on a slow model; on the
/// non-streaming path a client that has already gone away cannot even be detected (there is no send
/// to fail), so it burns its slot to completion. Why it is OFF by default: firing mid-generation
/// truncates a legitimate long reply, and only the operator knows their model's token rate.
fn request_timeout(cfg: &Config) -> Option<Duration> {
    let secs = cfg.serve.request_timeout_secs;
    (secs > 0).then(|| Duration::from_secs(secs))
}

/// Arm a wall-clock deadline over a generation that OUTLIVES the handler — i.e. the streaming path,
/// where the SSE response is returned immediately and the `spawn_blocking` task keeps producing
/// into a channel with nobody awaiting its join.
///
/// **It sets the generator's abort latch; it does not cancel anything.** `spawn_blocking` tasks are
/// NOT cancellable: `tokio::time::timeout` around a `JoinHandle` only stops the caller waiting —
/// the blocking thread runs on, still holding its `OwnedSemaphorePermit`, so the slot the deadline
/// exists to reclaim is exactly the thing that is not reclaimed. Latching `cancel` instead drives
/// the mechanism the decode loop already polls (see [`ChatGenerator::chat`]): the generator stops at
/// its next token boundary, returns normally, and the permit drops on the way out.
///
/// `hit` is a SECOND flag rather than a re-read of `cancel`, because `cancel` is also latched by a
/// client disconnect — the finish reason must say "the budget ran out", not "the socket died". It
/// is stored BEFORE `cancel` so a decode loop that observes the abort and returns instantly still
/// finds the label set.
///
/// **Teardown.** The returned [`oneshot::Sender`](tokio::sync::oneshot::Sender) is the "generation
/// finished" signal: move it into the blocking task and let it DROP there. A dropped sender
/// resolves the receiver (with `Err`), the `select!` takes that arm, and the watchdog task ends —
/// so a server that has served a million requests holds zero sleeping tasks. Drop is the right
/// trigger rather than an explicit `send`: it also fires while UNWINDING from a panic inside the
/// decode closure, which an explicit send at the end of the happy path would skip, leaking one task
/// per panicking request. (`JoinHandle::abort` would work too, but the handler would then have to
/// own and remember to abort a handle across two exit paths; a value whose destructor IS the signal
/// cannot be forgotten.)
fn arm_deadline(
    d: Duration,
    cancel: Arc<AtomicBool>,
    hit: Arc<AtomicBool>,
) -> tokio::sync::oneshot::Sender<()> {
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        tokio::select! {
            _ = tokio::time::sleep(d) => {
                hit.store(true, Ordering::Relaxed);
                cancel.store(true, Ordering::Relaxed);
                tracing::warn!(timeout_s = d.as_secs(), "request deadline hit — aborting generation");
            }
            _ = done_rx => {}
        }
    });
    done_tx
}

/// Optional bearer-token gate. `expected` is the configured API key (`INFR_API_KEY`), or `None`
/// when auth is DISABLED — in which case every request is allowed (default; preserves existing
/// localhost usage). When a key IS configured, the request must carry
/// `Authorization: Bearer <key>` with a matching token (audit finding 5).
///
/// The token comparison is CONSTANT TIME ([`subtle::ConstantTimeEq`]), not `==`. `str`/`[u8]`
/// equality bails out at the first differing byte, so the time to answer a request grows with how
/// many leading bytes of the key the attacker guessed right. Against a server reachable over a
/// network that is a byte-at-a-time oracle: guess byte 0 (256 tries), keep the value that answers
/// measurably slower, move to byte 1 — the key falls in time linear in its length instead of
/// exponential. `ct_eq` folds `|=` over the XOR of every byte pair and only then collapses the
/// accumulator to a `Choice`, so the answer costs the same whether the token differs in the first
/// byte or the last. It is used rather than a hand-rolled loop because "no early exit" also has to
/// survive the optimizer, which is exactly what `subtle` is written (and volatile-fenced) to do.
///
/// LENGTH IS STILL LEAKED — `ct_eq` short-circuits when the slices differ in length, so a wrong
/// token of the wrong length is rejected faster than a wrong token of the right one. That is an
/// ACCEPTED trade, not an oversight: an attacker learns only how many bytes to send, which does
/// not narrow the search of the CONTENT in any useful way (a 32-byte random key still has 32 bytes
/// of entropy to guess), and the alternatives — hashing both sides first, or padding to a fixed
/// buffer — buy nothing an operator can spend. What must not leak is a per-byte signal, and that
/// is what the fold removes.
fn authorize(expected: Option<&str>, auth_header: Option<&str>) -> bool {
    match expected {
        None => true,
        Some(key) => auth_header
            .and_then(|h| {
                h.strip_prefix("Bearer ")
                    .or_else(|| h.strip_prefix("bearer "))
            })
            .map(str::trim)
            .is_some_and(|tok| bool::from(tok.as_bytes().ct_eq(key.as_bytes()))),
    }
}

/// The optional bearer gate every PROTECTED route runs first: `Some(401)` when the request must be
/// refused, `None` when it may proceed.
///
/// Enforced ONLY when `serve.api_key` (`INFR_API_KEY`) is configured; with no key this returns
/// `None` for everything, so the default localhost server stays open exactly as it was.
///
/// It exists as ONE function because the gate is a policy, not a line of handler code: when
/// `/v1/models` was left ungated, an unauthenticated caller could still enumerate every hosted
/// model id off a server whose operator had set a key. A second handler spelling the check inline
/// is how that happens again, and a third would drift on the message or the status. The 401 body is
/// the shared [`error_body`] envelope, so it deserializes the same as every other failure this
/// server reports.
///
/// `/health` deliberately does NOT call this — see [`health_handler`].
fn auth_gate(cfg: &Config, headers: &HeaderMap) -> Option<Response> {
    let key = configured_api_key(cfg)?;
    let auth = headers.get("authorization").and_then(|v| v.to_str().ok());
    (!authorize(Some(key), auth)).then(|| {
        json_error(
            StatusCode::UNAUTHORIZED,
            "missing or invalid Authorization bearer token".into(),
        )
    })
}

/// The configured API key, or `None` when `serve.api_key` (`INFR_API_KEY`) is unset OR EMPTY —
/// auth disabled.
///
/// The empty-string filter is load-bearing and is NOT the `is_ok()` presence grammar every other
/// knob uses (§10.5): `INFR_API_KEY=` means "no auth", not "auth with the empty key". The env
/// layer already maps an empty value to `Some(None)`; the filter is kept here as well so a config
/// FILE saying `api_key = ""` means the same thing (S7 — this used to read the `INFR_API_KEY`
/// variable from the process environment and apply the same filter).
fn configured_api_key(cfg: &Config) -> Option<&str> {
    cfg.serve
        .api_key
        .as_deref()
        .filter(|k: &&str| !k.is_empty())
}

fn unix_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Flatten a DTO `content` field (string OR content-part array) to a plain `String`.
///
/// Mirrors the Python shim's `normalize_messages`: `"text"` parts are kept. Each `image_url`
/// part contributes an [`infr_engine::IMAGE_PART_PLACEHOLDER`] AT ITS ORIGINAL POSITION (the
/// template layer swaps each one for a vision marker before rendering, preserving the
/// client's text/image interleaving — PR #21 review fix). The image payloads themselves are
/// collected separately, in part order, by [`collect_images`] into [`ChatMessage::images`].
pub fn flatten_content(v: &Option<serde_json::Value>) -> String {
    match v {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| match p.get("type")?.as_str()? {
                "text" => p.get("text")?.as_str().map(str::to_owned),
                "image_url" => Some(infr_engine::IMAGE_PART_PLACEHOLDER.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(other) => other.to_string(),
    }
}

/// Collect a DTO `content` array's `image_url` part URLs (OpenAI shape:
/// `{"type":"image_url","image_url":{"url":…}}`), IN PART ORDER — the vision payloads the chat
/// layer pairs with the rendered prompt's `<|image_pad|>` markers (stage V5). A string content
/// (or any non-array) carries no images. Text joining (with per-image placeholders) stays in
/// [`flatten_content`].
fn collect_images(v: &Option<serde_json::Value>) -> Vec<String> {
    match v {
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                if p.get("type")?.as_str()? == "image_url" {
                    p.get("image_url")?.get("url")?.as_str().map(str::to_owned)
                } else {
                    None
                }
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn dto_to_engine(dto: &ChatMessageDto) -> ChatMessage {
    ChatMessage {
        role: dto.role.clone(),
        content: flatten_content(&dto.content),
        tool_calls: dto.tool_calls.as_ref().and_then(parse_oai_tool_calls),
        tool_call_id: dto.tool_call_id.clone(),
        name: dto.name.clone(),
        images: collect_images(&dto.content),
    }
}

/// Convert an inbound OpenAI `tool_calls` array (`[{function:{name, arguments}}]`, where `arguments`
/// is a JSON STRING) into engine [`ToolCall`]s with `arguments` parsed to a `Value`. Returns `None`
/// if the field isn't a non-empty array of valid calls.
fn parse_oai_tool_calls(v: &serde_json::Value) -> Option<Vec<ToolCall>> {
    let arr = v.as_array()?;
    let calls: Vec<ToolCall> = arr
        .iter()
        .filter_map(|c| {
            let f = c.get("function")?;
            let name = f.get("name")?.as_str()?.to_owned();
            let arguments = match f.get("arguments") {
                Some(serde_json::Value::String(s)) => {
                    serde_json::from_str(s).unwrap_or(serde_json::Value::String(s.clone()))
                }
                Some(other) => other.clone(),
                None => serde_json::Value::Object(serde_json::Map::new()),
            };
            Some(ToolCall { name, arguments })
        })
        .collect();
    (!calls.is_empty()).then_some(calls)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use infr_engine::IMAGE_PART_PLACEHOLDER;
    use tower::ServiceExt;

    /// Router backed by a headless state — no Engine, so /health and /v1/models work
    /// but /v1/chat/completions would return 500.  That's fine: we never call it here.
    fn test_router() -> Router {
        build_router(AppState::headless(
            "test-model",
            Arc::new(Config::default()),
        ))
    }

    // --- HTTP endpoint tests (no Engine required) ---------------------------

    #[tokio::test]
    async fn test_health_returns_200() {
        let resp = test_router()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_models_returns_200_with_expected_shape() {
        let resp = test_router()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(json["object"], "list");
        let card = &json["data"][0];
        assert_eq!(card["id"], "test-model");
        assert_eq!(card["object"], "model");
        assert_eq!(card["owned_by"], "local");
    }

    /// A stub generator that streams back exactly one content delta naming the model it belongs to,
    /// so a routing test can prove WHICH generator answered.
    struct EchoGen(&'static str);
    impl ChatGenerator for EchoGen {
        fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&serde_json::Value>,
            _tool_choice: Option<&str>,
            _params: &GenParams,
            _cancel: &AtomicBool,
            on_delta: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            on_delta(Delta::Content(format!("from:{}", self.0)));
            Ok(ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 3,
                cached_prompt_tokens: 0,
                completion_tokens: 2,
            })
        }
    }

    struct EchoEmbeddingGen;

    impl EmbeddingGenerator for EchoEmbeddingGen {
        fn embed(&self, inputs: &[String]) -> anyhow::Result<EmbeddingOutcome> {
            Ok(EmbeddingOutcome {
                embeddings: inputs
                    .iter()
                    .enumerate()
                    .map(|(index, input)| vec![index as f32, input.len() as f32])
                    .collect(),
                prompt_tokens: inputs.iter().map(|input| input.len() as u32).sum(),
            })
        }
    }

    fn embedding_router() -> Router {
        let generator: Arc<dyn EmbeddingGenerator> = Arc::new(EchoEmbeddingGen);
        build_router(AppState::embedding(
            generator,
            "test-embedding",
            2,
            Arc::new(Config::default()),
        ))
    }

    async fn post_embedding(body: &'static str) -> Response {
        embedding_router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn embeddings_accept_a_string_and_return_openai_shape() {
        let resp = post_embedding(
            r#"{"model":"test-embedding","input":"hello","encoding_format":"float"}"#,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["model"], "test-embedding");
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["data"][0]["index"], 0);
        assert_eq!(json["data"][0]["embedding"], serde_json::json!([0.0, 5.0]));
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["total_tokens"], 5);
    }

    #[tokio::test]
    async fn embeddings_accept_an_array_and_preserve_order() {
        let resp = post_embedding(r#"{"model":"test-embedding","input":["a","three"]}"#).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["data"][0]["embedding"], serde_json::json!([0.0, 1.0]));
        assert_eq!(json["data"][1]["embedding"], serde_json::json!([1.0, 5.0]));
        assert_eq!(json["usage"]["prompt_tokens"], 6);
    }

    #[tokio::test]
    async fn embeddings_reject_empty_input_and_base64_encoding() {
        let empty = post_embedding(r#"{"model":"test-embedding","input":[]}"#).await;
        assert_eq!(empty.status(), StatusCode::BAD_REQUEST);

        let base64 = post_embedding(
            r#"{"model":"test-embedding","input":"hello","encoding_format":"base64"}"#,
        )
        .await;
        assert_eq!(base64.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn configured_api_key_gates_the_embedding_endpoint() {
        let mut cfg = Config::default();
        cfg.serve.api_key = Some("s3cret".into());
        let generator: Arc<dyn EmbeddingGenerator> = Arc::new(EchoEmbeddingGen);
        let state = AppState::embedding(generator, "test-embedding", 1, Arc::new(cfg));
        let request = |authorization: Option<&str>| {
            let mut builder = Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json");
            if let Some(authorization) = authorization {
                builder = builder.header("authorization", authorization);
            }
            builder
                .body(Body::from(r#"{"model":"test-embedding","input":"hello"}"#))
                .unwrap()
        };

        let denied = build_router(state.clone())
            .oneshot(request(None))
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);

        let allowed = build_router(state)
            .oneshot(request(Some("Bearer s3cret")))
            .await
            .unwrap();
        assert_eq!(allowed.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn embedding_model_is_listed_alongside_chat_model() {
        let chat: Arc<dyn ChatGenerator> = Arc::new(EchoGen("chat"));
        let embedding: Arc<dyn EmbeddingGenerator> = Arc::new(EchoEmbeddingGen);
        let router = build_router(
            AppState::new(chat, "chat-model", 1, Arc::new(Config::default())).with_embedding(
                embedding,
                "embedding-model",
                1,
            ),
        );
        let resp = router
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let ids: Vec<&str> = json["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|card| card["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["chat-model", "embedding-model"]);
    }

    fn multi_router() -> Router {
        let a: Arc<dyn ChatGenerator> = Arc::new(EchoGen("alpha"));
        let b: Arc<dyn ChatGenerator> = Arc::new(EchoGen("beta"));
        build_router(AppState::multi(
            vec![("alpha".into(), a, 2), ("beta".into(), b, 2)],
            Arc::new(Config::default()),
        ))
    }

    #[tokio::test]
    async fn multi_models_are_all_listed() {
        let resp = multi_router()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let ids: Vec<&str> = json["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    /// A request naming `beta` must be answered by beta's generator; an unknown name falls to the
    /// default (first-listed `alpha`). This is the whole multi-model routing contract.
    async fn chat_model_field(router: Router, requested: &str) -> (String, String) {
        let body =
            format!(r#"{{"model":"{requested}","messages":[{{"role":"user","content":"hi"}}]}}"#);
        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        (
            json["model"].as_str().unwrap().to_string(),
            json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .to_string(),
        )
    }

    #[tokio::test]
    async fn request_routes_to_named_model() {
        let (model, content) = chat_model_field(multi_router(), "beta").await;
        assert_eq!(model, "beta");
        assert_eq!(content, "from:beta");
    }

    #[tokio::test]
    async fn unknown_model_falls_to_default() {
        let (model, content) = chat_model_field(multi_router(), "does-not-exist").await;
        assert_eq!(model, "alpha");
        assert_eq!(content, "from:alpha");
    }

    // --- ChatRequest serde round-trip tests --------------------------------

    #[test]
    fn chat_request_string_content_deserializes() {
        let raw = r#"{"model":"m","messages":[{"role":"user","content":"hello"}]}"#;
        let req: ChatRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(flatten_content(&req.messages[0].content), "hello");
        assert!(!req.stream);
        assert!(req.tools.is_none());
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn chat_request_array_content_flattens_text_and_collects_images() {
        let raw = r#"{
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type":"text","text":"hello"},
                    {"type":"image_url","image_url":{"url":"data:..."}},
                    {"type":"text","text":" world"}
                ]
            }]
        }"#;
        let req: ChatRequest = serde_json::from_str(raw).unwrap();
        // Text parts are concatenated; each image part leaves an IMAGE_PART_PLACEHOLDER AT its
        // position (the template layer substitutes the vision marker there), and the image
        // payload is COLLECTED onto the engine message (stage V5 — it used to be discarded).
        assert_eq!(
            flatten_content(&req.messages[0].content),
            format!("hello{IMAGE_PART_PLACEHOLDER} world")
        );
        let msg = dto_to_engine(&req.messages[0]);
        assert_eq!(msg.images, vec!["data:...".to_string()]);
    }

    #[test]
    fn chat_request_stream_flag() {
        let raw = r#"{"model":"m","messages":[],"stream":true}"#;
        let req: ChatRequest = serde_json::from_str(raw).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn chat_request_with_tools_and_max_tokens() {
        let raw = r#"{
            "model":"m","messages":[],
            "tools":[{"type":"function","function":{"name":"bash","description":"run bash"}}],
            "max_tokens":512
        }"#;
        let req: ChatRequest = serde_json::from_str(raw).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.max_tokens, Some(512));
    }

    #[test]
    fn chat_message_dto_with_tool_call_id_and_name() {
        let raw = r#"{"role":"tool","content":"result","tool_call_id":"tc_1","name":"bash"}"#;
        let msg: ChatMessageDto = serde_json::from_str(raw).unwrap();
        assert_eq!(msg.tool_call_id.as_deref(), Some("tc_1"));
        assert_eq!(msg.name.as_deref(), Some("bash"));
    }

    // --- Non-streaming response serialization tests ------------------------

    #[test]
    fn chat_completion_response_stop_serializes() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion",
            created: 1000,
            model: "test-model".into(),
            choices: vec![CompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: Some("hello".into()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: UsageInfo {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
            },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
        // skip_serializing_if = None → field absent in JSON → serde_json gives Null on access
        assert!(v["choices"][0]["message"]["reasoning_content"].is_null());
        assert!(v["choices"][0]["message"]["tool_calls"].is_null());
    }

    #[test]
    fn chat_completion_response_with_reasoning_content() {
        let resp = ChatCompletionResponse {
            id: "id".into(),
            object: "chat.completion",
            created: 0,
            model: "m".into(),
            choices: vec![CompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: Some("answer".into()),
                    reasoning_content: Some("I thought about it".into()),
                    tool_calls: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: UsageInfo {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
            },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(
            v["choices"][0]["message"]["reasoning_content"],
            "I thought about it"
        );
        assert_eq!(v["choices"][0]["message"]["content"], "answer");
    }

    #[test]
    fn chat_completion_response_tool_calls_finish_reason() {
        let resp = ChatCompletionResponse {
            id: "id".into(),
            object: "chat.completion",
            created: 0,
            model: "m".into(),
            choices: vec![CompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![OAIToolCall {
                        index: 0,
                        id: "call_0".into(),
                        kind: "function",
                        function: OAIFunction {
                            name: "bash".into(),
                            arguments: r#"{"command":"ls"}"#.into(),
                        },
                    }]),
                },
                finish_reason: "tool_calls".into(),
            }],
            usage: UsageInfo {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
            },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        let tc = &v["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "bash");
        assert_eq!(tc["function"]["arguments"], r#"{"command":"ls"}"#);
        assert_eq!(tc["index"], 0);
        // content: None serializes as null
        assert!(v["choices"][0]["message"]["content"].is_null());
    }

    // --- Streaming chunk serialization tests --------------------------------

    #[test]
    fn streaming_chunk_role_delta() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload {
                    role: Some("assistant".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        // Other delta fields absent
        assert!(v["choices"][0]["delta"]["content"].is_null());
        assert!(v["choices"][0]["delta"]["reasoning_content"].is_null());
        assert!(v["choices"][0]["delta"]["tool_calls"].is_null());
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn streaming_chunk_content_delta() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload {
                    content: Some("hello".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "hello");
        assert!(v["choices"][0]["delta"]["role"].is_null());
    }

    #[test]
    fn streaming_chunk_reasoning_content_delta() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload {
                    reasoning_content: Some("thinking...".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["choices"][0]["delta"]["reasoning_content"], "thinking...");
        assert!(v["choices"][0]["delta"]["content"].is_null());
    }

    #[test]
    fn streaming_chunk_tool_call_delta() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload {
                    tool_calls: Some(vec![OAIToolCall {
                        index: 0,
                        id: "call_0".into(),
                        kind: "function",
                        function: OAIFunction {
                            name: "bash".into(),
                            arguments: r#"{"cmd":"ls"}"#.into(),
                        },
                    }]),
                    ..Default::default()
                },
                finish_reason: None,
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        let tc = &v["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 0);
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "bash");
    }

    #[test]
    fn streaming_chunk_finish_reason() {
        let chunk = ChatCompletionChunk {
            id: "id".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaPayload::default(), // empty delta {}
                finish_reason: Some("stop".into()),
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        // Empty delta: all fields None → absent in JSON
        assert!(v["choices"][0]["delta"]["content"].is_null());
        assert!(v["choices"][0]["delta"]["role"].is_null());
    }

    // --- /v1/models serialization test ------------------------------------

    #[test]
    fn models_response_serializes() {
        let resp = ModelsResponse {
            object: "list",
            data: vec![ModelCard {
                id: "my-model".into(),
                object: "model",
                owned_by: "local",
            }],
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "my-model");
        assert_eq!(v["data"][0]["object"], "model");
        assert_eq!(v["data"][0]["owned_by"], "local");
    }

    // --- sampling param plumbing -------------------------------------------

    fn req(raw: &str) -> ChatRequest {
        serde_json::from_str(raw).unwrap()
    }

    #[test]
    fn absent_sampling_fields_stay_none() {
        // The whole point of Option-everything: a request that says nothing must not override the
        // process defaults (INFR_TEMP/TOP_K/TOP_P), i.e. today's behavior is preserved exactly.
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
        assert_eq!(p, GenParams::default());
    }

    #[test]
    fn sampling_fields_parse() {
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[],"temperature":0.0,"top_p":0.9,"top_k":40,"seed":42,
                "presence_penalty":0.5,"frequency_penalty":-0.25,"repeat_penalty":1.1,
                "stop":["\n\n","END"]}"#,
        ))
        .unwrap();
        assert_eq!(p.temperature, Some(0.0));
        assert_eq!(p.top_p, Some(0.9));
        assert_eq!(p.top_k, Some(40));
        assert_eq!(p.seed, Some(42));
        assert_eq!(p.presence_penalty, Some(0.5));
        assert_eq!(p.frequency_penalty, Some(-0.25));
        assert_eq!(p.repeat_penalty, Some(1.1));
        assert_eq!(p.stop, vec!["\n\n".to_string(), "END".to_string()]);
    }

    #[test]
    fn unknown_fields_are_ignored_not_rejected() {
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[],"n":1,"user":"bob","logit_bias":{},"logprobs":true}"#,
        ))
        .unwrap();
        assert_eq!(p, GenParams::default());
    }

    #[test]
    fn max_completion_tokens_is_preferred_alias() {
        let p = GenParams::from_request(&req(r#"{"model":"m","messages":[],"max_tokens":10}"#))
            .unwrap();
        assert_eq!(p.max_tokens, Some(10));
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[],"max_completion_tokens":20}"#,
        ))
        .unwrap();
        assert_eq!(p.max_tokens, Some(20));
        // Both present: the new name wins.
        let p = GenParams::from_request(&req(
            r#"{"model":"m","messages":[],"max_tokens":10,"max_completion_tokens":20}"#,
        ))
        .unwrap();
        assert_eq!(p.max_tokens, Some(20));
    }

    #[test]
    fn stop_accepts_a_bare_string() {
        let p =
            GenParams::from_request(&req(r#"{"model":"m","messages":[],"stop":"\n"}"#)).unwrap();
        assert_eq!(p.stop, vec!["\n".to_string()]);
    }

    #[test]
    fn empty_stop_strings_are_dropped() {
        // An empty stop would match at position 0 of every step — kill it at the door.
        let p =
            GenParams::from_request(&req(r#"{"model":"m","messages":[],"stop":["",""]}"#)).unwrap();
        assert!(p.stop.is_empty());
    }

    #[test]
    fn invalid_values_are_param_errors_not_clamps() {
        for (raw, param) in [
            (
                r#"{"model":"m","messages":[],"temperature":-1}"#,
                "temperature",
            ),
            (
                r#"{"model":"m","messages":[],"temperature":3}"#,
                "temperature",
            ),
            (r#"{"model":"m","messages":[],"top_p":5}"#, "top_p"),
            (r#"{"model":"m","messages":[],"top_k":-2}"#, "top_k"),
            (
                r#"{"model":"m","messages":[],"presence_penalty":-3}"#,
                "presence_penalty",
            ),
            (
                r#"{"model":"m","messages":[],"frequency_penalty":9}"#,
                "frequency_penalty",
            ),
            (
                r#"{"model":"m","messages":[],"repeat_penalty":0}"#,
                "repeat_penalty",
            ),
            (r#"{"model":"m","messages":[],"stop":[1,2]}"#, "stop"),
            (
                r#"{"model":"m","messages":[],"stop":["a","b","c","d","e"]}"#,
                "stop",
            ),
        ] {
            let e = GenParams::from_request(&req(raw)).unwrap_err();
            assert_eq!(e.param, param, "{raw}");
        }
    }

    #[tokio::test]
    async fn bad_temperature_returns_openai_shaped_400() {
        let resp = test_router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":-1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "temperature");
    }

    // --- StopMatcher: the token-boundary case -------------------------------

    /// Drive a matcher over a token sequence, returning (emitted text, hit).
    fn run_stops(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        for p in pieces {
            out.push_str(&m.push(p));
            if m.hit() {
                break;
            }
        }
        if !m.hit() {
            out.push_str(&m.flush());
        }
        (out, m.hit())
    }

    #[test]
    fn stop_within_one_token_fires_and_is_excluded() {
        let (out, hit) = run_stops(&["END"], &["hello ", "END", " more"]);
        assert!(hit);
        assert_eq!(out, "hello ");
    }

    #[test]
    fn stop_split_across_two_tokens_still_fires() {
        // THE boundary case: "END" arrives as "E" + "ND". It must fire, and the partial "E" must
        // NEVER have been emitted.
        let (out, hit) = run_stops(&["END"], &["hello ", "E", "ND", " more"]);
        assert!(hit, "a stop split across tokens must fire");
        assert_eq!(out, "hello ", "no partial stop prefix may leak");
    }

    #[test]
    fn stop_split_across_three_tokens_still_fires() {
        let (out, hit) = run_stops(&["<|done|>"], &["a", "<|", "do", "ne", "|>", "b"]);
        assert!(hit);
        assert_eq!(out, "a");
    }

    #[test]
    fn double_newline_stop_split_across_tokens() {
        let (out, hit) = run_stops(&["\n\n"], &["line", "\n", "\n", "next"]);
        assert!(hit);
        assert_eq!(out, "line");
    }

    #[test]
    fn stop_prefix_that_does_not_complete_is_eventually_emitted() {
        // "E" looked like the start of "END" but turned out to be "Every" — it must still be
        // delivered, exactly once, in order.
        let (out, hit) = run_stops(&["END"], &["E", "very", " day"]);
        assert!(!hit);
        assert_eq!(out, "Every day");
    }

    #[test]
    fn held_prefix_is_flushed_at_end_of_generation() {
        // Generation ends while the matcher still holds a partial prefix -> flush must emit it.
        let (out, hit) = run_stops(&["END"], &["done E"]);
        assert!(!hit);
        assert_eq!(out, "done E");
    }

    #[test]
    fn multiple_stops_take_the_earliest_match() {
        let (out, hit) = run_stops(&["World", "lo"], &["hel", "lo World"]);
        assert!(hit);
        assert_eq!(out, "hel");
    }

    #[test]
    fn no_stops_is_a_passthrough() {
        let (out, hit) = run_stops(&[], &["a", "b", "c"]);
        assert!(!hit);
        assert_eq!(out, "abc");
        assert!(!StopMatcher::new(vec![]).is_active());
    }

    #[test]
    fn multibyte_stop_split_mid_codepoint_is_safe() {
        // Pieces are always whole UTF-8, but a stop's own bytes may straddle them: "→END" arriving
        // as "→" + "END". Must fire without panicking on a char boundary.
        let (out, hit) = run_stops(&["→END"], &["x", "→", "END", "y"]);
        assert!(hit);
        assert_eq!(out, "x");
    }

    #[test]
    fn nothing_is_emitted_after_a_hit() {
        let mut m = StopMatcher::new(vec!["END".into()]);
        assert_eq!(m.push("aEND"), "a");
        assert!(m.hit());
        assert_eq!(m.push("more"), "");
        assert_eq!(m.flush(), "");
    }

    // --- finish_reason ------------------------------------------------------

    #[test]
    fn finish_reason_strings() {
        assert_eq!(Finish::Stop.as_str(), "stop");
        assert_eq!(Finish::Length.as_str(), "length");
        assert_eq!(Finish::ToolCalls.as_str(), "tool_calls");
    }

    // --- flatten_content unit tests ----------------------------------------

    #[test]
    fn flatten_content_string_value() {
        let v = Some(serde_json::Value::String("hello world".into()));
        assert_eq!(flatten_content(&v), "hello world");
    }

    #[test]
    fn flatten_content_array_skips_non_text_parts() {
        let v = Some(serde_json::json!([
            {"type": "text",      "text": "hello"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
            {"type": "text",      "text": " world"}
        ]));
        // Text parts are joined with an IMAGE_PART_PLACEHOLDER where each image part sat — the
        // payload itself is collected by `collect_images` (see the dto_to_engine test).
        assert_eq!(
            flatten_content(&v),
            format!("hello{IMAGE_PART_PLACEHOLDER} world")
        );
        assert_eq!(collect_images(&v), vec!["http://x".to_string()]);
    }

    /// PR #21 review fix: content-part ORDER must survive flattening. "text A → image 1 →
    /// text B → image 2" used to flatten to "A B" with both images appended after ALL text by
    /// the template, silently changing a multi-image prompt's semantics; now each image part
    /// leaves a placeholder at its original position, in part order.
    #[test]
    fn flatten_content_preserves_interleaved_image_positions() {
        let v = Some(serde_json::json!([
            {"type": "text",      "text": "A"},
            {"type": "image_url", "image_url": {"url": "data:one"}},
            {"type": "text",      "text": "B"},
            {"type": "image_url", "image_url": {"url": "data:two"}}
        ]));
        assert_eq!(
            flatten_content(&v),
            format!("A{IMAGE_PART_PLACEHOLDER}B{IMAGE_PART_PLACEHOLDER}")
        );
        assert_eq!(
            collect_images(&v),
            vec!["data:one".to_string(), "data:two".to_string()]
        );
    }

    /// dto_to_engine pairs the flattened text with the COLLECTED image payloads, in part order
    /// (stage V5): the text-only fields behave exactly as before, and a string content carries
    /// no images.
    #[test]
    fn dto_to_engine_collects_image_urls_in_part_order() {
        let dto = ChatMessageDto {
            role: "user".into(),
            content: Some(serde_json::json!([
                {"type": "image_url", "image_url": {"url": "data:first"}},
                {"type": "text",      "text": "two pics"},
                {"type": "image_url", "image_url": {"url": "data:second"}}
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let m = dto_to_engine(&dto);
        assert_eq!(
            m.content,
            format!("{IMAGE_PART_PLACEHOLDER}two pics{IMAGE_PART_PLACEHOLDER}")
        );
        assert_eq!(
            m.images,
            vec!["data:first".to_string(), "data:second".to_string()]
        );

        // A plain-string content (and a text-only array) carries no images.
        let text_dto = ChatMessageDto {
            role: "user".into(),
            content: Some(serde_json::json!("just text")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert!(dto_to_engine(&text_dto).images.is_empty());
    }

    #[test]
    fn flatten_content_none_gives_empty_string() {
        assert_eq!(flatten_content(&None), "");
    }

    #[test]
    fn flatten_content_null_json_gives_empty_string() {
        // An assistant tool-call message legally has `content: null` — it must flatten to "" (not the
        // literal "null", which would inject a stray word into the prompt).
        assert_eq!(flatten_content(&Some(serde_json::Value::Null)), "");
    }

    /// CHARACTERIZATION TEST — pins current behaviour, does not ask for a change.
    ///
    /// OpenAI's schema says `content` is a string or an array of content parts. A number, a
    /// bool or an object is a client bug. `flatten_content`'s fallback arm renders such a value
    /// with `Value::to_string()` (its JSON form) and feeds that to the prompt rather than
    /// erroring or dropping it. That leniency is DELIBERATE: a chat request that is merely
    /// sloppily typed still produces a sensible completion instead of a 400, and the rendered
    /// form is at least visible to the user in the echoed prompt rather than silently vanishing.
    ///
    /// It is pinned here because nothing else pins it: the arm is a one-liner a future refactor
    /// could plausibly "tidy" into `String::new()` or a rejection, silently changing what every
    /// such request produces. If this test fails, the change was intentional or it was not —
    /// decide, don't just re-bless.
    ///
    /// Note the exact rendering: `Value::to_string` is compact JSON, so a string INSIDE an
    /// object keeps its quotes, and `null` never reaches this arm (it is handled above).
    #[test]
    fn flatten_content_non_string_renders_json_leniently() {
        use serde_json::json;
        assert_eq!(flatten_content(&Some(json!(42))), "42");
        assert_eq!(flatten_content(&Some(json!(-1.5))), "-1.5");
        assert_eq!(flatten_content(&Some(json!(true))), "true");
        assert_eq!(flatten_content(&Some(json!(false))), "false");
        assert_eq!(flatten_content(&Some(json!({"a": 1}))), r#"{"a":1}"#);
        assert_eq!(
            flatten_content(&Some(json!({"a": "b", "c": [1, 2]}))),
            r#"{"a":"b","c":[1,2]}"#
        );
    }

    // --- make_id uniqueness (audit finding 4) ------------------------------

    #[test]
    fn make_id_is_unique_across_rapid_calls() {
        // Two completions minted in the same millisecond (routine under --parallel N) must NOT
        // collide — the monotonic suffix guarantees it even when the ms component is identical.
        let n = 10_000;
        let ids: std::collections::HashSet<String> = (0..n).map(|_| make_id()).collect();
        assert_eq!(ids.len(), n, "make_id produced a collision");
    }

    // --- usage totals (audit finding 3) ------------------------------------

    #[test]
    fn usage_total_is_prompt_plus_completion() {
        // The real fix: total == prompt + completion, from real counts, not content.len()/4.
        let outcome = ChatOutcome {
            finish: Finish::Stop,
            prompt_tokens: 17,
            cached_prompt_tokens: 0,
            completion_tokens: 5,
        };
        let usage = UsageInfo::from_outcome(outcome);
        assert_eq!(usage.total_tokens, 22);
        assert_eq!(usage.prompt_tokens_details.cached_tokens, 0);
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }

    #[test]
    fn usage_reports_full_prompt_and_cached_share() {
        let usage = UsageInfo::from_outcome(ChatOutcome {
            finish: Finish::Stop,
            prompt_tokens: 100,
            cached_prompt_tokens: 76,
            completion_tokens: 4,
        });
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.prompt_tokens_details.cached_tokens, 76);
        assert_eq!(usage.total_tokens, 104);
    }

    /// End-to-end: the non-streaming handler must surface the generator's REAL counts (EchoGen
    /// reports 3 prompt + 2 completion), not a byte-length estimate.
    #[tokio::test]
    async fn non_streaming_usage_comes_from_generator() {
        let resp = multi_router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"alpha","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["usage"]["prompt_tokens"], 3);
        assert_eq!(v["usage"]["completion_tokens"], 2);
        assert_eq!(v["usage"]["total_tokens"], 5);
    }

    // --- tool_choice parsing (audit finding 6) -----------------------------

    #[test]
    fn tool_choice_string_passes_through() {
        assert_eq!(
            tool_choice_str(&serde_json::json!("required")).unwrap(),
            Some("required".to_string())
        );
        assert_eq!(
            tool_choice_str(&serde_json::json!("auto")).unwrap(),
            Some("auto".to_string())
        );
    }

    /// B22: a forced `tool_choice` the server cannot honour is a 400, not free text.
    ///
    /// `tool_constraint_for` returns `None` as soon as `tools` is absent — without reading
    /// `tool_choice` at all — and `run_chat` treats that as "generate normally". So every case
    /// below used to produce ordinary assistant text while the client believed it had forced a
    /// call.
    #[test]
    fn a_forced_tool_choice_needs_tools_that_can_satisfy_it() {
        let tools = serde_json::json!([{"function": {"name": "bash"}}]);

        // Rejected: nothing to call.
        for choice in ["required", "bash"] {
            assert!(
                validate_tool_choice(Some(choice), None).is_err(),
                "{choice:?} with no tools must 400"
            );
            assert!(
                validate_tool_choice(Some(choice), Some(&serde_json::json!([]))).is_err(),
                "{choice:?} with an empty tools array must 400"
            );
        }
        // Rejected: names no tool that was offered. This is the case that built `{"anyOf": []}`.
        assert!(validate_tool_choice(Some("bogus"), Some(&tools)).is_err());

        // Accepted: policies that are satisfiable without tools, and a name that is present.
        assert!(validate_tool_choice(Some("auto"), None).is_ok());
        assert!(
            validate_tool_choice(Some("none"), None).is_ok(),
            "`none` asks for no tool call, which needs no tools"
        );
        assert!(validate_tool_choice(None, None).is_ok());
        assert!(validate_tool_choice(Some("required"), Some(&tools)).is_ok());
        assert!(validate_tool_choice(Some("bash"), Some(&tools)).is_ok());
    }

    /// The same rule over HTTP, so the wire really returns 400 and names the offending parameter.
    #[tokio::test]
    async fn forced_tool_choice_without_tools_is_a_400() {
        let resp = multi_router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"a","messages":[{"role":"user","content":"hi"}],"tool_choice":"required"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["param"], "tool_choice");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn tool_choice_named_function_forces_that_tool() {
        let v = serde_json::json!({"type":"function","function":{"name":"bash"}});
        assert_eq!(tool_choice_str(&v).unwrap(), Some("bash".to_string()));
    }

    #[test]
    fn tool_choice_object_without_name_is_an_error_not_auto() {
        // The bug: a forced-tool object lacking function.name silently downgraded to "auto". It must
        // now be a 400 (ParamError), distinguishable from ABSENT (which the caller maps to None).
        let v = serde_json::json!({"type":"function","function":{}});
        let e = tool_choice_str(&v).unwrap_err();
        assert_eq!(e.param, "tool_choice");
        let v = serde_json::json!({"type":"function"});
        assert!(tool_choice_str(&v).is_err());
    }

    #[test]
    fn tool_choice_wrong_shape_is_an_error() {
        assert!(tool_choice_str(&serde_json::json!(42)).is_err());
        assert!(tool_choice_str(&serde_json::json!(["auto"])).is_err());
    }

    // --- max_tokens clamp (audit finding 5) --------------------------------

    #[test]
    fn max_tokens_clamps_absurd_values_but_passes_unset_and_normal() {
        // Unset stays unset — the generator keeps applying its own INFR_MAX_NEW default (behaviour
        // unchanged for requests that don't ask for a budget).
        assert_eq!(clamp_max_tokens(None, 1000), None);
        // A sane value under the cap is untouched.
        assert_eq!(clamp_max_tokens(Some(512), 1000), Some(512));
        // An absurd value is capped, not rejected.
        assert_eq!(clamp_max_tokens(Some(10_000_000), 1000), Some(1000));
        // Exactly the cap passes.
        assert_eq!(clamp_max_tokens(Some(1000), 1000), Some(1000));
    }

    // --- optional bearer auth (audit finding 5) ----------------------------

    #[test]
    fn auth_disabled_allows_everything() {
        // No key configured (None) => open, regardless of what the client sends.
        assert!(authorize(None, None));
        assert!(authorize(None, Some("Bearer whatever")));
        assert!(authorize(None, Some("garbage")));
    }

    #[test]
    fn auth_enabled_requires_matching_bearer() {
        let key = Some("s3cret");
        // Correct token (either capitalisation of the scheme) => allowed.
        assert!(authorize(key, Some("Bearer s3cret")));
        assert!(authorize(key, Some("bearer s3cret")));
        // Wrong token, missing header, or malformed scheme => denied.
        assert!(!authorize(key, Some("Bearer wrong")));
        assert!(!authorize(key, None));
        assert!(!authorize(key, Some("s3cret")));
        assert!(!authorize(key, Some("Basic s3cret")));
    }

    /// The constant-time swap is a comparison, not a hash or a prefix check — so pin the two
    /// mistakes a hand-rolled "no early exit" loop typically makes.
    ///
    /// 1. SAME-LENGTH tokens must still be decided on CONTENT. A fold that ORs XORs together is
    ///    easy to get wrong in the direction of always-equal (e.g. accumulating `&=`, or seeding
    ///    the accumulator with the wrong identity), and same-length pairs are the only case where
    ///    that bug is invisible to the length check.
    /// 2. A correct PREFIX must be rejected. `ct_eq` answers `false` on a length mismatch, but a
    ///    zip-based loop silently compares only `min(len)` bytes and would accept `"s3c"` for
    ///    `"s3cret"` — the exact shape an attacker walks a key out with.
    #[test]
    fn auth_compares_full_token_not_a_prefix() {
        let key = Some("s3cret");
        // Same length as the wrong token below, and correct => allowed.
        assert!(authorize(key, Some("Bearer s3cret")));
        // Same length, differs only in the LAST byte => denied.
        assert!(!authorize(key, Some("Bearer s3creT")));
        // Same length, differs only in the FIRST byte => denied.
        assert!(!authorize(key, Some("Bearer S3cret")));
        // Correct prefix, short => denied (never "equal so far, therefore equal").
        assert!(!authorize(key, Some("Bearer s3c")));
        assert!(!authorize(key, Some("Bearer s")));
        assert!(!authorize(key, Some("Bearer ")));
        // Correct prefix, long => denied.
        assert!(!authorize(key, Some("Bearer s3cretx")));
        // The header's `str::trim` still runs before the compare (surrounding whitespace is not
        // part of the token), so a padded correct token is accepted and a padded wrong one is not.
        assert!(authorize(key, Some("Bearer   s3cret  ")));
        assert!(!authorize(key, Some("Bearer   s3cre  ")));
    }

    // --- the `serve.*` knobs, driven through a `Config` (S7, R7: never the environment) ------

    /// `serve.api_key` decides whether auth is on, and the EMPTY string still means OFF — the
    /// grammar `INFR_API_KEY=` has always had (§10.5). Getting this backwards would turn an empty
    /// key into a credential every request has to guess, so it is asserted explicitly.
    #[test]
    fn configured_api_key_reads_the_config_and_empty_still_means_no_auth() {
        let mut cfg = Config::default();
        assert_eq!(configured_api_key(&cfg), None, "unset => auth disabled");

        cfg.serve.api_key = Some(String::new());
        assert_eq!(configured_api_key(&cfg), None, "empty => auth DISABLED");

        cfg.serve.api_key = Some("hunter2".into());
        assert_eq!(configured_api_key(&cfg), Some("hunter2"));
        // …and a configured key really does gate the request.
        assert!(authorize(configured_api_key(&cfg), Some("Bearer hunter2")));
        assert!(!authorize(configured_api_key(&cfg), None));
    }

    /// `serve.max_tokens_cap` feeds the clamp, and a non-positive value falls back to the shipped
    /// default rather than clamping every request to zero.
    #[test]
    fn max_tokens_cap_reads_the_config_and_rejects_non_positive() {
        let mut cfg = Config::default();
        assert_eq!(max_tokens_cap(&cfg), DEFAULT_MAX_TOKENS_CAP);

        cfg.serve.max_tokens_cap = 4096;
        assert_eq!(max_tokens_cap(&cfg), 4096);
        assert_eq!(
            clamp_max_tokens(Some(10_000), max_tokens_cap(&cfg)),
            Some(4096)
        );

        cfg.serve.max_tokens_cap = 0;
        assert_eq!(max_tokens_cap(&cfg), DEFAULT_MAX_TOKENS_CAP);
    }

    /// End to end through the router: a state built with `serve.api_key` set answers 401 without a
    /// bearer token and gets past auth with one. Proves the handler reads the state's config, not
    /// the process environment.
    #[tokio::test]
    async fn configured_api_key_gates_the_chat_endpoint() {
        let mut cfg = Config::default();
        cfg.serve.api_key = Some("s3cret".into());
        let state = AppState::headless("test-model", Arc::new(cfg));
        let body = r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#;
        let req = |auth: Option<&str>| {
            let mut b = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json");
            if let Some(a) = auth {
                b = b.header("authorization", a);
            }
            b.body(Body::from(body)).unwrap()
        };
        let resp = build_router(state.clone())
            .oneshot(req(None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        // With the right token the request gets PAST auth (headless => 500 from the missing
        // engine, which is exactly "not 401").
        let resp = build_router(state)
            .oneshot(req(Some("Bearer s3cret")))
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    /// `/v1/models` enumerates what the process is hosting, so a configured key must gate it too —
    /// otherwise anyone who can reach the port learns every model id for free, on a server whose
    /// operator explicitly said it is not open. `/health` must stay OPEN through the same config: a
    /// load balancer holds no bearer token, and a bare 200 discloses nothing.
    #[tokio::test]
    async fn configured_api_key_gates_models_but_never_health() {
        let mut cfg = Config::default();
        cfg.serve.api_key = Some("s3cret".into());
        let state = AppState::headless("test-model", Arc::new(cfg));
        let get = |uri: &str, auth: Option<&str>| {
            let mut b = Request::builder().uri(uri);
            if let Some(a) = auth {
                b = b.header("authorization", a);
            }
            b.body(Body::empty()).unwrap()
        };

        // No bearer => 401, in the same envelope the chat handler returns.
        let resp = build_router(state.clone())
            .oneshot(get("/v1/models", None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "server_error");
        assert_eq!(
            v["error"]["message"],
            "missing or invalid Authorization bearer token"
        );
        // …and the model list did NOT leak into the error body.
        assert!(
            !String::from_utf8_lossy(&bytes).contains("test-model"),
            "a 401 must not disclose the model set: {bytes:?}"
        );

        // Wrong bearer => still 401.
        let resp = build_router(state.clone())
            .oneshot(get("/v1/models", Some("Bearer wrong")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Right bearer => the unchanged 200 body.
        let resp = build_router(state.clone())
            .oneshot(get("/v1/models", Some("Bearer s3cret")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "test-model");
        assert_eq!(v["data"][0]["object"], "model");
        assert_eq!(v["data"][0]["owned_by"], "local");

        // /health is open both ways.
        for auth in [None, Some("Bearer s3cret")] {
            let resp = build_router(state.clone())
                .oneshot(get("/health", auth))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "auth={auth:?}");
        }
    }

    // --- per-request wall-clock deadline ------------------------------------

    /// `serve.request_timeout_secs` is OFF (unbounded) by default, and `0` explicitly means the
    /// same thing — a deadline truncates legitimate long replies, so it is opt-in only.
    #[test]
    fn request_timeout_is_off_by_default_and_zero_means_unbounded() {
        let mut cfg = Config::default();
        assert_eq!(request_timeout(&cfg), None, "default => no deadline");

        cfg.serve.request_timeout_secs = 0;
        assert_eq!(request_timeout(&cfg), None, "0 => no deadline");

        cfg.serve.request_timeout_secs = 300;
        assert_eq!(request_timeout(&cfg), Some(Duration::from_secs(300)));
    }

    /// A generator that never stops on its own: it emits one delta and then polls `cancel` until
    /// something latches it — exactly the shape of a real decode loop, and the only way to test
    /// that the deadline reaches the abort mechanism rather than just abandoning the join. Returns
    /// `Finish::Stop`, because that is what a real generator reports when it is aborted: it cannot
    /// tell WHY the flag was set, which is why the handler has to relabel.
    struct LoopGen;
    impl ChatGenerator for LoopGen {
        fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&serde_json::Value>,
            _tool_choice: Option<&str>,
            _params: &GenParams,
            cancel: &AtomicBool,
            on_delta: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            on_delta(Delta::Content("partial".into()));
            while !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(2));
            }
            Ok(ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 3,
                cached_prompt_tokens: 0,
                completion_tokens: 1,
            })
        }
    }

    /// One entry hosting `gen`, routed as `"m"` — the handler-level fixture the deadline tests
    /// need, since `non_streaming`/`streaming` take a `ModelEntry` and a deadline directly (the
    /// knob is in whole SECONDS, so driving these through the router could not stay sub-second).
    fn deadline_entry(generator: Arc<dyn ChatGenerator>) -> ModelEntry {
        AppState::new(generator, "m", 1, Arc::new(Config::default())).route("m")
    }

    /// The per-request context the two generation paths take, with a fresh (undrained) stats sink —
    /// the handler builds this from the request; a test that calls `streaming`/`non_streaming`
    /// directly has to build its own.
    fn test_ctx(deadline: Option<Duration>, stream: bool) -> ReqCtx {
        ReqCtx {
            id: next_req_id(),
            cid: "cid".into(),
            model_id: "m".into(),
            created: 0,
            deadline,
            stream,
            stats: Arc::default(),
        }
    }

    fn user_msg() -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            images: Vec::new(),
        }]
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// The non-streaming deadline: it fires, the client still gets what was generated, and the
    /// finish reason is `length` — NOT a 500, and not the `stop` the generator itself reported.
    #[tokio::test]
    async fn non_streaming_deadline_returns_partial_content_as_length() {
        let started = std::time::Instant::now();
        let resp = non_streaming(
            deadline_entry(Arc::new(LoopGen)),
            user_msg(),
            None,
            None,
            GenParams::default(),
            test_ctx(Some(Duration::from_millis(150)), false),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK, "a deadline is not a failure");
        let v = body_json(resp).await;
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(
            v["choices"][0]["message"]["content"], "partial",
            "the partial completion must survive the deadline"
        );
        // The handler kept awaiting the join after latching the abort, so by the time it answered
        // the blocking task had really ended — and with it, its slot permit.
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the deadline did not stop the generator"
        );
    }

    /// The permit is genuinely back: the entry admits `--parallel 1` at a time, so a SECOND request
    /// on the same entry can only be served if the first one's blocking task actually ended. If the
    /// deadline had merely timed out the join and left the task running, this would hang.
    #[tokio::test]
    async fn deadline_frees_the_slot_for_the_next_request() {
        let entry = deadline_entry(Arc::new(LoopGen));
        for _ in 0..2 {
            let resp = non_streaming(
                entry.clone(),
                user_msg(),
                None,
                None,
                GenParams::default(),
                test_ctx(Some(Duration::from_millis(100)), false),
            )
            .await;
            assert_eq!(
                body_json(resp).await["choices"][0]["finish_reason"],
                "length"
            );
        }
        assert_eq!(
            entry.slots.available_permits(),
            1,
            "the slot must be back after a deadline hit"
        );
    }

    /// A request that finishes well inside its deadline is untouched: real content, the
    /// generator's OWN finish reason, and no relabelling.
    #[tokio::test]
    async fn generation_inside_the_deadline_is_untouched() {
        let resp = non_streaming(
            deadline_entry(Arc::new(EchoGen("alpha"))),
            user_msg(),
            None,
            None,
            GenParams::default(),
            test_ctx(Some(Duration::from_secs(30)), false),
        )
        .await;
        let v = body_json(resp).await;
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["choices"][0]["message"]["content"], "from:alpha");
    }

    /// The streaming deadline: the deltas already sent are kept, the finish chunk says `length`,
    /// and `[DONE]` still closes the stream.
    #[tokio::test]
    async fn streaming_deadline_finishes_with_length() {
        let resp = streaming(
            deadline_entry(Arc::new(LoopGen)),
            user_msg(),
            None,
            None,
            GenParams::default(),
            test_ctx(Some(Duration::from_millis(150)), true),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("partial"), "partial content lost: {text}");
        assert!(
            text.contains("\"finish_reason\":\"length\""),
            "a deadline hit must finish as `length`: {text}"
        );
        assert!(
            !text.contains("\"finish_reason\":\"stop\""),
            "the generator's `stop` must not reach the wire: {text}"
        );
        assert!(text.contains("[DONE]"), "sentinel missing: {text}");
    }

    /// The watchdog must not outlive its request. Dropping the "generation finished" sender — which
    /// is what the blocking task does when it returns, or unwinds — has to end the timer task, or a
    /// long-lived server accumulates one sleeping task per request. Asserted by waiting PAST the
    /// deadline and finding the flags still clear.
    #[tokio::test]
    async fn dropping_the_done_signal_disarms_the_watchdog() {
        let cancel = Arc::new(AtomicBool::new(false));
        let hit = Arc::new(AtomicBool::new(false));
        let done_tx = arm_deadline(Duration::from_millis(50), cancel.clone(), hit.clone());
        drop(done_tx); // the generation finished first
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            !cancel.load(Ordering::Relaxed) && !hit.load(Ordering::Relaxed),
            "a disarmed watchdog must never fire"
        );

        // …and one that is NOT disarmed does fire, on both flags (so the finish reason can tell a
        // deadline apart from a client disconnect, which latches only `cancel`).
        let cancel = Arc::new(AtomicBool::new(false));
        let hit = Arc::new(AtomicBool::new(false));
        let _done_tx = arm_deadline(Duration::from_millis(50), cancel.clone(), hit.clone());
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(cancel.load(Ordering::Relaxed) && hit.load(Ordering::Relaxed));
    }

    // --- finish-reason mapping: Err is an error frame, never `stop` (finding 1) ---

    /// A generator whose `chat` fails mid-stream. The streaming path must NOT relabel this as a
    /// clean `stop`; it must emit a terminal error frame, then `[DONE]`.
    struct FailGen;
    impl ChatGenerator for FailGen {
        fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&serde_json::Value>,
            _tool_choice: Option<&str>,
            _params: &GenParams,
            _cancel: &AtomicBool,
            on_delta: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            on_delta(Delta::Content("partial".into()));
            anyhow::bail!("boom mid-stream")
        }
    }

    #[tokio::test]
    async fn streaming_error_emits_error_frame_not_stop() {
        let g: Arc<dyn ChatGenerator> = Arc::new(FailGen);
        let router = build_router(AppState::new(g, "m", 1, Arc::new(Config::default())));
        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        // A terminal error frame is present, the stream is closed with [DONE], and NO finish chunk
        // ever claimed a clean "stop".
        assert!(text.contains("\"error\""), "missing error frame: {text}");
        assert!(
            text.contains("boom mid-stream"),
            "error message lost: {text}"
        );
        assert!(text.contains("[DONE]"), "sentinel missing: {text}");
        assert!(
            !text.contains("\"finish_reason\":\"stop\""),
            "an Err must not report a clean stop: {text}"
        );
    }

    /// A generator whose `chat` PANICS mid-stream, which is not the same as returning `Err`: the
    /// unwind skips every arm of the `match res` below it, so before B23 the only thing that ran
    /// was `DoneGuard`'s `[DONE]`.
    struct PanicGen;
    impl ChatGenerator for PanicGen {
        fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&serde_json::Value>,
            _tool_choice: Option<&str>,
            _params: &GenParams,
            _cancel: &AtomicBool,
            on_delta: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            on_delta(Delta::Content("partial".into()));
            panic!("boom: generator panicked mid-stream")
        }
    }

    /// B23: a panicking generator must terminate the stream like a FAILURE, not like a success.
    ///
    /// The bug this pins is one of omission, so assert on what was missing: an error frame, and a
    /// `failed` count. Both were absent while `[DONE]` was present, which is precisely what made a
    /// panic indistinguishable from a short completion on the wire.
    #[tokio::test]
    async fn streaming_panic_is_reported_as_a_failure() {
        let g: Arc<dyn ChatGenerator> = Arc::new(PanicGen);
        let state = AppState::new(g, "m", 1, Arc::new(Config::default()));
        let stats = state.stats.clone();
        let router = build_router(state);
        // The panic itself is expected; keep the default hook from spraying the test log with it.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let resp = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        std::panic::set_hook(prev);
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("\"error\""),
            "a panicking generator must still put a terminal error frame on the wire: {text}"
        );
        assert!(text.contains("[DONE]"), "sentinel missing: {text}");
        assert!(
            !text.contains("\"finish_reason\":\"stop\""),
            "a panic must never look like a clean stop: {text}"
        );

        let w = stats.drain(Duration::from_secs(1));
        assert_eq!(w.failed, 1, "the panic must be counted as a failure");
        assert_eq!(
            w.completed, 1,
            "a failure is still a completed request for the interval line"
        );
        assert_eq!(w.active, 0, "the slot gauge must be released by the unwind");
    }

    /// The abort-on-disconnect decision, unit-tested without a socket: when the SSE `send` returns
    /// Err (receiver dropped), the callback must latch the per-request cancel flag. Full
    /// disconnect→GPU-slot-release is integration-only (needs a live generator + real client drop).
    #[test]
    fn send_failure_latches_the_cancel_flag() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let cancel = AtomicBool::new(false);
        drop(rx); // simulate the client disconnecting: the receiver is gone.
        if tx.send(Ok(Event::default().data("x"))).is_err() {
            cancel.store(true, Ordering::Relaxed);
        }
        assert!(
            cancel.load(Ordering::Relaxed),
            "a failed send must latch the abort flag so the decode loop stops"
        );
    }

    // --- request / throughput logging (B10) ---------------------------------

    /// A finished request, spelled out so a test can fold one in without running a generator.
    ///
    /// Defaults to "all of its deltas are in window 0, which is still open" — the ordinary case
    /// where no drain intervened. [`rec_in_window`] is for the case that does.
    fn rec(prompt: u32, gen: u32, deltas: u64) -> ReqRecord {
        rec_in_window(prompt, gen, deltas, 0, deltas)
    }

    /// A finished request whose deltas landed in a specific stats window — the B24 shape.
    fn rec_in_window(
        prompt: u32,
        gen: u32,
        deltas: u64,
        window: u64,
        deltas_in_window: u64,
    ) -> ReqRecord {
        ReqRecord {
            prompt_tokens: prompt,
            cached_prompt_tokens: 0,
            gen_tokens: gen,
            deltas,
            window,
            deltas_in_window,
            prefill: Duration::from_millis(100),
            decode: Duration::from_millis(400),
            total: Duration::from_millis(500),
            finish: Finish::Stop,
        }
    }

    #[test]
    fn cached_prompt_tokens_are_excluded_from_prefill_rates() {
        let stats = ServeStats::default();
        let mut request = rec(100, 0, 0);
        request.cached_prompt_tokens = 80;

        assert_eq!(request.prefill_tps(), 200.0);
        stats.fold_completion(&request);
        assert_eq!(stats.drain(Duration::from_secs(1)).prompt_tokens, 20);
    }

    /// **The whole point of the periodic line.** Its numbers must describe the INTERVAL, so a drain
    /// has to leave the counters at zero — a cumulative counter would turn every rate into an
    /// average-since-boot, which looks plausible and is useless.
    ///
    /// Asserted by three consecutive intervals of the SAME wall length with DIFFERENT work in them:
    /// busy, half as busy, idle. Per-interval, the rates must go 10 → 5 → 0. Cumulative, they would
    /// go 10 → 15 → 15 (or 10 → 7.5 → 5 as an average), i.e. they could never fall — which is the
    /// exact failure this pins.
    #[test]
    fn stats_windows_cover_the_interval_and_never_accumulate() {
        let stats = ServeStats::default();
        let window = Duration::from_secs(2);

        for _ in 0..20 {
            stats.bump_gen(1);
        }
        stats.fold_completion(&rec(100, 20, 20));
        let busy = stats.drain(window);
        assert_eq!(
            (busy.prompt_tokens, busy.gen_tokens, busy.completed),
            (100, 20, 1)
        );
        assert!((busy.decode_tps() - 10.0).abs() < 1e-9, "{busy:?}");
        assert!((busy.prefill_tps() - 50.0).abs() < 1e-9, "{busy:?}");

        for _ in 0..10 {
            stats.bump_gen(1);
        }
        stats.fold_completion(&rec(20, 10, 10));
        let quieter = stats.drain(window);
        assert_eq!((quieter.prompt_tokens, quieter.gen_tokens), (20, 10));
        assert!(
            (quieter.decode_tps() - 5.0).abs() < 1e-9,
            "the second interval must report only ITS OWN 10 tokens: {quieter:?}"
        );
        assert!(
            quieter.decode_tps() < busy.decode_tps() && quieter.prefill_tps() < busy.prefill_tps(),
            "half the work in the same wall time must report a LOWER rate: {busy:?} then {quieter:?}"
        );

        let idle = stats.drain(window);
        assert_eq!(
            (idle.gen_tokens, idle.prompt_tokens, idle.completed),
            (0, 0, 0)
        );
        assert_eq!(idle.decode_tps(), 0.0);
        assert!(
            !idle.has_activity(),
            "an interval with nothing in it must emit no line: {idle:?}"
        );
    }

    /// The live per-delta count is an ESTIMATE (a delta is a text piece, not necessarily a token);
    /// `ChatOutcome` is authoritative. Whichever way they disagree, the interval total must end up
    /// on the authoritative number.
    #[test]
    fn the_live_delta_count_is_reconciled_against_the_real_token_count() {
        // Generator emitted 3 pieces but really decoded 5 tokens.
        let under = ServeStats::default();
        for _ in 0..3 {
            under.bump_gen(1);
        }
        under.fold_completion(&rec(1, 5, 3));
        assert_eq!(under.drain(Duration::from_secs(1)).gen_tokens, 5);

        // …and the other way: 6 pieces for 4 tokens.
        let over = ServeStats::default();
        for _ in 0..6 {
            over.bump_gen(1);
        }
        over.fold_completion(&rec(1, 4, 6));
        assert_eq!(over.drain(Duration::from_secs(1)).gen_tokens, 4);
    }

    /// B24: a correction must never be paid for out of a DIFFERENT request's interval.
    ///
    /// The reconciliation above only works because nothing drains between the deltas and the
    /// completion. When a reporter tick lands in that gap the deltas have already been published,
    /// and the retraction used to come out of whatever was generating next.
    #[test]
    fn a_correction_never_retracts_another_requests_tokens() {
        let s = ServeStats::default();

        // Request A emits two deltas, and the reporter ticks before A finishes.
        let mut a = ReqTally::new();
        a.on_text_delta(&s);
        a.on_text_delta(&s);
        let w1 = s.drain(Duration::from_secs(1));
        assert_eq!(w1.gen_tokens, 2, "the open window had A's two deltas");

        // Request B is now generating and puts three real tokens into the new window.
        let mut b = ReqTally::new();
        for _ in 0..3 {
            b.on_text_delta(&s);
        }

        // A completes: it really produced ONE token, so its live estimate was 1 too high. That
        // overcount is in a window that has already been reported and cannot be taken back.
        s.fold_completion(&a.finish(
            ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 1,
                cached_prompt_tokens: 0,
                completion_tokens: 1,
            },
            Finish::Stop,
        ));

        let w2 = s.drain(Duration::from_secs(1));
        assert_eq!(
            w2.gen_tokens, 3,
            "B's three tokens must survive A's correction — the −1 belongs to a closed window"
        );
    }

    /// The other half of B24: while the window is still open, a correction DOES apply — clamped to
    /// this request's own contribution, so it can never dig into tokens it did not put there.
    #[test]
    fn a_correction_applies_within_the_open_window_and_is_clamped_to_its_own_deltas() {
        let s = ServeStats::default();
        let mut a = ReqTally::new();
        for _ in 0..4 {
            a.on_text_delta(&s);
        }
        // No drain: A's deltas are still in the open window, so its −2 lands.
        s.fold_completion(&a.finish(
            ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 1,
                cached_prompt_tokens: 0,
                completion_tokens: 2,
            },
            Finish::Stop,
        ));
        assert_eq!(s.drain(Duration::from_secs(1)).gen_tokens, 2);

        // A request claiming FEWER tokens than it has deltas in this window can still only retract
        // what it contributed: 1 delta here, so a −3 becomes −1 and the other request keeps its 5.
        let s = ServeStats::default();
        let mut c = ReqTally::new();
        c.on_text_delta(&s);
        for _ in 0..5 {
            s.bump_gen(1); // another request's tokens, same window
        }
        let mut r = c.finish(
            ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 0,
                cached_prompt_tokens: 0,
                completion_tokens: 0,
            },
            Finish::Stop,
        );
        r.deltas = 4; // pretend 4 deltas were seen, only 1 of them in this window
        s.fold_completion(&r);
        assert_eq!(
            s.drain(Duration::from_secs(1)).gen_tokens,
            5,
            "the retraction is capped at this request's own contribution to the window"
        );
    }

    /// Activity-only, and what counts as activity. A request that is mid-generation has produced no
    /// completed tokens yet, but the server is plainly busy — so `active`/`queued` alone must be
    /// enough to emit a line, or a single long generation would log nothing until it ended. An
    /// interval with none of it emits nothing: there is no idle heartbeat, by decision.
    #[test]
    fn in_flight_requests_are_activity_and_an_empty_interval_is_not() {
        let stats = Arc::new(ServeStats::default());
        let w = Duration::from_secs(1);
        assert!(!stats.drain(w).has_activity(), "idle must be silent");

        let generating = ActiveGuard::new(stats.clone());
        assert!(
            stats.drain(w).has_activity(),
            "a generation in flight is activity even before it yields a token"
        );
        drop(generating);
        assert!(!stats.drain(w).has_activity());

        let waiting = QueuedGuard::new(stats.clone());
        assert!(
            stats.drain(w).has_activity(),
            "a request queued for a slot is activity"
        );
        drop(waiting);
        assert!(!stats.drain(w).has_activity());
    }

    /// The gauges are RAII, so the interesting case is the abnormal exit: a guard dropped while
    /// unwinding must still put the gauge back, or one panicking request leaves the server looking
    /// permanently busy.
    #[test]
    fn the_active_gauge_is_released_by_an_unwinding_panic() {
        let stats = Arc::new(ServeStats::default());
        let inner = stats.clone();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _active = ActiveGuard::new(inner);
            panic!("decode closure blew up");
        }));
        assert!(caught.is_err(), "the panic must have happened");
        assert_eq!(stats.active.load(Ordering::Relaxed), 0);
    }

    /// `serve.stats_interval_secs`: 5 s by default, `0` switches the periodic line OFF.
    #[test]
    fn stats_interval_defaults_to_five_seconds_and_zero_disables_it() {
        let mut cfg = Config::default();
        assert_eq!(stats_interval(&cfg), Some(Duration::from_secs(5)));
        cfg.serve.stats_interval_secs = 0;
        assert_eq!(stats_interval(&cfg), None, "0 => no periodic line at all");
        cfg.serve.stats_interval_secs = 30;
        assert_eq!(stats_interval(&cfg), Some(Duration::from_secs(30)));
    }

    /// KV slot occupancy is `busy/total` across every hosted model — the number the periodic line
    /// reports. It has to count HELD permits, not available ones.
    #[tokio::test]
    async fn slot_occupancy_counts_held_permits_across_models() {
        let a: Arc<dyn ChatGenerator> = Arc::new(EchoGen("alpha"));
        let b: Arc<dyn ChatGenerator> = Arc::new(EchoGen("beta"));
        let state = AppState::multi(
            vec![("alpha".into(), a, 3), ("beta".into(), b, 1)],
            Arc::new(Config::default()),
        );
        assert_eq!(state.slot_occupancy(), (0, 4));
        let held = state.models[0].slots.clone().acquire_owned().await.unwrap();
        assert_eq!(state.slot_occupancy(), (1, 4));
        let held2 = state.models[1].slots.clone().acquire_owned().await.unwrap();
        assert_eq!(state.slot_occupancy(), (2, 4));
        drop(held);
        drop(held2);
        assert_eq!(state.slot_occupancy(), (0, 4));
    }

    /// Request ids are a monotonic counter, NOT a clock reading: two requests admitted in the same
    /// millisecond must still be tellable apart in the log.
    #[test]
    fn request_ids_are_monotonic_and_distinct() {
        // STRICTLY increasing rather than dense: the counter is process-wide and the rest of the
        // suite is minting ids from other threads at the same time, so the gaps are the mechanism
        // working, not a failure. What must hold is that no id is ever reissued or reused.
        let ids: Vec<u64> = (0..1000).map(|_| next_req_id()).collect();
        assert!(
            ids.windows(2).all(|w| w[1] > w[0]),
            "ids must increase monotonically"
        );
        let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "ids must never repeat");
    }

    /// End to end through the real non-streaming path: the request's tallies reach the shared
    /// counters exactly once, with the generator's OWN token counts — `EchoGen` sends one delta but
    /// reports two completion tokens, so this also proves the reconciliation runs on the live path
    /// and not only in its unit test.
    #[tokio::test]
    async fn a_served_request_folds_its_counts_into_the_window() {
        let stats: Arc<ServeStats> = Arc::default();
        let ctx = ReqCtx {
            stats: stats.clone(),
            ..test_ctx(None, false)
        };
        let resp = non_streaming(
            deadline_entry(Arc::new(EchoGen("alpha"))),
            user_msg(),
            None,
            None,
            GenParams::default(),
            ctx,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let w = stats.drain(Duration::from_secs(1));
        assert_eq!(
            (w.prompt_tokens, w.gen_tokens, w.completed, w.failed),
            (3, 2, 1, 0),
            "EchoGen reports prompt_tokens=3, completion_tokens=2 for one request: {w:?}"
        );
        assert_eq!(
            (w.active, w.queued),
            (0, 0),
            "both gauges must be back at zero once the request is answered"
        );
    }

    /// A failed generation is still a completed request, and it is counted as a FAILURE — a server
    /// erroring on everything must not look idle.
    #[tokio::test]
    async fn a_failed_request_is_counted_as_a_failure() {
        let stats: Arc<ServeStats> = Arc::default();
        let ctx = ReqCtx {
            stats: stats.clone(),
            ..test_ctx(None, false)
        };
        let resp = non_streaming(
            deadline_entry(Arc::new(FailGen)),
            user_msg(),
            None,
            None,
            GenParams::default(),
            ctx,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let w = stats.drain(Duration::from_secs(1));
        assert_eq!((w.completed, w.failed), (1, 1), "{w:?}");
        assert!(
            w.has_activity(),
            "an interval of failures is not an idle one"
        );
    }
}
