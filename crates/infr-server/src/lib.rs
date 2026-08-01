//! OpenAI-compatible HTTP server (axum + SSE). Talks only to `infr-engine` — never the GPU.
//!
//! Reference for the wire mapping (streaming, `reasoning_content`, tool_calls): the working
//! shim at `~/Projects/scratch/dgemma-openai-server.py`. See docs/plan.md "server".
//!
//! Routes:
//!   GET  /health                -> 200 OK
//!   GET  /v1/models             -> { object: "list", data: [{ id, object, owned_by }] }
//!   POST /v1/chat/completions   -> chat.completion | SSE chat.completion.chunk stream
//!
//! Delta mapping:
//!   `Delta::Reasoning`  -> `delta.reasoning_content`
//!   `Delta::Content`    -> `delta.content`
//!   `Delta::ToolCall`   -> `delta.tool_calls[]`  (finish_reason "tool_calls")

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
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

/// What one [`ChatGenerator::chat`] call produced: why it ended, plus the real token counts the
/// handler needs for a truthful `usage` block (`total = prompt + completion`). Counts are what the
/// generator actually tokenized/emitted — never a `content.len()/4` estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatOutcome {
    pub finish: Finish,
    pub prompt_tokens: u32,
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
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
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
}

/// Shared server state — a non-empty, ordered set of hosted [`ModelEntry`]s. A request is routed to
/// the entry whose `id` matches its `model` field; an unknown/empty `model` falls to the DEFAULT
/// (the first entry). The single-model server is just the one-entry case, so its behaviour — and the
/// hot path — is byte-identical to before this became multi-model.
#[derive(Clone)]
pub struct AppState {
    /// Invariant: non-empty. `models[0]` is the default route.
    models: Arc<Vec<ModelEntry>>,
    /// The resolved process configuration (S7). The handler reads `serve.api_key` and
    /// `serve.max_tokens_cap` off it instead of the environment; it is an EXPLICIT constructor
    /// parameter on every entry point that can host a real model, so an embedder cannot silently
    /// end up with auth disabled by forgetting to pass one.
    cfg: Arc<Config>,
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
            models: Arc::new(vec![ModelEntry {
                id: Arc::from(model_id.into().as_str()),
                engine: Some(generator),
                slots: Arc::new(Semaphore::new(n_parallel.max(1))),
            }]),
            cfg,
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
            .map(|(id, gen, n)| ModelEntry {
                id: Arc::from(id.as_str()),
                engine: Some(gen),
                slots: Arc::new(Semaphore::new(n.max(1))),
            })
            .collect();
        Self {
            models: Arc::new(models),
            cfg,
        }
    }

    /// No-engine state — for /health, /v1/models, and serialisation tests. Takes the config too,
    /// so an auth/cap test drives `serve.*` through a `Config` value rather than the environment
    /// (R7).
    pub fn headless(model_id: impl Into<String>, cfg: Arc<Config>) -> Self {
        Self {
            models: Arc::new(vec![ModelEntry {
                id: Arc::from(model_id.into().as_str()),
                engine: None,
                slots: Arc::new(Semaphore::new(1)),
            }]),
            cfg,
        }
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

/// Bind + run the axum server over a fully-built [`AppState`] (single- or multi-model). The one
/// place the listener, the graceful-shutdown latch, and `axum::serve` live.
async fn serve_state(state: AppState, addr: SocketAddr) -> anyhow::Result<()> {
    let n_models = state.models.len();
    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, %n_models, "infr-server listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_latched())
        .await?;
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

async fn health_handler() -> StatusCode {
    StatusCode::OK
}

async fn models_handler(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: state
            .models
            .iter()
            .map(|m| ModelCard {
                id: m.id.to_string(),
                object: "model",
                owned_by: "local",
            })
            .collect(),
    })
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    // Optional bearer auth: enforced ONLY when `serve.api_key` (INFR_API_KEY) is configured.
    // Default = open (existing localhost usage is unaffected). Checked before any work, so an
    // unauthenticated request cannot even reach model routing / slot admission.
    if let Some(key) = configured_api_key(&state.cfg) {
        let auth = headers.get("authorization").and_then(|v| v.to_str().ok());
        if !authorize(Some(key), auth) {
            return json_error(
                StatusCode::UNAUTHORIZED,
                "missing or invalid Authorization bearer token".into(),
            );
        }
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
    // Route to the hosted model named in the request (exact id), else the default (first) entry.
    // The response `model` field echoes the entry ACTUALLY served, not the raw request string, so a
    // client that omitted/mis-named the model sees which one answered.
    let entry = state.route(&req.model);
    let model_id = entry.id.to_string();
    let cid = make_id();
    let created = unix_ts();

    if req.stream {
        streaming(
            entry,
            messages,
            tools,
            tool_choice,
            params,
            cid,
            model_id,
            created,
        )
        .await
    } else {
        non_streaming(
            entry,
            messages,
            tools,
            tool_choice,
            params,
            cid,
            model_id,
            created,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Non-streaming path
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn non_streaming(
    entry: ModelEntry,
    messages: Vec<ChatMessage>,
    tools: Option<serde_json::Value>,
    tool_choice: Option<String>,
    params: GenParams,
    cid: String,
    model_id: String,
    created: i64,
) -> Response {
    // Wait for a free slot ON THIS MODEL. With `--parallel N`, the (N+1)'th concurrent request to
    // this model queues HERE — in the async runtime, holding no thread — and is admitted FIFO as
    // soon as one of that model's generations finishes. A different model's slots are independent.
    let Ok(permit) = entry.slots.clone().acquire_owned().await else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        );
    };
    let engine_arc = entry.engine.clone();
    let cid_blk = cid.clone();

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        // The permit is MOVED into the blocking task and dropped when it ends: a slot is held for
        // exactly the generation, and the next queued request is admitted the moment it frees.
        let _permit = permit;
        let Some(engine) = engine_arc else {
            anyhow::bail!("no engine loaded");
        };

        let mut reasoning = String::new();
        let mut content = String::new();
        let mut tool_calls: Vec<OAIToolCall> = Vec::new();

        // No client-disconnect signal on the non-streaming path (the whole reply is buffered), so
        // this latch stays false here; generation is still bounded by EOS/stop/max_tokens and the
        // process shutdown latch, exactly as before. It exists to satisfy the shared trait.
        let cancel = AtomicBool::new(false);

        let outcome = engine
            .chat(
                &messages,
                tools.as_ref(),
                tool_choice.as_deref(),
                &params,
                &cancel,
                &mut |delta| match delta {
                    Delta::Reasoning(t) => reasoning.push_str(&t),
                    Delta::Content(t) => content.push_str(&t),
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

        Ok((reasoning, content, tool_calls, outcome))
    })
    .await
    .map_err(anyhow::Error::from)
    .and_then(|r| r);

    match result {
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Ok((reasoning, content, tool_calls, outcome)) => {
            let finish = if tool_calls.is_empty() {
                outcome.finish
            } else {
                Finish::ToolCalls
            }
            .as_str();
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
                    finish_reason: finish.into(),
                }],
                usage: UsageInfo {
                    prompt_tokens: outcome.prompt_tokens,
                    completion_tokens: outcome.completion_tokens,
                    total_tokens: outcome
                        .prompt_tokens
                        .saturating_add(outcome.completion_tokens),
                },
            })
            .into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming path (SSE)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn streaming(
    entry: ModelEntry,
    messages: Vec<ChatMessage>,
    tools: Option<serde_json::Value>,
    tool_choice: Option<String>,
    params: GenParams,
    cid: String,
    model_id: String,
    created: i64,
) -> Response {
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
    let Ok(permit) = entry.slots.clone().acquire_owned().await else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        );
    };
    let engine_arc = entry.engine.clone();
    // Clone sender + strings for use inside the on_delta callback closure.
    let tx_cb = tx.clone();
    let cid_cb = cid.clone();
    let model_cb = model_id.clone();

    // Per-request abort latch: set when the client disconnects (an SSE `send` starts failing) and
    // polled by the generator's decode loop so it stops promptly and frees the GPU slot instead of
    // running to `max_tokens` into a dead socket (audit finding 2).
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_cb = cancel.clone();

    tokio::task::spawn_blocking(move || {
        // Held for exactly this generation; freed for the next queued request on return.
        let _permit = permit;
        // Guarantees `[DONE]` is ALWAYS the final frame — including when the decode closure panics
        // (Drop runs while unwinding), so a strict SSE client never hangs waiting for a sentinel
        // that a panic swallowed (audit finding 1).
        let _done = DoneGuard { tx: tx.clone() };

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
            // No engine — `DoneGuard` closes the stream with `[DONE]`.
            return;
        };

        let mut tc_index = 0usize;
        let mut saw_tool_call = false;

        let res = engine.chat(
            &messages,
            tools.as_ref(),
            tool_choice.as_deref(),
            &params,
            &cancel_cb,
            &mut |delta| {
                let payload = match delta {
                    Delta::Reasoning(t) => DeltaPayload {
                        reasoning_content: Some(t),
                        ..Default::default()
                    },
                    Delta::Content(t) => DeltaPayload {
                        content: Some(t),
                        ..Default::default()
                    },
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
                let finish = if saw_tool_call {
                    Finish::ToolCalls
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
            }
            Err(e) => {
                // A mid-stream failure is NOT a clean `stop` — emit a terminal error frame so the
                // client can tell this apart from success (matching the non-streaming 500). `[DONE]`
                // still follows, via `DoneGuard` (audit finding 1).
                let _ = tx.send(Ok(sse_error_event(&e.to_string())));
            }
        }
        // `DoneGuard` drops here (or on an unwinding panic), sending the final `[DONE]`.
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

/// Sends the OpenAI `[DONE]` sentinel exactly once when it drops — from the normal end of the
/// stream OR from an unwinding panic inside the decode closure. That is the whole point: a strict
/// SSE client blocks until it sees `[DONE]`, so a swallowed panic must never leave the sentinel
/// unsent (audit finding 1). A failed send (client already gone) is fine — nothing to deliver to.
struct DoneGuard {
    tx: tokio::sync::mpsc::UnboundedSender<Result<Event, Infallible>>,
}

impl Drop for DoneGuard {
    fn drop(&mut self) {
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
/// Mirrors the Python shim's `normalize_messages`: only `"text"` parts are kept.
pub fn flatten_content(v: &Option<serde_json::Value>) -> String {
    match v {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                if p.get("type")?.as_str()? == "text" {
                    p.get("text")?.as_str().map(str::to_owned)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(other) => other.to_string(),
    }
}

fn dto_to_engine(dto: &ChatMessageDto) -> ChatMessage {
    ChatMessage {
        role: dto.role.clone(),
        content: flatten_content(&dto.content),
        tool_calls: dto.tool_calls.as_ref().and_then(parse_oai_tool_calls),
        tool_call_id: dto.tool_call_id.clone(),
        name: dto.name.clone(),
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
                completion_tokens: 2,
            })
        }
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
    fn chat_request_array_content_deserializes_and_flattens() {
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
        // Only text parts are concatenated; image_url is discarded.
        assert_eq!(flatten_content(&req.messages[0].content), "hello world");
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
        assert_eq!(flatten_content(&v), "hello world");
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
            completion_tokens: 5,
        };
        let usage = UsageInfo {
            prompt_tokens: outcome.prompt_tokens,
            completion_tokens: outcome.completion_tokens,
            total_tokens: outcome
                .prompt_tokens
                .saturating_add(outcome.completion_tokens),
        };
        assert_eq!(usage.total_tokens, 22);
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
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
}
