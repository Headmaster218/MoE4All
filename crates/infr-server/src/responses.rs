//! OpenAI Responses compatibility on top of the server's chat generation path.
//! Keep wire conversion here so slot admission, cancellation, progress and usage have one owner.
//! The request is stateless by default; `serve.response_store_entries` additionally enables an
//! LRU of response histories so `previous_response_id`/`conversation` chains replay server-side.
//! A chained request rebuilds the previous turn's exact token prefix, which the paged KV radix
//! tree hits via `best_match` — the integration point is the rebuilt history, not any KV handle.
//! `include:["reasoning.encrypted_content"]` rides the same store: the handle a reasoning item
//! carries is a reference into it (see [`reasoning_handle`]), never a copy of the text.

use super::*;
use axum::body::{to_bytes, Body, Bytes};
use axum::extract::{Path, Query};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use futures_util::{stream, Stream, StreamExt};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub(super) struct ResponsesRequest {
    model: String,
    input: Value,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    reasoning: Option<Value>,
    #[serde(default)]
    truncation: Option<String>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    previous_response_id: Option<Value>,
    #[serde(default)]
    conversation: Option<Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    prompt_cache_key: Option<String>,
    #[serde(default)]
    text: Option<Value>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    include: Option<Vec<String>>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Clone)]
struct Meta {
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    reasoning: Value,
    tools: Vec<Value>,
    tool_choice: Value,
    metadata: Value,
    prompt_cache_key: Option<String>,
    text: Value,
    parallel_tool_calls: bool,
    user: Option<String>,
    previous_response_id: Option<String>,
    store: bool,
    /// `include:["reasoning.encrypted_content"]` asked for a replayable reasoning handle, and this
    /// turn is stored so the handle resolves. See [`ResponseStore::resolve_item`].
    encrypted_reasoning: bool,
    /// The same include was requested on a STATELESS turn: no handle is issued (the item carries
    /// its summary only) and the response says so in `warnings`.
    handle_withheld: bool,
}

/// A converted request: the chat payload to dispatch, the wire metadata for the response
/// envelope, and what the handler needs to persist after a successful generation.
struct Prepared {
    chat: ChatRequest,
    meta: Meta,
    /// Input-side message history (this turn's assistant output is appended only on save).
    history: Arc<Vec<ChatMessageDto>>,
    /// The prompt in Responses-item shape (replayed history + this turn's input), kept only when
    /// the turn is stored.
    input_items: Vec<Value>,
    conversation: Option<String>,
    store: bool,
}

/// Server-side `/v1/responses` state: response id -> replayable message history AND the exact
/// envelope the client received, plus `conversation` name -> latest response id cursors.
/// KV reuse is implicit (chained requests rebuild the same token prefix and the paged radix
/// tree's `best_match` hits it), so this store never pins or references KV pages.
///
/// `serve.response_store_path` mirrors the whole store to one JSON file, loaded by [`open`] and
/// rewritten after every mutation; without it the store lives and dies with the process.
pub(super) struct ResponseStore {
    capacity: usize,
    entries: HashMap<String, StoredResponse>,
    conversations: HashMap<String, String>,
    /// Item id -> owning response id. `input` may replay any earlier item by reference, and a
    /// reference is only resolvable while its owning response is still in the LRU (see
    /// `resolve_item`), which is the same lifetime a chain has.
    items: HashMap<String, String>,
    /// Monotonic tick for LRU recency; no wall clock needed.
    clock: u64,
    /// Mirror of the store on disk (`serve.response_store_path`), rewritten after every mutation.
    /// `None` — the default — keeps everything in memory, so nothing a client said touches a disk.
    path: Option<PathBuf>,
}

struct StoredResponse {
    /// Full history INCLUDING this turn's assistant output — exactly what a stateless client
    /// would have sent back on the next request.
    history: Arc<Vec<ChatMessageDto>>,
    /// The response body as the client received it, so `GET /v1/responses/{id}` returns the
    /// same object it was handed at generation time (status, output items, usage included).
    response: Value,
    /// The Responses-shaped items this turn's `input` normalized to — what `input_items` lists.
    input_items: Vec<Value>,
    last_used: u64,
}

/// The on-disk shape of the store. `version` is checked on load so a future format change is a
/// named startup error instead of a silent mis-parse of history nobody can reconstruct.
#[derive(Deserialize)]
struct StoreFile {
    version: u32,
    clock: u64,
    conversations: HashMap<String, String>,
    entries: Vec<StoreFileEntry>,
}

const STORE_FILE_VERSION: u32 = 1;

#[derive(Deserialize)]
struct StoreFileEntry {
    id: String,
    last_used: u64,
    history: Vec<ChatMessageDto>,
    response: Value,
    input_items: Vec<Value>,
}

/// The write side borrows straight out of the live store — persisting must not clone every stored
/// history just to hand it to the serializer.
#[derive(Serialize)]
struct StoreFileRef<'a> {
    version: u32,
    clock: u64,
    conversations: &'a HashMap<String, String>,
    entries: Vec<StoreFileEntryRef<'a>>,
}

#[derive(Serialize)]
struct StoreFileEntryRef<'a> {
    id: &'a str,
    last_used: u64,
    history: &'a [ChatMessageDto],
    response: &'a Value,
    input_items: &'a [Value],
}

impl ResponseStore {
    /// Open the store, loading `path` when the operator asked for restart survival. A missing file
    /// is a fresh start; an unreadable or unparsable one is an ERROR — starting empty would turn
    /// every stored id into a "lost on restart" 400 while the data still sits on disk.
    pub(super) fn open(capacity: u32, path: Option<&std::path::Path>) -> Result<Self, String> {
        let mut store = Self {
            capacity: capacity as usize,
            entries: HashMap::new(),
            conversations: HashMap::new(),
            items: HashMap::new(),
            clock: 0,
            // `response_store_entries = 0` IS the stateless contract, so a path alongside it would
            // have nothing to mirror; see `ServeCfg::response_store_path`.
            path: path
                .filter(|_| capacity > 0)
                .map(std::path::Path::to_path_buf),
        };
        let Some(path) = store.path.clone() else {
            return Ok(store);
        };
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(store),
            Err(error) => return Err(format!("cannot read {}: {error}", path.display())),
        };
        let file: StoreFile = serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "{} is not a response store ({error}); move it aside to start empty",
                path.display()
            )
        })?;
        if file.version != STORE_FILE_VERSION {
            return Err(format!(
                "{} holds store version {}, this build reads {STORE_FILE_VERSION}",
                path.display(),
                file.version
            ));
        }
        store.clock = file.clock;
        store.conversations = file.conversations;
        let mut entries = file.entries;
        // Oldest first, so the capacity trim below keeps exactly the turns a running store would have.
        entries.sort_by_key(|entry| entry.last_used);
        for entry in entries {
            for item in item_ids(&entry.response["output"]) {
                store.items.insert(item, entry.id.clone());
            }
            store.entries.insert(
                entry.id,
                StoredResponse {
                    history: Arc::new(entry.history),
                    response: entry.response,
                    input_items: entry.input_items,
                    last_used: entry.last_used,
                },
            );
        }
        while store.entries.len() > store.capacity {
            let Some(victim) = store.oldest() else {
                break;
            };
            store.forget(&victim);
        }
        Ok(store)
    }

    /// Mirror the store to disk after a mutation. A failed write is logged, never propagated: the
    /// generation already succeeded, and turning a full disk into a failed request would lose the
    /// answer as well as the mirror.
    fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        if let Err(error) = self.write(path) {
            tracing::warn!(path = %path.display(), %error, "could not persist the response store");
        }
    }

    fn write(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut entries: Vec<StoreFileEntryRef<'_>> = self
            .entries
            .iter()
            .map(|(id, entry)| StoreFileEntryRef {
                id,
                last_used: entry.last_used,
                history: &entry.history,
                response: &entry.response,
                input_items: &entry.input_items,
            })
            .collect();
        entries.sort_by_key(|entry| entry.last_used);
        let file = StoreFileRef {
            version: STORE_FILE_VERSION,
            clock: self.clock,
            conversations: &self.conversations,
            entries,
        };
        let bytes = serde_json::to_vec(&file).map_err(std::io::Error::other)?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        // Write-then-rename: a crash mid-write leaves the previous store intact, never half of one.
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)
    }

    /// The least-recently-used entry id, or `None` when the store is empty.
    fn oldest(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(id, _)| id.clone())
    }

    /// Resolve the replay base for a chained request. `previous_response_id` wins over the
    /// `conversation` cursor. An unknown id is a hard 400 — NEVER a silent fresh context:
    /// a client that believes it is continuing a conversation must learn that it is not.
    fn chain(
        &mut self,
        previous: Option<&str>,
        conversation: Option<&str>,
    ) -> Result<Option<Arc<Vec<ChatMessageDto>>>, ParamError> {
        self.clock += 1;
        if let Some(id) = previous {
            let entry = self.entries.get_mut(id).ok_or_else(|| {
                invalid(
                    "previous_response_id",
                    format!(
                        "no stored response {id:?} (state is LRU-bounded; without \
                         serve.response_store_path a restart loses it)"
                    ),
                )
            })?;
            entry.last_used = self.clock;
            return Ok(Some(entry.history.clone()));
        }
        if let Some(name) = conversation {
            if let Some(id) = self.conversations.get(name) {
                // The cursor may point at an evicted entry: fail loudly (see `put`'s note on
                // why cursors survive eviction) instead of restarting the conversation blank.
                let entry = self.entries.get_mut(id).ok_or_else(|| {
                    invalid(
                        "conversation",
                        format!("conversation {name:?} points at evicted response {id:?}"),
                    )
                })?;
                entry.last_used = self.clock;
                return Ok(Some(entry.history.clone()));
            }
            return Ok(None); // unknown name = start a new conversation, per OpenAI semantics
        }
        Ok(None)
    }

    fn put(
        &mut self,
        id: &str,
        response: Value,
        history: Arc<Vec<ChatMessageDto>>,
        input_items: Vec<Value>,
        conversation: Option<&str>,
    ) {
        self.clock += 1;
        for item in item_ids(&response["output"]) {
            self.items.insert(item, id.to_owned());
        }
        self.entries.insert(
            id.to_owned(),
            StoredResponse {
                history,
                response,
                input_items,
                last_used: self.clock,
            },
        );
        if let Some(name) = conversation {
            self.conversations.insert(name.to_owned(), id.to_owned());
        }
        // Evict least-recently-used entries, but NEVER rewrite or drop conversation cursors:
        // a cursor left pointing at an evicted id makes the next chained request fail
        // explicitly (see `chain`) rather than silently lose the conversation's history.
        while self.entries.len() > self.capacity {
            let Some(victim) = self.oldest() else {
                break;
            };
            self.forget(&victim);
        }
        self.save();
    }

    /// Drop one response and everything only it could resolve: item references and, transitively,
    /// any chain that starts from it. Cursors are deliberately left alone (see `put`).
    fn forget(&mut self, id: &str) -> Option<StoredResponse> {
        self.items.retain(|_, owner| owner != id);
        self.entries.remove(id)
    }

    /// The stored envelope for `id`, touching its recency: reading a response through the REST
    /// surface keeps it alive exactly as chaining from it would.
    fn get(&mut self, id: &str) -> Option<Value> {
        self.clock += 1;
        let entry = self.entries.get_mut(id)?;
        entry.last_used = self.clock;
        Some(entry.response.clone())
    }

    fn remove(&mut self, id: &str) -> Option<Value> {
        let entry = self.forget(id)?;
        self.save();
        Some(entry.response)
    }

    /// Stored responses newest-first (`desc`) or oldest-first (`asc`), as OpenAI's list does.
    /// `last_used` is the recency signal; ties are impossible because every touch bumps the clock.
    fn list(&self, limit: usize, ascending: bool) -> Vec<Value> {
        let mut entries: Vec<&StoredResponse> = self.entries.values().collect();
        entries.sort_by_key(|entry| entry.last_used);
        if !ascending {
            entries.reverse();
        }
        entries
            .into_iter()
            .take(limit)
            .map(|entry| entry.response.clone())
            .collect()
    }

    /// Resolve an `{"type":"item_reference","id":...}` against a stored response. A reference to an
    /// evicted response fails loudly like a chain does — a client that believes it is replaying an
    /// earlier item must learn when it is not.
    ///
    /// Only `output` is searched: the items in a stored prompt are replayed by `previous_response_id`
    /// or by resending them, so indexing them would put the same conversation content under two
    /// different ids depending on which turn stored it.
    fn resolve_item(&mut self, id: &str) -> Option<Value> {
        self.clock += 1;
        let owner = self.items.get(id).cloned()?;
        let entry = self.entries.get_mut(&owner)?;
        entry.last_used = self.clock;
        entry.response["output"]
            .as_array()?
            .iter()
            .find(|item| item["id"].as_str() == Some(id))
            .cloned()
    }

    /// The complete item context of a stored response (replayed history + that turn's input).
    fn prompt_items(&mut self, id: &str) -> Option<Vec<Value>> {
        self.clock += 1;
        let entry = self.entries.get_mut(id)?;
        entry.last_used = self.clock;
        Some(entry.input_items.clone())
    }

    /// Stored responses, for `has_more` on a page.
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// The string ids carried by an array of Responses items — the keys `item_reference` resolves.
/// Only server-minted items (a response's `output`) carry ids, so those are what replay.
fn item_ids(items: &Value) -> Vec<String> {
    items
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item["id"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// Extract an optional non-empty string from a `previous_response_id` / `conversation` field.
/// `conversation` also accepts OpenAI's `{"id": ...}` object form.
fn chain_string(value: &Option<Value>, param: &'static str) -> Result<Option<String>, ParamError> {
    match value.as_ref() {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if !text.is_empty() => Ok(Some(text.clone())),
        Some(Value::Object(object)) if param == "conversation" => {
            match object.get("id").and_then(Value::as_str) {
                Some(id) if !id.is_empty() => Ok(Some(id.to_owned())),
                _ => Err(invalid(
                    "conversation",
                    "conversation object requires a non-empty id string",
                )),
            }
        }
        _ => Err(invalid(
            param,
            format!("{param} must be a non-empty string"),
        )),
    }
}

fn invalid(param: &'static str, message: impl Into<String>) -> ParamError {
    ParamError {
        param,
        message: message.into(),
    }
}

fn message(role: &str, content: Option<Value>) -> ChatMessageDto {
    ChatMessageDto {
        role: role.into(),
        content,
        reasoning_content: None,
        reasoning: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn input_content(value: &Value) -> Result<Value, ParamError> {
    match value {
        Value::String(_) => Ok(value.clone()),
        Value::Array(parts) => {
            let mut converted = Vec::with_capacity(parts.len());
            for part in parts {
                let kind = part
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("input", "each content part requires a string type"))?;
                match kind {
                    "input_text" | "output_text" | "text" => {
                        let text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                            invalid("input", format!("{kind} requires string text"))
                        })?;
                        converted.push(json!({"type":"text","text":text}));
                    }
                    "input_image" => {
                        if part.get("file_id").is_some_and(|v| !v.is_null()) {
                            return Err(invalid("input", "input_image.file_id is not supported"));
                        }
                        let url = part
                            .get("image_url")
                            .and_then(Value::as_str)
                            .ok_or_else(|| invalid("input", "input_image requires image_url"))?;
                        if !url.starts_with("data:") {
                            return Err(invalid(
                                "input",
                                "this server supports base64 data: image URLs, not remote URLs or file IDs",
                            ));
                        }
                        if let Some(detail) = part.get("detail").and_then(Value::as_str) {
                            if detail != "auto" {
                                return Err(invalid(
                                    "input",
                                    "only input_image.detail=auto is supported",
                                ));
                            }
                        }
                        converted.push(json!({"type":"image_url","image_url":{"url":url}}));
                    }
                    "input_file" => return Err(invalid("input", "input_file is not supported")),
                    other => {
                        return Err(invalid(
                            "input",
                            format!("unsupported content part {other:?}"),
                        ))
                    }
                }
            }
            Ok(Value::Array(converted))
        }
        _ => Err(invalid(
            "input",
            "message content must be text or an array of content parts",
        )),
    }
}

fn input_messages(input: &Value, messages: &mut Vec<ChatMessageDto>) -> Result<(), ParamError> {
    match input {
        Value::String(text) => messages.push(message("user", Some(Value::String(text.clone())))),
        Value::Array(items) => {
            for item in items {
                let kind = item
                    .get("type")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("role").and_then(Value::as_str).map(|_| "message"))
                    .ok_or_else(|| invalid("input", "each input item requires type or role"))?;
                match kind {
                    "message" => {
                        let role = item
                            .get("role")
                            .and_then(Value::as_str)
                            .ok_or_else(|| invalid("input", "message item requires role"))?;
                        if !matches!(role, "user" | "assistant" | "system" | "developer") {
                            return Err(invalid("input", format!("unsupported role {role:?}")));
                        }
                        let content = item
                            .get("content")
                            .ok_or_else(|| invalid("input", "message item requires content"))?;
                        messages.push(message(role, Some(input_content(content)?)));
                    }
                    "function_call" => {
                        let name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .ok_or_else(|| invalid("input", "function_call requires name"))?;
                        let arguments =
                            item.get("arguments")
                                .and_then(Value::as_str)
                                .ok_or_else(|| {
                                    invalid("input", "function_call requires string arguments")
                                })?;
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .ok_or_else(|| invalid("input", "function_call requires call_id"))?;
                        let mut m = message("assistant", None);
                        m.tool_calls = Some(
                            json!([{"id":call_id,"type":"function","function":{"name":name,"arguments":arguments}}]),
                        );
                        messages.push(m);
                    }
                    "function_call_output" => {
                        let call_id =
                            item.get("call_id").and_then(Value::as_str).ok_or_else(|| {
                                invalid("input", "function_call_output requires call_id")
                            })?;
                        let output = item.get("output").ok_or_else(|| {
                            invalid("input", "function_call_output requires output")
                        })?;
                        let content = input_content(output)?;
                        if !content.is_string()
                            && content
                                .as_array()
                                .is_some_and(|parts| parts.iter().any(|p| p["type"] != "text"))
                        {
                            return Err(invalid(
                                "input",
                                "function_call_output supports text only",
                            ));
                        }
                        let text = if let Some(text) = content.as_str() {
                            text.to_owned()
                        } else {
                            content
                                .as_array()
                                .unwrap()
                                .iter()
                                .filter_map(|p| p["text"].as_str())
                                .collect::<String>()
                        };
                        let mut m = message("tool", Some(Value::String(text)));
                        m.tool_call_id = Some(call_id.into());
                        messages.push(m);
                    }
                    "reasoning" => {
                        let summaries =
                            item.get("summary")
                                .and_then(Value::as_array)
                                .ok_or_else(|| {
                                    invalid("input", "reasoning item requires readable summary")
                                })?;
                        let mut text = String::new();
                        for summary in summaries {
                            if summary["type"] != "summary_text" {
                                return Err(invalid(
                                    "input",
                                    "only summary_text reasoning can be replayed",
                                ));
                            }
                            text.push_str(
                                summary["text"].as_str().ok_or_else(|| {
                                    invalid("input", "summary_text requires text")
                                })?,
                            );
                        }
                        if text.is_empty() {
                            return Err(invalid("input", "opaque reasoning cannot be replayed"));
                        }
                        let mut m = message("assistant", None);
                        m.reasoning_content = Some(text);
                        messages.push(m);
                    }
                    other => {
                        return Err(invalid(
                            "input",
                            format!("unsupported input item {other:?}"),
                        ))
                    }
                }
            }
        }
        _ => return Err(invalid("input", "input must be text or an array of items")),
    }
    Ok(())
}

/// The inverse of [`input_messages`]: the prompt's message DTOs back into Responses items, so
/// `GET /v1/responses/{id}/input_items` states what the model actually received rather than
/// echoing the client's own JSON at it. A DTO carrying both thinking and an answer becomes two
/// items, as it is on the wire.
fn to_items(messages: &[ChatMessageDto]) -> Vec<Value> {
    fn part(part: &Value) -> Value {
        if part["type"] == "image_url" {
            json!({"type":"input_image","image_url":part["image_url"]["url"]})
        } else {
            json!({"type":"input_text","text":part["text"]})
        }
    }
    let mut items = Vec::with_capacity(messages.len());
    for message in messages {
        if let Some(reasoning) = message
            .reasoning_content
            .as_deref()
            .filter(|text| !text.is_empty())
        {
            items.push(
                json!({"type":"reasoning","summary":[{"type":"summary_text","text":reasoning}]}),
            );
        }
        let assistant = message.role == "assistant";
        if message.role == "tool" {
            // A tool result has no message-item shape on the Responses wire; it is an item of its
            // own, keyed by the call it answers.
            items.push(json!({
                "type":"function_call_output","call_id":message.tool_call_id,
                "output":message.content.as_ref().and_then(Value::as_str).unwrap_or_default()
            }));
        } else {
            match &message.content {
                Some(Value::String(text)) => items.push(json!({
                    "type":"message","role":message.role,
                    "content":[{"type":if assistant{"output_text"}else{"input_text"},"text":text}]})),
                Some(Value::Array(parts)) => items.push(json!({
                    "type":"message","role":message.role,
                    "content":parts.iter().map(part).collect::<Vec<_>>()
                })),
                // Every DTO reaching here was built by `input_messages`, so content is text, an
                // array of parts, or absent — and a content-less assistant DTO carries only the
                // calls handled below.
                _ => {}
            }
        }
        if let Some(calls) = message.tool_calls.as_ref().and_then(Value::as_array) {
            for call in calls {
                items.push(json!({"type":"function_call","call_id":call["id"],
                    "name":call["function"]["name"],"arguments":call["function"]["arguments"]}));
            }
        }
    }
    items
}

fn instruction_text(content: &Option<Value>) -> Result<String, ParamError> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) if parts.iter().all(|part| part["type"] == "text") => Ok(parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n")),
        _ => Err(invalid(
            "input",
            "system and developer messages must contain text only",
        )),
    }
}

/// Fold the instruction surface — `instructions` plus any leading `system`/`developer` messages —
/// into the single system text a GGUF template accepts, draining those messages from `messages`
/// because they are not conversation turns. Returns `None` when the request carries no instruction
/// at all.
///
/// A `system`/`developer` item that arrives AFTER a conversation turn is rejected rather than moved
/// into the prefix: the templates cannot express an instruction that starts mid-conversation, and
/// relocating it silently rewrites the prompt the client wrote.
///
/// Both the fresh and the chained path call this, so a re-sent instruction is compared against the
/// stored prefix in exactly the shape the first turn put it there.
fn fold_instructions(
    instructions: Option<&str>,
    messages: &mut Vec<ChatMessageDto>,
) -> Result<Option<String>, ParamError> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(text) = instructions.filter(|text| !text.is_empty()) {
        parts.push(text.to_owned());
    }
    let prefix_len = messages
        .iter()
        .take_while(|m| matches!(m.role.as_str(), "system" | "developer"))
        .count();
    for m in &messages[..prefix_len] {
        parts.push(instruction_text(&m.content)?);
    }
    if messages[prefix_len..]
        .iter()
        .any(|m| matches!(m.role.as_str(), "system" | "developer"))
    {
        return Err(invalid(
            "input",
            "system and developer messages must precede conversation turns",
        ));
    }
    messages.drain(..prefix_len);
    Ok(if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    })
}

/// The system text a stored chain already replays — the prefix the first turn of the chain folded
/// into a single `system` DTO — or `None` when the chain has no system turn.
fn stored_system_prefix(base: &[ChatMessageDto]) -> Option<String> {
    let content = base
        .first()
        .filter(|message| message.role == "system")?
        .content
        .clone()?;
    instruction_text(&Some(content))
        .ok()
        .filter(|text| !text.is_empty())
}

/// Is `resent` covered by what the chain already replays? Many SDKs re-send the same `instructions`
/// (or the same system message) on every turn, so an exact duplicate — or a leading part of the
/// stored prefix, the shape a client takes when the first turn also carried a system *input* message
/// it now keeps to itself — means the tokenized prompt is unchanged. Anything else would edit the
/// prefix mid-conversation, which is what the chain exists to avoid.
fn prefix_already_stored(stored: &str, resent: &str) -> bool {
    stored
        .strip_prefix(resent)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with("\n\n"))
}

fn convert_tools(value: Option<&Value>) -> Result<Option<Value>, ParamError> {
    let Some(value) = value else { return Ok(None) };
    let tools = value
        .as_array()
        .ok_or_else(|| invalid("tools", "tools must be an array"))?;
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        if tool["type"] != "function" {
            return Err(invalid("tools", "only custom function tools are supported"));
        }
        if tool
            .get("strict")
            .is_some_and(|v| !v.is_null() && v != false)
        {
            return Err(invalid(
                "tools",
                "strict function schemas are not supported",
            ));
        }
        if tool.as_object().is_some_and(|o| {
            o.keys().any(|k| {
                !matches!(
                    k.as_str(),
                    "type" | "name" | "description" | "parameters" | "strict"
                )
            })
        }) {
            return Err(invalid("tools", "unsupported function tool setting"));
        }
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("tools", "function requires name"))?;
        let parameters = tool
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type":"object"}));
        out.push(json!({"type":"function","function":{"name":name,"description":tool.get("description"),"parameters":parameters}}));
    }
    Ok(Some(Value::Array(out)))
}

fn convert_tool_choice(value: Option<&Value>) -> Result<Option<Value>, ParamError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(choice)) if matches!(choice.as_str(), "auto" | "none" | "required") => {
            Ok(Some(Value::String(choice.clone())))
        }
        Some(Value::Object(choice))
            if choice.get("type").and_then(Value::as_str) == Some("function") =>
        {
            if choice.keys().any(|key| key != "type" && key != "name") {
                return Err(invalid(
                    "tool_choice",
                    "unsupported function choice setting",
                ));
            }
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("tool_choice", "function choice requires name"))?;
            Ok(Some(json!({"type":"function","function":{"name":name}})))
        }
        _ => Err(invalid("tool_choice", "unsupported tool_choice")),
    }
}

impl ResponsesRequest {
    /// Convert to the chat path. `base` is the stored history of a chained request (already
    /// resolved by [`ResponseStore::chain`]); `previous`/`conversation`/`store` are the
    /// feature-gated wire fields the handler validated, kept for the response envelope.
    fn into_chat(
        self,
        base: Option<Arc<Vec<ChatMessageDto>>>,
        previous: Option<String>,
        conversation: Option<String>,
        store: bool,
    ) -> Result<Prepared, ParamError> {
        if self.model.is_empty() {
            return Err(invalid("model", "model must not be empty"));
        }
        if self.truncation.as_deref().is_some_and(|s| s != "disabled") {
            return Err(invalid(
                "truncation",
                "only truncation=disabled is supported",
            ));
        }
        if self
            .user
            .as_ref()
            .is_some_and(|user| user.chars().count() > 64)
        {
            return Err(invalid("user", "user must be at most 64 characters"));
        }
        let wants_handle = self.include.as_ref().is_some_and(|include| {
            include
                .iter()
                .any(|item| item == "reasoning.encrypted_content")
        });
        if let Some(other) = self.include.as_ref().and_then(|include| {
            include
                .iter()
                .find(|item| item.as_str() != "reasoning.encrypted_content")
        }) {
            return Err(invalid(
                "include",
                format!(
                    "{other:?} cannot be served: this server has no hosted-tool data, no logprobs \
                     and no hosted reasoning to encrypt — reasoning.encrypted_content is the only \
                     accepted entry"
                ),
            ));
        }
        // A STATELESS turn cannot issue a store-backed handle. That is not a client error: OpenAI's
        // own servers omit `encrypted_content` when there is nothing to encrypt (Azure does), replay
        // only needs the reasoning item's readable `summary`, and pi-ai's Responses adapter asks for
        // the include on EVERY reasoning request — so 400ing here locks that client out of every
        // thinking level. Degrade to summary-only items and say so in `warnings`.
        let encrypted_reasoning = wants_handle && store;
        let handle_withheld = wants_handle && !store;
        if self
            .service_tier
            .as_deref()
            .is_some_and(|tier| !matches!(tier, "auto" | "default"))
        {
            return Err(invalid(
                "service_tier",
                "only the local default service tier is supported",
            ));
        }
        if let Some((key, _)) = self.extra.iter().find(|(_, v)| !v.is_null()) {
            return Err(invalid(
                "request",
                format!("unsupported request field {key:?}"),
            ));
        }
        let metadata = match self.metadata {
            None | Some(Value::Null) => json!({}),
            Some(v @ Value::Object(_)) => v,
            _ => return Err(invalid("metadata", "metadata must be an object")),
        };
        // `text.format` mirrors OpenAI's Responses shape: the json_schema fields sit at the TOP
        // level here (chat nests them under `json_schema`). Both API surfaces normalise into the
        // SAME `ResponseFormat` — see `response_format.rs` for the JSON Schema subset gate.
        let (text, response_format) = match self.text {
            None | Some(Value::Null) => (
                json!({"format":{"type":"text"},"verbosity":"medium"}),
                None,
            ),
            Some(Value::Object(object)) => {
                if let Some(key) = object
                    .keys()
                    .find(|key| key.as_str() != "format" && key.as_str() != "verbosity")
                {
                    return Err(invalid(
                        "text",
                        format!("text.{key} is not supported (allowed: format, verbosity)"),
                    ));
                }
                if object
                    .get("verbosity")
                    .is_some_and(|verbosity| !verbosity.is_null() && *verbosity != "medium")
                {
                    return Err(invalid(
                        "text.verbosity",
                        "only verbosity=medium is supported",
                    ));
                }
                let format = object
                    .get("format")
                    .filter(|format| !format.is_null())
                    .cloned()
                    .unwrap_or_else(|| json!({"type":"text"}));
                let parsed = super::response_format::parse_text_format(Some(&format), "text.format")?;
                let text = json!({"format":format,"verbosity":"medium"});
                (text, chat_wire_format(&parsed))
            }
            _ => {
                return Err(invalid(
                    "text",
                    "text must be an object with format/verbosity",
                ))
            }
        };
        let mut reasoning_effort = None;
        let mut reasoning_off = false;
        let reasoning = match self.reasoning {
            None | Some(Value::Null) => json!({"effort":null,"summary":null}),
            Some(v @ Value::Object(_)) => {
                // `minimal` is an OpenAI effort level, while local GGUF templates expose
                // `low` as their smallest non-off level. Report the effective level.
                let effective_effort = v.get("effort").map(|effort| {
                    if effort == "minimal" {
                        json!("low")
                    } else {
                        effort.clone()
                    }
                });
                if let Some(effort) = v.get("effort").filter(|x| !x.is_null()) {
                    // `none` = no reasoning: the templates have levels only, so it lands on the
                    // thinking switch (`chat_template_kwargs.enable_thinking`) below.
                    let (level, off) =
                        crate::parse_reasoning_effort(effort, "reasoning.effort")?;
                    reasoning_effort = level;
                    reasoning_off = off;
                }
                if v.get("summary")
                    .is_some_and(|s| !s.is_null() && s != "auto")
                {
                    return Err(invalid(
                        "reasoning.summary",
                        "only summary=auto is supported",
                    ));
                }
                if v.as_object()
                    .is_some_and(|o| o.keys().any(|k| k != "effort" && k != "summary"))
                {
                    return Err(invalid("reasoning", "unsupported reasoning setting"));
                }
                json!({"effort":effective_effort,"summary":v.get("summary")})
            }
            _ => return Err(invalid("reasoning", "reasoning must be an object")),
        };
        let tools = match self.tools {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(v)) => v,
            _ => return Err(invalid("tools", "tools must be an array")),
        };
        let tool_choice = self.tool_choice.unwrap_or_else(|| json!("auto"));
        let mut messages = Vec::new();
        input_messages(&self.input, &mut messages)?;
        if messages.is_empty() {
            return Err(invalid("input", "input must contain at least one item"));
        }
        let resent = fold_instructions(self.instructions.as_deref(), &mut messages)?;
        let messages = match base {
            Some(base) => {
                // Chained replay: the stored history already carries the system prefix and the
                // prior turns. Re-sending instructions or a system/developer message would change
                // the tokenized prompt prefix and defeat the paged-KV radix hit the chain exists to
                // get — so accept it only when the stored prefix already says exactly that, and
                // reject a real edit instead of silently diverging from either turn.
                if let Some(resent) = resent.as_deref() {
                    let stored = stored_system_prefix(&base);
                    let duplicate = stored
                        .as_deref()
                        .is_some_and(|stored| prefix_already_stored(stored, resent));
                    if !duplicate {
                        let param = if self.instructions.is_some() {
                            "instructions"
                        } else {
                            "input"
                        };
                        return Err(invalid(
                            param,
                            if stored.is_none() {
                                "this chain has no system prefix: send instructions on the first \
                                 turn of the chain, not on a chained request"
                                    .to_string()
                            } else {
                                "instructions do not match the system prefix already stored for \
                                 this chain; keep them byte-identical to the first turn or start a \
                                 new chain"
                                    .to_string()
                            },
                        ));
                    }
                }
                let mut chained = (*base).clone();
                chained.extend(messages);
                chained
            }
            None => {
                // Many GGUF templates accept only one initial system message. Preserve the
                // instruction prefix order, but do not silently move later instructions
                // across conversation turns.
                if let Some(text) = resent {
                    messages.insert(0, message("system", Some(json!(text))));
                }
                messages
            }
        };
        // Captured AFTER the history fold and BEFORE the assistant output is appended on save, so
        // this is the complete context the model was handed — replayed turns and this turn's input
        // — which is what `GET /v1/responses/{id}/input_items` has to report to be worth reading.
        let input_items = if store {
            to_items(&messages)
        } else {
            Vec::new()
        };
        // The store replays this exact vector; the chat path owns a clone. One Vec-of-DTO
        // clone per request is nothing next to a generation.
        let history = Arc::new(messages.clone());
        let chat = ChatRequest {
            model: self.model,
            messages,
            stream: self.stream,
            // The chat DTO keeps the wire value verbatim; `parse_reasoning_effort` reads it back.
            reasoning_effort: reasoning_effort.map(|level| json!(level)),
            // `reasoning.effort == "none"`: no reasoning — carry it as the thinking switch, which is
            // the engine's equivalent of OpenAI's `none` level.
            chat_template_kwargs: reasoning_off.then(|| ChatTemplateOptions {
                enable_thinking: Some(false),
                ..Default::default()
            }),
            tools: convert_tools(Some(&Value::Array(tools.clone())))?,
            tool_choice: convert_tool_choice(Some(&tool_choice))?,
            max_tokens: None,
            max_completion_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: None,
            seed: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            repeat_penalty: None,
            response_format,
        };
        let meta = Meta {
            instructions: self.instructions,
            // The wire object describes the request; dispatch applies its own safety cap.
            max_output_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            reasoning,
            tools,
            tool_choice,
            metadata,
            prompt_cache_key: self.prompt_cache_key,
            text,
            parallel_tool_calls: self.parallel_tool_calls.unwrap_or(true),
            user: self.user,
            previous_response_id: previous,
            store,
            encrypted_reasoning,
            handle_withheld,
        };
        Ok(Prepared {
            chat,
            meta,
            history,
            input_items,
            conversation,
            store,
        })
    }
}

/// Re-encode the normalised Responses format in the CHAT wire shape. The dispatch path
/// (`GenParams::from_request`) runs exactly one parser — `response_format` — so the flat Responses
/// `text.format` object must not reach it as-is.
fn chat_wire_format(format: &ResponseFormat) -> Option<Value> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(json!({"type":"json_object"})),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => Some(json!({
            "type":"json_schema",
            "json_schema":{
                "name":name.clone().unwrap_or_else(|| "response".to_string()),
                "schema":schema,
                "strict":strict
            }
        })),
    }
}

/// Replace input items that carry no content of their own — `{"type":"item_reference","id":...}`
/// and a `reasoning` item sent back with only its `encrypted_content` handle — with the stored items
/// they name, so `input_messages` converts the real item. A reasoning item that already has a
/// readable `summary` is left alone: it replays without the store, as it always has.
fn resolve_references(
    store: &Arc<Mutex<ResponseStore>>,
    enabled: bool,
    input: &mut Value,
) -> Result<(), ParamError> {
    let Some(items) = input.as_array_mut() else {
        return Ok(());
    };
    for item in items.iter_mut() {
        let lookup = match item["type"].as_str() {
            Some("item_reference") => {
                if !enabled {
                    return Err(invalid(
                        "input",
                        "item_reference needs server-side response state; enable serve.response_store_entries",
                    ));
                }
                item.get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("input", "item_reference requires a string id"))?
                    .to_owned()
            }
            Some("reasoning") if has_readable_summary(item) => continue,
            Some("reasoning") => {
                let handle = item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid("input", "reasoning item requires readable summary"))?;
                if !enabled {
                    return Err(invalid(
                        "input",
                        "a reasoning handle needs server-side response state; enable serve.response_store_entries",
                    ));
                }
                reasoning_item_id(handle).ok_or_else(|| {
                    invalid(
                        "input",
                        format!("{handle:?} is not a reasoning handle this server issued"),
                    )
                })?
            }
            _ => continue,
        };
        let mut guard = store.lock().unwrap();
        let resolved = guard.resolve_item(&lookup).ok_or_else(|| {
            invalid(
                "input",
                format!(
                    "no stored item {lookup:?} (state is LRU-bounded; without \
                     serve.response_store_path a restart loses it)"
                ),
            )
        })?;
        *item = resolved;
    }
    Ok(())
}

/// A reasoning item that can be replayed from its `summary` alone — the pre-handle wire shape.
fn has_readable_summary(item: &Value) -> bool {
    item.get("summary")
        .and_then(Value::as_array)
        .is_some_and(|summaries| {
            summaries
                .iter()
                .any(|summary| summary["type"] == "summary_text")
        })
}

pub(super) async fn handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<ResponsesRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    let Json(mut request) = match body {
        Ok(value) => value,
        Err(e) => return json_body_error(e),
    };
    let enabled = state.cfg.serve.response_store_entries > 0;
    let previous = match chain_string(&request.previous_response_id, "previous_response_id") {
        Ok(value) => value,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    let conversation = match chain_string(&request.conversation, "conversation") {
        Ok(value) => value,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    // OpenAI defaults `store` to true; a stateless server cannot honour replay, so the default
    // follows the feature: ON when the store is enabled, OFF (the old wire shape) when not.
    let store = request.store.unwrap_or(enabled);
    if !enabled {
        // Name the field the client actually sent: an SDK shows `error.param`, and blaming
        // `previous_response_id` for a `conversation` or `store` request sends the reader to the
        // wrong line of their own code. `include:["reasoning.encrypted_content"]` is NOT such a
        // request — it degrades to summary-only reasoning items (`handle_withheld`) instead.
        let (param, asked) = (
            if previous.is_some() {
                "previous_response_id"
            } else if conversation.is_some() {
                "conversation"
            } else {
                "store"
            },
            previous.is_some() || conversation.is_some() || request.store == Some(true),
        );
        if asked {
            return param_error(
                Some(param),
                "server-side response state is not supported; enable serve.response_store_entries"
                    .to_string(),
            );
        }
    }
    // A `conversation` cursor only advances with a stored turn, so asking to continue one while
    // refusing to store it would leave the next request of the same name at a blank prefix.
    // Replay is the whole point of the field — say so instead of quietly restarting the chat.
    if conversation.is_some() && !store {
        return param_error(
            Some("conversation"),
            "a conversation needs store=true: with store=false nothing advances the cursor and \
             the next turn starts from an empty context"
                .to_string(),
        );
    }
    if let Err(e) = resolve_references(&state.response_store, enabled, &mut request.input) {
        return param_error(Some(e.param), e.message);
    }
    let base = if enabled {
        match state
            .response_store
            .lock()
            .unwrap()
            .chain(previous.as_deref(), conversation.as_deref())
        {
            Ok(value) => value,
            Err(e) => return param_error(Some(e.param), e.message),
        }
    } else {
        None
    };
    let prepared = match request.into_chat(base, previous, conversation, store) {
        Ok(value) => value,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    let Prepared {
        chat,
        meta,
        history,
        input_items,
        conversation,
        store,
    } = prepared;
    let streaming = chat.stream;
    let save = (enabled && store).then(|| SaveCtx {
        store: state.response_store.clone(),
        history,
        input_items,
        conversation,
    });
    let response = dispatch_chat(state, chat, "/v1/responses").await;
    if response.status() != StatusCode::OK {
        return response;
    }
    if streaming {
        stream_response(response, meta, save)
    } else {
        non_stream_response(response, meta, save).await
    }
}

/// What a successful generation must persist: the shared store, the input-side history, the
/// items the input arrived as, and the conversation cursor to advance.
struct SaveCtx {
    store: Arc<Mutex<ResponseStore>>,
    history: Arc<Vec<ChatMessageDto>>,
    input_items: Vec<Value>,
    conversation: Option<String>,
}

/// Append this turn's assistant output to the replay history. The DTO mirrors what a
/// stateless client would send back next turn: content string (omitted when the turn is
/// tool-calls only), reasoning summary, and the tool_calls array.
fn output_history(
    history: &Arc<Vec<ChatMessageDto>>,
    reasoning: &str,
    content: &str,
    calls: &[Value],
) -> Arc<Vec<ChatMessageDto>> {
    let mut out = (**history).clone();
    let mut assistant = message(
        "assistant",
        if content.is_empty() && !calls.is_empty() {
            None
        } else {
            Some(Value::String(content.to_owned()))
        },
    );
    if !reasoning.is_empty() {
        assistant.reasoning_content = Some(reasoning.to_owned());
    }
    if !calls.is_empty() {
        assistant.tool_calls = Some(Value::Array(calls.to_vec()));
    }
    out.push(assistant);
    Arc::new(out)
}

fn response_id(chat_id: &str) -> String {
    format!(
        "resp_{}",
        chat_id.strip_prefix("chatcmpl-").unwrap_or(chat_id)
    )
}

fn response_base(id: &str, created: i64, model: &str, meta: &Meta) -> Value {
    json!({
        "id":id,"object":"response","created_at":created,"completed_at":null,
        "status":"in_progress","background":false,
        "error":null,"incomplete_details":null,"instructions":meta.instructions,
        "max_output_tokens":meta.max_output_tokens,"model":model,"output":[],
        "parallel_tool_calls":meta.parallel_tool_calls,
        "previous_response_id":meta.previous_response_id,
        "prompt_cache_key":meta.prompt_cache_key,"prompt_cache_retention":null,
        "reasoning":meta.reasoning,"service_tier":"default","store":meta.store,
        "temperature":meta.temperature,
        "text":meta.text,"tool_choice":meta.tool_choice,"tools":meta.tools,
        "top_p":meta.top_p,"truncation":"disabled","usage":null,"user":meta.user,
        "metadata":meta.metadata
    })
}

fn output_items(
    id: &str,
    reasoning: &str,
    content: &str,
    calls: &[Value],
    encrypted: bool,
) -> Vec<Value> {
    let mut items = Vec::new();
    if !reasoning.is_empty() {
        let mut item = json!({"id":format!("rs_{id}"),"type":"reasoning","status":"completed",
            "summary":[{"type":"summary_text","text":reasoning}]});
        if encrypted {
            item["encrypted_content"] = json!(reasoning_handle(id));
        }
        items.push(item);
    }
    if !content.is_empty() || calls.is_empty() {
        items.push(
            json!({"id":format!("msg_{id}"),"type":"message","status":"completed",
            "role":"assistant","content":[{"type":"output_text","text":content,"annotations":[]}]}),
        );
    }
    for (index, call) in calls.iter().enumerate() {
        items.push(json!({"id":format!("fc_{id}_{index}"),"type":"function_call","status":"completed",
            "call_id":call["id"],"name":call["function"]["name"],"arguments":call["function"]["arguments"]}));
    }
    items
}

/// The `encrypted_content` value handed to a client that asked for a replayable reasoning handle:
/// a pointer into the response store, NOT a secret. There is no crypto layer here and no hosted
/// reasoning to seal, so what this server can honestly offer is the reference a chained request
/// already resolves — the same text the client saw in `summary`, reachable by id while the LRU holds
/// the turn. Its own prefix keeps handles and `item_reference` ids in separate namespaces, so a
/// client that sends the wrong kind gets a precise error instead of a resolution that happens to work.
fn reasoning_handle(id: &str) -> String {
    format!("rehs_{id}")
}

/// The reasoning item id a handle points at, or `None` for anything that is not one of ours.
fn reasoning_item_id(handle: &str) -> Option<String> {
    handle
        .strip_prefix("rehs_")
        .filter(|id| !id.is_empty())
        .map(|id| format!("rs_{id}"))
}

/// The calls a chat completion's assistant message carries, in OpenAI chat shape. Read through one
/// helper because the envelope and the replayed history must apply the same `parallel_tool_calls`
/// shaping — a chain that replays calls the client never saw would have the model wait on results
/// nobody asked for.
fn chat_calls(chat: &Value) -> Vec<Value> {
    chat["choices"][0]["message"]["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// `parallel_tool_calls=false` promises the client at most one call per turn. The local decoder
/// has no such switch — the model decides how many calls to emit — so the promise is kept by
/// shaping the output instead. What was withheld is then named in the response: a client that sees
/// one call and no trace of the others cannot tell a short model answer from a server-side edit,
/// and an agent loop that quietly lost two of its tool calls is worse than one that got an error.
fn deliverable_calls(calls: &[Value], parallel: bool) -> (&[Value], usize) {
    if parallel || calls.len() < 2 {
        (calls, 0)
    } else {
        (&calls[..1], calls.len() - 1)
    }
}

fn note_suppressed(response: &mut Value, suppressed: usize) {
    if suppressed > 0 {
        push_warning(
            response,
            &format!(
                "{suppressed} additional tool call(s) were not returned because \
                 parallel_tool_calls=false"
            ),
        );
    }
}

/// Append one entry to the response's `warnings` array (created on first use).
fn push_warning(response: &mut Value, text: &str) {
    match response.get_mut("warnings").and_then(Value::as_array_mut) {
        Some(list) => list.push(json!(text)),
        None => response["warnings"] = json!([text]),
    }
}

/// Say that a requested `reasoning.encrypted_content` handle was not issued: the turn is stateless,
/// so reasoning items carry their readable summary only (OpenAI's own servers omit the field when a
/// model has nothing to encrypt, and replay only needs the summary).
fn note_handle_withheld(response: &mut Value, withheld: bool) {
    if withheld {
        push_warning(
            response,
            "reasoning.encrypted_content was requested but this server keeps no response state \
             (serve.response_store_entries); reasoning items carry their summary only",
        );
    }
}

fn responses_usage(chat_usage: &Value) -> Value {
    // Local model thinking is emitted as visible reasoning text and included in output_tokens.
    // The backend exposes no separately metered hidden reasoning tokens.
    json!({
        "input_tokens":chat_usage["prompt_tokens"],
        "input_tokens_details":{"cached_tokens":chat_usage["prompt_tokens_details"]["cached_tokens"]},
        "output_tokens":chat_usage["completion_tokens"],
        "output_tokens_details":{"reasoning_tokens":0},
        "total_tokens":chat_usage["total_tokens"]
    })
}

fn completed_response(chat: &Value, meta: &Meta) -> Value {
    let id = response_id(chat["id"].as_str().unwrap_or("unknown"));
    let mut response = response_base(
        &id,
        chat["created"].as_i64().unwrap_or_default(),
        chat["model"].as_str().unwrap_or(""),
        meta,
    );
    let choice = &chat["choices"][0];
    let message = &choice["message"];
    let all_calls = chat_calls(chat);
    let (calls, suppressed) = deliverable_calls(&all_calls, meta.parallel_tool_calls);
    let incomplete = choice["finish_reason"] == "length";
    response["status"] = json!(if incomplete {
        "incomplete"
    } else {
        "completed"
    });
    response["completed_at"] = if incomplete {
        Value::Null
    } else {
        json!(unix_ts())
    };
    response["incomplete_details"] = if incomplete {
        json!({"reason":"max_output_tokens"})
    } else {
        Value::Null
    };
    response["output"] = json!(output_items(
        &id,
        message["reasoning_content"].as_str().unwrap_or(""),
        message["content"].as_str().unwrap_or(""),
        calls,
        meta.encrypted_reasoning
    ));
    note_suppressed(&mut response, suppressed);
    note_handle_withheld(&mut response, meta.handle_withheld);
    response["usage"] = responses_usage(&chat["usage"]);
    response
}

/// Ceiling on the chat completion this layer will buffer to translate a NON-streaming
/// `/v1/responses` reply. The largest completion the server can produce is one context window of
/// text: 150 K tokens at the worst realistic JSON-escape ratio (~6 bytes per token for `\u00XX`
/// escapes) is under 1 MiB, so 64 MiB sits ~60x above anything a real request can reach while still
/// bounding what one in-flight request can hold in memory.
const MAX_NON_STREAM_TRANSLATION_BYTES: usize = 64 * 1024 * 1024;

async fn non_stream_response(response: Response, meta: Meta, save: Option<SaveCtx>) -> Response {
    let bytes = match to_bytes(response.into_body(), MAX_NON_STREAM_TRANSLATION_BYTES).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                format!(
                    "chat response exceeded the {} MiB non-streaming translation limit; \
                     send stream=true instead of buffering the completion: {e}",
                    MAX_NON_STREAM_TRANSLATION_BYTES / (1024 * 1024)
                ),
            )
        }
    };
    let chat: Value = match serde_json::from_slice(&bytes) {
        Ok(chat) => chat,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let body = completed_response(&chat, &meta);
    if let Some(save) = save {
        let choice = &chat["choices"][0];
        // Persist completed AND incomplete (length-truncated) turns — an incomplete response is
        // still replayable history; a missing terminal frame is not.
        if !choice["finish_reason"].is_null() {
            let message = &choice["message"];
            // Same shaping as the envelope above, so the chain replays exactly what was served.
            let all_calls = chat_calls(&chat);
            let (calls, _) = deliverable_calls(&all_calls, meta.parallel_tool_calls);
            let history = output_history(
                &save.history,
                message["reasoning_content"].as_str().unwrap_or(""),
                message["content"].as_str().unwrap_or(""),
                calls,
            );
            let id = response_id(chat["id"].as_str().unwrap_or("unknown"));
            save.store.lock().unwrap().put(
                &id,
                body.clone(),
                history,
                save.input_items,
                save.conversation.as_deref(),
            );
        }
    }
    Json(body).into_response()
}

struct StreamState {
    source: Pin<Box<dyn Stream<Item = Result<Bytes, axum::Error>> + Send>>,
    buffer: Vec<u8>,
    pending: VecDeque<Bytes>,
    meta: Meta,
    id: Option<String>,
    created: i64,
    model: String,
    sequence: u64,
    reasoning: String,
    content: String,
    calls: Vec<Value>,
    /// The tool calls the upstream stream has opened, index-aligned with `calls`.
    open_calls: Vec<OpenCall>,
    /// Upstream indexes dropped by `parallel_tool_calls=false`, so their later fragments stay out.
    dropped: Vec<u64>,
    suppressed: usize,
    reasoning_index: Option<usize>,
    content_index: Option<usize>,
    next_index: usize,
    terminal: bool,
    save: Option<SaveCtx>,
}

/// One tool call the upstream stream opened, in the shape the Responses events address it by.
struct OpenCall {
    /// The `delta.tool_calls[].index` this slot answers; `None` for an upstream that sends no index.
    upstream_index: Option<u64>,
    /// Position in `response.output` — allocated when the item opens, after the prose items.
    output_index: usize,
    /// Whether `response.output_item.added` has been emitted for this slot: the first fragment
    /// announces it (and closes the prose items ahead of it), later fragments only add arguments.
    added: bool,
}

impl StreamState {
    fn emit(&mut self, kind: &str, mut payload: Value) {
        payload["type"] = json!(kind);
        payload["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        self.pending
            .push_back(Bytes::from(format!("event: {kind}\ndata: {payload}\n\n")));
    }

    fn fail(&mut self, upstream: &Value) {
        let code = upstream["code"]
            .as_str()
            .or_else(|| upstream["type"].as_str())
            .unwrap_or("server_error");
        let message = upstream["message"].as_str().unwrap_or("generation failed");
        if let Some(id) = self.id.clone() {
            let mut response = response_base(&id, self.created, &self.model, &self.meta);
            response["status"] = json!("failed");
            response["error"] = json!({"code":code,"message":message});
            // No handle on a failed turn: it is never stored, so nothing could resolve one.
            let mut items = output_items(&id, &self.reasoning, &self.content, &self.calls, false);
            // A failed stream has no item-completion boundary. Keep only items
            // that were actually started, and leave unfinished items in progress.
            if self.content_index.is_none() && self.content.is_empty() {
                items.retain(|item| item["type"] != "message");
            }
            for item in &mut items {
                if (item["type"] == "reasoning" && self.reasoning_index.is_some())
                    || (item["type"] == "message" && self.content_index.is_some())
                {
                    item["status"] = json!("in_progress");
                }
            }
            response["output"] = json!(items);
            self.emit("response.failed", json!({"response":response}));
        } else {
            self.emit(
                "error",
                json!({"code":code,"message":message,"param":upstream.get("param")}),
            );
        }
        self.terminal = true;
    }

    fn start(&mut self, chunk: &Value) {
        if self.id.is_some() {
            return;
        }
        let id = response_id(chunk["id"].as_str().unwrap_or("unknown"));
        self.created = chunk["created"].as_i64().unwrap_or_default();
        self.model = chunk["model"].as_str().unwrap_or("").into();
        self.id = Some(id.clone());
        let mut response = response_base(&id, self.created, &self.model, &self.meta);
        response["status"] = json!("in_progress");
        self.emit("response.created", json!({"response":response}));
        self.emit("response.in_progress", json!({"response":response}));
    }

    fn reasoning_delta(&mut self, delta: &str) {
        let id = self.id.clone().unwrap();
        let index = if let Some(index) = self.reasoning_index {
            index
        } else {
            let index = self.next_index;
            self.next_index += 1;
            self.reasoning_index = Some(index);
            let mut item = json!({"id":format!("rs_{id}"),"type":"reasoning","status":"in_progress","summary":[]});
            if self.meta.encrypted_reasoning {
                item["encrypted_content"] = json!(reasoning_handle(&id));
            }
            self.emit(
                "response.output_item.added",
                json!({"output_index":index,"item":item}),
            );
            self.emit("response.reasoning_summary_part.added", json!({"item_id":format!("rs_{id}"),"output_index":index,"summary_index":0,"part":{"type":"summary_text","text":""}}));
            index
        };
        self.reasoning.push_str(delta);
        self.emit("response.reasoning_summary_text.delta", json!({"item_id":format!("rs_{id}"),"output_index":index,"summary_index":0,"delta":delta}));
    }

    fn content_delta(&mut self, delta: &str) {
        let id = self.id.clone().unwrap();
        let index = if let Some(index) = self.content_index {
            index
        } else {
            let index = self.next_index;
            self.next_index += 1;
            self.content_index = Some(index);
            self.emit("response.output_item.added", json!({"output_index":index,"item":{"id":format!("msg_{id}"),"type":"message","status":"in_progress","role":"assistant","content":[]}}));
            self.emit("response.content_part.added", json!({"item_id":format!("msg_{id}"),"output_index":index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}));
            index
        };
        self.content.push_str(delta);
        self.emit("response.output_text.delta", json!({"item_id":format!("msg_{id}"),"output_index":index,"content_index":0,"delta":delta}));
    }

    fn finish_items(&mut self, include_empty_message: bool) {
        let id = self.id.clone().unwrap();
        if let Some(index) = self.reasoning_index.take() {
            let text = self.reasoning.clone();
            self.emit("response.reasoning_summary_text.done", json!({"item_id":format!("rs_{id}"),"output_index":index,"summary_index":0,"text":text}));
            self.emit("response.reasoning_summary_part.done", json!({"item_id":format!("rs_{id}"),"output_index":index,"summary_index":0,"part":{"type":"summary_text","text":text}}));
            let mut item = json!({"id":format!("rs_{id}"),"type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":text}]});
            if self.meta.encrypted_reasoning {
                item["encrypted_content"] = json!(reasoning_handle(&id));
            }
            self.emit(
                "response.output_item.done",
                json!({"output_index":index,"item":item}),
            );
        }
        if self.content_index.is_none() && include_empty_message && self.calls.is_empty() {
            self.content_delta("");
        }
        if let Some(index) = self.content_index.take() {
            let text = self.content.clone();
            self.emit("response.output_text.done", json!({"item_id":format!("msg_{id}"),"output_index":index,"content_index":0,"text":text}));
            self.emit("response.content_part.done", json!({"item_id":format!("msg_{id}"),"output_index":index,"content_index":0,"part":{"type":"output_text","text":text,"annotations":[]}}));
            self.emit("response.output_item.done", json!({"output_index":index,"item":{"id":format!("msg_{id}"),"type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":text,"annotations":[]}]}}));
        }
    }

    fn tool_call(&mut self, call: Value) {
        // OpenAI chat-completions shape: fragments of one call share its `index`, and only the
        // first carries `id`/`name`. Keying on the index is what turns a call that arrives in
        // pieces into ONE item with several argument deltas; treating every fragment as a whole
        // call would hand the client a row of half-argument calls to dispatch.
        let Some(slot) = self.open_slot(&call) else {
            return; // suppressed by `parallel_tool_calls=false`, counted in `open_slot`
        };
        if let Some(name) = call["function"]["name"]
            .as_str()
            .filter(|name| !name.is_empty())
        {
            if self.calls[slot]["function"]["name"].is_null() {
                self.calls[slot]["function"]["name"] = json!(name);
            }
        }
        if call["id"].is_string() && self.calls[slot]["id"].is_null() {
            self.calls[slot]["id"] = call["id"].clone();
        }
        let id = self.id.clone().unwrap();
        let item_id = format!("fc_{id}_{slot}");
        let output_index = self.open_calls[slot].output_index;
        if !self.open_calls[slot].added {
            // The prose items this call follows close first, so the events arrive in `output_index`
            // order — the ordering guarantee a client uses to interleave thinking and calls.
            self.finish_items(false);
            self.open_calls[slot].added = true;
            self.emit("response.output_item.added", json!({"output_index":output_index,"item":{"id":item_id,"type":"function_call","status":"in_progress","call_id":self.calls[slot]["id"].clone(),"name":self.calls[slot]["function"]["name"].clone(),"arguments":""}}));
        }
        if let Some(fragment) = call["function"]["arguments"]
            .as_str()
            .filter(|text| !text.is_empty())
        {
            let mut accumulated = self.calls[slot]["function"]["arguments"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            accumulated.push_str(fragment);
            self.calls[slot]["function"]["arguments"] = json!(accumulated);
            self.emit(
                "response.function_call_arguments.delta",
                json!({"item_id":item_id,"output_index":output_index,"delta":fragment}),
            );
        }
    }

    /// The slot a `delta.tool_calls[]` fragment belongs to, opening a new one when it does not.
    /// `None` = drop: `parallel_tool_calls=false` already delivered this turn's single call. A drop
    /// is remembered by upstream index so the later fragments of a suppressed call cannot leak into
    /// the arguments of the call the client did receive.
    fn open_slot(&mut self, call: &Value) -> Option<usize> {
        let upstream = call["index"].as_u64();
        if let Some(key) = upstream {
            if let Some(slot) = self
                .open_calls
                .iter()
                .position(|open| open.upstream_index == Some(key))
            {
                return Some(slot);
            }
            if self.dropped.contains(&key) {
                return None;
            }
        } else if let Some(last) = self.open_calls.len().checked_sub(1) {
            // An index-less, name-less fragment continues the call already open. Once anything has
            // been suppressed there is no safe target, so it is dropped instead of appended.
            return if self.dropped.is_empty() {
                Some(last)
            } else {
                None
            };
        }
        if !self.meta.parallel_tool_calls && !self.calls.is_empty() {
            if let Some(key) = upstream {
                self.dropped.push(key);
            }
            self.suppressed += 1;
            return None;
        }
        let mut opened = call.clone();
        // The stored object is the accumulation target, so the arguments start empty and every
        // fragment — this one included — is added by the caller.
        opened["function"]["arguments"] = json!("");
        self.open_calls.push(OpenCall {
            upstream_index: upstream,
            output_index: self.next_index,
            added: false,
        });
        self.next_index += 1;
        self.calls.push(opened);
        Some(self.calls.len() - 1)
    }

    /// Close every tool call that was opened: the terminal events that move an item from "in
    /// progress" to something the client can dispatch.
    fn finish_calls(&mut self) {
        let id = match self.id.clone() {
            Some(id) => id,
            None => return,
        };
        for slot in 0..self.calls.len() {
            if !self.open_calls[slot].added {
                continue; // never surfaced, so there is nothing to close
            }
            let output_index = self.open_calls[slot].output_index;
            let item_id = format!("fc_{id}_{slot}");
            let arguments = self.calls[slot]["function"]["arguments"]
                .as_str()
                .unwrap_or_default()
                .to_owned();
            self.emit(
                "response.function_call_arguments.done",
                json!({"item_id":item_id,"output_index":output_index,"arguments":arguments}),
            );
            self.emit("response.output_item.done", json!({"output_index":output_index,"item":{"id":item_id,"type":"function_call","status":"completed","call_id":self.calls[slot]["id"].clone(),"name":self.calls[slot]["function"]["name"].clone(),"arguments":arguments}}));
        }
    }

    fn consume(&mut self, data: &[u8]) {
        if data == b"[DONE]" {
            return;
        }
        let Ok(chunk) = serde_json::from_slice::<Value>(data) else {
            return;
        };
        if chunk.get("error").is_some() {
            self.fail(&chunk["error"]);
            return;
        }
        self.start(&chunk);
        let choice = &chunk["choices"][0];
        if let Some(delta) = choice["delta"]["reasoning_content"].as_str() {
            self.reasoning_delta(delta);
        }
        if let Some(delta) = choice["delta"]["content"].as_str() {
            self.content_delta(delta);
        }
        if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
            for call in calls {
                self.tool_call(call.clone());
            }
        }
        if !choice["finish_reason"].is_null() {
            self.finish_items(true);
            self.finish_calls();
            let id = self.id.clone().unwrap();
            let mut response = response_base(&id, self.created, &self.model, &self.meta);
            let incomplete = choice["finish_reason"] == "length";
            response["status"] = json!(if incomplete {
                "incomplete"
            } else {
                "completed"
            });
            response["completed_at"] = if incomplete {
                Value::Null
            } else {
                json!(unix_ts())
            };
            response["incomplete_details"] = if incomplete {
                json!({"reason":"max_output_tokens"})
            } else {
                Value::Null
            };
            response["output"] = json!(output_items(
                &id,
                &self.reasoning,
                &self.content,
                &self.calls,
                self.meta.encrypted_reasoning
            ));
            response["usage"] = responses_usage(&chunk["usage"]);
            note_suppressed(&mut response, self.suppressed);
            note_handle_withheld(&mut response, self.meta.handle_withheld);
            // Same save rule as the non-streaming path: completed AND incomplete persist;
            // `fail()` (below the terminal boundary) never does. The stored envelope is the exact
            // object this event hands the client, so `retrieve` cannot drift from what was served.
            if let Some(save) = self.save.take() {
                let history =
                    output_history(&save.history, &self.reasoning, &self.content, &self.calls);
                save.store.lock().unwrap().put(
                    &id,
                    response.clone(),
                    history,
                    save.input_items,
                    save.conversation.as_deref(),
                );
            }
            self.emit(
                if incomplete {
                    "response.incomplete"
                } else {
                    "response.completed"
                },
                json!({"response":response}),
            );
            self.terminal = true;
        }
    }

    fn consume_frames(&mut self) {
        while let Some(end) = self.buffer.windows(2).position(|w| w == b"\n\n") {
            let frame: Vec<u8> = self.buffer.drain(..end + 2).collect();
            let mut has_data = false;
            let mut has_comment = false;
            for line in frame.split(|b| *b == b'\n') {
                if let Some(data) = line.strip_prefix(b"data: ") {
                    has_data = true;
                    self.consume(data.strip_suffix(b"\r").unwrap_or(data));
                } else if line.starts_with(b":") {
                    has_comment = true;
                }
            }
            if has_comment && !has_data {
                self.pending.push_back(Bytes::from(frame));
            }
        }
    }
}

fn stream_response(response: Response, meta: Meta, save: Option<SaveCtx>) -> Response {
    let source = Box::pin(response.into_body().into_data_stream());
    let state = StreamState {
        source,
        buffer: Vec::new(),
        pending: VecDeque::new(),
        meta,
        id: None,
        created: 0,
        model: String::new(),
        sequence: 0,
        reasoning: String::new(),
        content: String::new(),
        calls: Vec::new(),
        open_calls: Vec::new(),
        dropped: Vec::new(),
        suppressed: 0,
        reasoning_index: None,
        content_index: None,
        next_index: 0,
        terminal: false,
        save,
    };
    let translated = stream::unfold(state, |mut state| async move {
        loop {
            if let Some(bytes) = state.pending.pop_front() {
                return Some((Ok::<Bytes, axum::Error>(bytes), state));
            }
            if state.terminal {
                return None;
            }
            match state.source.next().await {
                Some(Ok(bytes)) => {
                    state.buffer.extend_from_slice(&bytes);
                    state.consume_frames();
                }
                Some(Err(error)) => {
                    state.fail(&json!({"code":"server_error","message":error.to_string()}));
                }
                None => {
                    state.fail(&json!({"code":"server_error","message":"generation ended without a terminal frame"}));
                }
            }
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-store")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(translated))
        .expect("Responses SSE response has valid headers")
}

// -- stored-response REST surface -------------------------------------------------------------
//
// These read what `POST /v1/responses` already wrote: the envelope the client received and the
// item context behind it. Nothing here re-runs a generation, so a stored id resolves to the same
// bytes that were served the first time.

/// Response ids are server-minted (`resp_<ms>-<seq>`); a path id longer than this is a probe, and
/// the store is keyed by strings, so bound the work before the lookup.
const MAX_ID_LEN: usize = 512;

/// `limit`/`order` as OpenAI's list endpoints spell them. Both stay strings: a typed `u32` field
/// would turn `?limit=abc` into axum's plain-text rejection, and every error on this API is a JSON
/// body that names the offending `param`.
#[derive(Deserialize)]
pub(super) struct PageQuery {
    #[serde(default)]
    limit: Option<String>,
    #[serde(default)]
    order: Option<String>,
}

const DEFAULT_PAGE_LIMIT: usize = 20;
const MAX_PAGE_LIMIT: usize = 100;

fn store_enabled(state: &AppState) -> bool {
    state.cfg.serve.response_store_entries > 0
}

/// The state-off answer for every read surface. An empty page would assert "no responses stored",
/// which is a different fact than "this server stores none" — same reasoning as the POST path.
/// `param` names what the client addressed: `response_id` on the id-bearing routes, null on the
/// bare list route where none of its query parameters caused this.
fn state_disabled(param: Option<&'static str>) -> Response {
    param_error(
        param,
        "server-side response state is not supported; enable serve.response_store_entries"
            .to_string(),
    )
}

fn bad_id() -> Response {
    param_error(
        Some("response_id"),
        format!("response id must be between 1 and {MAX_ID_LEN} characters"),
    )
}

/// OpenAI's 404 body: the same error object as a 400, with `param`/`code` present and null.
fn unknown_id(id: &str) -> Response {
    let mut body = error_body(
        &format!("Response with id '{id}' not found."),
        "invalid_request_error",
    );
    body["error"]["param"] = Value::Null;
    body["error"]["code"] = Value::Null;
    (StatusCode::NOT_FOUND, Json(body)).into_response()
}

fn page_window(query: &PageQuery) -> Result<(usize, bool), ParamError> {
    let limit = match query.limit.as_deref() {
        None => DEFAULT_PAGE_LIMIT,
        // A non-number and an out-of-range number get the same answer: the range the server
        // accepts is the whole fact the client needs.
        Some(raw) => match raw.parse::<usize>() {
            Ok(value) if (1..=MAX_PAGE_LIMIT).contains(&value) => value,
            _ => {
                return Err(invalid(
                    "limit",
                    format!("limit must be an integer between 1 and {MAX_PAGE_LIMIT}"),
                ))
            }
        },
    };
    let ascending = match query.order.as_deref() {
        None | Some("desc") => false,
        Some("asc") => true,
        Some(_) => return Err(invalid("order", "order must be \"asc\" or \"desc\"")),
    };
    Ok((limit, ascending))
}

fn page(data: Vec<Value>, has_more: bool) -> Value {
    // Borrowed before `data` moves into the envelope; items the server never minted an id for
    // (a replayed input item carries the client's own id, if any) page as `null` bounds.
    let id_of = |item: &Value| item.get("id").and_then(Value::as_str).map(str::to_owned);
    let first_id = data.first().and_then(id_of);
    let last_id = data.last().and_then(id_of);
    json!({
        "object":"list",
        "data":data,
        "first_id":first_id,
        "last_id":last_id,
        "has_more":has_more
    })
}

pub(super) async fn retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    if !store_enabled(&state) {
        return state_disabled(Some("response_id"));
    }
    if id.is_empty() || id.len() > MAX_ID_LEN {
        return bad_id();
    }
    match state.response_store.lock().unwrap().get(&id) {
        Some(response) => Json(response).into_response(),
        None => unknown_id(&id),
    }
}

pub(super) async fn remove(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    if !store_enabled(&state) {
        return state_disabled(Some("response_id"));
    }
    if id.is_empty() || id.len() > MAX_ID_LEN {
        return bad_id();
    }
    // Deleting also orphans anything chained from this response and any item reference into its
    // output: both then fail loudly (400), which is the honest answer — never a blank replay.
    match state.response_store.lock().unwrap().remove(&id) {
        Some(_) => Json(json!({"id":id,"object":"response","deleted":true})).into_response(),
        None => unknown_id(&id),
    }
}

/// A rejected query string still has to answer in the JSON error shape, like the POST body does:
/// axum's own rejection is plain text, which an SDK cannot read as an API error.
fn page_query(
    query: Result<Query<PageQuery>, axum::extract::rejection::QueryRejection>,
) -> Result<PageQuery, String> {
    match query {
        Ok(Query(query)) => Ok(query),
        Err(e) => Err(e.body_text()),
    }
}

pub(super) async fn list(
    State(state): State<AppState>,
    headers: HeaderMap,
    query: Result<Query<PageQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    let query = match page_query(query) {
        Ok(query) => query,
        Err(message) => return param_error(None, message),
    };
    if !store_enabled(&state) {
        return state_disabled(None);
    }
    let (limit, ascending) = match page_window(&query) {
        Ok(window) => window,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    let store = state.response_store.lock().unwrap();
    let data = store.list(limit, ascending);
    Json(page(data, store.len() > limit)).into_response()
}

pub(super) async fn input_items(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    query: Result<Query<PageQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    let query = match page_query(query) {
        Ok(query) => query,
        Err(message) => return param_error(None, message),
    };
    if !store_enabled(&state) {
        return state_disabled(Some("response_id"));
    }
    if id.is_empty() || id.len() > MAX_ID_LEN {
        return bad_id();
    }
    let (limit, ascending) = match page_window(&query) {
        Ok(window) => window,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    let mut store = state.response_store.lock().unwrap();
    // The prompt order is oldest-first; OpenAI pages items newest-first unless asked otherwise.
    let mut items = match store.prompt_items(&id) {
        Some(items) => items,
        None => return unknown_id(&id),
    };
    if !ascending {
        items.reverse();
    }
    let has_more = items.len() > limit;
    items.truncate(limit);
    Json(page(items, has_more)).into_response()
}

pub(super) async fn cancel(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    if !store_enabled(&state) {
        return state_disabled(Some("response_id"));
    }
    if id.is_empty() || id.len() > MAX_ID_LEN {
        return bad_id();
    }
    if state.response_store.lock().unwrap().get(&id).is_none() {
        return unknown_id(&id);
    }
    // Every response on this server is generated synchronously: by the time a client can address
    // it by id, it is already terminal, so there is nothing to cancel and nothing queued to drop.
    param_error(
        Some("background"),
        "only a background response can be cancelled; this server generates synchronously, and \
         disconnecting the stream is the way to stop one"
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    struct Fixture;

    struct Limited;

    impl ChatGenerator for Limited {
        fn chat(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&Value>,
            _tool_choice: Option<&str>,
            _params: &GenParams,
            _cancel: &Arc<AtomicBool>,
            _progress: Option<infr_core::GenerationProgressCallback>,
            on_delta: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            on_delta(Delta::Content("partial".into()));
            Ok(ChatOutcome {
                finish: Finish::Length,
                prompt_tokens: 8,
                cached_prompt_tokens: 2,
                completion_tokens: 1,
            })
        }
    }

    impl ChatGenerator for Fixture {
        fn chat(
            &self,
            messages: &[ChatMessage],
            tools: Option<&Value>,
            _tool_choice: Option<&str>,
            params: &GenParams,
            _cancel: &Arc<AtomicBool>,
            _progress: Option<infr_core::GenerationProgressCallback>,
            on_delta: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            if messages.last().is_some_and(|m| !m.images.is_empty()) {
                on_delta(Delta::Content(format!(
                    "image:{}",
                    messages.last().unwrap().images[0]
                )));
            } else if params.response_format.is_structured() {
                on_delta(Delta::Content(r#"{"answer":"hello"}"#.into()));
            } else if tools.is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty())) {
                on_delta(Delta::ToolCall {
                    name: "shell".into(),
                    arguments: r#"{"cmd":"dir"}"#.into(),
                });
            } else {
                on_delta(Delta::Reasoning("think".into()));
                on_delta(Delta::Content("hello".into()));
            }
            Ok(ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 20,
                cached_prompt_tokens: 12,
                completion_tokens: 3,
            })
        }
    }

    async fn post(input: Value) -> Response {
        let router = build_router(AppState::new(
            Arc::new(Fixture),
            "fixture",
            1,
            Arc::new(Config::default()),
        ));
        router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(input.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn json_body(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn non_streaming_preserves_usage_and_metadata() {
        let response = post(
            json!({"model":"fixture","input":"hi","instructions":"brief",
            "temperature":0.25,"metadata":{"trace":"local"},"max_output_tokens":32,
            "prompt_cache_key":"harness-test","user":"local-user",
            "service_tier":"auto"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["output"][0]["type"], "reasoning");
        assert_eq!(body["output"][1]["content"][0]["text"], "hello");
        assert_eq!(body["usage"]["input_tokens"], 20);
        assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 12);
        assert_eq!(body["temperature"], 0.25);
        assert_eq!(body["metadata"]["trace"], "local");
        assert_eq!(body["prompt_cache_key"], "harness-test");
        assert_eq!(body["max_output_tokens"], 32);
        assert_eq!(body["user"], "local-user");
        assert_eq!(body["service_tier"], "default");
    }

    #[test]
    fn instruction_roles_form_one_portable_system_prefix() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","instructions":"primary instruction",
            "input":[
                {"role":"developer","content":"developer instruction"},
                {"role":"system","content":[{"type":"input_text","text":"system instruction"}]},
                {"role":"user","content":"hello"}
            ]
        }))
        .unwrap();
        let chat = request.into_chat(None, None, None, false).unwrap().chat;
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(
            chat.messages[0].content.as_ref().unwrap(),
            "primary instruction\n\ndeveloper instruction\n\nsystem instruction"
        );
        let template = include_str!("../../infr-chat/tests/fixtures/qwen38_chat_template.jinja");
        let prompt = infr_chat::render_template(
            template,
            chat.messages
                .iter()
                .map(|m| json!({"role":m.role,"content":m.content}))
                .collect(),
            Value::Null,
            "",
            "",
            true,
            true,
        )
        .unwrap();
        assert!(prompt.contains("developer instruction"));
    }

    #[test]
    fn late_developer_message_is_rejected_instead_of_moved() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":[
                {"role":"user","content":"hello"},
                {"role":"developer","content":"change the rules"}
            ]
        }))
        .unwrap();
        assert_eq!(
            request
                .into_chat(None, None, None, false)
                .err()
                .unwrap()
                .param,
            "input"
        );
    }

    /// Turn one of a chain, as the store would have kept it: the folded history the next request
    /// replays as its base.
    fn chained_base(request: &Value) -> Arc<Vec<ChatMessageDto>> {
        let prepared: ResponsesRequest = serde_json::from_value(request.clone()).unwrap();
        Arc::new(
            prepared
                .into_chat(None, None, None, false)
                .unwrap()
                .chat
                .messages,
        )
    }

    #[test]
    fn identical_instructions_replay_the_chain_without_a_second_system_turn() {
        // The shape an SDK sends every turn: the same `instructions` on a chained request. The
        // stored prefix already says it, so the duplicate is dropped and the tokenized prompt
        // keeps the one system turn the chain was built on.
        let first = json!({"model":"fixture","instructions":"answer briefly","input":"hi"});
        let base = chained_base(&first);
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","instructions":"answer briefly","input":"again",
            "previous_response_id":"resp_1"
        }))
        .unwrap();
        let chat = request
            .into_chat(Some(base), None, None, false)
            .unwrap()
            .chat;
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(
            chat.messages.iter().filter(|m| m.role == "system").count(),
            1
        );
        assert_eq!(chat.messages[0].content.as_ref().unwrap(), "answer briefly");
        assert_eq!(chat.messages[1].content.as_ref().unwrap(), "hi");
        assert_eq!(chat.messages[2].content.as_ref().unwrap(), "again");
    }

    #[test]
    fn instructions_matching_a_leading_part_of_the_stored_prefix_are_accepted() {
        // Turn one folded `instructions` AND a developer message into one system turn; a client
        // that re-sends only its instructions still matches the prefix it stored.
        let first = json!({
            "model":"fixture","instructions":"primary",
            "input":[
                {"role":"developer","content":"tool rules"},
                {"role":"user","content":"hi"}
            ]
        });
        let base = chained_base(&first);
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","instructions":"primary","input":"again"
        }))
        .unwrap();
        let chat = request
            .into_chat(Some(base), None, None, false)
            .unwrap()
            .chat;
        assert_eq!(
            chat.messages[0].content.as_ref().unwrap(),
            "primary\n\ntool rules"
        );
        assert_eq!(chat.messages.len(), 3);
    }

    #[test]
    fn edited_instructions_on_a_chain_are_rejected_with_the_field_that_changed() {
        let base =
            chained_base(&json!({"model":"fixture","instructions":"answer briefly","input":"hi"}));
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","instructions":"answer at length","input":"again"
        }))
        .unwrap();
        let error = request
            .into_chat(Some(base), None, None, false)
            .err()
            .unwrap();
        assert_eq!(error.param, "instructions");
        assert!(error.message.contains("do not match the system prefix"));

        // A chain with no system turn at all: the instruction would be new context inserted after
        // the replayed turns, which no template can express.
        let base = chained_base(&json!({"model":"fixture","input":"hi"}));
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","instructions":"answer briefly","input":"again"
        }))
        .unwrap();
        let error = request
            .into_chat(Some(base), None, None, false)
            .err()
            .unwrap();
        assert_eq!(error.param, "instructions");
        assert!(error.message.contains("no system prefix"));
    }

    #[test]
    fn repeated_system_input_message_is_dropped_but_a_new_one_is_not() {
        let first = json!({
            "model":"fixture",
            "input":[{"role":"system","content":"answer briefly"},{"role":"user","content":"hi"}]
        });
        let base = chained_base(&first);
        let same: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture",
            "input":[{"role":"system","content":"answer briefly"},{"role":"user","content":"again"}]
        }))
        .unwrap();
        let chat = same
            .into_chat(Some(base.clone()), None, None, false)
            .unwrap()
            .chat;
        assert_eq!(
            chat.messages.iter().filter(|m| m.role == "system").count(),
            1
        );
        assert_eq!(chat.messages.len(), 3);

        let edited: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture",
            "input":[{"role":"system","content":"answer at length"},{"role":"user","content":"again"}]
        }))
        .unwrap();
        let error = edited
            .into_chat(Some(base), None, None, false)
            .err()
            .unwrap();
        assert_eq!(error.param, "input");
    }

    #[test]
    fn minimal_effort_and_requested_token_limit_are_reported_honestly() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hello","reasoning":{"effort":"minimal"},
            "max_output_tokens":100
        }))
        .unwrap();
        let Prepared { chat, meta, .. } = request.into_chat(None, None, None, false).unwrap();
        assert_eq!(chat.reasoning_effort, Some(json!("low")));
        assert_eq!(meta.reasoning["effort"], "low");
        assert_eq!(meta.max_output_tokens, Some(100));
    }

    /// `reasoning.effort: "none"` is OpenAI's "no reasoning" level. The local templates only have
    /// levels, so it must land on the thinking switch instead of a 400 (a real client — DeepSeek
    /// Harness — sent exactly this and could not talk to the server at all).
    #[test]
    fn effort_none_turns_thinking_off_instead_of_failing() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hello","reasoning":{"effort":"none"}
        }))
        .unwrap();
        let Prepared { chat, meta, .. } = request.into_chat(None, None, None, false).unwrap();
        assert_eq!(chat.reasoning_effort, None, "no template level is requested");
        assert_eq!(
            chat.chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.enable_thinking),
            Some(false),
            "`none` must reach the template as enable_thinking=false"
        );
        assert_eq!(meta.reasoning["effort"], "none", "echo what was asked");

        // Anything else still names the parameter rather than being silently clamped.
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hello","reasoning":{"effort":"turbo"}
        }))
        .unwrap();
        let error = request
            .into_chat(None, None, None, false)
            .err()
            .unwrap();
        assert_eq!(error.param, "reasoning.effort");
        assert!(
            error.message.contains("none, minimal, low, medium, high, xhigh or max"),
            "the error must list the accepted values, got {:?}",
            error.message
        );
    }

    #[test]
    fn user_limit_counts_characters_not_utf8_bytes() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hi","user":"本地用户"
        }))
        .unwrap();
        assert!(request.into_chat(None, None, None, false).is_ok());
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hi","user":"x".repeat(65)
        }))
        .unwrap();
        assert_eq!(
            request
                .into_chat(None, None, None, false)
                .err()
                .unwrap()
                .param,
            "user"
        );
    }

    #[tokio::test]
    async fn supported_image_is_passed_to_current_vision_path() {
        let response = post(json!({"model":"fixture","input":[{"role":"user","content":[
            {"type":"input_text","text":"describe"},
            {"type":"input_image","image_url":"data:image/png;base64,AA=="}
        ]}]}))
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(
            body["output"][0]["content"][0]["text"],
            "image:data:image/png;base64,AA=="
        );
    }

    #[tokio::test]
    async fn unsupported_semantics_are_rejected_before_generation() {
        for (field, value) in [
            ("previous_response_id", json!("resp_old")),
            ("conversation", json!("conv_old")),
            ("store", json!(true)),
            ("truncation", json!("auto")),
            ("background", json!(true)),
            ("include", json!(["message.output_text.logprobs"])),
            ("service_tier", json!("priority")),
            ("text", json!({"verbosity":"low"})),
        ] {
            let mut input = json!({"model":"fixture","input":"hi"});
            input[field] = value;
            let response = post(input).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
        }
        for part in [
            json!({"type":"input_file","file_data":"ignored"}),
            json!({"type":"input_image","file_id":"file_1"}),
            json!({"type":"input_image","image_url":"https://example.com/image.png"}),
        ] {
            let response =
                post(json!({"model":"fixture","input":[{"role":"user","content":[part]}]})).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    /// Responses `text.format` used to be a 400 for every non-text value. It now reaches the SAME
    /// parser/grammar as chat's `response_format`; the flat request shape is echoed back verbatim.
    #[tokio::test]
    async fn text_format_json_schema_and_json_object_take_effect() {
        let format = json!({"type":"json_schema","name":"answer","strict":true,
            "schema":{"type":"object","properties":{"answer":{"type":"string"}},
                      "required":["answer"],"additionalProperties":false}});
        let response = post(json!({"model":"fixture","input":"hi","text":{"format":format}})).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(
            body["output"][0]["content"][0]["text"],
            r#"{"answer":"hello"}"#
        );
        assert_eq!(body["text"]["format"], format);

        let response = post(
            json!({"model":"fixture","input":"hi","text":{"format":{"type":"json_object"}}}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        let text = body["output"][0]["content"][0]["text"].as_str().unwrap();
        assert!(
            serde_json::from_str::<Value>(text).unwrap().is_object(),
            "{text}"
        );
    }

    #[tokio::test]
    async fn unsupported_text_format_is_rejected_with_text_format_param() {
        for format in [
            json!({"type":"json_schema","name":"x","schema":{"$ref":"#/$defs/x"}}),
            json!({"type":"json_schema","name":"x"}),
            json!({"type":"json_schema","schema":{"type":"object"}}),
            json!({"type":"json_schema","name":"x","schema":{"type":"object"},"extra":1}),
            json!({"type":"nope"}),
        ] {
            let response =
                post(json!({"model":"fixture","input":"hi","text":{"format":format}})).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{format}");
            let body = json_body(response).await;
            assert_eq!(body["error"]["param"], "text.format", "{format}");
        }
        let response = post(json!({"model":"fixture","input":"hi","text":{"format":
            {"type":"json_schema","name":"x","schema":{"$ref":"#/$defs/x"}}}}))
        .await;
        let body = json_body(response).await;
        assert!(
            body["error"]["message"].as_str().unwrap().contains("$ref"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn structured_text_format_with_tools_is_a_400_naming_text_format() {
        let response = post(json!({"model":"fixture","input":"hi",
            "text":{"format":{"type":"json_object"}},
            "tools":[{"type":"function","name":"shell","parameters":{"type":"object"}}]}))
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "text.format");
    }

    #[tokio::test]
    async fn structured_stream_completes_with_the_json_text() {
        let format = json!({"type":"json_schema","name":"answer",
            "schema":{"type":"object","properties":{"answer":{"type":"string"}},
                      "required":["answer"]}});
        let response =
            post(json!({"model":"fixture","input":"hi","stream":true,"text":{"format":format}}))
                .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let wire = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(wire.contains(r#""type":"response.completed""#), "{wire}");
        assert!(wire.contains(r#"{\"answer\":\"hello\"}"#), "{wire}");
    }

    #[test]
    fn stateless_replay_and_reasoning_effort_reach_chat_path() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture",
            "reasoning":{"effort":"high","summary":"auto"},
            "input":[
                {"type":"message","role":"user","content":[{"type":"input_text","text":"run"}]},
                {"type":"reasoning","summary":[{"type":"summary_text","text":"plan"}]},
                {"type":"function_call","call_id":"call_1","name":"shell","arguments":"{\"cmd\":\"dir\"}"},
                {"type":"function_call_output","call_id":"call_1","output":"ok"}
            ],
            "tools":[{"type":"function","name":"shell","parameters":{"type":"object"}}]
        })).unwrap();
        let Prepared { chat, meta, .. } = request.into_chat(None, None, None, false).unwrap();
        assert_eq!(chat.reasoning_effort, Some(json!("high")));
        assert_eq!(chat.messages[1].reasoning_content.as_deref(), Some("plan"));
        assert_eq!(
            chat.messages[2].tool_calls.as_ref().unwrap()[0]["id"],
            "call_1"
        );
        assert_eq!(chat.messages[3].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(meta.reasoning["effort"], "high");
    }

    #[tokio::test]
    async fn stream_emits_typed_text_and_function_call_lifecycles() {
        for tools in [
            Value::Null,
            json!([{"type":"function","name":"shell","parameters":{"type":"object"}}]),
        ] {
            let response =
                post(json!({"model":"fixture","input":"hi","stream":true,"tools":tools})).await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let wire = String::from_utf8(bytes.to_vec()).unwrap();
            let events: Vec<Value> = wire
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|data| serde_json::from_str(data).ok())
                .collect();
            let types: Vec<&str> = events
                .iter()
                .filter_map(|event| event["type"].as_str())
                .collect();
            assert_eq!(types.first(), Some(&"response.created"), "{wire}");
            assert_eq!(types.last(), Some(&"response.completed"), "{wire}");
            assert!(events
                .windows(2)
                .all(|pair| pair[1]["sequence_number"].as_u64().unwrap()
                    == pair[0]["sequence_number"].as_u64().unwrap() + 1));
            assert_eq!(
                events.last().unwrap()["response"]["usage"]["input_tokens_details"]
                    ["cached_tokens"],
                12
            );
            if tools.is_null() {
                assert!(types.contains(&"response.output_text.delta"), "{wire}");
                assert!(
                    types.contains(&"response.reasoning_summary_text.delta"),
                    "{wire}"
                );
            } else {
                assert!(
                    types.contains(&"response.function_call_arguments.delta"),
                    "{wire}"
                );
                assert_eq!(
                    events.last().unwrap()["response"]["output"][0]["name"],
                    "shell"
                );
            }
            assert!(!wire.contains("data: [DONE]"));
        }
    }

    #[tokio::test]
    async fn generation_limit_maps_to_incomplete_in_both_modes() {
        let router = build_router(AppState::new(
            Arc::new(Limited),
            "fixture",
            1,
            Arc::new(Config::default()),
        ));
        for streaming in [false, true] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/responses")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            json!({"model":"fixture","input":"hi","stream":streaming}).to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            if streaming {
                let bytes = response.into_body().collect().await.unwrap().to_bytes();
                let wire = String::from_utf8(bytes.to_vec()).unwrap();
                assert!(wire.contains("event: response.incomplete"), "{wire}");
                assert!(wire.contains("max_output_tokens"), "{wire}");
            } else {
                let body = json_body(response).await;
                assert_eq!(body["status"], "incomplete");
                assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
            }
        }
    }

    #[tokio::test]
    async fn stream_failure_uses_typed_terminal_event_and_uncached_headers() {
        let request: ResponsesRequest =
            serde_json::from_value(json!({"model":"fixture","input":"hi"})).unwrap();
        let meta = request.into_chat(None, None, None, false).unwrap().meta;
        let upstream = Response::new(Body::from(concat!(
            "data: {\"id\":\"chatcmpl-test\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
            "data: {\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}\n\n"
        )));
        let response = stream_response(upstream, meta, None);
        assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
        assert_eq!(response.headers()["x-accel-buffering"], "no");
        let wire = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let events: Vec<Value> = wire
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).unwrap())
            .collect();
        assert_eq!(events[0]["type"], "response.created");
        assert_eq!(events.last().unwrap()["type"], "response.failed");
        assert!(!events.iter().any(|event| {
            event["type"] == "response.output_item.done"
                || event["type"] == "response.content_part.done"
                || event["type"] == "response.output_text.done"
        }));
        assert_eq!(events.last().unwrap()["response"]["status"], "failed");
        assert_eq!(
            events.last().unwrap()["response"]["output"][0]["status"],
            "in_progress"
        );
        assert_eq!(
            events.last().unwrap()["response"]["output"][1]["status"],
            "in_progress"
        );
        assert_eq!(
            events.last().unwrap()["response"]["error"]["message"],
            "boom"
        );
    }

    #[tokio::test]
    async fn stream_error_before_creation_has_flat_error_shape() {
        let request: ResponsesRequest =
            serde_json::from_value(json!({"model":"fixture","input":"hi"})).unwrap();
        let meta = request.into_chat(None, None, None, false).unwrap().meta;
        let upstream = Response::new(Body::from(
            "data: {\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}\n\n",
        ));
        let response = stream_response(upstream, meta, None);
        let wire = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let event: Value = serde_json::from_str(
            wire.lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(event["type"], "error");
        assert_eq!(event["message"], "boom");
        assert!(event.get("error").is_none());
    }

    #[tokio::test]
    async fn oversized_non_streaming_translation_is_a_clear_gateway_error() {
        let request: ResponsesRequest =
            serde_json::from_value(json!({"model":"fixture","input":"hi"})).unwrap();
        let meta = request.into_chat(None, None, None, false).unwrap().meta;
        let upstream = Response::new(Body::from(vec![b' '; MAX_NON_STREAM_TRANSLATION_BYTES + 1]));
        let response = non_stream_response(upstream, meta, None).await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = json_body(response).await;
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("64 MiB"), "{message}");
        // The way out has to be in the message: this is a buffering limit, not a generation one.
        assert!(message.contains("stream=true"), "{message}");
    }

    /// The `data:` payloads of a translated stream, in order.
    async fn sse_events(response: Response) -> Vec<Value> {
        let wire = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        wire.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    fn upstream_stream(frames: &[&str]) -> Response {
        Response::new(Body::from(
            frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>(),
        ))
    }

    #[tokio::test]
    async fn fragmented_tool_call_stream_becomes_one_item_with_argument_deltas() {
        // The OpenAI chat-completions shape: fragments sharing an `index`, only the first carrying
        // `id`/`name`. One call, three argument fragments, one completed item.
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hi",
            "tools":[{"type":"function","name":"shell","parameters":{"type":"object"}}]
        }))
        .unwrap();
        let meta = request.into_chat(None, None, None, false).unwrap().meta;
        let response = stream_response(
            upstream_stream(&[
                r#"{"id":"chatcmpl-1","created":1,"model":"fixture","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":""}}]}}]}"#,
                r#"{"id":"chatcmpl-1","created":1,"model":"fixture","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":\""}}]}}]}"#,
                r#"{"id":"chatcmpl-1","created":1,"model":"fixture","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"dir\"}"}}]}}]}"#,
                r#"{"id":"chatcmpl-1","created":1,"model":"fixture","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}"#,
            ]),
            meta,
            None,
        );
        let events = sse_events(response).await;
        let added: Vec<&Value> = events
            .iter()
            .filter(|event| event["type"] == "response.output_item.added")
            .collect();
        assert_eq!(added.len(), 1, "one call, one item announcement");
        assert_eq!(added[0]["item"]["type"], "function_call");
        assert_eq!(added[0]["item"]["name"], "shell");
        assert_eq!(added[0]["item"]["call_id"], "call_1");
        let deltas: Vec<&Value> = events
            .iter()
            .filter(|event| event["type"] == "response.function_call_arguments.delta")
            .collect();
        assert_eq!(deltas.len(), 2, "empty first fragment emits nothing");
        assert_eq!(
            deltas
                .iter()
                .map(|event| event["delta"].as_str().unwrap())
                .collect::<String>(),
            r#"{"cmd":"dir"}"#
        );
        let done = events
            .iter()
            .find(|event| event["type"] == "response.output_item.done")
            .unwrap();
        assert_eq!(done["item"]["arguments"], r#"{"cmd":"dir"}"#);
        let completed = events.last().unwrap();
        assert_eq!(completed["type"], "response.completed");
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["arguments"], r#"{"cmd":"dir"}"#);
        assert_eq!(output[0]["id"], added[0]["item"]["id"]);
    }

    #[tokio::test]
    async fn parallel_tool_calls_false_returns_one_call_and_says_what_was_dropped() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hi","parallel_tool_calls":false,
            "tools":[{"type":"function","name":"shell","parameters":{"type":"object"}}]
        }))
        .unwrap();
        let meta = request.into_chat(None, None, None, false).unwrap().meta;
        assert!(!meta.parallel_tool_calls);
        let response = stream_response(
            upstream_stream(&[
                r#"{"id":"chatcmpl-1","created":1,"model":"fixture","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{}"}}]}}]}"#,
                r#"{"id":"chatcmpl-1","created":1,"model":"fixture","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"shell","arguments":"{}"}}]}}]}"#,
                r#"{"id":"chatcmpl-1","created":1,"model":"fixture","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"cmd\":\"late\"}"}}]}}]}"#,
                r#"{"id":"chatcmpl-1","created":1,"model":"fixture","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}"#,
            ]),
            meta,
            None,
        );
        let events = sse_events(response).await;
        let items: Vec<&Value> = events
            .iter()
            .filter(|event| event["type"] == "response.output_item.done")
            .collect();
        assert_eq!(items.len(), 1, "the second call never opens an item");
        assert_eq!(items[0]["item"]["call_id"], "call_1");
        assert!(!events.iter().any(|event| {
            event["type"] == "response.function_call_arguments.delta"
                && event["delta"] == r#"{"cmd":"late"#
        }));
        let completed = events.last().unwrap();
        assert_eq!(completed["response"]["parallel_tool_calls"], false);
        assert!(completed["response"]["warnings"][0]
            .as_str()
            .unwrap()
            .contains("parallel_tool_calls=false"));

        // Non-streaming applies the same shaping, so the two paths cannot disagree.
        let chat = json!({"id":"chatcmpl-2","created":2,"model":"fixture","choices":[{"index":0,
            "message":{"role":"assistant","tool_calls":[
                {"index":0,"id":"call_a","type":"function","function":{"name":"shell","arguments":"{}"}},
                {"index":1,"id":"call_b","type":"function","function":{"name":"shell","arguments":"{}"}}
            ]},"finish_reason":"tool_calls"}],
            "usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}});
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hi","parallel_tool_calls":false
        }))
        .unwrap();
        let meta = request.into_chat(None, None, None, false).unwrap().meta;
        let body = json_body(
            non_stream_response(Response::new(Body::from(chat.to_string())), meta, None).await,
        )
        .await;
        let output = body["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["call_id"], "call_a");
        assert!(body["warnings"][0]
            .as_str()
            .unwrap()
            .contains("parallel_tool_calls=false"));
    }

    // -- response store: chained replay ----------------------------------------------------

    /// Records the message count each generation actually receives — the replay proof.
    struct Recorder {
        lengths: Arc<Mutex<Vec<usize>>>,
    }

    impl ChatGenerator for Recorder {
        fn chat(
            &self,
            messages: &[ChatMessage],
            _tools: Option<&Value>,
            _tool_choice: Option<&str>,
            _params: &GenParams,
            _cancel: &Arc<AtomicBool>,
            _progress: Option<infr_core::GenerationProgressCallback>,
            on_delta: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            self.lengths.lock().unwrap().push(messages.len());
            on_delta(Delta::Content("ok".into()));
            Ok(ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 10,
                cached_prompt_tokens: 0,
                completion_tokens: 1,
            })
        }
    }

    fn recorder() -> (Arc<Recorder>, Arc<Mutex<Vec<usize>>>) {
        let lengths = Arc::new(Mutex::new(Vec::new()));
        (
            Arc::new(Recorder {
                lengths: lengths.clone(),
            }),
            lengths,
        )
    }

    fn router_entries(generator: Arc<dyn ChatGenerator>, entries: u32) -> Router {
        let mut cfg = Config::default();
        cfg.serve.response_store_entries = entries;
        build_router(AppState::new(generator, "fixture", 1, Arc::new(cfg)))
    }

    async fn post_to(router: &Router, input: Value) -> Response {
        router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(input.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    fn first_id(body: &Value) -> String {
        body["id"].as_str().unwrap().to_owned()
    }

    #[tokio::test]
    async fn chained_response_replays_history_and_grows_the_prefix() {
        let (rec, lengths) = recorder();
        let router = router_entries(rec, 8);
        let first =
            json_body(post_to(&router, json!({"model":"fixture","input":"first"})).await).await;
        assert_eq!(first["status"], "completed");
        assert_eq!(first["store"], true);
        let second = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"second","previous_response_id":first_id(&first)}),
            )
            .await,
        )
        .await;
        assert_eq!(second["previous_response_id"], first["id"]);
        // R1 sent [user]; R2 replays [user, assistant] and appends [user] — the rebuilt prompt
        // carries R1's exact token prefix, which is what the paged radix tree hits.
        assert_eq!(*lengths.lock().unwrap(), vec![1, 3]);
    }

    /// Answers with the reasoning every prompt message carries, so a test can prove what the model
    /// saw instead of inferring it from message counts.
    struct ReasoningEcho;

    impl ChatGenerator for ReasoningEcho {
        fn chat(
            &self,
            messages: &[ChatMessage],
            _tools: Option<&Value>,
            _tool_choice: Option<&str>,
            _params: &GenParams,
            _cancel: &Arc<AtomicBool>,
            _progress: Option<infr_core::GenerationProgressCallback>,
            on_delta: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            let seen: Vec<&str> = messages
                .iter()
                .filter_map(|m| m.reasoning_content.as_deref())
                .collect();
            on_delta(Delta::Reasoning("think".into()));
            on_delta(Delta::Content(seen.join("|")));
            Ok(ChatOutcome {
                finish: Finish::Stop,
                prompt_tokens: 1,
                cached_prompt_tokens: 0,
                completion_tokens: 1,
            })
        }
    }

    #[tokio::test]
    async fn encrypted_reasoning_handle_round_trips_into_the_next_prompt() {
        let router = router_entries(Arc::new(ReasoningEcho), 8);
        let first = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"first","include":["reasoning.encrypted_content"]}),
            )
            .await,
        )
        .await;
        assert_eq!(first["output"][0]["type"], "reasoning");
        let handle = first["output"][0]["encrypted_content"]
            .as_str()
            .expect("include asked for a handle")
            .to_owned();
        assert!(!handle.is_empty());

        // The handle replays the reasoning even though the client no longer sends its text.
        let second = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":[
                    {"type":"reasoning","encrypted_content":handle},
                    {"role":"user","content":"second"}
                ]}),
            )
            .await,
        )
        .await;
        assert_eq!(second["status"], "completed");
        let replayed = second["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "message")
            .expect("the answer item")
            .clone();
        assert_eq!(replayed["content"][0]["text"], "think");

        // Without the include, the same turn carries no handle at all.
        let plain =
            json_body(post_to(&router, json!({"model":"fixture","input":"third"})).await).await;
        assert!(plain["output"][0].get("encrypted_content").is_none());
    }

    #[tokio::test]
    async fn streaming_reasoning_item_carries_the_handle_when_asked() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hi","include":["reasoning.encrypted_content"],"store":true
        }))
        .unwrap();
        let meta = request.into_chat(None, None, None, true).unwrap().meta;
        let response = stream_response(
            upstream_stream(&[
                r#"{"id":"chatcmpl-9","created":1,"model":"fixture","choices":[{"index":0,"delta":{"reasoning_content":"plan"},"finish_reason":null}]}"#,
                r#"{"id":"chatcmpl-9","created":1,"model":"fixture","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#,
            ]),
            meta,
            None,
        );
        let events = sse_events(response).await;
        let handles: Vec<&str> = events
            .iter()
            .filter(|event| event["item"]["type"] == "reasoning")
            .filter(|event| {
                matches!(
                    event["type"].as_str(),
                    Some("response.output_item.added" | "response.output_item.done")
                )
            })
            .map(|event| event["item"]["encrypted_content"].as_str().unwrap())
            .collect();
        assert_eq!(handles, vec!["rehs_resp_9", "rehs_resp_9"]);
        let completed = events.last().unwrap();
        assert_eq!(
            completed["response"]["output"][0]["encrypted_content"],
            "rehs_resp_9"
        );
    }

    /// `include:["reasoning.encrypted_content"]` on a turn the server cannot store is a CAPABILITY
    /// request, not a contradiction: OpenAI's own servers omit the field when there is nothing to
    /// encrypt, replay only needs the reasoning item's readable summary, and pi-ai's Responses
    /// adapter asks for the include on every reasoning request (so 400ing here locks a client out
    /// of every thinking level — measured with DeepSeek Harness).
    #[tokio::test]
    async fn include_without_response_state_degrades_with_a_warning() {
        // Store feature off entirely: the request succeeds and says why no handle came back.
        let response = post(json!({
            "model":"fixture","input":"hi","include":["reasoning.encrypted_content"]
        }))
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert!(
            body["warnings"][0]
                .as_str()
                .unwrap()
                .contains("response_store_entries"),
            "the warning names the switch that would issue a handle: {body}"
        );
        assert!(
            body["output"]
                .as_array()
                .unwrap()
                .iter()
                .all(|item| item["type"] != "reasoning" || item["encrypted_content"].is_null()),
            "a stateless turn must not mint a handle it cannot resolve"
        );

        // Store available but this turn declines it (`store:false`): same degradation, not a 400.
        let (rec, _) = recorder();
        let router = router_entries(rec, 8);
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"hi","store":false,
                   "include":["reasoning.encrypted_content"]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert!(body["warnings"][0]
            .as_str()
            .unwrap()
            .contains("response_store_entries"));

        // An include this server genuinely cannot honour still names `include`.
        let response = post(json!({
            "model":"fixture","input":"hi","include":["message.output_text.logprobs"]
        }))
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(response).await["error"]["param"], "include");
    }

    #[tokio::test]
    async fn an_unresolvable_reasoning_handle_is_a_loud_400() {
        let (rec, _) = recorder();
        let router = router_entries(rec, 8);
        for handle in ["rehs_resp_missing", "not-a-handle"] {
            let response = post_to(
                &router,
                json!({"model":"fixture","input":[
                    {"type":"reasoning","encrypted_content":handle},
                    {"role":"user","content":"next"}
                ]}),
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{handle}");
            assert_eq!(json_body(response).await["error"]["param"], "input");
        }
    }

    fn temp_store_path(name: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("infr-responses-{name}-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn store_path_round_trips_history_items_and_cursors() {
        let path = temp_store_path("roundtrip");
        {
            let mut store = ResponseStore::open(4, Some(&path)).unwrap();
            let history = Arc::new(vec![message("user", Some(json!("hello")))]);
            store.put(
                "resp_1",
                json!({"id":"resp_1","output":[{"id":"msg_1","type":"message","status":"completed"}]}),
                history,
                vec![json!({"role":"user","content":"hello"})],
                Some("conv"),
            );
            assert!(path.exists(), "a mutation is mirrored immediately");
        }
        // A fresh store over the same file stands in for a restarted process.
        let mut reopened = ResponseStore::open(4, Some(&path)).unwrap();
        let chained = reopened.chain(Some("resp_1"), None).unwrap().unwrap();
        assert_eq!(chained.len(), 1);
        assert_eq!(chained[0].content, Some(json!("hello")));
        let via_cursor = reopened.chain(None, Some("conv")).unwrap().unwrap();
        assert_eq!(via_cursor.len(), 1);
        assert_eq!(reopened.get("resp_1").unwrap()["id"], "resp_1");
        let item = reopened.resolve_item("msg_1").expect("item index rebuilt");
        assert_eq!(item["id"], "msg_1");
        assert_eq!(reopened.prompt_items("resp_1").unwrap().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn store_path_keeps_the_lru_window_and_drops_evicted_turns() {
        let path = temp_store_path("lru");
        {
            let mut store = ResponseStore::open(2, Some(&path)).unwrap();
            for id in ["resp_a", "resp_b", "resp_c"] {
                store.put(
                    id,
                    json!({"id":id,"output":[]}),
                    Arc::new(vec![message("user", Some(json!(id)))]),
                    Vec::new(),
                    None,
                );
            }
        }
        let mut reopened = ResponseStore::open(2, Some(&path)).unwrap();
        assert!(reopened.get("resp_a").is_none(), "oldest turn was evicted");
        assert!(reopened.get("resp_c").is_some());
        assert_eq!(reopened.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_corrupt_store_file_refuses_to_open() {
        let path = temp_store_path("corrupt");
        std::fs::write(&path, b"{not a store").unwrap();
        let error = ResponseStore::open(4, Some(&path))
            .err()
            .expect("a corrupt store must not open");
        assert!(error.contains("move it aside"), "{error}");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn responses_chain_survives_a_restart_when_a_path_is_configured() {
        let path = temp_store_path("restart");
        let cfg = || {
            let mut cfg = Config::default();
            cfg.serve.response_store_entries = 8;
            cfg.serve.response_store_path = Some(path.clone());
            Arc::new(cfg)
        };
        let first = {
            let router = build_router(AppState::new(Arc::new(ReasoningEcho), "fixture", 1, cfg()));
            json_body(post_to(&router, json!({"model":"fixture","input":"first"})).await).await
        };
        // A second router over the same file: nothing but the mirror is shared with the first.
        let router = build_router(AppState::new(Arc::new(ReasoningEcho), "fixture", 1, cfg()));
        let second = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"second","previous_response_id":first_id(&first)}),
            )
            .await,
        )
        .await;
        assert_eq!(second["previous_response_id"], first["id"]);
        assert_eq!(second["status"], "completed");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn unknown_previous_response_id_is_a_loud_400() {
        let (rec, _) = recorder();
        let router = router_entries(rec, 8);
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"hi","previous_response_id":"resp_missing"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "previous_response_id");
    }

    #[tokio::test]
    async fn conversation_cursor_chains_without_an_explicit_id() {
        let (rec, lengths) = recorder();
        let router = router_entries(rec, 8);
        post_to(
            &router,
            json!({"model":"fixture","input":"first","conversation":"chat-1"}),
        )
        .await;
        let body = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"second","conversation":"chat-1"}),
            )
            .await,
        )
        .await;
        assert_eq!(body["status"], "completed");
        assert_eq!(*lengths.lock().unwrap(), vec![1, 3]);
        // The `{"id": ...}` object form resolves the same cursor.
        json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"third","conversation":{"id":"chat-1"}}),
            )
            .await,
        )
        .await;
        assert_eq!(*lengths.lock().unwrap(), vec![1, 3, 5]);
    }

    #[tokio::test]
    async fn store_false_response_cannot_be_replayed() {
        let (rec, _) = recorder();
        let router = router_entries(rec, 8);
        let first = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"hi","store":false}),
            )
            .await,
        )
        .await;
        assert_eq!(first["store"], false);
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"again","previous_response_id":first_id(&first)}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn lru_eviction_breaks_old_chains_explicitly() {
        let (rec, _) = recorder();
        let router = router_entries(rec, 1);
        let first = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"one","conversation":"c"}),
            )
            .await,
        )
        .await;
        let first_id = first_id(&first);
        let second = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"two","previous_response_id":first_id.clone()}),
            )
            .await,
        )
        .await;
        assert_eq!(second["status"], "completed");
        // capacity 1: storing the second response evicted the first.
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"three","previous_response_id":first_id}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        // The cursor still points at the evicted id — fail, do not silently restart.
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"three","conversation":"c"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "conversation");
    }

    #[tokio::test]
    async fn streaming_chain_stores_history() {
        let (rec, lengths) = recorder();
        let router = router_entries(rec, 8);
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"first","stream":true}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let wire = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let id = wire
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .and_then(|data| serde_json::from_str::<Value>(data).ok())
            .and_then(|event| event["response"]["id"].as_str().map(str::to_owned))
            .unwrap();
        let body = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"second","previous_response_id":id}),
            )
            .await,
        )
        .await;
        assert_eq!(body["status"], "completed");
        assert_eq!(*lengths.lock().unwrap(), vec![1, 3]);
    }

    #[tokio::test]
    async fn identical_instructions_replay_a_chain_and_edited_ones_do_not() {
        let (rec, lengths) = recorder();
        let router = router_entries(rec, 8);
        let first = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"one","instructions":"be brief"}),
            )
            .await,
        )
        .await;
        // The SDK shape: the same instructions on every turn. Accepted, and the prompt the engine
        // sees still carries ONE system turn — the prefix hit the chain exists to get survives.
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"two","previous_response_id":first_id(&first),
                   "instructions":"be brief"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(json_body(response).await["status"], "completed");
        assert_eq!(*lengths.lock().unwrap(), vec![2, 4]);

        let response = post_to(
            &router,
            json!({"model":"fixture","input":"three","previous_response_id":first_id(&first),
                   "instructions":"be verbose"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["param"], "instructions");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("do not match the system prefix"));
    }

    // -- stored-response REST surface -------------------------------------------------------

    async fn ask(router: &Router, method: &str, uri: &str) -> Response {
        router
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    fn stored_id(body: &Value) -> String {
        body["id"].as_str().unwrap().to_owned()
    }

    #[tokio::test]
    async fn retrieve_returns_the_envelope_that_was_served() {
        let (rec, _) = recorder();
        let router = router_entries(rec, 8);
        let first =
            json_body(post_to(&router, json!({"model":"fixture","input":"one"})).await).await;
        let retrieved = ask(
            &router,
            "GET",
            &format!("/v1/responses/{}", stored_id(&first)),
        )
        .await;
        assert_eq!(retrieved.status(), StatusCode::OK);
        // Byte-for-byte the same object the client got at generation time: retrieve must not
        // rebuild history and drift from what was actually served.
        assert_eq!(json_body(retrieved).await, first);

        let missing = ask(&router, "GET", "/v1/responses/resp_nobody").await;
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        let body = json_body(missing).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        // SDKs read `error.code` even on a 404; it is present and null, never absent.
        assert!(body["error"]["code"].is_null());
        assert_eq!(body["error"]["param"], Value::Null);
    }

    #[tokio::test]
    async fn retrieve_returns_the_streamed_turn_too() {
        let (rec, _) = recorder();
        let router = router_entries(rec, 8);
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"one","stream":true}),
        )
        .await;
        let wire = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let completed = wire
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        let id = completed["response"]["id"].as_str().unwrap().to_owned();
        let retrieved = json_body(ask(&router, "GET", &format!("/v1/responses/{id}")).await).await;
        assert_eq!(retrieved["status"], "completed");
        assert_eq!(retrieved["output"][0]["content"][0]["text"], "ok");
        assert_eq!(retrieved["usage"]["input_tokens"], 10);
    }

    #[tokio::test]
    async fn list_pages_stored_responses_and_its_window_is_validated() {
        let (rec, _) = recorder();
        let router = router_entries(rec, 8);
        let first =
            json_body(post_to(&router, json!({"model":"fixture","input":"one"})).await).await;
        let second = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"two","previous_response_id":stored_id(&first)}),
            )
            .await,
        )
        .await;

        let body = json_body(ask(&router, "GET", "/v1/responses").await).await;
        assert_eq!(body["object"], "list");
        // Newest first, like OpenAI's default order.
        assert_eq!(body["data"][0]["id"], second["id"]);
        assert_eq!(body["data"][1]["id"], first["id"]);
        assert_eq!(body["has_more"], false);

        let body = json_body(ask(&router, "GET", "/v1/responses?limit=1").await).await;
        assert_eq!(body["data"].as_array().unwrap().len(), 1);
        assert_eq!(body["has_more"], true, "one page cannot hold both");
        assert_eq!(body["first_id"], second["id"]);
        assert_eq!(body["last_id"], second["id"]);

        let body = json_body(ask(&router, "GET", "/v1/responses?order=asc").await).await;
        assert_eq!(body["data"][0]["id"], first["id"]);

        // A non-number and a negative are rejected here rather than by the extractor: `limit` is
        // read as a string, so `?limit=abc` answers with the JSON error body an SDK can parse.
        for query in [
            "limit=0",
            "limit=101",
            "limit=-1",
            "limit=abc",
            "order=sideways",
        ] {
            let response = ask(&router, "GET", &format!("/v1/responses?{query}")).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
            let body = json_body(response).await;
            assert_eq!(
                body["error"]["param"],
                if query.starts_with("limit") {
                    "limit"
                } else {
                    "order"
                },
                "{query}"
            );
        }
    }

    #[tokio::test]
    async fn delete_forgets_the_response_and_breaks_its_chain_loudly() {
        let (rec, lengths) = recorder();
        let router = router_entries(rec, 8);
        let first =
            json_body(post_to(&router, json!({"model":"fixture","input":"one"})).await).await;
        let id = stored_id(&first);
        let deleted = json_body(ask(&router, "DELETE", &format!("/v1/responses/{id}")).await).await;
        assert_eq!(deleted["id"], first["id"]);
        assert_eq!(deleted["object"], "response");
        assert_eq!(deleted["deleted"], true);
        assert_eq!(
            ask(&router, "GET", &format!("/v1/responses/{id}"))
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
        // The chain does NOT fall back to a blank prefix: it fails where the deleted response was.
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"two","previous_response_id":id}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            json_body(response).await["error"]["param"],
            "previous_response_id"
        );
        assert_eq!(
            *lengths.lock().unwrap(),
            vec![1],
            "the deleted chain never generated"
        );
    }

    #[tokio::test]
    async fn input_items_reports_the_prompt_the_model_received() {
        let (rec, lengths) = recorder();
        let router = router_entries(rec, 8);
        let first = json_body(
            post_to(
                &router,
                json!({"model":"fixture","instructions":"be brief","input":"one"}),
            )
            .await,
        )
        .await;
        let body = json_body(
            ask(
                &router,
                "GET",
                &format!("/v1/responses/{}/input_items?order=asc", stored_id(&first)),
            )
            .await,
        )
        .await;
        // Instructions are folded into one system prefix, so the items are what the template saw.
        assert_eq!(body["data"][0]["role"], "system");
        assert_eq!(body["data"][0]["content"][0]["text"], "be brief");
        assert_eq!(body["data"][1]["role"], "user");
        assert_eq!(body["data"][1]["content"][0]["text"], "one");
        assert_eq!(body["has_more"], false);

        let second = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":"two","previous_response_id":stored_id(&first)}),
            )
            .await,
        )
        .await;
        let body = json_body(
            ask(
                &router,
                "GET",
                &format!("/v1/responses/{}/input_items?order=asc", stored_id(&second)),
            )
            .await,
        )
        .await;
        assert_eq!(
            body["data"].as_array().unwrap().len(),
            4,
            "replayed prefix + this turn's input"
        );
        assert_eq!(body["data"][2]["role"], "assistant");
        assert_eq!(body["data"][2]["content"][0]["text"], "ok");
        assert_eq!(body["data"][3]["role"], "user");
        assert_eq!(body["data"][3]["content"][0]["text"], "two");
        assert_eq!(*lengths.lock().unwrap(), vec![2, 4]);
    }

    #[tokio::test]
    async fn item_reference_replays_a_stored_output_item() {
        let (rec, lengths) = recorder();
        let router = router_entries(rec, 8);
        let first =
            json_body(post_to(&router, json!({"model":"fixture","input":"one"})).await).await;
        let item_id = first["output"][0]["id"].as_str().unwrap().to_owned();
        let second = json_body(
            post_to(
                &router,
                json!({"model":"fixture","input":[
                    {"type":"item_reference","id":item_id},
                    {"type":"message","role":"user","content":"and now?"}
                ]}),
            )
            .await,
        )
        .await;
        assert_eq!(second["status"], "completed");
        // The reference became the assistant message it names: [hello, and now?] reach the model.
        assert_eq!(*lengths.lock().unwrap(), vec![1, 2]);

        let response = post_to(
            &router,
            json!({"model":"fixture","input":[{"type":"item_reference","id":"msg_missing"}]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(response).await["error"]["param"], "input");
    }

    #[tokio::test]
    async fn cancel_has_nothing_to_cancel_but_only_for_a_stored_id() {
        let (rec, _) = recorder();
        let router = router_entries(rec, 8);
        let first =
            json_body(post_to(&router, json!({"model":"fixture","input":"one"})).await).await;
        let response = ask(
            &router,
            "POST",
            &format!("/v1/responses/{}/cancel", stored_id(&first)),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            json_body(response).await["error"]["param"],
            "background",
            "generation is synchronous: the response is terminal before it is addressable"
        );
        assert_eq!(
            ask(&router, "POST", "/v1/responses/resp_absent/cancel")
                .await
                .status(),
            StatusCode::NOT_FOUND,
            "an unknown id is not the same answer as an unstorable one"
        );
    }

    #[tokio::test]
    async fn state_off_errors_name_the_field_the_client_sent() {
        // The `post` router has the store OFF, which is the config a chain request hits by mistake.
        for (field, body) in [
            (
                "previous_response_id",
                json!({"model":"fixture","input":"hi","previous_response_id":"resp_x"}),
            ),
            (
                "conversation",
                json!({"model":"fixture","input":"hi","conversation":"c"}),
            ),
            (
                "store",
                json!({"model":"fixture","input":"hi","store":true}),
            ),
            (
                "input",
                json!({"model":"fixture","input":[{"type":"item_reference","id":"msg_x"}]}),
            ),
        ] {
            let response = post(body).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{field}");
            let body = json_body(response).await;
            assert_eq!(body["error"]["param"], field, "{field}");
        }
        let router = build_router(AppState::new(
            Arc::new(Fixture),
            "fixture",
            1,
            Arc::new(Config::default()),
        ));
        for (uri, param) in [
            ("/v1/responses", Value::Null),
            ("/v1/responses/resp_x", json!("response_id")),
            ("/v1/responses/resp_x/input_items", json!("response_id")),
        ] {
            let response = ask(&router, "GET", uri).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
            assert_eq!(
                json_body(response).await["error"]["param"],
                param,
                "{uri} must not answer an empty page for a feature that is off"
            );
        }
    }

    #[tokio::test]
    async fn a_conversation_needs_a_stored_turn() {
        let (rec, lengths) = recorder();
        let router = router_entries(rec, 8);
        let response = post_to(
            &router,
            json!({"model":"fixture","input":"one","conversation":"c","store":false}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(
            body["error"]["param"], "conversation",
            "an unstored turn cannot advance the cursor the next turn reads"
        );
        assert_eq!(
            *lengths.lock().unwrap(),
            Vec::<usize>::new(),
            "never dispatched"
        );
    }

    #[tokio::test]
    async fn stored_state_reads_are_gated_by_the_api_key() {
        let mut cfg = Config::default();
        cfg.serve.response_store_entries = 4;
        cfg.serve.api_key = Some("s3cret".into());
        let (rec, _) = recorder();
        let router = build_router(AppState::new(rec, "fixture", 1, Arc::new(cfg)));
        for (method, uri) in [
            ("GET", "/v1/responses"),
            ("GET", "/v1/responses/resp_x"),
            ("GET", "/v1/responses/resp_x/input_items"),
            ("DELETE", "/v1/responses/resp_x"),
            ("POST", "/v1/responses/resp_x/cancel"),
        ] {
            let denied = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(denied.status(), StatusCode::UNAUTHORIZED, "{method} {uri}");
        }
        let authorized = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/responses")
                    .header("authorization", "Bearer s3cret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);
        assert_eq!(
            json_body(authorized).await["data"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn an_absurd_path_id_is_a_400_before_the_store_is_consulted() {
        let (rec, _) = recorder();
        let router = router_entries(rec, 8);
        let response = ask(
            &router,
            "GET",
            &format!("/v1/responses/resp_{}", "x".repeat(MAX_ID_LEN)),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(json_body(response).await["error"]["param"], "response_id");
    }
}
