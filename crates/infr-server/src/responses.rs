//! Stateless OpenAI Responses compatibility on top of the server's chat generation path.
//! Keep wire conversion here so slot admission, cancellation, progress and usage have one owner.

use super::*;
use axum::body::{to_bytes, Body, Bytes};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use futures_util::{stream, Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};

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
    fn into_chat(self) -> Result<(ChatRequest, Meta), ParamError> {
        if self.model.is_empty() {
            return Err(invalid("model", "model must not be empty"));
        }
        if self
            .previous_response_id
            .as_ref()
            .is_some_and(|v| !v.is_null())
        {
            return Err(invalid(
                "previous_response_id",
                "server-side response state is not supported",
            ));
        }
        if self.conversation.as_ref().is_some_and(|v| !v.is_null()) {
            return Err(invalid(
                "conversation",
                "server-side conversations are not supported",
            ));
        }
        if self.store == Some(true) {
            return Err(invalid("store", "store=true is not supported"));
        }
        if self.truncation.as_deref().is_some_and(|s| s != "disabled") {
            return Err(invalid(
                "truncation",
                "only truncation=disabled is supported",
            ));
        }
        if self.parallel_tool_calls == Some(false) {
            return Err(invalid(
                "parallel_tool_calls",
                "parallel_tool_calls=false is not supported",
            ));
        }
        if self
            .user
            .as_ref()
            .is_some_and(|user| user.chars().count() > 64)
        {
            return Err(invalid("user", "user must be at most 64 characters"));
        }
        if self.include.as_ref().is_some_and(|include| {
            include
                .iter()
                .any(|item| item != "reasoning.encrypted_content")
        }) {
            return Err(invalid(
                "include",
                "only reasoning.encrypted_content is accepted as a compatibility hint; this server cannot supply hosted-tool data, logprobs, or encrypted reasoning",
            ));
        }
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
        let text = match self.text {
            None | Some(Value::Null) => json!({"format":{"type":"text"},"verbosity":"medium"}),
            Some(v @ Value::Object(_))
                if v.get("format")
                    .is_none_or(|f| f.is_null() || f["type"] == "text")
                    && v.get("verbosity")
                        .is_none_or(|x| x.is_null() || x == "medium")
                    && v.as_object()
                        .is_some_and(|o| o.keys().all(|k| k == "format" || k == "verbosity"))
                    && v.get("format").is_none_or(|f| {
                        f.is_null() || f.as_object().is_some_and(|o| o.keys().all(|k| k == "type"))
                    }) =>
            {
                json!({"format":{"type":"text"},"verbosity":v.get("verbosity").and_then(Value::as_str).unwrap_or("medium")})
            }
            _ => {
                return Err(invalid(
                    "text",
                    "only plain text with default verbosity is supported",
                ))
            }
        };
        let mut reasoning_effort = None;
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
                if let Some(effort) = effective_effort.as_ref().filter(|x| !x.is_null()) {
                    reasoning_effort =
                        Some(serde_json::from_value(effort.clone()).map_err(|_| {
                            invalid("reasoning.effort", "unsupported reasoning effort")
                        })?);
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
        // Many GGUF templates accept only one initial system message. Preserve the
        // instruction prefix order, but do not silently move later instructions
        // across conversation turns.
        let mut system_parts = Vec::new();
        if let Some(instructions) = self.instructions.as_ref().filter(|s| !s.is_empty()) {
            system_parts.push(instructions.clone());
        }
        let prefix_len = messages
            .iter()
            .take_while(|m| matches!(m.role.as_str(), "system" | "developer"))
            .count();
        for m in &messages[..prefix_len] {
            system_parts.push(instruction_text(&m.content)?);
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
        if !system_parts.is_empty() {
            messages.insert(0, message("system", Some(json!(system_parts.join("\n\n")))));
        }
        let chat = ChatRequest {
            model: self.model,
            messages,
            stream: self.stream,
            reasoning_effort,
            chat_template_kwargs: None,
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
        };
        Ok((chat, meta))
    }
}

pub(super) async fn handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<ResponsesRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, &headers) {
        return denied;
    }
    let Json(request) = match body {
        Ok(value) => value,
        Err(e) => return param_error(None, e.body_text()),
    };
    let (chat, meta) = match request.into_chat() {
        Ok(value) => value,
        Err(e) => return param_error(Some(e.param), e.message),
    };
    let streaming = chat.stream;
    let response = dispatch_chat(state, chat, "/v1/responses").await;
    if response.status() != StatusCode::OK {
        return response;
    }
    if streaming {
        stream_response(response, meta)
    } else {
        non_stream_response(response, meta).await
    }
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
        "parallel_tool_calls":meta.parallel_tool_calls,"previous_response_id":null,
        "prompt_cache_key":meta.prompt_cache_key,"prompt_cache_retention":null,
        "reasoning":meta.reasoning,"service_tier":"default","store":false,
        "temperature":meta.temperature,
        "text":meta.text,"tool_choice":meta.tool_choice,"tools":meta.tools,
        "top_p":meta.top_p,"truncation":"disabled","usage":null,"user":meta.user,
        "metadata":meta.metadata
    })
}

fn output_items(id: &str, reasoning: &str, content: &str, calls: &[Value]) -> Vec<Value> {
    let mut items = Vec::new();
    if !reasoning.is_empty() {
        items.push(
            json!({"id":format!("rs_{id}"),"type":"reasoning","status":"completed",
            "summary":[{"type":"summary_text","text":reasoning}]}),
        );
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
    let calls = message["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default();
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
        &calls
    ));
    response["usage"] = responses_usage(&chat["usage"]);
    response
}

async fn non_stream_response(response: Response, meta: Meta) -> Response {
    let bytes = match to_bytes(response.into_body(), 16 * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                format!("unable to translate chat response (16 MiB limit): {e}"),
            )
        }
    };
    let chat: Value = match serde_json::from_slice(&bytes) {
        Ok(chat) => chat,
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    Json(completed_response(&chat, &meta)).into_response()
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
    reasoning_index: Option<usize>,
    content_index: Option<usize>,
    next_index: usize,
    terminal: bool,
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
            let mut items = output_items(&id, &self.reasoning, &self.content, &self.calls);
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
            self.emit("response.output_item.added", json!({"output_index":index,"item":{"id":format!("rs_{id}"),"type":"reasoning","status":"in_progress","summary":[]}}));
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
            self.emit("response.output_item.done", json!({"output_index":index,"item":{"id":format!("rs_{id}"),"type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":text}]}}));
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
        if self.calls.is_empty() {
            self.finish_items(false);
        }
        let id = self.id.clone().unwrap();
        let index = self.next_index;
        self.next_index += 1;
        let call_index = self.calls.len();
        let item_id = format!("fc_{id}_{call_index}");
        let call_id = call["id"].clone();
        let name = call["function"]["name"].clone();
        let arguments = call["function"]["arguments"].clone();
        self.emit("response.output_item.added", json!({"output_index":index,"item":{"id":item_id,"type":"function_call","status":"in_progress","call_id":call_id,"name":name,"arguments":""}}));
        self.emit(
            "response.function_call_arguments.delta",
            json!({"item_id":item_id,"output_index":index,"delta":arguments}),
        );
        self.emit(
            "response.function_call_arguments.done",
            json!({"item_id":item_id,"output_index":index,"arguments":arguments}),
        );
        self.emit("response.output_item.done", json!({"output_index":index,"item":{"id":item_id,"type":"function_call","status":"completed","call_id":call_id,"name":name,"arguments":arguments}}));
        self.calls.push(call);
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
                &self.calls
            ));
            response["usage"] = responses_usage(&chunk["usage"]);
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

fn stream_response(response: Response, meta: Meta) -> Response {
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
        reasoning_index: None,
        content_index: None,
        next_index: 0,
        terminal: false,
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
            _params: &GenParams,
            _cancel: &Arc<AtomicBool>,
            _progress: Option<infr_core::GenerationProgressCallback>,
            on_delta: &mut dyn FnMut(Delta),
        ) -> anyhow::Result<ChatOutcome> {
            if messages.last().is_some_and(|m| !m.images.is_empty()) {
                on_delta(Delta::Content(format!(
                    "image:{}",
                    messages.last().unwrap().images[0]
                )));
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
            "include":["reasoning.encrypted_content"],"service_tier":"auto"}),
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
        let (chat, _) = request.into_chat().unwrap();
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
        assert_eq!(request.into_chat().err().unwrap().param, "input");
    }

    #[test]
    fn minimal_effort_and_requested_token_limit_are_reported_honestly() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hello","reasoning":{"effort":"minimal"},
            "max_output_tokens":100
        }))
        .unwrap();
        let (chat, meta) = request.into_chat().unwrap();
        assert_eq!(
            chat.reasoning_effort,
            Some(infr_core::config::ReasoningEffort::Low)
        );
        assert_eq!(meta.reasoning["effort"], "low");
        assert_eq!(meta.max_output_tokens, Some(100));
    }

    #[test]
    fn user_limit_counts_characters_not_utf8_bytes() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hi","user":"本地用户"
        }))
        .unwrap();
        assert!(request.into_chat().is_ok());
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture","input":"hi","user":"x".repeat(65)
        }))
        .unwrap();
        assert_eq!(request.into_chat().err().unwrap().param, "user");
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
            ("parallel_tool_calls", json!(false)),
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
        let (chat, meta) = request.into_chat().unwrap();
        assert_eq!(
            chat.reasoning_effort,
            Some(infr_core::config::ReasoningEffort::High)
        );
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
        let (_, meta) = request.into_chat().unwrap();
        let upstream = Response::new(Body::from(concat!(
            "data: {\"id\":\"chatcmpl-test\",\"created\":1,\"model\":\"fixture\",\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\",\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
            "data: {\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}\n\n"
        )));
        let response = stream_response(upstream, meta);
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
        let (_, meta) = request.into_chat().unwrap();
        let upstream = Response::new(Body::from(
            "data: {\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}\n\n",
        ));
        let response = stream_response(upstream, meta);
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
        let (_, meta) = request.into_chat().unwrap();
        let upstream = Response::new(Body::from(vec![b' '; 16 * 1024 * 1024 + 1]));
        let response = non_stream_response(upstream, meta).await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = json_body(response).await;
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("16 MiB"));
    }
}
