//! Chat-template rendering: turn `(role, content)` messages into a prompt string via the GGUF's
//! embedded `tokenizer.chat_template` (a Jinja2 string). The single source of truth — every prompt
//! path funnels through [`render_chat_jinja`].

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use infr_core::config::Config;
use infr_core::loader::MetaValue;
use infr_core::WeightSource; // brings `Gguf::metadata()` into scope
use infr_gguf::Gguf;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::ChatMessage;

/// Compiled-environment cache keyed by the raw template source. A GGUF's chat template never
/// changes across a process, but `serve` re-renders it on every request/turn — building the
/// minijinja `Environment` and re-parsing the (often large, HF tool-calling) template each time is
/// pure waste. Keyed by source so distinct templates don't collide; entry count is bounded by the
/// number of distinct templates loaded (one per model), so no eviction is needed.
type SharedEnv = Arc<minijinja::Environment<'static>>;
static ENV_CACHE: LazyLock<Mutex<HashMap<String, SharedEnv>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Build a minijinja `Environment` with the full infr jinja surface (pycompat, `raise_exception`,
/// `strftime_now`, `tojson` with `indent=`) and the given chat template compiled under `"chat"`.
fn build_env(template: &str) -> Result<minijinja::Environment<'static>, minijinja::Error> {
    let mut env = minijinja::Environment::new();
    // HF chat templates lean on Python str/dict/list methods (`.get`, `.items`, `.strip`, …) that
    // minijinja core doesn't implement — pycompat supplies them (e.g. gemma4's tool-calling template).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    // `strftime_now(format)` — llama.cpp-minja parity: Llama-3.x templates stamp
    // `Today Date: {{ strftime_now("%d %b %Y") }}` (guarded by `is defined`, so defining it
    // switches those templates from their hardcoded fallback date to the real one).
    env.add_function("strftime_now", |fmt: String| {
        chrono::Local::now().format(&fmt).to_string()
    });
    // `tojson` with the optional `indent=` kwarg (Llama-3.x uses `tojson(indent=4)` for the tool
    // definitions; Qwen-family uses the bare compact form). Not minijinja's built-in `json` filter:
    // that one HTML-escapes (`<` → `<`), which llama.cpp/HF renders don't.
    env.add_filter(
        "tojson",
        |v: minijinja::Value,
         kwargs: minijinja::value::Kwargs|
         -> Result<String, minijinja::Error> {
            let indent: Option<usize> = kwargs.get("indent")?;
            kwargs.assert_all_used()?;
            let out = match indent {
                Some(n) => {
                    let pad = " ".repeat(n);
                    let fmt = serde_json::ser::PrettyFormatter::with_indent(pad.as_bytes());
                    let mut buf = Vec::new();
                    let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
                    serde::Serialize::serialize(&v, &mut ser)
                        .ok()
                        .and_then(|()| String::from_utf8(buf).ok())
                }
                None => serde_json::to_string(&v).ok(),
            };
            // Unserializable values degrade to "null" (pre-existing lenient behavior).
            Ok(out.unwrap_or_else(|| "null".to_owned()))
        },
    );
    env.add_template_owned("chat", template.to_owned())?;
    Ok(env)
}

/// Fetch (or build + cache) the compiled `Environment` for `template`.
fn cached_env(template: &str) -> Result<SharedEnv, minijinja::Error> {
    if let Some(env) = ENV_CACHE.lock().unwrap().get(template) {
        return Ok(env.clone());
    }
    let env: SharedEnv = Arc::new(build_env(template)?);
    ENV_CACHE
        .lock()
        .unwrap()
        .insert(template.to_owned(), env.clone());
    Ok(env)
}

/// Why a chat-template render failed — so serve/CLI callers can surface the ACTUAL jinja error
/// (e.g. a template construct the renderer doesn't support) instead of a generic "no template".
#[derive(Debug)]
pub enum TemplateError {
    /// The GGUF has no `tokenizer.chat_template` metadata at all.
    NoTemplate,
    /// The embedded template failed to parse or render (the minijinja error says why).
    Render(minijinja::Error),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::NoTemplate => {
                write!(f, "model GGUF has no `tokenizer.chat_template`")
            }
            TemplateError::Render(e) => write!(f, "chat template failed to render: {e:#}"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// THE jinja chat renderer — turns `(role, content)` messages into a prompt via the GGUF's embedded
/// `tokenizer.chat_template`. Template handling (pycompat, `enable_thinking`, bos/eos, tools) lives
/// here so every caller (single-turn, multi-turn, CPU + GPU backends) shares it. Returns `None` if
/// there's no template or it fails to render (caller falls back to [`chatml`]).
///
/// `cfg` supplies the two knobs this renderer reads — `sampling.no_think` (`INFR_NO_THINK`) and
/// `debug.chat` (`INFR_DEBUG_CHAT`). It is a BORROWED parameter rather than a field on a renderer
/// struct because `infr-chat` deliberately owns no state at all (no model, no backend, no cache
/// beyond the compiled-template memo): every entry point here is a pure function of its inputs, and
/// the config is one of those inputs. Callers that DO own a renderer — `SeamModel`,
/// `infr_llama::chat::OaiRenderer` — hold the `Arc<Config>` and hand out a borrow (§5, R6).
pub fn render_chat_jinja(
    gguf: &Gguf,
    tokenizer: &Tokenizer,
    eos: u32,
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
    cfg: &Config,
) -> Option<String> {
    let msgs: Vec<Value> = messages
        .iter()
        .map(|(r, c)| serde_json::json!({ "role": r, "content": c }))
        .collect();
    render_core(
        gguf,
        tokenizer,
        eos,
        msgs,
        Value::Null,
        add_generation_prompt,
        cfg,
    )
    .ok()
}

/// Render OpenAI-shaped [`ChatMessage`]s (full multi-turn history WITH tool calls + results) plus an
/// optional `tools` spec through the GGUF's embedded chat template. This is the tool-calling entry
/// point: the model's OWN template renders the tool definitions and wraps prior `tool_calls` /
/// `tool` results in its native format — so infr never hardcodes a per-model tool syntax.
///
/// `tools` is the request's OpenAI `tools` array (`[{type:"function", function:{name, parameters}}]`)
/// or `None`. Assistant `tool_calls` are emitted as `{type:"function", function:{name, arguments}}`
/// with `arguments` as a JSON object (templates `| tojson` it).
///
/// Errors carry the real cause ([`TemplateError`]) so serve can return the render error message
/// instead of a bare 500.
#[allow(clippy::too_many_arguments)]
pub fn render_chat_oai(
    gguf: &Gguf,
    tokenizer: &Tokenizer,
    eos: u32,
    messages: &[ChatMessage],
    tools: Option<&Value>,
    add_generation_prompt: bool,
    cfg: &Config,
) -> Result<String, TemplateError> {
    let msgs: Vec<Value> = messages.iter().map(message_to_json).collect();
    let tools = tools.cloned().unwrap_or(Value::Null);
    render_core(
        gguf,
        tokenizer,
        eos,
        msgs,
        tools,
        add_generation_prompt,
        cfg,
    )
}

/// Build the template's per-message dict, preserving the tool round-trip fields the HF chat templates
/// read (`tool_calls`, `tool_call_id`, `name`).
fn message_to_json(m: &ChatMessage) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), m.role.clone().into());
    obj.insert("content".into(), m.content.clone().into());
    if let Some(calls) = &m.tool_calls {
        let arr: Vec<Value> = calls
            .iter()
            .map(|c| {
                serde_json::json!({
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments },
                })
            })
            .collect();
        obj.insert("tool_calls".into(), Value::Array(arr));
    }
    if let Some(id) = &m.tool_call_id {
        obj.insert("tool_call_id".into(), id.clone().into());
    }
    if let Some(name) = &m.name {
        obj.insert("name".into(), name.clone().into());
    }
    Value::Object(obj)
}

/// Core renderer over a GGUF: pull the template + bos/eos out of the metadata, then delegate to
/// [`render_template`]. Shared by every entry point so template handling lives in ONE place.
#[allow(clippy::too_many_arguments)]
fn render_core(
    gguf: &Gguf,
    tokenizer: &Tokenizer,
    eos: u32,
    msgs: Vec<Value>,
    tools: Value,
    add_generation_prompt: bool,
    cfg: &Config,
) -> Result<String, TemplateError> {
    let template = gguf
        .metadata()
        .str("tokenizer.chat_template")
        .ok_or(TemplateError::NoTemplate)?;
    // BOS: use the metadata id if present, else fall back to the tokenizer's own BOS token — never
    // a hardcoded id (the old `2` default is EOS for Llama-family GGUFs, so a missing-metadata
    // model would inject the EOS string at the prompt head).
    let bos = gguf
        .metadata()
        .get("tokenizer.ggml.bos_token_id")
        .and_then(MetaValue::as_u64)
        .and_then(|id| tokenizer.id_to_token(id as u32))
        .unwrap_or_default();
    let eos_s = tokenizer.id_to_token(eos).unwrap_or_default();
    // Thinking is ON by default for every model whose template supports it — the key is simply
    // ignored by non-thinking templates, and thinking-capable models (Qwen3, Qwen3.5)
    // then behave the same under `infr run`/`serve` regardless of what their template's own
    // default is (Qwen3.5 defaults itself OFF via `enable_thinking is defined and is true`).
    let think = thinking_enabled(cfg);
    match render_template(
        template,
        msgs,
        tools,
        &bos,
        &eos_s,
        add_generation_prompt,
        think,
    ) {
        Ok(s) => {
            if cfg.debug.chat {
                eprintln!("[chat-template] rendered:\n{s}\n[/chat-template]");
            }
            Ok(s)
        }
        Err(e) => {
            if cfg.debug.chat {
                eprintln!("[chat-template] render error: {e:#}");
            }
            Err(TemplateError::Render(e))
        }
    }
}

/// The `enable_thinking` a render gets, from `sampling.no_think` (S7 — [`render_core`] used to
/// read the `INFR_NO_THINK` variable from the process environment right here).
///
/// `INFR_NO_THINK=1` turns thinking OFF and `INFR_NO_THINK=0` is a NO-OP, matching the other
/// `INFR_NO_*` toggles — that is the `SetNotZero` env grammar the config layer parses, so the
/// polarity lives in exactly one place and this is a plain negation.
fn thinking_enabled(cfg: &Config) -> bool {
    !cfg.sampling.no_think
}

/// Render a raw chat-template STRING with the full infr jinja environment (pycompat,
/// `raise_exception`, `tojson` with `indent=`, `strftime_now`) and prompt context (`messages`,
/// `tools`, bos/eos, `enable_thinking`). This is the GGUF-free seam — [`render_core`] wraps it, and
/// template-compat regression tests feed known templates (e.g. Llama-3.x) straight through it.
#[allow(clippy::too_many_arguments)]
pub fn render_template(
    template: &str,
    msgs: Vec<Value>,
    tools: Value,
    bos_token: &str,
    eos_token: &str,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String, minijinja::Error> {
    let env = cached_env(template)?;
    let tmpl = env
        .get_template("chat")
        .expect("template was just added under this name");
    let mut ctx = serde_json::Map::new();
    ctx.insert("messages".into(), Value::Array(msgs));
    ctx.insert("tools".into(), tools);
    ctx.insert("add_generation_prompt".into(), add_generation_prompt.into());
    ctx.insert("bos_token".into(), bos_token.into());
    ctx.insert("eos_token".into(), eos_token.into());
    ctx.insert("enable_thinking".into(), enable_thinking.into());
    tmpl.render(serde_json::Value::Object(ctx))
}

/// Single user turn through [`render_chat_jinja`] (`add_generation_prompt=true`). Shared by the GPU
/// and CPU one-shot paths so an instruct model answers coherently.
pub fn render_chat_user(
    gguf: &Gguf,
    tokenizer: &Tokenizer,
    eos: u32,
    user: &str,
    cfg: &Config,
) -> Option<String> {
    render_chat_jinja(gguf, tokenizer, eos, &[("user", user)], true, cfg)
}

#[cfg(test)]
mod template_tests {
    use super::*;

    const TMPL: &str =
        "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}bos={{ bos_token }}";

    fn msgs() -> Vec<Value> {
        vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "content": "yo"}),
        ]
    }

    #[test]
    fn cache_returns_identical_renders() {
        // Rendering the same template twice (second hits the compiled-env cache) is byte-identical.
        let a = render_template(TMPL, msgs(), Value::Null, "<s>", "</s>", true, true).unwrap();
        let b = render_template(TMPL, msgs(), Value::Null, "<s>", "</s>", true, true).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, "user:hi\nassistant:yo\nbos=<s>");
    }

    #[test]
    fn cache_is_keyed_by_source() {
        // A distinct template source must not collide with a previously-cached one.
        let other = "ONLY:{{ messages[0].content }}";
        let a = render_template(TMPL, msgs(), Value::Null, "<s>", "</s>", true, true).unwrap();
        let b = render_template(other, msgs(), Value::Null, "<s>", "</s>", true, true).unwrap();
        assert_ne!(a, b);
        assert_eq!(b, "ONLY:hi");
    }

    /// `sampling.no_think` drives the template's `enable_thinking`, read off a `Config` VALUE —
    /// never the environment (S7, R7). The truth table the env layer feeds this with:
    ///
    /// | `INFR_NO_THINK` | env layer emits | `sampling.no_think` | `enable_thinking` |
    /// | --------------- | --------------- | ------------------- | ----------------- |
    /// | unset           | `None`          | `false`             | `true`  (thinking) |
    /// | `"0"`           | `None`          | `false`             | `true`  (NO-OP)    |
    /// | `""`            | `Some(true)`    | `true`              | `false` (off)      |
    /// | `"1"`           | `Some(true)`    | `true`              | `false` (off)      |
    #[test]
    fn no_think_config_drives_enable_thinking() {
        let mut cfg = Config::default();
        assert!(thinking_enabled(&cfg), "default = thinking ON");
        cfg.sampling.no_think = true;
        assert!(!thinking_enabled(&cfg), "sampling.no_think = thinking OFF");

        // …and the flag really reaches the template context.
        const PROBE: &str = "think={{ enable_thinking }}";
        let mut cfg = Config::default();
        let on = render_template(
            PROBE,
            msgs(),
            Value::Null,
            "",
            "",
            true,
            thinking_enabled(&cfg),
        )
        .unwrap();
        cfg.sampling.no_think = true;
        let off = render_template(
            PROBE,
            msgs(),
            Value::Null,
            "",
            "",
            true,
            thinking_enabled(&cfg),
        )
        .unwrap();
        assert_eq!(on, "think=true");
        assert_eq!(off, "think=false");
    }
}
