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

/// Instruction budget for ONE render ([`minijinja::Environment::set_fuel`], per-render, not
/// per-process).
///
/// The chat template is UNTRUSTED INPUT: it arrives as `tokenizer.chat_template` inside a GGUF
/// somebody published, and infr executes it on every prompt. Unbounded, a template that loops —
/// `{% for i in range(100000) %}{% for j in range(100000) %}x{% endfor %}{% endfor %}` is 10^10
/// instructions — pins a core forever on the FIRST prompt. Under `infr serve` that render happens
/// in a `spawn_blocking` while the request still holds its `--parallel` slot permit, so the hang
/// also costs a generation slot that never comes back. (minijinja's own 100_000-element cap on
/// `range()` does not help: nested loops multiply, and macro recursion sidesteps it entirely.)
///
/// 100M is picked from measured consumption of the real templates this repo renders — Llama-3.x
/// (`tojson(indent=4)`, list slicing), Qwen3/Qwen3.6 tool-calling, and gemma-4's, the worst of the
/// bunch because it runs a backward pre-scan over prior messages FOR each message and so costs
/// ~3.5·n² for n messages:
///
/// | render                                    | fuel used  | margin  |
/// | ----------------------------------------- | ---------- | ------- |
/// | 2 messages, no tools (any template)       |    150-420 | ~250000x |
/// | 2 messages, 16 tools (gemma-4, worst)     |     10_071 |   ~9900x |
/// | 100 turns, 16 tools (gemma-4, worst)      |    181_440 |    ~550x |
/// | 1000 turns, 32 tools (gemma-4, worst)     | 14_336_388 |      ~7x |
/// | 4001 messages, 64 tools (gemma-4, worst)  | 56_672_684 |    ~1.8x |
///
/// The bottom row is a 276 KB prompt — past any context window infr serves — and it still renders.
/// The ceiling is also self-limiting in wall-clock terms: minijinja runs ~90M instructions/sec in
/// release, so a render that actually exhausts 100M fuel has already burned a full CPU-second
/// building a PROMPT, which is itself the pathology, not the work.
///
/// Symptom when the limit IS hit: `tmpl.render` returns `minijinja::Error` with
/// `ErrorKind::OutOfFuel` ("ran out of fuel"), which [`render_core`] wraps as
/// [`TemplateError::Render`] — so `serve` answers with the template error message and `infr run`
/// falls back to [`chatml`](crate::chatml), exactly as for any other malformed template. Nothing
/// hangs and no slot leaks.
pub(crate) const CHAT_TEMPLATE_FUEL: u64 = 100_000_000;

/// Build a minijinja `Environment` with the full infr jinja surface (pycompat, `raise_exception`,
/// `strftime_now`, `tojson` with `indent=`), a [`CHAT_TEMPLATE_FUEL`] execution bound, and the
/// given chat template compiled under `"chat"`.
fn build_env(template: &str) -> Result<minijinja::Environment<'static>, minijinja::Error> {
    let mut env = minijinja::Environment::new();
    // Bound the render before anything else — see CHAT_TEMPLATE_FUEL for why an untrusted template
    // must never be handed an unbounded interpreter. The tracker is created per render, so the
    // budget is not consumed across the cached environment's lifetime.
    env.set_fuel(Some(CHAT_TEMPLATE_FUEL));
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
///
/// ONE lock acquisition covers both the lookup and the insert. Split across two (get, drop, build,
/// re-lock, insert) the miss path is a race: under `serve --parallel N` the first N requests for a
/// freshly loaded model arrive together, all miss, all parse the (16 KB, for gemma-4) template, and
/// the last insert overwrites the rest — N-1 parses thrown away, and N distinct `Arc`s briefly
/// handed out for what is documented to be a shared compiled environment.
///
/// The cost of holding the lock across the fallible `build_env` is that concurrent FIRST renders
/// serialize behind one parse. That is the intended trade: it happens once per distinct template
/// (i.e. once per model, per process) and one parse is precisely what the shared cache exists to
/// buy. `entry().or_insert_with` cannot express this — the closure has no way to return the
/// `minijinja::Error` a malformed template produces, and swallowing it would cache a bad
/// environment or panic — so the check-then-insert stays explicit inside a single `lock()` scope.
fn cached_env(template: &str) -> Result<SharedEnv, minijinja::Error> {
    let mut cache = ENV_CACHE.lock().unwrap();
    if let Some(env) = cache.get(template) {
        return Ok(env.clone());
    }
    let env: SharedEnv = Arc::new(build_env(template)?);
    cache.insert(template.to_owned(), env.clone());
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
                tracing::info!("[chat-template] rendered:\n{s}\n[/chat-template]");
            }
            Ok(s)
        }
        Err(e) => {
            if cfg.debug.chat {
                tracing::info!("[chat-template] render error: {e:#}");
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

    /// Every environment this crate hands out carries the [`CHAT_TEMPLATE_FUEL`] bound. Asserted on
    /// `build_env` directly (and instantly) so a future edit that reorders or drops the
    /// `set_fuel` call fails HERE, loudly, instead of only in the slow runaway test below.
    #[test]
    fn build_env_sets_the_fuel_bound() {
        let env = build_env(TMPL).expect("fixture template parses");
        assert_eq!(env.fuel(), Some(CHAT_TEMPLATE_FUEL));
    }

    /// A malicious template — one nested loop, 10^10 iterations — must FAIL rather than hang the
    /// process. This is the whole point of the fuel bound: the template ships inside a downloaded
    /// GGUF, and `serve` renders it inside a `spawn_blocking` that holds a `--parallel` slot.
    ///
    /// The loop is nested deliberately: minijinja caps a single `range()` at 100_000 elements, so
    /// the flat `{% for i in range(100000000) %}` form is rejected by minijinja itself and would
    /// prove nothing about fuel. Two nested 100_000 ranges are individually legal and multiply.
    ///
    /// Cost note: this test burns the REAL production budget (there is no smaller-fuel back door,
    /// on purpose — the test must exercise the environment `render_template` actually builds), so
    /// it runs for ~1s in release and ~5s in a debug `cargo test`. That is the price of proving the
    /// bound terminates; without it the same test never returns at all.
    #[test]
    fn runaway_template_runs_out_of_fuel_instead_of_hanging() {
        const RUNAWAY: &str =
            "{% for i in range(100000) %}{% for j in range(100000) %}x{% endfor %}{% endfor %}";
        let err = render_template(RUNAWAY, msgs(), Value::Null, "<s>", "</s>", true, true)
            .expect_err("a 10^10-iteration template must be cut off, not rendered");
        assert_eq!(
            err.kind(),
            minijinja::ErrorKind::OutOfFuel,
            "must fail on the fuel bound, not incidentally: {err:#}"
        );
    }

    /// …and the bound does not clip legitimate work: an ordinary template renders byte-identically
    /// to what it produced before the limit existed. Pairs with the runaway test — a fuel limit
    /// that rejected real templates would "pass" that one for the wrong reason.
    #[test]
    fn fuel_bound_does_not_clip_a_normal_render() {
        let out = render_template(TMPL, msgs(), Value::Null, "<s>", "</s>", true, true).unwrap();
        assert_eq!(out, "user:hi\nassistant:yo\nbos=<s>");

        // A loop with real (but sane) trip count — 1000 iterations of string building — is far
        // under the budget too, so per-message work in a long conversation is never the thing that
        // trips the limit.
        const LOOPY: &str = "{% for i in range(1000) %}{{ i }},{% endfor %}";
        let out = render_template(LOOPY, msgs(), Value::Null, "", "", true, true)
            .expect("1000 iterations is legitimate work, not a runaway");
        assert!(out.starts_with("0,1,2,"), "{out}");
        assert!(out.ends_with(",999,"), "{out}");
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
