//! Parsing model OUTPUT back into structured pieces — channel splitting (reasoning vs answer)
//! and `<|tool_call>…<tool_call|>` block parsing. Pure logic, no IO.
//!
//! Reference: `~/Projects/scratch/dgemma-openai-server.py` (Python shim).
//! Token formats: docs/plan.md "DiffusionGemma spec".

use serde_json::Value;

// ---------------------------------------------------------------------------
// Channel splitting
// ---------------------------------------------------------------------------

/// The marker that begins the final-answer channel.
const FINAL_MARK: &str = "<channel|>";

/// All marker substrings that should be stripped when cleaning channel text.
const MARKERS: &[&str] = &[
    "<|channel>thought",
    "<|channel|>thought",
    "<|channel>",
    "<channel|>",
    "<|channel|>",
];

fn strip_markers(s: &str) -> String {
    let mut out = s.to_owned();
    for m in MARKERS {
        out = out.replace(m, "");
    }
    // Trim leading newlines / whitespace (mirrors Python lstrip("\n").strip())
    out.trim_start_matches('\n').trim().to_owned()
}

/// Split cumulative model output into `(reasoning, content)`.
///
/// Reasoning = text before `<channel|>` (markers stripped).
/// Content   = text after  `<channel|>` (markers stripped).
/// If the marker is absent, reasoning = full stripped text, content = `""`.
pub fn split_channels(full: &str) -> (String, String) {
    if let Some(idx) = full.find(FINAL_MARK) {
        let head = &full[..idx];
        let tail = &full[idx + FINAL_MARK.len()..];
        (strip_markers(head), strip_markers(tail))
    } else {
        (strip_markers(full), String::new())
    }
}

// ---------------------------------------------------------------------------
// Tool-call parsing
// ---------------------------------------------------------------------------

/// One parsed tool invocation from the model's `<|tool_call>…<tool_call|>` block.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

// Regex-free: scan for literal open/close delimiters.
const TC_OPEN: &str = "<|tool_call>";
const TC_CLOSE: &str = "<tool_call|>";

/// Read exactly 4 hex digits from `s` at `i` into a code point; `None` if fewer than 4 remain or a
/// digit is non-hex.
fn read_hex4(s: &[u8], i: usize) -> Option<u32> {
    if i + 4 > s.len() {
        return None;
    }
    let mut v = 0u32;
    for k in 0..4 {
        v = v * 16 + (s[i + k] as char).to_digit(16)?;
    }
    Some(v)
}

/// Remove the given byte `spans` (non-overlapping, on char boundaries) from `text`. Sorted then
/// excised back-to-front so earlier indices stay valid — the shared removal machinery both
/// tool-call parsers use.
fn remove_spans(text: &str, mut spans: Vec<(usize, usize)>) -> String {
    spans.sort_unstable_by_key(|&(start, _)| start);
    let mut out = text.to_owned();
    for (start, end) in spans.into_iter().rev() {
        out.replace_range(start..end, "");
    }
    out
}

/// Hard ceiling on `{`/`[` nesting in [`parse_value`].
///
/// This parser runs on model OUTPUT, and in `infr serve` that output is steerable by whoever
/// sent the request — "reply with `<|tool_call>call:x{a:{a:{a:…`" is enough. Without a ceiling
/// each `{` costs the attacker one byte and us one stack frame, so a single HTTP request
/// stack-overflows the server process (SIGSEGV, not a catchable panic — every other in-flight
/// request dies with it). 64 is far above any real tool schema, whose arguments are one or two
/// levels of object/array; legitimate calls never reach it.
const MAX_VALUE_DEPTH: usize = 64;

/// Index of the next non-whitespace byte at or after `i` (or `s.len()`). Used both to enter a value
/// and to look ahead at one without committing to parsing it.
fn skip_ws(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && matches!(s[i], b' ' | b'\n' | b'\t' | b'\r') {
        i += 1;
    }
    i
}

/// Recursive-descent value parser — mirrors `_parse_value` in the Python shim.
///
/// `s` has already had `<|"|>` replaced with `"`.
///
/// Returns `None` in exactly two cases: nesting would exceed [`MAX_VALUE_DEPTH`], or the value
/// position holds a byte that cannot begin one (`,`, `}`, `]`). `None` propagates all the way out,
/// so the caller drops the WHOLE call rather than acting on a partial one.
///
/// Everything else stays lenient-by-design — missing quotes, unterminated strings and an input
/// that simply runs out all yield a best-effort `Value`, which callers rely on.
///
/// The delimiter rejection is not merely strictness, and it is why the degradation could NOT be
/// "return `Value::Null` in place and stop consuming": both container arms loop until they reach
/// their closing bracket, so a value that returns WITHOUT advancing `i` leaves the loop on the
/// same byte forever, pushing values until the process dies. That is reachable from model output
/// (`{a:[}]}` is enough), so it is a hang and an unbounded allocation, not a parse quirk.
/// Propagating `None` is the only exit that terminates every loop.
///
/// A VALUELESS OBJECT ENTRY never reaches here: the object arm looks ahead past the colon and, on
/// finding a `,` or `}`, drops that one entry and carries on. `{a:,b:1}` is `{"b": 1}`, not a
/// dropped call — one malformed pair must not cost the pairs that parsed. So the `,`/`}` half of
/// the rejection above is reachable only from an ARRAY (`[}]`) or the top level, where there is no
/// entry to drop and no way to make progress.
fn parse_value(s: &[u8], mut i: usize, depth: usize) -> Option<(Value, usize)> {
    i = skip_ws(s, i);
    if i >= s.len() {
        return Some((Value::Null, i));
    }
    // A closing delimiter is not a value. Returning a successful empty bareword without consuming
    // it leaves the surrounding container loop on the same byte forever.
    if matches!(s[i], b',' | b'}' | b']') {
        return None;
    }
    // Refuse before descending into a container, so the frame budget is checked once per level.
    if matches!(s[i], b'{' | b'[') && depth >= MAX_VALUE_DEPTH {
        return None;
    }
    match s[i] {
        b'{' => {
            i += 1;
            let mut obj = serde_json::Map::new();
            loop {
                // skip whitespace and commas
                while i < s.len() && matches!(s[i], b' ' | b'\n' | b'\t' | b'\r' | b',') {
                    i += 1;
                }
                if i >= s.len() || s[i] == b'}' {
                    if i < s.len() {
                        i += 1;
                    }
                    return Some((Value::Object(obj), i));
                }
                // Find the `key:` separator, bounded by this object's own punctuation.
                //
                // An unbounded `position(|b| b == b':')` scans past the closing `}` into
                // unrelated text, so a body like `{foo} see http://x` produced the key
                // "foo} see http" and resumed the value parse mid-token — a tool call that
                // LOOKS well-formed but carries a garbage argument name, which is worse than
                // a rejected parse because the caller acts on it. Stop at the first `}` or
                // `,` seen before any colon, and stay quote-aware so a quoted key may still
                // legitimately contain those bytes (`{"a,b": 1}`).
                let mut j = i;
                let mut quote: Option<u8> = None;
                let mut colon: Option<usize> = None;
                let mut stop: Option<usize> = None;
                while j < s.len() {
                    let b = s[j];
                    match quote {
                        Some(q) => {
                            if b == b'\\' {
                                j += 2; // skip the escaped byte, whatever it is
                                continue;
                            }
                            if b == q {
                                quote = None;
                            }
                        }
                        None => match b {
                            b'"' | b'\'' => quote = Some(b),
                            b':' => {
                                colon = Some(j);
                                break;
                            }
                            b'}' | b',' => {
                                stop = Some(j);
                                break;
                            }
                            _ => {}
                        },
                    }
                    j += 1;
                }
                let colon = match colon {
                    Some(c) => c,
                    // `}` closes the object; `,` ends a valueless (malformed) entry — drop it
                    // and carry on with the next one rather than scanning onward for a colon.
                    None => {
                        let at = stop.unwrap_or(s.len());
                        if at < s.len() && s[at] == b'}' {
                            return Some((Value::Object(obj), at + 1));
                        }
                        if at >= s.len() {
                            // No colon and no punctuation left: nothing parseable remains.
                            return Some((Value::Object(obj), i));
                        }
                        i = at + 1;
                        continue;
                    }
                };
                let raw_key = String::from_utf8_lossy(&s[i..colon])
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_owned();
                i = colon + 1;
                // A valueless entry (`{a:,b:1}`, `{a:}`) loses THE ENTRY, not the call: binding a
                // value the model never wrote would hand the tool an argument it never asked for,
                // and dropping the whole call over one bad pair throws away the pairs that ARE
                // well-formed. Resume on the delimiter itself so the loop above consumes it —
                // advancing past it here would skip whatever follows.
                let vi = skip_ws(s, i);
                if vi < s.len() && matches!(s[vi], b',' | b'}') {
                    i = vi;
                    continue;
                }
                let (val, ni) = parse_value(s, i, depth + 1)?;
                i = ni;
                obj.insert(raw_key, val);
            }
        }
        b'[' => {
            i += 1;
            let mut arr = Vec::new();
            loop {
                while i < s.len() && matches!(s[i], b' ' | b'\n' | b'\t' | b'\r' | b',') {
                    i += 1;
                }
                if i >= s.len() || s[i] == b']' {
                    if i < s.len() {
                        i += 1;
                    }
                    return Some((Value::Array(arr), i));
                }
                let (val, ni) = parse_value(s, i, depth + 1)?;
                i = ni;
                arr.push(val);
            }
        }
        q @ b'"' | q @ b'\'' => {
            i += 1;
            let mut buf: Vec<u8> = Vec::new();
            while i < s.len() {
                let b = s[i];
                if b == b'\\' && i + 1 < s.len() {
                    let esc = s[i + 1];
                    i += 2;
                    match esc {
                        b'n' => buf.push(b'\n'),
                        b't' => buf.push(b'\t'),
                        b'r' => buf.push(b'\r'),
                        b'"' => buf.push(b'"'),
                        b'\'' => buf.push(b'\''),
                        b'\\' => buf.push(b'\\'),
                        b'/' => buf.push(b'/'),
                        b'b' => buf.push(0x08),
                        b'f' => buf.push(0x0C),
                        b'u' => {
                            if let Some(cp) = read_hex4(s, i) {
                                i += 4;
                                // High surrogate → consume the following `\uXXXX` low surrogate.
                                let ch = if (0xD800..=0xDBFF).contains(&cp)
                                    && s.get(i) == Some(&b'\\')
                                    && s.get(i + 1) == Some(&b'u')
                                {
                                    match read_hex4(s, i + 2) {
                                        Some(lo) if (0xDC00..=0xDFFF).contains(&lo) => {
                                            i += 6;
                                            let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                            char::from_u32(c).unwrap_or('\u{FFFD}')
                                        }
                                        _ => char::from_u32(cp).unwrap_or('\u{FFFD}'),
                                    }
                                } else {
                                    char::from_u32(cp).unwrap_or('\u{FFFD}')
                                };
                                let mut tmp = [0u8; 4];
                                buf.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                            } else {
                                buf.push(b'u'); // malformed `\u` — keep the letter
                            }
                        }
                        // Unknown escape: drop the backslash, keep the escaped byte.
                        other => buf.push(other),
                    }
                } else if b == q {
                    i += 1;
                    break;
                } else {
                    buf.push(b);
                    i += 1;
                }
            }
            let text = String::from_utf8_lossy(&buf).into_owned();
            Some((Value::String(text), i))
        }
        _ => {
            // bareword / number / bool / null — read until delimiter
            let j_rel = s[i..]
                .iter()
                .position(|&b| matches!(b, b',' | b'}' | b']'))
                .unwrap_or(s.len() - i);
            let j = i + j_rel;
            let tok = String::from_utf8_lossy(&s[i..j]);
            let tok = tok.trim();
            i = j;
            // Leaf values are always parseable (lenient by design) — only the container arms
            // above can return `None`, and only on depth overflow.
            Some(match tok {
                "true" => (Value::Bool(true), i),
                "false" => (Value::Bool(false), i),
                "null" => (Value::Null, i),
                _ => {
                    if let Ok(n) = tok.parse::<i64>() {
                        (Value::Number(n.into()), i)
                    } else if let Ok(f) = tok.parse::<f64>() {
                        // `from_f64` returns None only for non-finite (`inf`/`NaN`), which JSON
                        // can't represent — keep those as the literal string, don't coerce to 0.
                        match serde_json::Number::from_f64(f) {
                            Some(num) => (Value::Number(num), i),
                            None => (Value::String(tok.to_owned()), i),
                        }
                    } else {
                        (Value::String(tok.to_owned()), i)
                    }
                }
            })
        }
    }
}

/// Find all `<|tool_call>…<tool_call|>` blocks, parse them, and return:
/// - `clean`: the input text with all tool-call blocks (and stray markers) removed.
/// - the parsed [`ToolCall`] list.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();

    // Work on the original text for finding blocks; collect spans to remove.
    let mut search_from = 0usize;
    let mut spans: Vec<(usize, usize)> = Vec::new();

    while let Some(open_pos) = text[search_from..].find(TC_OPEN) {
        let open_abs = search_from + open_pos;
        let body_start = open_abs + TC_OPEN.len();
        let Some(close_rel) = text[body_start..].find(TC_CLOSE) else {
            // Unterminated opener: strip the dangling markup through end-of-text so it can't leak
            // into `clean` as raw `<|tool_call>…`.
            spans.push((open_abs, text.len()));
            break;
        };
        let body = &text[body_start..body_start + close_rel];
        let close_abs = body_start + close_rel + TC_CLOSE.len();
        spans.push((open_abs, close_abs));
        search_from = close_abs;

        // parse body: strip leading "call:"
        let body = body.trim();
        let body = body.strip_prefix("call:").unwrap_or(body);

        let Some(brace) = body.find('{') else {
            continue;
        };
        let name = body[..brace].trim().to_owned();
        if name.is_empty() {
            continue;
        }
        // Replace the model's string-quote escape with real double-quotes
        let argstr = body[brace..].replace("<|\"|>", "\"");
        // `None` ⇒ the arguments nested past `MAX_VALUE_DEPTH`. Drop the call entirely rather
        // than reporting the truncated prefix as a successful `ToolCall`: the caller executes
        // what it is handed, and a call whose arguments we refused to finish parsing is not
        // one we can vouch for. The block's span is already queued for removal above, so the
        // markup still never leaks into `clean`.
        let Some((val, _)) = parse_value(argstr.as_bytes(), 0, 0) else {
            continue;
        };
        let arguments = match val {
            Value::Object(_) => val,
            other => {
                let mut m = serde_json::Map::new();
                m.insert("value".to_owned(), other);
                Value::Object(m)
            }
        };
        calls.push(ToolCall { name, arguments });
    }

    let clean = strip_markers(&remove_spans(text, spans));
    (clean, calls)
}

// ---------------------------------------------------------------------------
// Hermes / Qwen tool-call parsing (`<tool_call>{json}</tool_call>`)
// ---------------------------------------------------------------------------

const HERMES_OPEN: &str = "<tool_call>";
const HERMES_CLOSE: &str = "</tool_call>";

/// Parse the Hermes / Qwen tool-call format — `<tool_call>{"name":..,"arguments":{..}}</tool_call>`
/// (the format Qwen3 and most OSS function-calling models emit, and what the GGUF chat templates
/// render). Returns `(clean, calls)`: `clean` is the text with all tool-call blocks removed; `calls`
/// the parsed invocations. The body is real JSON, so it's parsed with serde (no hand-rolled scanner);
/// `arguments` is accepted as either an object or a JSON string and normalised to a `Value`.
///
/// A tool-call block's MARKUP never reaches `clean`, whether or not its body parsed — the same
/// policy [`parse_tool_calls`] applies to the pipe-marker dialect. Two cases used to leak it, and
/// both are model output the user then saw verbatim as if it were prose:
/// - an unterminated `<tool_call>` (the turn hit the token budget mid-call) — the dangling opener
///   and everything after it is dropped, exactly as the pipe-marker parser drops its own;
/// - a body that failed to parse as either dialect — the span is removed even though no call is
///   produced. Emitting nothing is the honest outcome for a call we could not read; emitting the
///   raw `<tool_call>…</tool_call>` as assistant content is not.
///
/// Note the span is queued BEFORE the body is parsed, so the two cases share one rule instead of
/// the removal being conditional on the parse succeeding.
pub fn parse_hermes_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut from = 0usize;
    while let Some(open) = text[from..].find(HERMES_OPEN) {
        let open_abs = from + open;
        let body_start = open_abs + HERMES_OPEN.len();
        let Some(close_rel) = text[body_start..].find(HERMES_CLOSE) else {
            // Unterminated opener: strip the dangling markup through end-of-text.
            spans.push((open_abs, text.len()));
            break;
        };
        let body = text[body_start..body_start + close_rel].trim();
        let close_abs = body_start + close_rel + HERMES_CLOSE.len();
        from = close_abs;
        spans.push((open_abs, close_abs));
        if let Some(call) = parse_hermes_body(body) {
            calls.push(call);
        }
    }
    let clean = remove_spans(text, spans);
    (clean.trim().to_owned(), calls)
}

/// Parse one `<tool_call>` body into a [`ToolCall`]. Two body dialects exist in the wild:
/// - Hermes/Qwen3 JSON: `{"name":..,"arguments":..}` (`arguments` tolerated as a nested object
///   or an embedded JSON string);
/// - the XML-parameter format Qwen3.5/3.6-class templates mandate (llama.cpp's "qwen3-coder"
///   handler): `<function=NAME><parameter=KEY>VALUE</parameter>…</function>` — the model follows
///   its template, so a JSON-only parser silently DROPPED these calls (empty serve replies with
///   `finish_reason:"stop"`, found via hrdr against Qwen3.6-27B).
fn parse_hermes_body(body: &str) -> Option<ToolCall> {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        let name = v.get("name")?.as_str()?.to_owned();
        let arguments = match v.get("arguments") {
            Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(Value::String(s.clone())),
            Some(other) => other.clone(),
            None => Value::Object(serde_json::Map::new()),
        };
        return Some(ToolCall { name, arguments });
    }
    parse_xml_function_body(body)
}

/// The XML-parameter dialect: `<function=NAME>` then zero or more
/// `<parameter=KEY>\nVALUE\n</parameter>` blocks (values may span lines), `</function>`.
/// Values are JSON-coerced when they parse (numbers, booleans, objects), else kept as strings —
/// matching llama.cpp's qwen3-coder chat handler.
fn parse_xml_function_body(body: &str) -> Option<ToolCall> {
    let body = body.trim();
    let rest = body.strip_prefix("<function=")?;
    let name_end = rest.find('>')?;
    let name = rest[..name_end].trim().to_owned();
    if name.is_empty() {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut rest = &rest[name_end + 1..];
    while let Some(pstart) = rest.find("<parameter=") {
        let after = &rest[pstart + "<parameter=".len()..];
        let Some(key_end) = after.find('>') else {
            break;
        };
        let key = after[..key_end].trim().to_owned();
        let val_area = &after[key_end + 1..];
        let Some(vend) = val_area.find("</parameter>") else {
            break;
        };
        // The template frames values with newlines; strip exactly the framing whitespace.
        let raw = val_area[..vend].trim();
        let value =
            serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_owned()));
        args.insert(key, value);
        rest = &val_area[vend + "</parameter>".len()..];
    }
    Some(ToolCall {
        name,
        arguments: Value::Object(args),
    })
}

/// Dialect-aware tool-call extraction — THE entry point consumers should use. Supported models
/// emit three tool-call dialects, per their GGUF chat templates:
/// - Hermes/Qwen3 JSON and the Qwen3.5/3.6 XML-parameter body, both inside `<tool_call>` tags
///   ([`parse_hermes_tool_calls`]);
/// - the gemma-4 / E2B / DiffusionGemma pipe-marker form `<|tool_call>call:NAME{..}<tool_call|>`
///   ([`parse_tool_calls`]) — serve previously never tried this one, silently dropping those
///   models' calls;
/// - Llama-3.x's bare-JSON form: the template instructs `{"name": .., "parameters": {..}}` as
///   the WHOLE response, no markers at all.
///
/// Dialects are tried in that order; the bare-JSON form only counts when the entire (trimmed)
/// body is a single such object, so ordinary prose mentioning JSON is never misparsed. Callers
/// gate on tools-present (`allow_tools`) as before.
pub fn parse_any_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let (clean, calls) = parse_hermes_tool_calls(text);
    if !calls.is_empty() {
        return (clean, calls);
    }
    let (clean, calls) = parse_tool_calls(text);
    if !calls.is_empty() {
        return (clean, calls);
    }
    // The bare-JSON arm returns an EMPTY `clean` while the two arms above preserve the
    // surrounding prose through their own `clean`. That asymmetry is deliberate, not an
    // oversight: the other two dialects delimit the call with markers, so text outside the
    // markers is genuine assistant prose that must survive. Llama-3.x has no markers — its
    // template instructs the model to make the WHOLE response the JSON object, and
    // `parse_bare_json_call` only matches when the entire trimmed body is that object. So
    // by construction there is no surrounding text to preserve here: everything `text`
    // holds IS the call, and echoing it back as content would print the raw JSON to the
    // user on top of dispatching the call.
    if let Some(call) = parse_bare_json_call(text) {
        return (String::new(), vec![call]);
    }
    (text.trim().to_owned(), Vec::new())
}

/// Llama-3.x dialect: the whole body is one JSON object `{"name": .., "parameters"|"arguments":
/// {..}}` (the template literally instructs 'respond with JSON for a function call' in exactly
/// this format). Anything else — prose, JSON missing "name", arrays — is not a call.
pub(crate) fn parse_bare_json_call(text: &str) -> Option<ToolCall> {
    let t = text.trim();
    if !t.starts_with('{') || !t.ends_with('}') {
        return None;
    }
    let v: Value = serde_json::from_str(t).ok()?;
    let name = v.get("name")?.as_str()?.to_owned();
    let arguments = match v.get("parameters").or_else(|| v.get("arguments")) {
        Some(Value::Object(m)) => Value::Object(m.clone()),
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(Value::String(s.clone())),
        None => Value::Object(serde_json::Map::new()),
        Some(_) => return None,
    };
    Some(ToolCall { name, arguments })
}

// ---------------------------------------------------------------------------
// Reasoning split (`<think>…</think>`)
// ---------------------------------------------------------------------------

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Format-aware reasoning split: channel-marker output (E2B/gpt-oss `<|channel>thought…<channel|>`)
/// goes through [`split_channels`], everything else through [`split_think`]. THE one entry point
/// for "what part of this reply was reasoning" — history stripping and batch consumers use this.
pub fn split_reasoning(text: &str) -> (String, String) {
    if MARKERS.iter().any(|m| text.contains(m)) {
        split_channels(text)
    } else {
        split_think(text)
    }
}

/// Split output into `(reasoning, content)` on the `<think>…</think>` markers Qwen3/DeepSeek-R1
/// emit — stripping EVERY reasoning span, not just the first (a constrained/looped turn can emit
/// several). Handles the prefilled-`<think>` case where the open marker was added by the template
/// (output then starts mid-reasoning and only the close marker appears) and an unterminated open
/// (truncated turn — the tail is reasoning). No marker ⇒ all content, empty reasoning. This is
/// THE reasoning grammar: `infr run`'s display, `infr serve`'s deltas and the chat history
/// stripper all resolve reasoning-vs-content through this module.
pub fn split_think(text: &str) -> (String, String) {
    let (mut reasoning, mut content) = (String::new(), String::new());
    let mut rest = text;
    while !rest.is_empty() {
        let open = rest.find(THINK_OPEN);
        let close = rest.find(THINK_CLOSE);
        match (open, close) {
            // Close before (or without) an open: prefilled-`<think>` — the head is reasoning.
            // `trim_start_matches` tolerates a stray duplicate open marker.
            (o, Some(c)) if o.is_none_or(|o| c < o) => {
                reasoning.push_str(rest[..c].trim_start_matches(THINK_OPEN));
                rest = &rest[c + THINK_CLOSE.len()..];
            }
            // Open..close pair (the guard above rules out c < o).
            (Some(o), Some(c)) => {
                content.push_str(&rest[..o]);
                reasoning.push_str(&rest[o + THINK_OPEN.len()..c]);
                rest = &rest[c + THINK_CLOSE.len()..];
            }
            // Open with no close (truncated / max tokens) — the tail is reasoning.
            (Some(o), None) => {
                content.push_str(&rest[..o]);
                reasoning.push_str(&rest[o + THINK_OPEN.len()..]);
                rest = "";
            }
            (None, None) => {
                content.push_str(rest);
                rest = "";
            }
            // Unreachable: `(None, Some(_))` is consumed by the first arm's guard.
            (None, Some(_)) => unreachable!("guarded above"),
        }
    }
    (reasoning.trim().to_owned(), content.trim().to_owned())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- split_channels --------------------------------------------------

    #[test]
    fn split_channels_reasoning_and_answer() {
        let full = "<|channel>thought\nI need to think about this.\n<channel|>The answer is 42.";
        let (r, c) = split_channels(full);
        assert_eq!(r, "I need to think about this.", "reasoning mismatch");
        assert_eq!(c, "The answer is 42.", "content mismatch");
    }

    #[test]
    fn split_channels_no_marker_returns_reasoning_only() {
        let full = "<|channel>thought\nOnly reasoning here, no final marker.";
        let (r, c) = split_channels(full);
        assert_eq!(r, "Only reasoning here, no final marker.");
        assert_eq!(c, "", "content should be empty when marker absent");
    }

    #[test]
    fn split_channels_strips_all_marker_variants() {
        // Both head and tail may contain stray markers from the model.
        let full = "<|channel|>thought\nsome thought<|channel|><channel|>the answer";
        let (r, c) = split_channels(full);
        assert!(!r.contains("<|channel"), "stray markers in reasoning");
        assert!(!c.contains("<channel|"), "stray markers in content");
        assert_eq!(r, "some thought");
        assert_eq!(c, "the answer");
    }

    #[test]
    fn split_channels_empty_input() {
        let (r, c) = split_channels("");
        assert_eq!(r, "");
        assert_eq!(c, "");
    }

    // --- parse_tool_calls ------------------------------------------------

    #[test]
    fn parse_tool_calls_bash_single_arg() {
        // Format from docs/plan.md: strings wrapped in <|"|>…<|"|>
        let text = r#"<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|>"#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments, json!({"command": "ls"}));
        assert_eq!(
            clean.trim(),
            "",
            "clean should have no leftover tool-call text"
        );
    }

    #[test]
    fn parse_tool_calls_multi_arg() {
        let text = r#"<|tool_call>call:write_file{path:<|"|>/tmp/x.txt<|"|>,content:<|"|>hello<|"|>}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(
            calls[0].arguments,
            json!({"path": "/tmp/x.txt", "content": "hello"})
        );
    }

    #[test]
    fn parse_tool_calls_nested_values() {
        // Nested object + array
        let text = r#"<|tool_call>call:query{filter:{field:<|"|>name<|"|>,values:[<|"|>a<|"|>,<|"|>b<|"|>]}}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "query");
        let args = &calls[0].arguments;
        assert_eq!(args["filter"]["field"], json!("name"));
        assert_eq!(args["filter"]["values"], json!(["a", "b"]));
    }

    #[test]
    fn parse_tool_calls_numeric_and_bool_args() {
        let text = r#"<|tool_call>call:set_config{timeout:30,enabled:true,ratio:0.5}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["timeout"], json!(30));
        assert_eq!(calls[0].arguments["enabled"], json!(true));
        // f64 comparison via as_f64
        let ratio = calls[0].arguments["ratio"].as_f64().unwrap();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn parse_tool_calls_no_tool_call_empty_vec() {
        let text = "Just a plain answer with no tool calls.";
        let (clean, calls) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(clean, text);
    }

    #[test]
    fn unexpected_container_delimiter_is_not_a_value() {
        assert!(
            parse_value(b"}", 0, 0).is_none(),
            "a value parser must reject a delimiter without consuming it"
        );
    }

    #[test]
    fn malformed_array_delimiter_does_not_loop() {
        let (_, calls) = parse_tool_calls("<|tool_call>call:x{a:[}]}<tool_call|>");
        assert!(calls.is_empty());
    }

    /// Where the delimiter rejection draws its line, asserted rather than described. A valueless
    /// entry costs THAT ENTRY and nothing else — the surrounding call survives with the pairs that
    /// did parse, which is the whole point: a model that fumbles one argument should not lose the
    /// other four. Only a malformation with no entry to drop (a stray delimiter inside an ARRAY)
    /// takes the call down, because there the loop cannot otherwise advance.
    #[test]
    fn a_valueless_entry_costs_only_itself() {
        let args = |t: &str| {
            let (_, calls) = parse_tool_calls(t);
            calls.first().map(|c| c.arguments.clone())
        };

        // The entry goes; its siblings, on both sides of it, stay.
        assert_eq!(
            args(r#"<|tool_call>call:x{a:,b:<|"|>We have something<|"|>}<tool_call|>"#),
            Some(serde_json::json!({ "b": "We have something" }))
        );
        assert_eq!(
            args("<|tool_call>call:x{a:1,b:,c:3}<tool_call|>"),
            Some(serde_json::json!({ "a": 1, "c": 3 }))
        );
        // Sole entry valueless → the call still fires, with no arguments.
        assert_eq!(
            args("<|tool_call>call:x{a:}<tool_call|>"),
            Some(serde_json::json!({}))
        );

        // Truncated rather than valueless: the value position runs off the end.
        assert_eq!(
            args("<|tool_call>call:x{a:<tool_call|>"),
            Some(serde_json::json!({ "a": null })),
            "a stream cut mid-value must still yield a best-effort call"
        );

        // Well-formed input is untouched, including empty containers.
        assert_eq!(
            args("<|tool_call>call:x{a:1,b:[1,2],c:{}}<tool_call|>"),
            Some(serde_json::json!({ "a": 1, "b": [1, 2], "c": {} }))
        );

        // No entry to drop and no way forward: the array arm still takes the whole call down,
        // which is what stops the unbounded-allocation loop.
        assert!(parse_tool_calls("<|tool_call>call:x{a:[}]}<tool_call|>")
            .1
            .is_empty());
    }

    /// Every container loop must advance its cursor on every path.
    ///
    /// This parser runs on model output, which in `infr serve` is steerable by whoever sent the
    /// request, so a body that leaves the cursor parked is not a parse quirk — it is a hang and an
    /// unbounded allocation reachable from a single HTTP request. That shipped once already
    /// (`{a:[}]}`, where the bareword arm returned at `}` without consuming it), and the object
    /// arm's valueless-entry skip is a second place that has to get this right.
    ///
    /// Cheap enough to be exhaustive rather than sampled, which is what makes it a regression test
    /// and not a fuzz run: every 6-byte body over the punctuation that drives those loops. There is
    /// nothing to assert about the RESULTS — the property is only that each call returns at all, so
    /// a regression shows up as this test hanging or exhausting memory rather than failing.
    #[test]
    fn delimiter_soup_always_terminates() {
        let alphabet = *b"{}[],:a\"";
        let mut n = 0u64;
        for w in 0..alphabet.len().pow(6) {
            let mut body = Vec::new();
            let mut x = w;
            for _ in 0..6 {
                body.push(alphabet[x % alphabet.len()]);
                x /= alphabet.len();
            }
            let text = format!(
                "<|tool_call>call:x{}<tool_call|>",
                String::from_utf8_lossy(&body)
            );
            let _ = parse_tool_calls(&text);
            n += 1;
        }
        assert_eq!(n, 262_144, "every case must have returned");
    }

    #[test]
    fn parse_tool_calls_multiple_calls() {
        let text = "<|tool_call>call:foo{x:<|\"|>1<|\"|>}<tool_call|> middle <|tool_call>call:bar{y:<|\"|>2<|\"|>}<tool_call|> suffix.";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "foo");
        assert_eq!(calls[1].name, "bar");
        assert!(clean.contains("middle"));
        assert!(clean.contains("suffix"));
    }

    #[test]
    fn parse_tool_calls_text_preserved_around_blocks() {
        let text = "Before.<|tool_call>call:ping{}<tool_call|>After.";
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert!(
            clean.contains("Before.") || clean.contains("After."),
            "surrounding text should survive: {clean:?}"
        );
    }

    // --- parse_hermes_tool_calls (Qwen3 / Hermes) ------------------------

    #[test]
    fn hermes_single_call_object_args() {
        let text =
            "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>";
        let (clean, calls) = parse_hermes_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments, json!({"command": "ls"}));
        assert_eq!(clean, "");
    }

    #[test]
    fn hermes_string_args_are_reparsed() {
        // Some templates emit arguments as an embedded JSON string.
        let text = r#"<tool_call>{"name":"get","arguments":"{\"id\":7}"}</tool_call>"#;
        let (_, calls) = parse_hermes_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({"id": 7}));
    }

    #[test]
    fn hermes_multiple_calls_and_surrounding_text() {
        let text = "ok <tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call> then <tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>";
        let (clean, calls) = parse_hermes_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert!(clean.contains("ok") && clean.contains("then"));
    }

    #[test]
    fn hermes_no_calls_passes_text_through() {
        let text = "Just an answer.";
        let (clean, calls) = parse_hermes_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(clean, "Just an answer.");
    }

    #[test]
    fn hermes_malformed_body_is_skipped() {
        let text = "<tool_call>not json</tool_call>";
        let (_, calls) = parse_hermes_tool_calls(text);
        assert!(calls.is_empty());
    }

    /// An unterminated `<tool_call>` (the turn ran out of budget mid-call) must not leak its opener
    /// into `clean` — the same guarantee `parse_tool_calls_dangling_opener_stripped` pins for the
    /// pipe-marker dialect. The hermes parser used to just `break`, so the user was shown the raw
    /// markup and the half-written JSON as if the model had said it.
    #[test]
    fn hermes_dangling_opener_stripped() {
        let text = "Answer text.<tool_call>{\"name\":\"foo\",\"argum";
        let (clean, calls) = parse_hermes_tool_calls(text);
        assert!(calls.is_empty());
        assert!(!clean.contains("tool_call"), "dangling opener: {clean:?}");
        assert_eq!(clean, "Answer text.");
    }

    /// A body that parses as NEITHER dialect still produces no call — that half is
    /// `hermes_malformed_body_is_skipped` — but the block's markup is now removed as well. A call
    /// we could not read is worth nothing; the raw `<tool_call>…</tool_call>` shown to the user as
    /// assistant prose is worse than nothing.
    #[test]
    fn hermes_unparseable_body_markup_is_stripped() {
        let text = "Sure.<tool_call>not json</tool_call>Done.";
        let (clean, calls) = parse_hermes_tool_calls(text);
        assert!(calls.is_empty());
        assert!(!clean.contains("tool_call"), "markup leaked: {clean:?}");
        assert!(!clean.contains("not json"), "body leaked: {clean:?}");
        assert_eq!(clean, "Sure.Done.");
    }

    // --- split_think -----------------------------------------------------

    #[test]
    fn think_splits_reasoning_and_content() {
        let (r, c) = split_think("<think>\nplanning\n</think>\nThe answer is 42.");
        assert_eq!(r, "planning");
        assert_eq!(c, "The answer is 42.");
    }

    #[test]
    fn think_prefilled_open_marker() {
        // Template prefilled `<think>`, so output begins mid-reasoning with only the close marker.
        let (r, c) = split_think("reasoning here</think>answer");
        assert_eq!(r, "reasoning here");
        assert_eq!(c, "answer");
    }

    #[test]
    fn think_absent_is_all_content() {
        let (r, c) = split_think("plain answer");
        assert_eq!(r, "");
        assert_eq!(c, "plain answer");
    }

    #[test]
    fn parses_xml_function_tool_call_qwen36() {
        let text = "<tool_call>\n<function=ls>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>";
        let (clean, calls) = parse_hermes_tool_calls(text);
        assert!(clean.is_empty(), "clean: {clean:?}");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ls");
        assert_eq!(calls[0].arguments["path"], serde_json::json!("."));
    }

    #[test]
    fn any_dialect_pipe_marker_gemma4() {
        let text = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";
        let (_, calls) = parse_any_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn any_dialect_xml_function_qwen36() {
        let text = "<tool_call>\n<function=ls>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_any_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ls");
    }

    #[test]
    fn any_dialect_bare_json_llama3() {
        let text = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}";
        let (clean, calls) = parse_any_tool_calls(text);
        assert!(clean.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], serde_json::json!("Paris"));
    }

    #[test]
    fn any_dialect_prose_with_json_is_not_a_call() {
        let text = "The config looks like {\"name\": \"x\"} but I did not call anything.";
        let (clean, calls) = parse_any_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(clean, text);
    }

    // --- parse_value corruption fixes ------------------------------------

    #[test]
    fn parse_tool_calls_translates_json_escapes() {
        let text = r#"<|tool_call>call:write{content:<|"|>a\nb\tcA<|"|>}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["content"], json!("a\nb\tcA"));
    }

    #[test]
    fn parse_tool_calls_surrogate_pair_escape() {
        // 😀 == U+1F600 (😀)
        let text = r#"<|tool_call>call:emoji{c:<|"|>😀<|"|>}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls[0].arguments["c"], json!("\u{1F600}"));
    }

    #[test]
    fn parse_tool_calls_non_finite_becomes_string() {
        let text = r#"<|tool_call>call:f{a:inf,b:NaN}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls[0].arguments["a"], json!("inf"));
        assert_eq!(calls[0].arguments["b"], json!("NaN"));
    }

    #[test]
    fn parse_tool_calls_multibyte_key_preserved() {
        let text = r#"<|tool_call>call:f{café:<|"|>x<|"|>}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls[0].arguments["café"], json!("x"));
    }

    #[test]
    fn parse_tool_calls_dangling_opener_stripped() {
        // Unterminated `<|tool_call>` (no close) must not leak opener markup into `clean`.
        let text = "Answer text.<|tool_call>call:foo{x:1}";
        let (clean, calls) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert!(
            !clean.contains("tool_call"),
            "dangling opener leaked: {clean:?}"
        );
        assert!(clean.contains("Answer text"));
    }

    // --- depth / scan-bound hardening ------------------------------------

    /// Deeply nested arguments must not recurse `parse_value` off the stack. In `infr serve`
    /// the model's output is steerable by the requesting client, so this body is one HTTP
    /// request away; without the ceiling it is a SIGSEGV that takes the whole process down.
    /// The call is dropped rather than reported with truncated arguments, and the markup
    /// still never leaks into `clean`.
    #[test]
    fn parse_tool_calls_deeply_nested_object_is_rejected() {
        let depth = 50_000;
        let mut body = String::from("<|tool_call>call:x{");
        for _ in 0..depth {
            body.push_str("a:{");
        }
        body.push_str("}".repeat(depth + 1).as_str());
        body.push_str("<tool_call|>");

        let (clean, calls) = parse_tool_calls(&body);
        assert!(
            calls.is_empty(),
            "over-deep arguments must not yield a ToolCall, got {calls:?}"
        );
        assert!(
            !clean.contains("tool_call"),
            "tool-call markup leaked into clean: {clean:?}"
        );
    }

    /// Same for arrays — `[` recurses through the other container arm, and a value that
    /// bailed without advancing the cursor there would spin the array loop forever, so this
    /// also pins that the rejection terminates rather than hangs.
    #[test]
    fn parse_tool_calls_deeply_nested_array_is_rejected() {
        let depth = 50_000;
        let mut body = String::from("<|tool_call>call:x{a:");
        body.push_str("[".repeat(depth).as_str());
        body.push_str("]".repeat(depth).as_str());
        body.push_str("}<tool_call|>");

        let (_, calls) = parse_tool_calls(&body);
        assert!(
            calls.is_empty(),
            "over-deep array arguments must not yield a ToolCall, got {calls:?}"
        );
    }

    /// Nesting that stays under the ceiling is untouched — the guard must not cost leniency
    /// for the one or two levels real tool schemas use.
    #[test]
    fn parse_tool_calls_moderate_nesting_still_parses() {
        let text = r#"<|tool_call>call:q{a:{b:{c:{d:[1,2,[3]]}}}}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["a"]["b"]["c"]["d"], json!([1, 2, [3]]));
    }

    /// A valueless entry (`{foo}`) followed by a colon somewhere later in the body must not
    /// make the key scan swallow everything up to that far-away colon. Before the bound the
    /// key came out as `foo} note` and the value parse resumed mid-token, producing a call
    /// that LOOKS well-formed but carries a garbage argument — worse than no parse, because
    /// the caller acts on it.
    #[test]
    fn parse_tool_calls_key_scan_stops_at_closing_brace() {
        let text = "<|tool_call>call:f{foo} note: see http://example.com<tool_call|>";
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(
            calls[0].arguments,
            json!({}),
            "no garbage key may be invented from text past the closing brace"
        );
    }

    /// The same bound at a `,`: a valueless entry is dropped and the following well-formed
    /// pair still parses, instead of the key scan running through the comma to a later colon.
    #[test]
    fn parse_tool_calls_key_scan_stops_at_comma() {
        let text = r#"<|tool_call>call:f{foo,bar:<|"|>ok<|"|>}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({"bar": "ok"}));
    }

    /// The scan is quote-aware, so a quoted key containing `,` or `}` is still one key.
    #[test]
    fn parse_tool_calls_quoted_key_with_punctuation() {
        let text = r#"<|tool_call>call:f{"a,b}c":<|"|>v<|"|>}<tool_call|>"#;
        let (_, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({"a,b}c": "v"}));
    }

    #[test]
    fn remove_spans_matches_manual_removal() {
        assert_eq!(
            remove_spans("abcDEFghiJKLmno", vec![(3, 6), (9, 12)]),
            "abcghimno"
        );
        assert_eq!(remove_spans("hello", vec![]), "hello");
        assert_eq!(remove_spans("xyz", vec![(0, 3)]), "");
    }
}
