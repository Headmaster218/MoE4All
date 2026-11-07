//! Streaming reasoning/content splitter — THE single place model output is separated into
//! reasoning vs answer (vs tool calls) as it streams. `infr run` (display styling), `infr serve`
//! (OpenAI deltas) and the history stripper all consume this module, so every thinking model is
//! exposed the same way on every surface; a new reasoning format (e.g. channel markers) lands
//! here once and everywhere at once.

/// Does this rendered prompt end with a `<think>` PREFILL (the template opened the reasoning
/// block itself — Qwen3.5 style — so the model's output starts MID-reasoning with only
/// the close marker)? Callers inject a synthetic `"<think>"` piece at the head of the stream when
/// this is true, so the streaming splitter, the display styler and the history stripper all see a
/// well-formed block through the one shared grammar.
pub fn prompt_prefills_think(prompt: &str) -> bool {
    prompt.trim_end().ends_with("<think>")
}

/// A streamed piece of a response.
#[derive(Clone, Debug)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    ToolCall { name: String, arguments: String },
}

/// Incremental splitter for the streaming server path. Accumulates the raw decoded text and, on each
/// piece, emits the newly-stable Reasoning (`<think>…</think>`) and Content deltas — holding back a
/// marker-length tail so a partial `<think>`/`</think>`/`<tool_call>` marker is never emitted. Once a
/// tool-call opener appears (and tool calls are allowed) it stops streaming content — matching every
/// dialect `finish()` parses (Hermes `<tool_call>`, the pipe form `<|tool_call>`, and a Llama-3
/// bare-JSON `{…}` body), so non-Hermes calls can't leak as content while also firing as a ToolCall.
/// `finish()` flushes the held-back tails and parses the buffered tool call(s) into ToolCall deltas.
pub struct ChatStream {
    raw: String,
    sent_r: usize, // reasoning bytes already emitted (offset within the reasoning region)
    sent_c: usize, // content bytes already emitted (offset within the content region)
    allow_tools: bool,
    // Per-marker cursors so `emit` doesn't re-`find` from offset 0 over the whole buffer on every
    // pushed piece (that was O(n²) across a response). Each caches the first hit once found and
    // otherwise resumes scanning from the previously-scanned tail.
    cur_think_open: Cursor,
    cur_think_close: Cursor,
    cur_tool: Cursor,  // Hermes/Qwen `<tool_call>`
    cur_pipe: Cursor,  // gemma4/E2B/DG `<|tool_call>`
    cur_final: Cursor, // channel-format `<channel|>`
}

/// A forward-only search for a fixed `needle` in an append-only buffer: once the needle is found its
/// byte offset is cached; until then, each call resumes scanning from just before the previously
/// scanned tail (so a needle straddling the append boundary is still caught).
#[derive(Default)]
struct Cursor {
    off: Option<usize>,
    scanned: usize,
}

impl Cursor {
    fn find(&mut self, raw: &str, needle: &str) -> Option<usize> {
        self.find_from(raw, needle, 0)
    }

    fn find_from(&mut self, raw: &str, needle: &str, region_start: usize) -> Option<usize> {
        if let Some(off) = self.off {
            if off >= region_start {
                return Some(off);
            }
            self.scanned = region_start;
        }
        self.off = None;
        let mut start = self
            .scanned
            .saturating_sub(needle.len().saturating_sub(1))
            .max(region_start);
        while start > region_start && !raw.is_char_boundary(start) {
            start -= 1;
        }
        if let Some(p) = raw[start..].find(needle) {
            self.off = Some(start + p);
        }
        self.scanned = raw.len();
        self.off
    }
}

impl ChatStream {
    pub fn new(allow_tools: bool) -> Self {
        Self {
            raw: String::new(),
            sent_r: 0,
            sent_c: 0,
            allow_tools,
            cur_think_open: Cursor::default(),
            cur_think_close: Cursor::default(),
            cur_tool: Cursor::default(),
            cur_pipe: Cursor::default(),
            cur_final: Cursor::default(),
        }
    }

    pub fn push(&mut self, piece: &str, on_delta: &mut dyn FnMut(Delta)) {
        self.raw.push_str(piece);
        self.emit(false, on_delta);
    }

    pub fn finish(&mut self, on_delta: &mut dyn FnMut(Delta)) {
        self.emit(true, on_delta); // flush the held-back tails (no marker can still be forming)
        if self.allow_tools {
            let (_r, body) = crate::split_reasoning(&self.raw);
            // Dialect-aware: hermes JSON + qwen3.5/3.6 XML + gemma pipe-marker + llama3 bare JSON
            // (hermes-only here silently dropped every non-qwen3 model's calls over serve).
            let (_content, calls) = crate::parse_any_tool_calls(&body);
            for call in calls {
                on_delta(Delta::ToolCall {
                    name: call.name,
                    arguments: serde_json::to_string(&call.arguments)
                        .unwrap_or_else(|_| "{}".to_string()),
                });
            }
        }
    }

    /// The earliest tool-call opener at/after `cs`, across every dialect `finish()` parses — Hermes
    /// `<tool_call>`, the pipe form `<|tool_call>`, and (when the content region begins with `{`) a
    /// Llama-3 bare-JSON call. Returns `None` when tools aren't allowed or no opener is present, so
    /// the content region runs to the buffer end. Matching `finish()`'s dialect set here is what
    /// stops non-Hermes calls from leaking as `Delta::Content` while also firing as `Delta::ToolCall`.
    fn tool_open_at(&mut self, cs: usize, final_flush: bool) -> Option<usize> {
        if !self.allow_tools {
            return None;
        }
        const TL: &str = "<tool_call>";
        const PIPE: &str = "<|tool_call>";
        let hermes = self.cur_tool.find_from(&self.raw, TL, cs);
        let pipe = self.cur_pipe.find_from(&self.raw, PIPE, cs);
        let bare = self.bare_json_open(cs, final_flush);
        [hermes, pipe, bare].into_iter().flatten().min()
    }

    /// Position of a bare-JSON (`{…}`) tool-call opener in the content region starting at `cs`, or
    /// `None`. Scoped to the leading `{` of the region (skipping only whitespace) so ordinary prose
    /// that merely *contains* a brace is never suppressed. During streaming a leading `{` is held
    /// back optimistically; on the final flush it's confirmed a real call via
    /// [`crate::tools::parse_bare_json_call`] so a leading brace that isn't a call still streams as
    /// content instead of being dropped.
    fn bare_json_open(&self, cs: usize, final_flush: bool) -> Option<usize> {
        let bytes = self.raw.as_bytes();
        let mut i = cs;
        while i < bytes.len() && matches!(bytes[i], b' ' | b'\n' | b'\t' | b'\r') {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'{' {
            return None;
        }
        if final_flush && crate::tools::parse_bare_json_call(&self.raw[i..]).is_none() {
            return None; // leading `{` but not actually a call — let it stream as content
        }
        Some(i)
    }

    fn emit(&mut self, final_flush: bool, on_delta: &mut dyn FnMut(Delta)) {
        const TO: &str = "<think>";
        const TC: &str = "</think>";
        // Channel-format head (E2B/gpt-oss: `<|channel>thought…<channel|>answer`): reasoning runs
        // from after the thought marker to the final-answer marker, content after it. While the
        // head could still be a forming marker, hold everything back.
        use crate::tools::{channel_thought_start, CHANNEL_THOUGHT_MARKERS};
        const FINAL: &str = "<channel|>";
        if let Some(hm) = channel_thought_start(&self.raw) {
            let (r_end, hold, c_start) = match self.cur_final.find(&self.raw, FINAL) {
                Some(f) => (f, false, Some(f + FINAL.len())),
                None => (self.raw.len(), !final_flush, None),
            };
            emit_region(&self.raw, hm, r_end, hold, &mut self.sent_r, true, on_delta);
            if let Some(cs) = c_start {
                let tool_open = self.tool_open_at(cs, final_flush);
                let (c_end, hold) = match tool_open {
                    Some(t) if t >= cs => (t, false),
                    _ => (self.raw.len(), !final_flush),
                };
                emit_region(
                    &self.raw,
                    cs,
                    c_end,
                    hold,
                    &mut self.sent_c,
                    false,
                    on_delta,
                );
            }
            return;
        }
        if !final_flush
            && CHANNEL_THOUGHT_MARKERS.iter().any(|m| {
                m.as_bytes()
                    .starts_with(&self.raw.as_bytes()[..self.raw.len().min(m.len())])
            })
            && self.raw.len() < CHANNEL_THOUGHT_MARKERS[0].len()
        {
            return; // head could still become a channel marker — hold
        }
        let think_open = self.cur_think_open.find(&self.raw, TO);
        let think_close = self.cur_think_close.find(&self.raw, TC);
        // Reasoning region: between `<think>` and `</think>` (or end, while still thinking).
        if let Some(to) = think_open {
            let rs = to + TO.len();
            let (r_end, hold) = match think_close {
                Some(tc) => (tc, false),
                None => (self.raw.len(), !final_flush),
            };
            emit_region(&self.raw, rs, r_end, hold, &mut self.sent_r, true, on_delta);
        }
        // Content region: after `</think>` (or from the start when there's no `<think>` at all), up to
        // a tool-call opener (whose block is buffered, not streamed) or the end.
        let c_start = match think_close {
            Some(tc) => Some(tc + TC.len()),
            None if think_open.is_none() => Some(0),
            None => None,
        };
        if let Some(cs) = c_start {
            let tool_open = self.tool_open_at(cs, final_flush);
            let (c_end, hold) = match tool_open {
                Some(t) if t >= cs => (t, false),
                _ => (self.raw.len(), !final_flush),
            };
            emit_region(
                &self.raw,
                cs,
                c_end,
                hold,
                &mut self.sent_c,
                false,
                on_delta,
            );
        }
    }
}

/// Emit the not-yet-sent slice of `raw[region_start .. region_end]` (past `*sent` bytes), holding back
/// a marker-length tail when `hold` (so a partial marker isn't emitted mid-stream), clamped to a UTF-8
/// boundary. Advances `*sent`.
fn emit_region(
    raw: &str,
    region_start: usize,
    region_end: usize,
    hold: bool,
    sent: &mut usize,
    reasoning: bool,
    on_delta: &mut dyn FnMut(Delta),
) {
    const HOLD: usize = 12; // > the longest marker, so a partial one never streams
    let abs = region_start + *sent;
    if abs >= region_end {
        return;
    }
    let mut end = if hold {
        region_end.saturating_sub(HOLD).max(abs)
    } else {
        region_end
    };
    while end > abs && !raw.is_char_boundary(end) {
        end -= 1;
    }
    if end <= abs {
        return;
    }
    let text = &raw[abs..end];
    if text.is_empty() {
        return;
    }
    on_delta(if reasoning {
        Delta::Reasoning(text.to_owned())
    } else {
        Delta::Content(text.to_owned())
    });
    *sent += text.len();
}

#[cfg(test)]
mod chat_stream_tests {
    use super::*;

    /// Feed `pieces` through a `ChatStream` and collect every emitted delta (streaming + finish).
    fn run(pieces: &[&str], allow_tools: bool) -> (String, String, Vec<(String, String)>) {
        let mut out: Vec<Delta> = Vec::new();
        let mut s = ChatStream::new(allow_tools);
        {
            let mut od = |d: Delta| out.push(d);
            for p in pieces {
                s.push(p, &mut od);
            }
        }
        s.finish(&mut |d: Delta| out.push(d));
        let (mut r, mut c, mut t) = (String::new(), String::new(), Vec::new());
        for d in &out {
            match d {
                Delta::Reasoning(x) => r.push_str(x),
                Delta::Content(x) => c.push_str(x),
                Delta::ToolCall { name, arguments } => t.push((name.clone(), arguments.clone())),
            }
        }
        (r, c, t)
    }

    fn check_fragmentation(
        text: &str,
        allow_tools: bool,
    ) -> (String, String, Vec<(String, String)>) {
        let expected = run(&[text], allow_tools);
        for split in text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()))
        {
            assert_eq!(
                run(&[&text[..split], &text[split..]], allow_tools),
                expected,
                "split at {split}"
            );
        }
        let chars: Vec<&str> = text
            .char_indices()
            .map(|(i, c)| &text[i..i + c.len_utf8()])
            .collect();
        assert_eq!(run(&chars, allow_tools), expected, "character fragments");
        expected
    }

    #[test]
    fn channel_tool_examples_are_reasoning_not_calls() {
        for head in crate::tools::CHANNEL_THOUGHT_MARKERS {
            let text = format!("{head}\nExample <|tool_call>call:example{{}}<tool_call|><channel|>No call is needed.\u{4e2d}");
            let (r, c, calls) = check_fragmentation(&text, true);
            assert!(r.contains("call:example"));
            assert_eq!(c.trim(), "No call is needed.\u{4e2d}");
            assert!(calls.is_empty());
            assert_eq!(
                crate::split_reasoning(&text),
                (r.trim().into(), c.trim().into())
            );
        }
    }

    #[test]
    fn unfinished_channel_reasoning_cannot_dispatch_tools() {
        let (r, c, calls) = check_fragmentation(
            "<|channel>thought\n<tool_call>{\"name\":\"example\",\"arguments\":{}}</tool_call>",
            true,
        );
        assert!(r.contains("example"));
        assert!(c.is_empty());
        assert!(calls.is_empty());
    }

    #[test]
    fn tools_after_reasoning_examples_are_buffered_and_dispatched_once() {
        for (head, close) in [("<|channel>thought", "<channel|>"), ("<think>", "</think>")] {
            for call in [
                "<|tool_call>call:real{}<tool_call|>",
                "<tool_call>{\"name\":\"real\",\"arguments\":{}}</tool_call>",
                "{\"name\":\"real\",\"parameters\":{}}",
            ] {
                let text = format!("{head}<|tool_call>call:example{{}}<tool_call|><tool_call>{{\"name\":\"example\",\"arguments\":{{}}}}</tool_call>{close}{call}");
                let (r, c, calls) = check_fragmentation(&text, true);
                assert!(r.contains("example"));
                assert!(c.trim().is_empty(), "call leaked as content: {c}");
                assert_eq!(calls, [("real".into(), "{}".into())]);
                let (_, body) = crate::split_reasoning(&text);
                let (_, batch_calls) = crate::parse_any_tool_calls(&body);
                assert_eq!(batch_calls.len(), 1);
                assert_eq!(batch_calls[0].name, "real");
            }
        }
    }

    #[test]
    fn channel_markers_in_content_remain_literal() {
        for text in [
            "Example: `<|channel>thought ... <channel|>` is literal text.",
            "<|channel>thought\nplan<channel|>Use `<channel|>` and `<|channel>thought` literally.",
        ] {
            let (r, c, calls) = check_fragmentation(text, true);
            assert!(calls.is_empty());
            assert_eq!(
                crate::split_reasoning(text),
                (r.trim().into(), c.trim().into())
            );
        }
    }

    #[test]
    fn channel_markers_in_tool_arguments_are_preserved() {
        let text = "<|channel>thought\nplan<channel|><tool_call>{\"name\":\"real\",\"arguments\":{\"text\":\"<|channel>thought <channel|>\"}}</tool_call>";
        let (_, c, calls) = check_fragmentation(text, true);
        assert!(c.trim().is_empty());
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["text"], "<|channel>thought <channel|>");
    }

    #[test]
    fn channel_tools_disabled_remain_final_content() {
        let text = "<|channel>thought\nplan<channel|><|tool_call>call:real{}<tool_call|>";
        let (r, c, calls) = check_fragmentation(text, false);
        assert_eq!(r.trim(), "plan");
        assert_eq!(c, "<|tool_call>call:real{}<tool_call|>");
        assert!(calls.is_empty());
    }

    #[test]
    fn cursor_skips_cached_matches_before_the_content_region() {
        let mut cursor = Cursor::default();
        let raw = "<tool_call>reasoning<tool_call>answer";
        assert_eq!(cursor.find(raw, "<tool_call>"), Some(0));
        let content_start = "<tool_call>reasoning".len();
        assert_eq!(
            cursor.find_from(raw, "<tool_call>", content_start),
            Some(content_start)
        );
    }

    #[test]
    fn streams_channel_thought_then_final() {
        // E2B/gpt-oss channel format: reasoning after the thought marker, content after the
        // final-answer marker — streamed as Reasoning/Content like the <think> form.
        let (r, c, t) = run(
            &[
                "<|channel>th",
                "ought\nreaso",
                "ning<chan",
                "nel|>the answer",
            ],
            false,
        );
        assert_eq!(r.trim(), "reasoning");
        assert_eq!(c.trim(), "the answer");
        assert!(t.is_empty());
    }

    #[test]
    fn prefilled_think_via_injected_opener() {
        // Template-prefilled thinking (Qwen3.5): the caller injects "<think>" before the model's
        // mid-reasoning output; the splitter then treats the head as Reasoning.
        let (r, c, t) = run(&["<think>", "reasoning here", "</think>", "answer"], false);
        assert_eq!(r.trim(), "reasoning here");
        assert_eq!(c.trim(), "answer");
        assert!(t.is_empty());
    }

    #[test]
    fn streams_think_then_content() {
        let (r, c, t) = run(
            &["<think>", "reason", "ing", "</think>", "the ", "answer"],
            true,
        );
        assert_eq!(r.trim(), "reasoning");
        assert_eq!(c.trim(), "the answer");
        assert!(t.is_empty());
    }

    #[test]
    fn plain_content_no_think() {
        let (r, c, _) = run(&["hello ", "world, this is a longer reply"], true);
        assert!(r.trim().is_empty());
        assert_eq!(c.trim(), "hello world, this is a longer reply");
    }

    #[test]
    fn tool_call_buffered_and_parsed() {
        let (r, _c, t) = run(
            &[
                "<think>plan</think>",
                "<tool_call>\n{\"name\": \"run_bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>",
            ],
            true,
        );
        assert_eq!(r.trim(), "plan");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].0, "run_bash");
        assert!(t[0].1.contains("ls"));
    }

    #[test]
    fn tool_choice_none_keeps_tool_text_as_content() {
        // allow_tools=false → a <tool_call> block is NOT extracted; it stays content.
        let (_r, c, t) = run(&["<tool_call>{\"name\":\"x\"}</tool_call>"], false);
        assert!(t.is_empty());
        assert!(c.contains("tool_call"));
    }

    #[test]
    fn pipe_form_tool_call_not_streamed_as_content() {
        // gemma4/E2B/DG pipe form has no `<tool_call>` substring — it must still gate content
        // and fire exactly once as a ToolCall (not leak as Content AND duplicate).
        let (_r, c, t) = run(
            &[
                "<|tool_",
                "call>call:get_weather{city:<|\"|>Par",
                "is<|\"|>}",
                "<tool_call|>",
            ],
            true,
        );
        assert!(
            !c.contains("tool_call") && !c.contains("get_weather"),
            "pipe markup leaked as content: {c:?}"
        );
        assert_eq!(t.len(), 1, "expected exactly one tool call, got {t:?}");
        assert_eq!(t[0].0, "get_weather");
    }

    #[test]
    fn bare_json_tool_call_not_streamed_as_content() {
        // Llama-3 bare-JSON form: the whole body is one JSON object — no markers at all.
        let (_r, c, t) = run(
            &[
                "{\"name\": \"get_weather\", ",
                "\"parameters\": {\"city\": \"Paris\"}}",
            ],
            true,
        );
        assert!(c.trim().is_empty(), "bare json leaked as content: {c:?}");
        assert_eq!(t.len(), 1, "expected exactly one tool call, got {t:?}");
        assert_eq!(t[0].0, "get_weather");
    }

    #[test]
    fn content_with_stray_brace_not_swallowed() {
        // Ordinary prose that merely contains a brace mid-text must NOT be treated as a call.
        let (_r, c, t) = run(
            &[
                "The config is {",
                "\"a\": 1} and that is the whole story here, nothing more.",
            ],
            true,
        );
        assert!(t.is_empty(), "stray brace misparsed as tool call: {t:?}");
        assert!(c.contains("The config is"), "content lost: {c:?}");
        assert!(c.contains("whole story"), "content truncated: {c:?}");
    }

    #[test]
    fn leading_brace_non_call_survives_as_content() {
        // A leading `{` that is NOT a valid bare-JSON call must be flushed as content, not dropped.
        let (_r, c, t) = run(
            &["{not json here, just prose that opens with a brace}"],
            true,
        );
        assert!(t.is_empty(), "non-call misparsed: {t:?}");
        assert!(c.contains("not json here"), "content dropped: {c:?}");
    }
}
