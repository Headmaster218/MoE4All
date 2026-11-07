//! Grammar-constrained decoding via [`llguidance`] — the reliability tier of tool calling (llama.cpp
//! parity). A [`Constraint`] wraps an llguidance `Matcher` over the model's tokenizer and, each decode
//! step, (a) masks the logits to the grammatically-allowed tokens and (b) consumes the sampled token
//! to advance the grammar (plus any deterministically-forced "fast-forward" tokens). When tools are in
//! play the grammar forces syntactically-valid, schema-conforming `<tool_call>` JSON, so even tiny
//! models can't emit malformed calls.
//!
//! The tokenizer bridge uses [`ByteTokenizer::from_json_bytes`] (the tokenizer's serialized JSON)
//! rather than `from_tokenizer`, so it's immune to the `tokenizers` crate-version skew between infr
//! (0.20) and toktrie (0.21).
//!
//! KNOWN ISSUE — the GGUF→`tokenizers`→serialized-JSON tokenizer bridge is NOT canonical in toktrie's
//! sense, so llguidance's token-level `compute_mask` returns a SUPERSET that the byte-forcing parser
//! then rejects ("forced bytes: got '{'"). Mitigations applied: (1) [`NonCanonicalEnv`] forces the
//! canonical flag false; (2) the decode loop validate-before-commits each pick ([`Constraint::try_accept`])
//! and re-picks on rejection instead of erroring. These stop the crash, but on the live model the mask
//! is still inconsistent enough that the forced JSON can mask out entirely (empty body) — so the server
//! treats an empty/unparseable constrained call as a miss and FALLS BACK to unconstrained generation.
//! Net today: ≥1.7B models get reliable tool calls via the (unconstrained) auto path; the constrained
//! reliability tier for tiny models needs the principled fix — driving the grammar at BYTE level
//! (`Matcher::compute_ff_bytes`, the non-canonical-safe path) instead of token level. TODO.

use anyhow::{anyhow, Result};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{TokEnv, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use serde_json::Value;
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

/// Wraps a [`ByteTokenizerEnv`] to report itself NON-canonical. `ByteTokenizerEnv` inherits the trait
/// default `tokenize_is_canonical() == true`, which makes llguidance enable token-healing: it computes
/// a byte-forced prefix and returns a HEALED mask that's only exact for a truly-canonical tokenizer.
/// The GGUF→`tokenizers`→serialized-JSON bridge is NOT canonical, so the healed mask is a SUPERSET that
/// `consume_token` then rejects (the live "forced bytes: got '{'; applying '!'" bug). Forcing the flag
/// to `false` makes llguidance use the conservative, non-canonical-safe masking path where
/// `compute_mask` and `consume_token` agree.
struct NonCanonicalEnv(ByteTokenizerEnv);

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl TokenizerEnv for NonCanonicalEnv {
    fn tok_trie(&self) -> &TokTrie {
        self.0.tok_trie()
    }
    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.0.tokenize_bytes(s)
    }
    fn tokenize_is_canonical(&self) -> bool {
        false
    }
}

/// Build an llguidance [`TokEnv`] from infr's in-memory tokenizer. Serializes the tokenizer to JSON
/// and reparses it on toktrie's side (decoupling the `tokenizers` versions); `eos_ids` mark stop
/// tokens; `vocab` is the model's logit width so the token trie matches the logits exactly. Wrapped in
/// [`NonCanonicalEnv`] so the mask is consistent with `consume` (see that type's docs).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn build_tok_env(tokenizer: &Tokenizer, vocab: usize, eos_ids: &[u32]) -> Result<TokEnv> {
    let json = tokenizer
        .to_string(false)
        .map_err(|e| anyhow!("serialize tokenizer: {e}"))?;
    let mut bt = ByteTokenizer::from_json_bytes(json.as_bytes())
        .map_err(|e| anyhow!("byte tokenizer: {e}"))?;
    if !eos_ids.is_empty() {
        bt.set_eos_tokens(eos_ids);
    }
    let env = ByteTokenizerEnv::new(bt, Some(vocab)).map_err(|e| anyhow!("tok env: {e}"))?;
    Ok(Arc::new(NonCanonicalEnv(env)))
}

/// Force every id at or beyond `vocab` to -inf. Those ids sit past the token trie, so they can
/// never be grammar-legal, yet a padded-vocab model's logits row is wider than `vocab`. Split out
/// so the tail-masking is unit-testable without a live tokenizer/matcher.
fn mask_padding_tail(logits: &mut [f32], vocab: usize) {
    for l in logits.iter_mut().skip(vocab) {
        *l = f32::NEG_INFINITY;
    }
}

/// A live grammar constraint over a decode. Cheap-ish to construct (parser build); one per request.
pub struct Constraint {
    matcher: Matcher,
    vocab: usize,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Constraint {
    /// Construct a constraint for `grammar` over `tok_env`.
    pub fn new(tok_env: TokEnv, grammar: TopLevelGrammar) -> Result<Self> {
        let vocab = tok_env.tok_trie().vocab_size();
        let factory =
            ParserFactory::new_simple(&tok_env).map_err(|e| anyhow!("parser factory: {e}"))?;
        let factory = Arc::new(factory);
        let parser = factory
            .create_parser(grammar)
            .map_err(|e| anyhow!("create parser: {e}"))?;
        let matcher = Matcher::new(Ok(parser));
        Ok(Self { matcher, vocab })
    }

    /// Mask `logits` in place to the grammar's allowed tokens (disallowed → -inf). Then the caller
    /// samples as usual and feeds the chosen token to [`accept`](Self::accept).
    pub fn apply_mask(&mut self, logits: &mut [f32]) -> Result<()> {
        let mask = self.matcher.compute_mask().map_err(|e| anyhow!("{e}"))?;
        let n = logits.len().min(self.vocab);
        for (id, l) in logits.iter_mut().enumerate().take(n) {
            if !mask.is_allowed(id as u32) {
                *l = f32::NEG_INFINITY;
            }
        }
        // Padded-vocab GGUFs give a logits row WIDER than the token trie (`logits.len() > vocab`).
        // Those tail ids have no trie entry, so the mask can never allow them — but the loop above
        // stops at `vocab`, leaving them finite. Force them to -inf so a downstream full-slice
        // argmax/sample can't emit an out-of-grammar, out-of-trie token (fail CLOSED, not open).
        mask_padding_tail(logits, self.vocab);
        Ok(())
    }

    /// The grammar's deterministically-FORCED continuation at the current state (e.g. the literal
    /// `{`/`"`/`:` bytes a JSON object must emit). These must be consumed WITHOUT sampling — the right
    /// llguidance flow is to drain forced tokens first each step, then mask+sample only a free token.
    /// Returns empty when the next token is a real choice.
    pub fn forced(&mut self) -> Vec<u32> {
        self.matcher.compute_ff_tokens()
    }

    /// Try to advance the grammar by one freely-sampled `token`, validating BEFORE committing. Returns
    /// `true` if accepted, `false` if the token isn't actually grammar-legal (no state change). This is
    /// the guard against llguidance's token-healing returning a SUPERSET mask: `compute_mask` can allow
    /// a token that the parser then rejects (the GGUF tokenizer bridge isn't truly canonical), so the
    /// decode loop re-picks instead of failing the whole request.
    pub fn try_accept(&mut self, token: u32) -> Result<bool> {
        Ok(self
            .matcher
            .try_consume_tokens(&[token])
            .map_err(|e| anyhow!("{e}"))?
            == 1)
    }

    /// Whether the grammar is in an accepting state (a legal place to stop — EOS allowed here).
    pub fn accepting(&mut self) -> Result<bool> {
        self.matcher.is_accepting().map_err(|e| anyhow!("{e}"))
    }

    /// Advance the grammar by a run of forced tokens.
    pub fn consume(&mut self, tokens: &[u32]) -> Result<()> {
        self.matcher
            .consume_tokens(tokens)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Whether the grammar has reached an accepting stop state (no further tokens required).
    pub fn stopped(&mut self) -> bool {
        self.matcher.is_stopped()
    }
}

/// One CONSTRAINED decode step over `logits`: drain llguidance's deterministically-forced tokens
/// first (no sampling — the intended flow, keeps compute_mask/consume consistent); otherwise mask
/// the logits and pick a grammar-legal token with validate-before-commit (the healed mask can be a
/// SUPERSET on the non-canonical GGUF tokenizer bridge — a rejected candidate is dropped and
/// re-picked, never failed). EOS terminates only in an accepting state.
///
/// The free token is drawn with the SAME [`Sampler`]/`rng` the unconstrained decode path uses, so a
/// request that sets `temperature`/`top_p`/`seed` alongside `tool_choice` is honoured instead of
/// being silently forced to greedy. At `temp<=0` [`sample_logits`](crate::sampling::sample_logits)
/// collapses to `argmax`, so the deterministic tool-call path is byte-for-byte unchanged; a masked
/// (-inf) token has probability 0 in the softmax, so sampling can never leave the grammar.
///
/// Returns `(tokens emitted this step, grammar finished)`; an EMPTY step means the constrained
/// span is done (accepting EOS or mask exhausted). ONE implementation shared by the bespoke
/// decode loop and both seam decode paths.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn constrained_step(
    c: &mut Constraint,
    logits: &mut [f32],
    eos_ids: &[u32],
    sampler: crate::sampling::Sampler,
    rng: &mut u64,
) -> Result<(Vec<u32>, bool)> {
    let forced = c.forced();
    if !forced.is_empty() {
        c.consume(&forced)?;
        let done = c.stopped();
        return Ok((forced, done));
    }
    c.apply_mask(logits)?;
    loop {
        let cand = crate::sampling::sample_logits(logits, sampler, rng);
        if !logits[cand as usize].is_finite() {
            return Ok((Vec::new(), true)); // mask exhausted — nothing grammar-legal left
        }
        if eos_ids.contains(&cand) {
            if c.accepting()? {
                return Ok((Vec::new(), true)); // legal end of the constrained span
            }
            logits[cand as usize] = f32::NEG_INFINITY; // EOS not allowed yet
            continue;
        }
        if c.try_accept(cand)? {
            let done = c.stopped();
            return Ok((vec![cand], done));
        }
        logits[cand as usize] = f32::NEG_INFINITY; // superset member — drop + retry
    }
}

/// Build the grammar [`Constraint`] that FORCES a valid, schema-conforming tool call, for
/// `tool_choice` values that require one (`"required"`, or a named function). `None` for
/// `"auto"`/`"none"`/absent. Tokenizer-based (no `Llama` needed) so the seam backends share it.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn tool_constraint_for(
    tokenizer: &Tokenizer,
    vocab: usize,
    eos_ids: &[u32],
    tools: Option<&Value>,
    tool_choice: Option<&str>,
) -> Result<Option<Constraint>> {
    let Some(tools) = tools else {
        return Ok(None);
    };
    let (force, only): (bool, Option<&str>) = match tool_choice {
        Some("required") => (true, None),
        Some("auto") | Some("none") | None => (false, None),
        Some(name) => (true, Some(name.trim_matches('"'))),
    };
    if !force {
        return Ok(None);
    }
    let filtered;
    let tools = if let Some(name) = only {
        let arr = tools.as_array().cloned().unwrap_or_default();
        filtered = Value::Array(
            arr.into_iter()
                .filter(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        == Some(name)
                })
                .collect(),
        );
        &filtered
    } else {
        tools
    };
    let grammar = forced_tool_call_grammar(tools)?;
    let env = build_tok_env(tokenizer, vocab, eos_ids)?;
    Ok(Some(Constraint::new(env, grammar)?))
}

/// Build a JSON-schema grammar constraining the tool-call BODY — a single JSON object
/// `{"name": <one-of-the-tool-names>, "arguments": <that tool's parameter schema>}` (the union over
/// the request's tools). The caller prefills the `<tool_call>` opener and constrains only this body,
/// so the grammar stays pure JSON over normal byte tokens (no special-token / byte-grammar mismatch).
/// Used for `tool_choice: "required"` / a named tool — the model MUST emit one valid, schema-conforming
/// call. `tools` is the OpenAI `tools` array.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn forced_tool_call_grammar(tools: &Value) -> Result<TopLevelGrammar> {
    let arr = tools
        .as_array()
        .ok_or_else(|| anyhow!("`tools` is not an array"))?;
    // One JSON-schema alternative per tool: { name: const, arguments: <params> }.
    let mut alts: Vec<Value> = Vec::new();
    for t in arr {
        let f = t
            .get("function")
            .ok_or_else(|| anyhow!("tool missing `function`"))?;
        let name = f
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool missing `function.name`"))?;
        let params = f
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({"type": "object"}));
        alts.push(serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "const": name },
                "arguments": params,
            },
            "required": ["name", "arguments"],
            "additionalProperties": false,
        }));
    }
    let call_schema = serde_json::json!({ "anyOf": alts });
    Ok(TopLevelGrammar::from_json_schema(call_schema))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn qwen3_06b() -> Option<PathBuf> {
        let base =
            dirs_home()?.join(".cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots");
        std::fs::read_dir(&base)
            .ok()?
            .filter_map(|e| e.ok())
            .find_map(|e| {
                let p = e.path().join("Qwen3-0.6B-Q4_K_M.gguf");
                p.exists().then_some(p)
            })
    }
    fn dirs_home() -> Option<PathBuf> {
        std::env::var_os("HOME").map(PathBuf::from)
    }

    /// **Finding 2 — `apply_mask` must fail CLOSED past `vocab`.** On a padded-vocab model the
    /// logits row is wider than the token trie; ids `>= vocab` have no trie entry and must be forced
    /// to -inf so a downstream full-slice argmax can't emit an out-of-grammar token. Tests the tail
    /// masker directly (no live matcher needed).
    #[test]
    fn padding_tail_is_masked_to_neg_inf() {
        let mut logits = vec![1.0f32; 10];
        mask_padding_tail(&mut logits, 6);
        for (i, &l) in logits.iter().enumerate() {
            if i < 6 {
                assert_eq!(l, 1.0, "id {i} within vocab must be untouched");
            } else {
                assert!(
                    l.is_infinite() && l.is_sign_negative(),
                    "id {i} beyond vocab must be -inf"
                );
            }
        }
        // vocab >= len is a no-op (no padding tail).
        let mut exact = vec![2.0f32; 4];
        mask_padding_tail(&mut exact, 4);
        assert!(exact.iter().all(|&l| l == 2.0));
    }

    /// **Finding 3 — constrained decoding honours the `Sampler`.** At `temp==0` `constrained_step`
    /// stays deterministic (greedy within the mask); at `temp>0` with a fixed seed it draws — and a
    /// masked token has probability 0, so every emitted token is still grammar-legal. Model-gated
    /// (needs the real GGUF tokenizer); self-skips when the model isn't cached.
    #[test]
    fn constrained_step_honours_sampler_and_stays_in_grammar() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not cached");
            return;
        };
        let g = infr_gguf::Gguf::open(&path).expect("open gguf");
        let cfg = crate::Config::from_gguf(&g).expect("config");
        let tok = crate::build_tokenizer(&g).expect("tokenizer");
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                },
            },
        }]);

        // Deterministic (temp==0) run — collapses to greedy within the mask.
        let run = |sampler: crate::sampling::Sampler, seed: u64| -> Vec<u32> {
            let env = build_tok_env(&tok, cfg.vocab, &[cfg.eos]).expect("tok env");
            let grammar = forced_tool_call_grammar(&tools).expect("grammar");
            let mut c = Constraint::new(env, grammar).expect("constraint");
            let mut rng = seed;
            let mut out = Vec::new();
            for _ in 0..40 {
                if c.stopped() {
                    break;
                }
                let mut logits: Vec<f32> =
                    (0..cfg.vocab).map(|i| (i as f32 * 0.01).sin()).collect();
                let (step, done) =
                    constrained_step(&mut c, &mut logits, &[cfg.eos], sampler, &mut rng)
                        .expect("constrained step");
                if step.is_empty() {
                    break;
                }
                out.extend(step);
                if done {
                    break;
                }
            }
            out
        };

        let greedy = crate::sampling::Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        };
        // temp==0 is deterministic regardless of seed.
        assert_eq!(
            run(greedy, 1),
            run(greedy, 999),
            "temp==0 must stay greedy/deterministic"
        );

        // temp>0, seeded: reproducible, AND every emitted token is grammar-legal (a masked token
        // could never be drawn). We re-validate the run against a fresh grammar to prove the tokens
        // stayed inside the constraint.
        let sampled = crate::sampling::Sampler {
            temp: 1.2,
            top_k: 0,
            top_p: 0.95,
        };
        let a = run(sampled, 12345);
        let b = run(sampled, 12345);
        assert_eq!(a, b, "a seeded temp>0 constrained run must be reproducible");
        let env = build_tok_env(&tok, cfg.vocab, &[cfg.eos]).expect("tok env");
        let grammar = forced_tool_call_grammar(&tools).expect("grammar");
        let mut check = Constraint::new(env, grammar).expect("constraint");
        for &t in &a {
            if t == cfg.eos {
                continue;
            }
            assert!(
                check.try_accept(t).expect("try_accept"),
                "sampled token {t} escaped the grammar mask"
            );
        }
    }

    /// End-to-end: build the TokEnv from a real GGUF tokenizer, build the forced tool-call grammar,
    /// and verify the constraint actually restricts the first token to a strict subset of the vocab
    /// (the grammar must START a `<tool_call>` — not "anything goes"). Self-skips without the model.
    #[test]
    fn forced_grammar_constrains_first_token() {
        let Some(path) = qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not cached");
            return;
        };
        let g = infr_gguf::Gguf::open(&path).expect("open gguf");
        let cfg = crate::Config::from_gguf(&g).expect("config");
        let tok = crate::build_tokenizer(&g).expect("tokenizer");
        let env = build_tok_env(&tok, cfg.vocab, &[cfg.eos]).expect("tok env");

        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                },
            },
        }]);
        let grammar = forced_tool_call_grammar(&tools).expect("grammar");
        let mut c = Constraint::new(env, grammar).expect("constraint");

        // Drive the constraint, greedily picking the first allowed token each step. (This naive picker
        // wanders inside JSON strings and may never choose a closing quote, so we don't require full
        // termination — only that every step stays constrained and `accept` never rejects, i.e. the
        // mask and the parser agree. Full termination is covered by the live-model server test.)
        // Mirror the real decode loop: drain forced tokens first, else mask + argmax-pick a free token.
        let mut out: Vec<u32> = Vec::new();
        let mut drops = 0usize;
        for step in 0..60 {
            if c.stopped() {
                break;
            }
            let forced = c.forced();
            if !forced.is_empty() {
                c.consume(&forced).expect("consume forced");
                out.extend(forced);
                continue;
            }
            // ADVERSARIAL logits: bias toward high token ids (and away from the genuinely-valid low-id
            // delimiters). The healed mask can be a SUPERSET, so a naive argmax here would pick a
            // superset member that `consume_token` rejects — exactly the live-server failure. The
            // retry-on-reject loop (mirroring `decode_loop`) must recover and still produce valid JSON.
            let mut logits: Vec<f32> = (0..cfg.vocab).map(|i| i as f32).collect();
            c.apply_mask(&mut logits).expect("mask");
            let allowed = logits.iter().filter(|l| l.is_finite()).count();
            assert!(allowed > 0, "step {step}: grammar masked out EVERY token");
            assert!(
                allowed < cfg.vocab,
                "step {step}: grammar allowed everything"
            );
            let picked = loop {
                let cand = crate::sampling::argmax(&logits) as u32;
                assert!(
                    logits[cand as usize].is_finite(),
                    "step {step}: mask exhausted — every allowed token was rejected"
                );
                if c.try_accept(cand).expect("try_accept") {
                    break cand;
                }
                drops += 1;
                logits[cand as usize] = f32::NEG_INFINITY; // superset member — drop + retry
            };
            out.push(picked);
        }
        let text = tok.decode(&out, false).expect("decode");
        // `drops` counts superset-member rejections recovered via retry. With the canonical GGUF-built
        // tokenizer the mask is usually exact (drops==0); the SUPERSET only appears with the live model
        // tokenizer (the server test exercises that), so we don't assert drops>0 here — we assert the
        // retry path is sound: never deadlocks (handled in-loop) and keeps the output valid JSON.
        eprintln!("constrained prefix ({drops} superset drops): {text:?}");
        assert!(
            text.trim_start().starts_with('{'),
            "constrained output must stay valid JSON: {text:?}"
        );
    }
}
