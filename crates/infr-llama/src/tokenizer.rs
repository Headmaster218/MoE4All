//! GGUF-embedded tokenizer construction (byte-level BPE + SentencePiece).
//! Mechanically split out of `lib.rs` (no logic change).
use crate::{LLAMA4_PRE_RE, QWEN2_PRE_RE};
use anyhow::{anyhow, bail, Context, Result};
use infr_core::loader::{MetaValue, Metadata};
use infr_core::WeightSource;
use infr_gguf::Gguf;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, DecoderWrapper, SplitDelimiterBehavior, Tokenizer};

/// The `tokenizer.ggml.merges` line grammar: each entry is `"left right"`, ONE space, and the
/// right half may itself contain spaces (SPM pieces are `▁`-prefixed, but a byte-level BPE piece
/// can be a literal space) — hence `splitn(2, ' ')`, not `split`. Entries that are not strings, or
/// that have no space at all, are dropped rather than erroring: a merge we cannot read is one the
/// BPE simply never applies, which degrades tokenization instead of refusing to load the model.
///
/// Single-sources the grammar for both tokenizer families. `build_tokenizer` (gpt2 BPE) and
/// `build_spm_tokenizer` (gemma4's explicit merges) each spelled this expression out; a fix to one
/// copy — the `splitn` bound in particular — silently did not reach the other, and the two families
/// would then disagree about what a merge line means while both claiming to read the same key.
fn parse_merges(arr: &[MetaValue]) -> Vec<(String, String)> {
    arr.iter()
        .filter_map(|m| {
            let s = m.as_str()?;
            let mut it = s.splitn(2, ' ');
            Some((it.next()?.to_string(), it.next()?.to_string()))
        })
        .collect()
}

/// Register `tokenizer.ggml.token_type` 3 (CONTROL) and 4 (USER_DEFINED) tokens on `tok`, matching
/// HF: control tokens are SPECIAL (`AddedToken::from(s, true)`), user-defined ones are normal added
/// tokens. Both encode atomically; only the special ones are dropped by
/// `decode(.., skip_special=true)`, which is why `<think>`/`</think>` (type 4) must NOT be special —
/// the reasoning block has to stay visible in the output for the reasoning split to find it.
///
/// Absent `token_type` ⇒ nothing to register. The empty-vec guards are kept because `add_tokens` /
/// `add_special_tokens` on an empty slice still bump the tokenizer's added-vocab bookkeeping, and
/// the ORDER (added before specials) is load-bearing: a string appearing in both lists must end up
/// special, and the later call wins.
///
/// This is the single copy of a policy that used to be written out verbatim in both
/// [`build_tokenizer`] and [`build_spm_tokenizer`]. The two copies were already drifting in their
/// comments about which type meant what, and any new token_type the GGUF spec grows would have had
/// to be added twice — with the SPM family (which loads llama/gemma) the easy one to miss.
fn register_token_types(md: &Metadata, toks: &[MetaValue], tok: &mut Tokenizer) {
    let Some(types) = md
        .get("tokenizer.ggml.token_type")
        .and_then(MetaValue::as_arr)
    else {
        return;
    };
    let mut specials = Vec::new();
    let mut added = Vec::new();
    for (i, ty) in types.iter().enumerate() {
        let Some(s) = toks.get(i).and_then(MetaValue::as_str) else {
            continue;
        };
        match ty.as_u64() {
            Some(3) => specials.push(AddedToken::from(s.to_string(), true)),
            Some(4) => added.push(AddedToken::from(s.to_string(), false)),
            _ => {}
        }
    }
    if !added.is_empty() {
        tok.add_tokens(&added);
    }
    if !specials.is_empty() {
        tok.add_special_tokens(&specials);
    }
}

/// Build an HF `Tokenizer` from the GGUF's embedded vocab (`tokenizer.ggml.*`). Supports the
/// GPT-2 byte-level BPE family (Qwen/Llama-3/SmolLM etc., `tokenizer.ggml.model == "gpt2"`):
/// vocab from `.tokens`, merges from `.merges`, ByteLevel pre-tokenizer + decoder, and control /
/// user-defined tokens (token_type 3/4, e.g. `<|im_start|>`) registered as special so they encode
/// atomically. SentencePiece (`model == "llama"`) isn't built here — pass a `tokenizer.json`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn build_tokenizer(g: &Gguf) -> Result<Tokenizer> {
    let md = g.metadata();
    let model = md.str("tokenizer.ggml.model").unwrap_or("");
    match model {
        "gpt2" => {}
        // SentencePiece (llama/gemma3/gemma4): a byte-fallback BPE with a Metaspace (▁) word-boundary
        // scheme. gemma4 ships explicit merges; llama/gemma3 reconstruct them from the token scores.
        "llama" | "gemma4" => return build_spm_tokenizer(g),
        other => bail!(
            "can't derive a tokenizer from tokenizer.ggml.model={other:?} \
             (only gpt2 BPE / llama SPM); pass a tokenizer.json sidecar instead"
        ),
    }
    let toks = md
        .get("tokenizer.ggml.tokens")
        .and_then(MetaValue::as_arr)
        .context("gguf missing tokenizer.ggml.tokens")?;
    let vocab: std::collections::HashMap<String, u32> = toks
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.as_str().map(|s| (s.to_string(), i as u32)))
        .collect();
    let merges = parse_merges(
        md.get("tokenizer.ggml.merges")
            .and_then(MetaValue::as_arr)
            .context("gguf missing tokenizer.ggml.merges")?,
    );
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!("build bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    let add_prefix = matches!(
        md.get("tokenizer.ggml.add_space_prefix"),
        Some(MetaValue::Bool(true))
    );
    let pre = md.str("tokenizer.ggml.pre").unwrap_or("default");
    // qwen2/llama4 both split on a regex before ByteLevel (matching HF); only the pattern differs.
    let split_re = match pre {
        "qwen2" => Some(QWEN2_PRE_RE),
        "llama4" => Some(LLAMA4_PRE_RE),
        _ => None,
    };
    if let Some(re) = split_re {
        // Sequence[ Split(pre regex, Isolated), ByteLevel(use_regex=false) ] — matches HF.
        let split = Split::new(
            SplitPattern::Regex(re.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| anyhow!("split pretokenizer: {e}"))?;
        let seq = PreSequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ]);
        tok.with_pre_tokenizer(Some(seq));
    } else {
        tok.with_pre_tokenizer(Some(ByteLevel::new(add_prefix, true, true)));
    }
    tok.with_decoder(Some(ByteLevelDecoder::default()));
    // Control (type 3, e.g. <|im_end|>) as SPECIAL, user-defined (type 4, e.g. <think>) as normal
    // added tokens — see [`register_token_types`].
    register_token_types(md, toks, &mut tok);
    Ok(tok)
}

/// The global ordering of reconstructed SPM merge candidates, keyed by `(score, id_l, id_r)`:
/// score DESCENDING (a higher GGUF score is an EARLIER merge), ties broken by `(id_l, id_r)`
/// ASCENDING. That exact ordering is load-bearing — it is what reproduces HF `SpmConverter`'s
/// merge ranking, and flipping either half silently re-tokenizes every prompt on a llama/gemma3
/// GGUF (no merges array, so the ranks come from here). Factored out of
/// [`build_spm_tokenizer`] only so a test can pin it without a model file.
///
/// Scores compare with [`f64::total_cmp`], NOT `partial_cmp().unwrap_or(Equal)`. A NaN score — a
/// corrupt or hand-edited `tokenizer.ggml.scores` — makes the `unwrap_or(Equal)` comparator
/// non-transitive (NaN is "equal" to everything, but the things it ties with are not equal to each
/// other), and Rust's sort DETECTS that and panics with "user-provided comparison function does not
/// correctly implement a total order". So a bad score used to abort model load inside the sort with
/// an error about our comparator instead of about the file. `total_cmp` is a genuine total order
/// over every f64 including NaN, so the sort completes and the corrupt entry merely ranks
/// consistently (NaN sorts to one end) rather than taking the process down.
fn spm_merge_order(a: (f64, u32, u32), b: (f64, u32, u32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2)))
}

/// Build a SentencePiece (Unigram) tokenizer from a GGUF's embedded vocab (`tokenizer.ggml.model
/// == "llama"`, used by llama/gemma). The token strings + `scores` become the Unigram lattice;
/// `<0xXX>` byte tokens (token_type 6) are handled by Unigram byte-fallback; CONTROL tokens
/// (type 3, e.g. `<bos>`/`<start_of_turn>`) register as special so they encode atomically. The
/// Metaspace replacement (▁) maps spaces; `add_space_prefix` controls the leading dummy space.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn build_spm_tokenizer(g: &Gguf) -> Result<Tokenizer> {
    let md = g.metadata();
    let toks = md
        .get("tokenizer.ggml.tokens")
        .and_then(MetaValue::as_arr)
        .context("gguf missing tokenizer.ggml.tokens")?;
    let token_strs: Vec<String> = toks
        .iter()
        .map(|t| t.as_str().unwrap_or("").to_string())
        .collect();
    let vocab: std::collections::HashMap<String, u32> = token_strs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u32))
        .collect();
    // Merge list for the greedy BPE. gemma4 ships explicit `merges` ("left right", ▁ for spaces);
    // llama/gemma3 don't, so reconstruct them from the token scores the same way HF's SpmConverter
    // builds `LlamaTokenizerFast` (the GGUF scores are negative merge RANKS, not unigram log-probs —
    // a Unigram model would maximize their sum and wrongly split common words). For every piece, each
    // split into two existing pieces is a candidate merge, globally ordered by the merged piece's
    // score (descending = earliest), ties broken by piece ids; greedy BPE over these reproduces SPM.
    let merges: Vec<(String, String)> =
        if let Some(arr) = md.get("tokenizer.ggml.merges").and_then(MetaValue::as_arr) {
            parse_merges(arr)
        } else {
            let scores = md
                .get("tokenizer.ggml.scores")
                .and_then(MetaValue::as_arr)
                .context("gguf needs tokenizer.ggml.merges or .scores for the SPM tokenizer")?;
            // (score, id_l, id_r, l, r) per candidate — global sort by score desc, then (id_l, id_r).
            let mut cand: Vec<(f64, u32, u32, &str, &str)> = Vec::new();
            for (i, piece) in token_strs.iter().enumerate() {
                if piece.len() < 2 {
                    continue;
                }
                let score = scores.get(i).and_then(MetaValue::as_f64).unwrap_or(0.0);
                for (b, _) in piece.char_indices().skip(1) {
                    let (l, r) = piece.split_at(b);
                    if let (Some(&il), Some(&ir)) = (vocab.get(l), vocab.get(r)) {
                        cand.push((score, il, ir, l, r));
                    }
                }
            }
            cand.sort_by(|a, b| spm_merge_order((a.0, a.1, a.2), (b.0, b.1, b.2)));
            cand.into_iter()
                .map(|(_, _, _, l, r)| (l.to_string(), r.to_string()))
                .collect()
        };
    let unk = md
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(MetaValue::as_u64)
        .and_then(|i| token_strs.get(i as usize).cloned())
        .unwrap_or_else(|| "<unk>".to_string());
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token(unk)
        .byte_fallback(true)
        .fuse_unk(true)
        .build()
        .map_err(|e| anyhow!("build spm bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    // SPM: spaces → ▁. add_space_prefix=true prepends a dummy ▁ (PrependScheme::First); gemma3
    // sets it false. `split` keeps Metaspace's word splitting on the replacement char.
    let add_prefix = matches!(
        md.get("tokenizer.ggml.add_space_prefix"),
        Some(MetaValue::Bool(true))
    );
    let scheme = if add_prefix {
        PrependScheme::First
    } else {
        PrependScheme::Never
    };
    tok.with_pre_tokenizer(Some(Metaspace::new('▁', scheme, true)));
    // Decode: reassemble byte-fallback bytes, fuse, then map ▁→space (Metaspace decoder).
    let dec = DecoderSequence::new(vec![
        DecoderWrapper::ByteFallback(ByteFallback::default()),
        DecoderWrapper::Fuse(Fuse::default()),
        DecoderWrapper::Metaspace(Metaspace::new('▁', scheme, true)),
    ]);
    tok.with_decoder(Some(dec));
    // CONTROL tokens (type 3, e.g. <bos>/<start_of_turn>/<end_of_turn>) encode atomically as
    // special; USER_DEFINED (type 4) as normal added tokens — see [`register_token_types`].
    register_token_types(md, toks, &mut tok);
    Ok(tok)
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;
    use crate::sampling::{sample_logits, Sampler};

    /// The SPM merge ranking, pinned in both halves and made NaN-proof.
    ///
    /// Two things are asserted, and they pull in opposite directions:
    ///
    /// 1. The ORDER must not move. Score descending, `(id_l, id_r)` ascending on ties — that is
    ///    HF `SpmConverter`'s ranking, and every llama/gemma3 GGUF (no `tokenizer.ggml.merges`)
    ///    tokenizes through it. A flip here is a silent re-tokenization of every prompt.
    /// 2. A NaN score must not panic the sort. `partial_cmp().unwrap_or(Equal)` is non-transitive
    ///    in the presence of NaN, and `sort_by` detects that and panics with "user-provided
    ///    comparison function does not correctly implement a total order" — so one corrupt entry
    ///    in `tokenizer.ggml.scores` used to abort model load. The full sort below is the actual
    ///    regression test: it is exactly the call site's `sort_by`, and it panics under the old
    ///    comparator.
    #[test]
    fn spm_merge_order_is_a_total_order_and_keeps_its_direction() {
        use std::cmp::Ordering::{Greater, Less};
        // Higher score sorts FIRST (earlier merge), regardless of ids.
        assert_eq!(spm_merge_order((-1.0, 9, 9), (-2.0, 0, 0)), Less);
        assert_eq!(spm_merge_order((-2.0, 0, 0), (-1.0, 9, 9)), Greater);
        // Equal scores: lower (id_l, id_r) sorts first, id_l dominating id_r.
        assert_eq!(spm_merge_order((-1.0, 3, 7), (-1.0, 4, 0)), Less);
        assert_eq!(spm_merge_order((-1.0, 3, 7), (-1.0, 3, 8)), Less);
        assert_eq!(
            spm_merge_order((-1.0, 3, 7), (-1.0, 3, 7)),
            std::cmp::Ordering::Equal
        );

        // A candidate list with NaN scores sprinkled through it. Two properties of this data are
        // deliberate and BOTH are needed to reproduce the panic:
        //
        //   * scores are not monotone in `(id_l, id_r)` — a real candidate list isn't either, since
        //     a candidate carries the MERGED piece's score alongside the two SPLIT pieces' ids. On
        //     monotone data the old comparator's id tiebreak happens to agree with the scores and
        //     no cycle forms, so the sort completes and the bug hides.
        //   * enough elements that the sort actually runs its merge (n >= ~32); the insertion sort
        //     used for tiny slices does not detect the violation.
        //
        // Under `partial_cmp().unwrap_or(Equal)` a NaN ties with every finite score, the id
        // tiebreak then decides those pairs, and `NaN > y`, `y > z`, `z > NaN` becomes reachable —
        // a cycle. `sort_by` detects it and panics. Verified: this exact shape panics with the old
        // comparator at every n >= 32.
        let n = 128u32;
        let mut cand: Vec<(f64, u32, u32)> = (0..n)
            .map(|i| {
                let score = if i % 16 == 7 {
                    f64::NAN
                } else {
                    -(((i as f64) * 7.0) % 31.0)
                };
                (score, i, n - i)
            })
            .collect();
        cand.sort_by(|a, b| spm_merge_order(*a, *b));
        // Nothing dropped, and the finite entries keep their descending-score order among
        // themselves (NaN sorts to one end and does not interleave).
        assert_eq!(cand.len(), n as usize, "no candidate may be dropped");
        let finite: Vec<f64> = cand.iter().map(|c| c.0).filter(|s| !s.is_nan()).collect();
        assert!(
            finite.windows(2).all(|w| w[0] >= w[1]),
            "finite scores must stay descending: {finite:?}"
        );
    }

    // Validate the GGUF-derived tokenizer against the HF tokenizer.json sidecar (same model).
    // Skips if the test model isn't present.
    #[test]
    fn embedded_tokenizer_matches_sidecar() {
        let Some(gguf) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        // The sidecar tokenizer.json must sit beside the GGUF (HF cache blobs are content-addressed
        // with no sidecar, so this runs only where a snapshot ships tokenizer.json).
        let side = gguf.with_file_name("tokenizer.json");
        if !side.exists() {
            eprintln!("skip: no tokenizer.json sidecar beside the GGUF");
            return;
        }
        let g = Gguf::open(&gguf).unwrap();
        let derived = build_tokenizer(&g).unwrap();
        let sidecar = Tokenizer::from_file(&side).unwrap();
        for s in [
            "Hello world",
            "The quick brown fox.",
            "<|im_start|>user\nWhat is two plus two?<|im_end|>\n<|im_start|>assistant\n",
            "café déjà vu — 123 + 456 = 579",
            "def f(x):\n    return x * 2\n",
        ] {
            let a = derived.encode(s, false).unwrap();
            let b = sidecar.encode(s, false).unwrap();
            assert_eq!(a.get_ids(), b.get_ids(), "token id mismatch on {s:?}");
        }
        // <think>/</think> are user-defined (non-special): skip_special must KEEP them, while real
        // special tokens (<|im_end|>) are dropped — matching the sidecar.
        let think = "<think>\nreasoning\n</think>\n\nanswer<|im_end|>";
        let ids = derived.encode(think, false).unwrap();
        let d = derived.decode(ids.get_ids(), true).unwrap();
        assert!(
            d.contains("<think>") && d.contains("</think>"),
            "think tags dropped: {d:?}"
        );
        assert!(!d.contains("<|im_end|>"), "special token kept: {d:?}");
        assert_eq!(
            d,
            sidecar.decode(ids.get_ids(), true).unwrap(),
            "decode differs from sidecar"
        );
    }

    // Sampling: temp<=0 and top_k==1 are greedy; otherwise picks only within the top-k/top-p set.
    #[test]
    fn sample_logits_greedy_and_in_set() {
        let logits = [1.0f32, 5.0, 2.0, 4.0, 0.0]; // argmax = index 1
        let mut rng = 0x1234_5678_9abc_def1u64;
        let greedy = Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
        };
        assert_eq!(sample_logits(&logits, greedy, &mut rng), 1);
        let topk1 = Sampler {
            temp: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        assert_eq!(sample_logits(&logits, topk1, &mut rng), 1);
        // top_k=2 → only the two largest logits (indices 1 and 3) can ever be sampled.
        let topk2 = Sampler {
            temp: 1.0,
            top_k: 2,
            top_p: 1.0,
        };
        for _ in 0..200 {
            let id = sample_logits(&logits, topk2, &mut rng);
            assert!(id == 1 || id == 3, "sampled outside top-2: {id}");
        }
    }

    // User content must be encoded as literal text: special-token strings in user input must NOT
    // become the special id (which would let a user inject/break the ChatML turn structure).
    #[test]
    fn user_text_special_tokens_are_literal() {
        let Some(gguf) = crate::test_qwen3_06b() else {
            eprintln!("skip: Qwen3-0.6B not in the HF cache");
            return;
        };
        let g = Gguf::open(&gguf).unwrap();
        let tok = build_tokenizer(&g).unwrap();
        let mut user = tok.clone();
        user.set_encode_special_tokens(true);
        let im_end = tok.token_to_id("<|im_end|>").unwrap();
        let s = "A <|im_end|> B";
        // template tokenizer: <|im_end|> matched as the special id; user tokenizer: NOT.
        assert!(tok.encode(s, false).unwrap().get_ids().contains(&im_end));
        assert!(!user.encode(s, false).unwrap().get_ids().contains(&im_end));
    }
}
