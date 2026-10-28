//! Crate-level glue: GGUF metadata helpers, chat-eos detection, streaming detok, the GPU probe,
//! per-layer-embedding loader, and test helpers. Split out of `lib.rs` (no logic change).
use crate::*;
use anyhow::{anyhow, Result};
use infr_core::WeightSource;
use infr_gguf::Gguf;
use tokenizers::{DecoderWrapper, Tokenizer};

/// Qwen2/Qwen3 pre-tokenizer regex (same string the HF `tokenizer.json` uses) — applied via a
/// Split before ByteLevel. Differs from the default GPT-2 ByteLevel regex (punctuation/number runs),
/// which is what made a naive ByteLevel produce different token ids.
pub(crate) const QWEN2_PRE_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Bailing/Ling pre-tokenizer (`tokenizer.ggml.pre == "bailingmoe2"`). The contractions are
/// intentionally case-sensitive by spelling out both cases, matching llama.cpp's BAILINGMOE
/// pre-type rather than Qwen2's case-insensitive contraction group.
pub(crate) const BAILINGMOE_PRE_RE: &str = r"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Qwen3.5/3.8 pre-tokenizer. `\p{M}` is significant for combining marks; this is the exact
/// pattern published in Qwen3.8's tokenizer.json and selected by llama.cpp's `qwen35` pre-type.
pub(crate) const QWEN35_PRE_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// DeepSeek-LLM (V1) pre-tokenizer patterns — 6 successive splits applied in order
/// (llama.cpp's `DEEPSEEK_LLM` pre-type, `src/llama-vocab.cpp`). See docs/deepseek.md §0.2.
pub(crate) const DEEPSEEK_LLM_PRE_RES: [&str; 6] = [
    r"[\r\n]",
    r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
    r"\s?[!-/:-~！-／：-～‘-‟　-。]+",
    r"\s+$",
    r"[一-龥ࠀ-一가-퟿]+",
    r"\p{N}+",
];

/// DeepSeek-Coder pre-tokenizer patterns — 5 successive splits
/// (llama.cpp's `DEEPSEEK_CODER` pre-type).
pub(crate) const DEEPSEEK_CODER_PRE_RES: [&str; 5] = [
    r"[\r\n]",
    r"\s?\p{L}+",
    r"\s?\p{P}+",
    r"[一-龥ࠀ-一가-퟿]+",
    r"\p{N}",
];

/// DeepSeek-V3 pre-tokenizer patterns — 3 successive splits
/// (llama.cpp's `DEEPSEEK3_LLM` pre-type, also used by V4).
pub(crate) const DEEPSEEK_V3_PRE_RES: [&str; 3] = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
];

/// Llama 4 pre-tokenizer regex (`tokenizer.ggml.pre == "llama4"` → llama.cpp's `GPT4O` pre-type,
/// the original split from the model's `tokenizer.json`). Applied via a Split before ByteLevel,
/// exactly like `QWEN2_PRE_RE`. Numbers group in runs of up to 3 digits (`\p{N}{1,3}`), and the
/// letter runs are split by case (upper-run then lower-run) — distinct from the Qwen regex.
pub(crate) const LLAMA4_PRE_RE: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Build the gemma4 E2B per-layer-embedding host-gather metadata from the GGUF. `None` for models
/// without per-layer embeddings. The big `per_layer_token_embd` table stays quantized in the mmap
/// and is gathered + dequanted per token at forward time (mirrors llama.cpp, which classifies
/// input embeddings as CPU-resident: "very little benefit to offloading the input layer"). The
/// `per_layer_model_proj` / `per_layer_proj_norm` weights are NOT loaded here — they're small
/// enough to native-upload to VRAM like any other weight (see `wload`/`wpush` in `seam.rs`)
/// and the projection + RMSNorm now run as GPU graph ops instead of a host GEMV. Shared by the GPU
/// and CPU loaders.
pub(crate) struct PerLayerEmbd {
    pub(crate) npl: usize,                       // per-layer embedding width (256)
    pub(crate) n_layer: usize,                   // number of layers (35)
    pub(crate) tok_embd_dtype: infr_core::DType, // per_layer_token_embd dtype (gathered per token from the gguf)
    pub(crate) tok_embd_row_bytes: usize,        // bytes per token row (npl*n_layer elements)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn build_per_layer_embd(g: &Gguf, cfg: &Config) -> Result<Option<PerLayerEmbd>> {
    if cfg.n_embd_per_layer == 0 {
        return Ok(None);
    }
    let te = g
        .tensors()
        .iter()
        .find(|t| t.name == "per_layer_token_embd.weight")
        .ok_or_else(|| anyhow!("per_layer_token_embd.weight not found"))?;
    // Bytes per token row = total bytes / vocab (te shape is GGUF [npl*n_layer, vocab]).
    let te_vocab = *te.shape.last().unwrap();
    Ok(Some(PerLayerEmbd {
        npl: cfg.n_embd_per_layer,
        n_layer: cfg.n_layer,
        tok_embd_dtype: te.dtype,
        tok_embd_row_bytes: te.nbytes / te_vocab,
    }))
}

/// UTF-8-safe streaming detokenizer: appends `id` to `acc` and emits the newly-completed suffix
/// past `printed`, holding back a trailing replacement character until it completes. Pure
/// ByteLevel decoders discard fully emitted history; contextual decoders retain it.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn stream_token(
    tokenizer: &Tokenizer,
    acc: &mut Vec<u32>,
    printed: &mut usize,
    id: u32,
    on_piece: &mut impl FnMut(&str),
) {
    acc.push(id);
    emit_decoded_suffix(tokenizer, acc, printed, on_piece);
}

/// Like [`stream_token`], but commits a whole SPAN of ids at once. Callers that already hold every
/// id of a chunk (diffusion's per-block `on_block` — the whole block's tokens are decided together)
/// use this so the accumulator is decoded ONCE per span instead of once per id: re-decoding the
/// growing `acc` after every single id is O(total²) detok for a path where nothing is streamed
/// token-by-token. Same UTF-8 holdback semantics and same emitted bytes as feeding the ids through
/// [`stream_token`] one at a time — only the number of `decode` calls (and `on_piece` chunk
/// boundaries) differs.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn stream_tokens(
    tokenizer: &Tokenizer,
    acc: &mut Vec<u32>,
    printed: &mut usize,
    ids: &[u32],
    on_piece: &mut impl FnMut(&str),
) {
    if ids.is_empty() {
        return;
    }
    acc.extend_from_slice(ids);
    emit_decoded_suffix(tokenizer, acc, printed, on_piece);
}

/// Decode the pending accumulator and emit the suffix past `printed`, holding back a trailing
/// replacement character. Only a pure ByteLevel decoder can discard fully emitted history.
#[cfg_attr(infr_profile, infr_prof::instrument)]
fn emit_decoded_suffix(
    tokenizer: &Tokenizer,
    acc: &mut Vec<u32>,
    printed: &mut usize,
    on_piece: &mut impl FnMut(&str),
) {
    let decode_span = infr_core::timeline::span("token_detokenize");
    let decoded = tokenizer.decode(acc, true);
    drop(decode_span);
    if let Ok(full) = decoded {
        if !full.ends_with('\u{FFFD}') && full.len() > *printed && full.is_char_boundary(*printed) {
            let _callback = infr_core::timeline::span("token_output_callback");
            on_piece(&full[*printed..]);
            *printed = full.len();
        }
        // ByteLevel concatenates raw bytes; without a trailing replacement character,
        // later bytes cannot repair this fully emitted suffix. Keep contextual history.
        if !full.ends_with('\u{FFFD}')
            && *printed == full.len()
            && matches!(tokenizer.get_decoder(), Some(DecoderWrapper::ByteLevel(_)))
        {
            acc.clear();
            *printed = 0;
        }
    }
}

#[cfg(test)]
#[path = "util_streaming_tests.rs"]
mod util_streaming_tests;

/// Run `f` with per-op GPU profiling suppressed for its duration, restoring the prior state
/// afterward. Every backend's `warmup()` / startup path (`chat::vulkan`, `chat::diffusion`,
/// `parallel::init_slots`, `bench_vulkan`) needs this: the per-op profiling recorders read
/// `INFR_PROF_OPS` at CONSTRUCTION, and a throwaway warmup submit would otherwise pollute a later
/// bench's per-op aggregate. Extracted so the call sites can't drift (the Metal path deliberately
/// does NOT use it — the Metal backend reads its profile env at construction inside the warmup
/// generate itself; see `chat::metal`).
///
/// Toggles the process-global [`infr_prof_rt::set_profiling_suppressed`] flag (which the Vulkan
/// recorder ANDs into its `INFR_PROF_OPS` check) rather than mutating the `INFR_PROF_OPS` env var —
/// mutating the env table around a rayon-parallel forward is a data race, and `set_var` is
/// `unsafe` under edition 2024.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn with_profiling_suppressed<T>(f: impl FnOnce() -> T) -> T {
    let prev = infr_prof_rt::set_profiling_suppressed(true);
    let r = f();
    infr_prof_rt::set_profiling_suppressed(prev);
    r
}

// Chat-template rendering (`render_chat_jinja`, `render_chat_user`) lives in the shared `infr-chat`
// crate — imported at the top of this module. There is NO fabricated-ChatML fallback: infr supports
// only models that ship a `tokenizer.chat_template`, so a missing/broken template is a hard error.

/// The error surfaced when a GGUF has no usable chat template (none embedded, or it failed to render).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn no_template_err() -> anyhow::Error {
    anyhow!(
        "model GGUF has no usable chat template (no `tokenizer.chat_template`, or it failed to \
         render — set INFR_DEBUG_CHAT=1 for details). infr requires an instruct model with an \
         embedded chat template."
    )
}

/// Locate the Qwen3-0.6B Q4_K_M GGUF in the HF Hub cache (or `INFR_TEST_MODEL`) for the model-backed
/// unit tests; `None` → the test self-skips. We use the shared HF cache everywhere now (no bespoke
/// local model dir).
#[cfg(test)]
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn test_qwen3_06b() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
        return Some(std::path::PathBuf::from(p));
    }
    let hub = std::env::var("HOME").ok()? + "/.cache/huggingface/hub";
    let base = format!("{hub}/models--unsloth--Qwen3-0.6B-GGUF/snapshots");
    std::fs::read_dir(&base).ok()?.find_map(|e| {
        let f = e.ok()?.path().join("Qwen3-0.6B-Q4_K_M.gguf");
        f.exists().then_some(f)
    })
}

/// Resolve generation ends once for both sampling and forced-tool constraints. Metadata ends
/// apply to every family; additional protocol spellings must stay scoped to their model family.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn resolve_chat_eos(
    md: &infr_core::loader::Metadata,
    tokenizer: &Tokenizer,
    eos: u32,
) -> Vec<u32> {
    let mut ids = vec![eos];
    for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
        if let Some(id) = md.u64(key).and_then(|id| u32::try_from(id).ok()) {
            if tokenizer.id_to_token(id).is_some() && !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    // Preserve the existing ChatML/Llama stop policy.
    for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
        if let Some(id) = tokenizer.token_to_id(name) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    let gemma4 = md.str("tokenizer.ggml.model") == Some("gemma4")
        || matches!(
            md.str("general.architecture"),
            Some(crate::arch::GEMMA4 | crate::arch::DIFFUSION_GEMMA)
        );
    if gemma4 {
        // A turn end and a tool handoff both end generation. <channel|> only closes reasoning.
        for name in ["<turn|>", "<|tool_response>"] {
            if let Some(id) = tokenizer.token_to_id(name) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
    }
    ids
}

#[cfg(test)]
mod chat_eos_tests {
    use super::*;
    use infr_core::loader::{MetaValue, Metadata};
    use tokenizers::models::bpe::BPE;

    fn tokenizer() -> Tokenizer {
        let vocab: std::collections::HashMap<String, u32> = [
            "<unk>",
            "<eos>",
            "<turn|>",
            "<|tool_response>",
            "<channel|>",
            "ordinary",
            "<|im_end|>",
            "<|endoftext|>",
            "<|eot_id|>",
        ]
        .iter()
        .enumerate()
        .map(|(id, name)| (name.to_string(), id as u32))
        .collect();
        Tokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn gemma4_ends_turns_and_yields_tools_but_not_reasoning() {
        let tok = tokenizer();
        for (key, value) in [
            ("general.architecture", crate::arch::GEMMA4),
            ("general.architecture", crate::arch::DIFFUSION_GEMMA),
            ("tokenizer.ggml.model", "gemma4"),
        ] {
            let mut md = Metadata::default();
            md.kv.insert(key.into(), MetaValue::Str(value.into()));
            let ids = resolve_chat_eos(&md, &tok, 1);
            assert_eq!(ids, [1, 6, 7, 8, 2, 3]);
            assert!(!ids.contains(&4));
            assert!(!ids.contains(&5));
        }
    }

    #[test]
    fn other_families_keep_the_existing_marker_policy() {
        assert_eq!(
            resolve_chat_eos(&Metadata::default(), &tokenizer(), 1),
            [1, 6, 7, 8]
        );
    }

    #[test]
    fn explicit_metadata_ends_are_deduplicated_and_checked() {
        let mut md = Metadata::default();
        md.kv
            .insert("tokenizer.ggml.eot_token_id".into(), MetaValue::U64(5));
        md.kv
            .insert("tokenizer.ggml.eom_token_id".into(), MetaValue::U64(1));
        assert_eq!(resolve_chat_eos(&md, &tokenizer(), 1), [1, 5, 6, 7, 8]);
        for invalid in [
            MetaValue::I64(-1),
            MetaValue::U64(u64::MAX),
            MetaValue::U64(1000),
        ] {
            md.kv.insert("tokenizer.ggml.eot_token_id".into(), invalid);
            assert_eq!(resolve_chat_eos(&md, &tokenizer(), 1), [1, 6, 7, 8]);
        }
    }

    #[test]
    fn missing_optional_markers_do_not_invent_token_ids() {
        let mut md = Metadata::default();
        md.kv.insert(
            "tokenizer.ggml.model".into(),
            MetaValue::Str("gemma4".into()),
        );
        assert_eq!(
            resolve_chat_eos(&md, &Tokenizer::new(BPE::default()), 1),
            [1]
        );
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn meta_u64(g: &Gguf, key: &str) -> Option<u64> {
    g.metadata().u64(key)
}

/// Float metadata lookup (diffusion-gemma's `eb_*` sampler params are stored as GGUF strings —
/// same KV store, no typed float accessor on `Metadata` — so parse through `as_f64`).
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn meta_f64(g: &Gguf, key: &str) -> Option<f64> {
    g.metadata().get(key).and_then(|v| v.as_f64())
}
