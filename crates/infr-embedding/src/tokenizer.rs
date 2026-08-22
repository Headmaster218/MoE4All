//! GGUF-backed BERT WordPiece tokenizer.
//!
//! llama.cpp stores WordPiece word starts with a SentencePiece-style `▁` marker and removes the
//! original Hugging Face `##` continuation prefix. Reversing that representation lets the mature
//! `tokenizers` BERT normalizer/pre-tokenizer execute the exact same longest-prefix algorithm.

use anyhow::{anyhow, bail, Context, Result};
use infr_core::loader::{MetaValue, Metadata};
use std::collections::HashMap;
use tokenizers::{
    models::wordpiece::WordPiece, normalizers::bert::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer, processors::bert::BertProcessing, AddedToken,
    Tokenizer,
};

pub(crate) struct BertWordPiece {
    tokenizer: Tokenizer,
    max_context: usize,
}

impl BertWordPiece {
    pub(crate) fn from_metadata(md: &Metadata, max_context: usize) -> Result<Self> {
        if md.str("tokenizer.ggml.model") != Some("bert") {
            bail!(
                "native Nomic-BERT requires tokenizer.ggml.model=\"bert\" (got {:?})",
                md.str("tokenizer.ggml.model")
            );
        }
        let tokens = md
            .get("tokenizer.ggml.tokens")
            .and_then(MetaValue::as_arr)
            .context("GGUF missing tokenizer.ggml.tokens")?;
        let token_types = md
            .get("tokenizer.ggml.token_type")
            .and_then(MetaValue::as_arr);

        let id = |key: &str, default: u32| -> Result<u32> {
            let value = md.u64(key).unwrap_or(default as u64);
            let value = u32::try_from(value).with_context(|| format!("GGUF {key} is too large"))?;
            if value as usize >= tokens.len() {
                bail!(
                    "GGUF {key}={value} is outside the {}-token vocabulary",
                    tokens.len()
                );
            }
            Ok(value)
        };
        let bos = id("tokenizer.ggml.bos_token_id", 101)?;
        let sep = id("tokenizer.ggml.seperator_token_id", 102)?;
        let unk = id("tokenizer.ggml.unknown_token_id", 100)?;

        let token_text = |id: u32| -> Result<&str> {
            tokens[id as usize]
                .as_str()
                .with_context(|| format!("tokenizer token {id} is not a string"))
        };
        let bos_text = token_text(bos)?.to_owned();
        let sep_text = token_text(sep)?.to_owned();
        let unk_text = token_text(unk)?.to_owned();

        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut specials = Vec::new();
        for (index, value) in tokens.iter().enumerate() {
            let raw = value
                .as_str()
                .with_context(|| format!("tokenizer token {index} is not a string"))?;
            let ty = token_types
                .and_then(|types| types.get(index))
                .and_then(MetaValue::as_u64)
                .unwrap_or(1);
            let wordpiece = if ty == 1 {
                raw.strip_prefix('▁')
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("##{raw}"))
            } else {
                specials.push(AddedToken::from(raw.to_owned(), true));
                raw.to_owned()
            };
            if let Some(previous) = vocab.insert(wordpiece.clone(), index as u32) {
                bail!(
                    "GGUF WordPiece vocabulary collision for {wordpiece:?}: ids {previous} and {index}"
                );
            }
        }

        let model = WordPiece::builder()
            .vocab(vocab)
            .unk_token(unk_text)
            .continuing_subword_prefix("##".to_owned())
            .build()
            .map_err(|error| anyhow!("build BERT WordPiece vocabulary: {error}"))?;
        let lowercase = md
            .get("tokenizer.ggml.normalizer.lowercase")
            .and_then(|value| match value {
                MetaValue::Bool(value) => Some(*value),
                _ => None,
            })
            .unwrap_or(true);
        let strip_accents = md
            .get("tokenizer.ggml.normalizer.strip_accents")
            .and_then(|value| match value {
                MetaValue::Bool(value) => Some(*value),
                _ => None,
            })
            .unwrap_or(lowercase);

        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_normalizer(Some(BertNormalizer::new(
            true,
            true,
            Some(strip_accents),
            lowercase,
        )));
        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
        tokenizer.with_post_processor(Some(BertProcessing::new((sep_text, sep), (bos_text, bos))));
        if !specials.is_empty() {
            tokenizer.add_special_tokens(&specials);
        }

        Ok(Self {
            tokenizer,
            max_context,
        })
    }

    pub(crate) fn encode(&self, input: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(input, true)
            .map_err(|error| anyhow!("tokenize embedding input: {error}"))?;
        let ids = encoding.get_ids();
        if ids.len() > self.max_context {
            bail!(
                "embedding input has {} tokens, exceeding model context {}",
                ids.len(),
                self.max_context
            );
        }
        Ok(ids.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::WeightSource;

    fn metadata() -> Metadata {
        let tokens = [
            "[PAD]", "[unused]", "[UNK]", "[CLS]", "[SEP]", "▁hello", "s", "▁世", "▁界", "▁!",
        ];
        let mut md = Metadata::default();
        md.kv
            .insert("tokenizer.ggml.model".into(), MetaValue::Str("bert".into()));
        md.kv.insert(
            "tokenizer.ggml.tokens".into(),
            MetaValue::Arr(
                tokens
                    .iter()
                    .map(|token| MetaValue::Str((*token).into()))
                    .collect(),
            ),
        );
        md.kv.insert(
            "tokenizer.ggml.token_type".into(),
            MetaValue::Arr(
                [3, 3, 3, 3, 3, 1, 1, 1, 1, 1]
                    .into_iter()
                    .map(MetaValue::I64)
                    .collect(),
            ),
        );
        md.kv
            .insert("tokenizer.ggml.bos_token_id".into(), MetaValue::U64(3));
        md.kv.insert(
            "tokenizer.ggml.seperator_token_id".into(),
            MetaValue::U64(4),
        );
        md.kv
            .insert("tokenizer.ggml.unknown_token_id".into(), MetaValue::U64(2));
        md
    }

    #[test]
    fn restores_gguf_word_starts_and_continuations() {
        let tokenizer = BertWordPiece::from_metadata(&metadata(), 16).unwrap();
        assert_eq!(
            tokenizer.encode("HELLOS 世界!").unwrap(),
            [3, 5, 6, 7, 8, 9, 4]
        );
    }

    #[test]
    fn rejects_inputs_past_model_context() {
        let tokenizer = BertWordPiece::from_metadata(&metadata(), 3).unwrap();
        let error = tokenizer.encode("hello world").unwrap_err();
        assert!(error.to_string().contains("exceeding model context"));
    }

    #[test]
    fn real_nomic_matches_llama_cpp_token_counts_when_available() {
        let Ok(path) = std::env::var("INFR_EMBEDDING_TEST_MODEL") else {
            return;
        };
        let gguf = infr_gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
        let tokenizer = BertWordPiece::from_metadata(gguf.metadata(), 2048).unwrap();
        let cases = [
            ("今天天气很好，我们去公园散步。", 17),
            ("A quick brown fox jumps over the lazy dog.", 12),
        ];
        for (text, expected) in cases {
            assert_eq!(tokenizer.encode(text).unwrap().len(), expected, "{text}");
        }
        let pair = [
            "怎样提高本地大模型推理速度？",
            "How can local LLM inference be accelerated?",
        ];
        let pair_tokens: usize = pair
            .iter()
            .map(|text| tokenizer.encode(text).unwrap().len())
            .sum();
        assert_eq!(pair_tokens, 27);
        let long = "长文本语义检索基准。".repeat(40);
        assert_eq!(tokenizer.encode(&long).unwrap().len(), 402);
    }
}
