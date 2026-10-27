use super::{emit_decoded_suffix, stream_token, stream_tokens};
use std::collections::HashMap;
use tokenizers::decoders::{
    byte_level::ByteLevel, metaspace::Metaspace, sequence::Sequence, wordpiece::WordPiece,
};
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, DecoderWrapper, Tokenizer};

fn byte_level_tokenizer() -> Tokenizer {
    // GPT-2 byte alphabet: give each raw byte its own token ID for split-byte tests.
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars: Vec<u32> = bytes.iter().map(|&byte| u32::from(byte)).collect();
    let mut extra = 256;
    for byte in 0..=255u8 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            chars.push(extra);
            extra += 1;
        }
    }
    let vocab: HashMap<String, u32> = bytes
        .into_iter()
        .zip(chars)
        .map(|(byte, ch)| (char::from_u32(ch).unwrap().to_string(), u32::from(byte)))
        .collect();
    let model = BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer.add_special_tokens(&[AddedToken::from("<|skip|>", true)]);
    tokenizer.add_tokens(&[
        AddedToken::from("<think>", false),
        AddedToken::from("\u{4E2D}", false),
    ]);
    tokenizer
}

#[derive(Default)]
struct Comparison {
    pending: Vec<u32>,
    printed: usize,
    chunks: Vec<String>,
    history: Vec<u32>,
    old_printed: usize,
    old_chunks: Vec<String>,
}

impl Comparison {
    fn token(&mut self, tokenizer: &Tokenizer, id: u32) {
        stream_token(
            tokenizer,
            &mut self.pending,
            &mut self.printed,
            id,
            &mut |piece| self.chunks.push(piece.to_owned()),
        );
        self.reference(tokenizer, &[id]);
    }

    fn span(&mut self, tokenizer: &Tokenizer, ids: &[u32]) {
        stream_tokens(
            tokenizer,
            &mut self.pending,
            &mut self.printed,
            ids,
            &mut |piece| self.chunks.push(piece.to_owned()),
        );
        self.reference(tokenizer, ids);
    }

    fn reference(&mut self, tokenizer: &Tokenizer, ids: &[u32]) {
        if !ids.is_empty() {
            self.history.extend_from_slice(ids);
            // The pre-compaction algorithm, including its whole-output holdback rule.
            if let Ok(full) = tokenizer.decode(&self.history, true) {
                if !full.ends_with('\u{FFFD}')
                    && full.len() > self.old_printed
                    && full.is_char_boundary(self.old_printed)
                {
                    self.old_chunks.push(full[self.old_printed..].to_owned());
                    self.old_printed = full.len();
                }
            }
        }
        assert_eq!(self.chunks, self.old_chunks, "history: {:?}", self.history);
        assert_eq!(
            self.chunks.concat().as_bytes(),
            self.old_chunks.concat().as_bytes()
        );
    }

    fn assert_cleared(&self) {
        assert!(self.pending.is_empty());
        assert_eq!(self.printed, 0);
    }
}

#[test]
fn byte_level_every_prefix_matches_old_chunks_for_unicode_and_invalid_bytes() {
    let tokenizer = byte_level_tokenizer();
    let cases: &[&[u8]] = &[
        b"hello  world\t\r\n!",
        "\u{4E2D}\u{6587} \u{1F642} e\u{301} \u{200D}\u{FE0F}".as_bytes(),
        &[0x80, b'X', 0xC0, 0xAF, b'y', 0xED, 0xA0, 0x80, b'Z'],
        &[0xF4, 0x90, 0x80, 0x80, b'!', 0xE2, 0x28, 0xA1, b'.'],
        &[b'a', 0xC3],
        &[0xEF, 0xBF, 0xBD],
        &[0xEF, 0xBF, 0xBD, b'!'],
    ];
    for &bytes in cases {
        let mut comparison = Comparison::default();
        for &byte in bytes {
            comparison.token(&tokenizer, u32::from(byte));
        }
    }
    // Cover every byte value, including NUL and invalid UTF-8, at a compaction boundary.
    for byte in 0..=255u8 {
        let mut comparison = Comparison::default();
        comparison.token(&tokenizer, u32::from(byte));
        comparison.token(&tokenizer, u32::from(b'!'));
        comparison.assert_cleared();
    }
}

#[test]
fn byte_level_keeps_split_utf8_until_the_complete_chunk_is_emitted() {
    let tokenizer = byte_level_tokenizer();
    let mut comparison = Comparison::default();
    comparison.token(&tokenizer, u32::from(b'A'));
    comparison.assert_cleared();
    let emoji = [0xF0, 0x9F, 0x99, 0x82];
    for (index, &byte) in emoji.iter().enumerate() {
        comparison.token(&tokenizer, byte);
        if index < 3 {
            assert_eq!(comparison.pending.len(), index + 1);
            assert_eq!(comparison.chunks, ["A"]);
        }
    }
    comparison.assert_cleared();
    assert_eq!(comparison.chunks, ["A", "\u{1F642}"]);
}

#[test]
fn byte_level_preserves_invalid_and_literal_replacement_holdback() {
    let tokenizer = byte_level_tokenizer();
    let mut comparison = Comparison::default();
    comparison.token(&tokenizer, u32::from(b'A'));
    comparison.token(&tokenizer, 0x80);
    assert_eq!(comparison.pending, [0x80]);
    assert_eq!(comparison.chunks, ["A"]);
    comparison.token(&tokenizer, u32::from(b'B'));
    comparison.assert_cleared();
    assert_eq!(comparison.chunks, ["A", "\u{FFFD}B"]);

    comparison.span(&tokenizer, &[0xEF, 0xBF, 0xBD]);
    assert_eq!(comparison.pending, [0xEF, 0xBF, 0xBD]);
    let skipped = tokenizer.token_to_id("<|skip|>").unwrap();
    comparison.token(&tokenizer, skipped);
    assert_eq!(comparison.pending.len(), 4);
    comparison.token(&tokenizer, u32::from(b'C'));
    comparison.assert_cleared();
    assert_eq!(comparison.chunks, ["A", "\u{FFFD}B", "\u{FFFD}C"]);
}

#[test]
fn byte_level_special_unknown_and_added_tokens_preserve_every_prefix() {
    let tokenizer = byte_level_tokenizer();
    let skipped = tokenizer.token_to_id("<|skip|>").unwrap();
    let normal = tokenizer.token_to_id("<think>").unwrap();
    let unicode = tokenizer.token_to_id("\u{4E2D}").unwrap();
    let mut comparison = Comparison::default();
    for id in [skipped, u32::MAX, skipped] {
        comparison.token(&tokenizer, id);
        comparison.assert_cleared();
        assert!(comparison.chunks.is_empty());
    }
    for id in [0xC3, skipped, u32::MAX, 0xA9, normal, unicode, skipped] {
        comparison.token(&tokenizer, id);
    }
    comparison.assert_cleared();
    assert_eq!(comparison.chunks, ["\u{E9}", "<think>", "\u{4E2D}"]);
}

#[test]
fn byte_level_empty_spans_and_mixed_token_spans_match_old_call_boundaries() {
    let tokenizer = byte_level_tokenizer();
    let skipped = tokenizer.token_to_id("<|skip|>").unwrap();
    let mut comparison = Comparison::default();
    comparison.span(&tokenizer, &[]);
    comparison.token(&tokenizer, u32::from(b'a'));
    comparison.span(&tokenizer, &[]);
    comparison.span(&tokenizer, &[0xF0, 0x9F]);
    comparison.span(&tokenizer, &[]);
    assert_eq!(comparison.pending, [0xF0, 0x9F]);
    comparison.token(&tokenizer, skipped);
    comparison.token(&tokenizer, 0x99);
    comparison.span(&tokenizer, &[0x82, u32::from(b' '), 0xC3]);
    comparison.span(&tokenizer, &[]);
    // Even the completed emoji remains held when this call ends in incomplete UTF-8.
    assert_eq!(comparison.chunks, ["a"]);
    comparison.span(&tokenizer, &[0xA9, u32::from(b'!')]);
    comparison.assert_cleared();
    assert_eq!(comparison.chunks, ["a", "\u{1F642} \u{E9}!"]);
    comparison.span(&tokenizer, &[skipped, u32::MAX]);
    comparison.assert_cleared();
}

#[test]
fn non_byte_level_decoders_keep_the_complete_old_history() {
    let decoders: Vec<Option<DecoderWrapper>> = vec![
        None,
        Some(DecoderWrapper::Sequence(Sequence::new(vec![
            DecoderWrapper::ByteLevel(ByteLevel::default()),
        ]))),
        Some(DecoderWrapper::Metaspace(Metaspace::default())),
        Some(DecoderWrapper::WordPiece(WordPiece::default())),
    ];
    for decoder in decoders {
        let mut tokenizer = byte_level_tokenizer();
        tokenizer.with_decoder(decoder);
        let skipped = tokenizer.token_to_id("<|skip|>").unwrap();
        let mut comparison = Comparison::default();
        for id in [u32::from(b'a'), skipped, u32::MAX, u32::from(b'b')] {
            comparison.token(&tokenizer, id);
            assert_eq!(comparison.pending, comparison.history);
            assert_eq!(comparison.printed, comparison.old_printed);
        }
        comparison.span(&tokenizer, &[]);
        comparison.span(&tokenizer, &[u32::from(b'c'), u32::from(b'd')]);
        assert_eq!(comparison.pending, comparison.history);
        assert_eq!(comparison.printed, comparison.old_printed);
        assert_eq!(comparison.pending.len(), 6);
    }
}

#[test]
fn byte_level_long_stream_clears_complete_history_at_every_prefix() {
    let tokenizer = byte_level_tokenizer();
    let mut comparison = Comparison::default();
    let unit = [u32::from(b'a'), 0xF0, 0x9F, 0x99, 0x82, u32::from(b'!')];
    let mut max_pending = 0;
    for _ in 0..128 {
        for &id in &unit {
            comparison.token(&tokenizer, id);
            max_pending = max_pending.max(comparison.pending.len());
            assert!(comparison.pending.len() <= 3);
            assert_eq!(comparison.printed, 0);
        }
        comparison.assert_cleared();
    }
    assert_eq!(max_pending, 3);
    assert_eq!(comparison.history.len(), 768);
    assert_eq!(comparison.chunks.concat(), "a\u{1F642}!".repeat(128));
}

#[test]
fn byte_level_skipped_only_long_stream_does_not_accumulate_history() {
    let tokenizer = byte_level_tokenizer();
    let skipped = tokenizer.token_to_id("<|skip|>").unwrap();
    let mut comparison = Comparison::default();
    for _ in 0..512 {
        comparison.token(&tokenizer, skipped);
        comparison.token(&tokenizer, u32::MAX);
        comparison.assert_cleared();
    }
    assert!(comparison.chunks.is_empty());
}

#[test]
fn byte_level_pending_utf8_retains_intervening_skipped_tokens_until_completion() {
    let tokenizer = byte_level_tokenizer();
    let skipped = tokenizer.token_to_id("<|skip|>").unwrap();
    let mut comparison = Comparison::default();
    comparison.token(&tokenizer, 0xF0);
    for _ in 0..64 {
        comparison.token(&tokenizer, skipped);
    }
    // No universal fixed bound: an unfinished character must retain skipped token IDs too.
    assert_eq!(comparison.pending.len(), 65);
    comparison.span(&tokenizer, &[0x9F, 0x99, 0x82]);
    comparison.assert_cleared();
    assert_eq!(comparison.chunks, ["\u{1F642}"]);
}

#[test]
fn byte_level_callback_panic_does_not_discard_pending_output() {
    let tokenizer = byte_level_tokenizer();
    let mut pending = Vec::new();
    let mut printed = 0;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stream_token(
            &tokenizer,
            &mut pending,
            &mut printed,
            u32::from(b'a'),
            &mut |_| panic!("injected output callback failure"),
        );
    }));
    assert!(result.is_err());
    assert_eq!(pending, [u32::from(b'a')]);
    assert_eq!(printed, 0);
    let mut chunks = Vec::new();
    emit_decoded_suffix(&tokenizer, &mut pending, &mut printed, &mut |piece| {
        chunks.push(piece.to_owned());
    });
    assert_eq!(chunks, ["a"]);
    assert!(pending.is_empty());
    assert_eq!(printed, 0);
}
