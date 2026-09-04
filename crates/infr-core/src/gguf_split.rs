//! `gguf-split` shard naming — the `-NNNNN-of-MMMMM.gguf` convention llama.cpp's `gguf-split`
//! writes, parsed and enumerated in one place.
//!
//! Two crates need the same convention and neither can be the other's dependency-free home for it:
//! `infr-hub` decides whether a cached model is COMPLETE (every shard present, no dangling blob)
//! before a download or a load, and `infr-gguf` follows a first shard's `split.count` to the rest of
//! the set at open. A second copy of the parser is where those two silently drift — one accepting a
//! name the other rejects means a model that downloads and then fails to load, or worse, a set
//! enumerated two different ways. Nothing here touches the filesystem or the GGUF binary format; it
//! is string logic over a filename.

/// A parsed `-NNNNN-of-MMMMM.gguf` shard suffix (llama.cpp's split naming).
#[derive(Debug, PartialEq, Eq)]
pub struct Shard {
    /// Everything before `-NNNNN-of-MMMMM.gguf`.
    pub base: String,
    /// This shard's 1-based position (`NNNNN`). Note that GGUF's own `split.no` metadata key is
    /// ZERO-based, so a well-formed shard has `split.no == index - 1`.
    pub index: u32,
    /// Total shard count (`MMMMM`).
    pub total: u32,
    /// Zero-pad width of the index/total fields (usually 5).
    pub width: usize,
}

/// The widest shard field this parser will accept, and therefore the largest `total`
/// [`shard_set`] can ever be asked to enumerate.
///
/// llama.cpp's `gguf-split` writes five digits (`-00001-of-00003`), so six or more cannot name a
/// real split. The bound matters because `total` is REMOTE input: `pull_repo_latest` runs
/// `shard_set` on a filename taken from the HuggingFace sibling list, before any download and
/// before `check_relative`, and `shard_set` materialises every name in `1..=total` into a `Vec`.
/// Ten digits parse happily as `u32::MAX`, which is 4.29e9 `String`s — roughly 100 GiB — allocated
/// from one hostile filename (backlog B19).
///
/// Rejected in the PARSER rather than in `shard_set` so every call site is covered by one check:
/// `Store::resolve_repo` expands local snapshot names through the same helper, and `Gguf::open`
/// bounds a `split.count` read out of a downloaded file's metadata by requiring it to agree with
/// the filename's own total.
const MAX_SHARD_WIDTH: usize = 5;

/// Parse a `<base>-NNNNN-of-MMMMM.gguf` shard filename. `None` when `fname` is not a shard.
pub fn parse_shard(fname: &str) -> Option<Shard> {
    let stem = fname.strip_suffix(".gguf").or_else(|| {
        fname
            .to_lowercase()
            .ends_with(".gguf")
            .then(|| &fname[..fname.len() - 5])
    })?;
    let (left, total_s) = stem.rsplit_once("-of-")?;
    let (base, idx_s) = left.rsplit_once('-')?;
    if total_s.is_empty()
        || idx_s.is_empty()
        || !total_s.bytes().all(|b| b.is_ascii_digit())
        || !idx_s.bytes().all(|b| b.is_ascii_digit())
        || total_s.len() != idx_s.len()
        || total_s.len() > MAX_SHARD_WIDTH
    {
        return None;
    }
    Some(Shard {
        base: base.to_string(),
        index: idx_s.parse().ok()?,
        total: total_s.parse().ok()?,
        width: total_s.len(),
    })
}

/// The FULL set of shard filenames for `chosen` (all `-00001-of-MMMMM` … `-MMMMM-of-MMMMM`), in
/// order. A non-sharded file returns just `[chosen]`. All must be present for the model to load.
pub fn shard_set(chosen: &str) -> Vec<String> {
    match parse_shard(chosen) {
        Some(s) if s.total >= 1 => (1..=s.total)
            .map(|i| {
                format!(
                    "{}-{:0width$}-of-{:0width$}.gguf",
                    s.base,
                    i,
                    s.total,
                    width = s.width
                )
            })
            .collect(),
        _ => vec![chosen.to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_shard_pattern() {
        let s = parse_shard("model-Q4_K_M-00001-of-00003.gguf").unwrap();
        assert_eq!(s.base, "model-Q4_K_M");
        assert_eq!(s.index, 1);
        assert_eq!(s.total, 3);
        assert_eq!(s.width, 5);
        // The index is the FILENAME's 1-based position, not `split.no`.
        assert_eq!(parse_shard("m-00003-of-00005.gguf").unwrap().index, 3);
        // Not a shard.
        assert_eq!(parse_shard("model-Q4_K_M.gguf"), None);
        // Mismatched field widths / non-numeric → not a shard.
        assert_eq!(parse_shard("m-1-of-003.gguf"), None);
        assert_eq!(parse_shard("m-000ab-of-00003.gguf"), None);
    }

    /// B19: the shard total is REMOTE input and `shard_set` enumerates every name in `1..=total`
    /// before anything is downloaded, so an over-wide field must not parse at all.
    ///
    /// The 10-digit case is the one that mattered: both fields are the same length, so the
    /// width-equality guard passed it, and it parsed as `u32::MAX` — 4.29e9 filenames. The
    /// expansion itself is deliberately not run here; that is the bug, and it would take the
    /// runner with it.
    #[test]
    fn an_over_wide_shard_total_is_not_a_shard() {
        assert_eq!(
            parse_shard("m-Q4_K_M-0000000001-of-4294967295.gguf"),
            None,
            "a 10-digit shard field must be rejected before shard_set can expand it"
        );
        assert_eq!(parse_shard("m-000001-of-999999.gguf"), None, "6 digits");
        // The real llama.cpp shapes still parse: 5 digits is gguf-split's own width, and narrower
        // fields are accepted as long as both sides agree.
        assert_eq!(parse_shard("m-99999-of-99999.gguf").unwrap().total, 99999);
        assert_eq!(parse_shard("m-01-of-12.gguf").unwrap().total, 12);
        assert_eq!(
            shard_set("m-Q4_K_M-00001-of-00002.gguf").len(),
            2,
            "the ordinary sharded case must be unaffected"
        );
    }

    #[test]
    fn shard_set_enumerates_full_set() {
        assert_eq!(
            shard_set("m-Q4_K_M-00001-of-00003.gguf"),
            vec![
                "m-Q4_K_M-00001-of-00003.gguf",
                "m-Q4_K_M-00002-of-00003.gguf",
                "m-Q4_K_M-00003-of-00003.gguf",
            ]
        );
        // The set is the same whichever member names it — a loader handed shard 3 still finds
        // shard 1, which is the only shard carrying the model's metadata.
        assert_eq!(
            shard_set("m-Q4_K_M-00003-of-00003.gguf"),
            shard_set("m-Q4_K_M-00001-of-00003.gguf")
        );
        // A non-shard file is its own singleton set.
        assert_eq!(shard_set("m-Q4_K_M.gguf"), vec!["m-Q4_K_M.gguf"]);
    }
}
