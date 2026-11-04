//! Small, deterministic frontend for the external Melo Chinese voice package.
//!
//! Uses the voice's own phones/tones lexicon. Unknown text fails explicitly;
//! user-supplied `custom-lexicon.txt` entries override the base lexicon. English
//! acronyms in uppercase may be spelled using the lexicon's letter readings.
use anyhow::{bail, Context, Result};
use std::{collections::HashMap, fs, path::Path};

#[derive(Clone, Debug, PartialEq)]
pub struct FrontendOutput {
    pub phones: Vec<i32>,
    pub tones: Vec<i32>,
    pub languages: Vec<i32>,
}

#[derive(Clone)]
struct Entry {
    phones: Vec<i32>,
    tones: Vec<i32>,
}

pub struct Frontend {
    lexicon: HashMap<String, Entry>,
    tokens: HashMap<String, i32>,
    max_word_chars: usize,
    language_id: i32,
    add_blank: bool,
    num_tones: i32,
}

impl Frontend {
    /// Accept a model package directory, or any file inside that directory.
    pub fn load(path: &Path) -> Result<Self> {
        let dir = if path.is_dir() {
            path
        } else {
            path.parent().context("Model has no package directory")?
        };
        let config: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
        let mut tokens = HashMap::new();
        for (line_index, line) in fs::read_to_string(dir.join("tokens.txt"))?
            .lines()
            .enumerate()
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() != 2 {
                bail!("Invalid tokens.txt line {}", line_index + 1);
            }
            let id: i32 = fields[1].parse()?;
            if id < 0 || tokens.insert(fields[0].to_owned(), id).is_some() {
                bail!("Duplicate token or invalid ID at line {}", line_index + 1);
            }
        }
        if tokens.get("_") != Some(&0) {
            bail!("Melo Chinese package must use token '_' as blank ID 0");
        }
        let mut this = Self {
            lexicon: HashMap::new(),
            tokens,
            max_word_chars: 1,
            language_id: 3,
            add_blank: config["data"]["add_blank"].as_bool().unwrap_or(true),
            num_tones: 16,
        };
        let frontend_config = dir.join("frontend.json");
        if frontend_config.exists() {
            let frontend: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(frontend_config)?)?;
            this.num_tones = frontend["num_tones"].as_i64().unwrap_or(16) as i32;
            if frontend["language_id"].as_i64().unwrap_or(3) != 3 {
                bail!("This frontend supports the Melo ZH_MIX_EN language ID 3 only");
            }
        }
        this.add_lexicon(&fs::read_to_string(dir.join("lexicon.txt"))?, false)?;
        let custom = dir.join("custom-lexicon.txt");
        if custom.exists() {
            this.add_lexicon(&fs::read_to_string(custom)?, true)?;
        }
        for (alias, original) in [("嗯", "恩"), ("呣", "母")] {
            if !this.lexicon.contains_key(alias) {
                if let Some(entry) = this.lexicon.get(original).cloned() {
                    this.lexicon.insert(alias.to_owned(), entry);
                }
            }
        }
        Ok(this)
    }

    fn add_lexicon(&mut self, text: &str, replace: bool) -> Result<()> {
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() < 3 || (fields.len() - 1) % 2 != 0 {
                bail!(
                    "Invalid Melo lexicon line {}: expected word, phones, tones",
                    i + 1
                );
            }
            let count = (fields.len() - 1) / 2;
            let phones = fields[1..1 + count]
                .iter()
                .map(|p| {
                    self.tokens
                        .get(*p)
                        .copied()
                        .with_context(|| format!("Unknown phone {p:?} at lexicon line {}", i + 1))
                })
                .collect::<Result<Vec<_>>>()?;
            let tones = fields[1 + count..]
                .iter()
                .map(|t| {
                    let tone: i32 = t.parse()?;
                    if !(0..self.num_tones).contains(&tone) {
                        bail!("Invalid tone {tone} at lexicon line {}", i + 1);
                    }
                    Ok(tone)
                })
                .collect::<Result<Vec<_>>>()?;
            let word = fields[0].to_lowercase();
            self.max_word_chars = self.max_word_chars.max(word.chars().count());
            if replace || !self.lexicon.contains_key(&word) {
                self.lexicon.insert(word, Entry { phones, tones });
            }
        }
        Ok(())
    }

    /// Encode as one sequence; calls may choose sentence-level scheduling instead.
    pub fn encode(&self, text: &str) -> Result<FrontendOutput> {
        let normalized = normalize_text(text)?;
        self.encode_normalized(&normalized)
    }

    pub fn encode_sentences(&self, text: &str) -> Result<Vec<FrontendOutput>> {
        let normalized = normalize_text(text)?;
        let mut result = Vec::new();
        let mut start = 0;
        for (i, c) in normalized.char_indices() {
            if matches!(c, '.' | '!' | '?' | ',' | ';') {
                let end = i + c.len_utf8();
                let sentence = normalized[start..end].trim();
                if sentence.chars().any(|c| c.is_alphanumeric()) {
                    result.push(self.encode_normalized(sentence)?);
                }
                start = end;
            }
        }
        let tail = normalized[start..].trim();
        if tail.chars().any(|c| c.is_alphanumeric()) {
            result.push(self.encode_normalized(tail)?);
        }
        if result.is_empty() {
            bail!("No speakable text");
        }
        Ok(result)
    }

    fn encode_normalized(&self, text: &str) -> Result<FrontendOutput> {
        let mut phones = Vec::new();
        let mut tones = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            let tail = &text[pos..];
            let c = tail.chars().next().unwrap();
            if c.is_whitespace() {
                if !phones.is_empty() && !tail.trim().is_empty() && phones.last() != Some(&0) {
                    phones.push(0);
                    tones.push(0);
                }
                pos += c.len_utf8();
                continue;
            }
            if c.is_ascii_alphabetic() {
                let len = tail
                    .bytes()
                    .take_while(|b| b.is_ascii_alphabetic() || *b == b'\'')
                    .count();
                let word = &tail[..len];
                let lower = word.to_ascii_lowercase();
                if let Some(e) = self.lexicon.get(&lower) {
                    phones.extend_from_slice(&e.phones);
                    tones.extend_from_slice(&e.tones);
                } else if word.len() <= 12 && word.bytes().all(|b| b.is_ascii_uppercase()) {
                    // Explicit acronym spelling, using the voice's actual English
                    // letter entries; unrecognized ordinary English never disappears.
                    for letter in lower.chars() {
                        let e = self.lexicon.get(&letter.to_string()).with_context(|| {
                            format!("Cannot spell acronym {word:?}; add it to custom-lexicon.txt")
                        })?;
                        phones.extend_from_slice(&e.phones);
                        tones.extend_from_slice(&e.tones);
                    }
                } else {
                    bail!("Unknown English term {word:?}; add its pronunciation to custom-lexicon.txt");
                }
                pos += len;
                continue;
            }
            let mut ends: Vec<_> = tail
                .char_indices()
                .take(self.max_word_chars)
                .map(|(i, c)| i + c.len_utf8())
                .collect();
            let mut found = None;
            while let Some(end) = ends.pop() {
                if let Some(e) = self.lexicon.get(&tail[..end]) {
                    found = Some((end, e));
                    break;
                }
            }
            if let Some((len, e)) = found {
                phones.extend_from_slice(&e.phones);
                tones.extend_from_slice(&e.tones);
                pos += len;
            } else if let Some(id) = self.tokens.get(&c.to_string()) {
                phones.push(*id);
                tones.push(0);
                pos += c.len_utf8();
            } else {
                bail!("Unknown character {c:?} at normalized byte {pos}; add its pronunciation to custom-lexicon.txt");
            }
        }
        if phones.is_empty() {
            bail!("No speakable text");
        }
        let mut languages = vec![self.language_id; phones.len()];
        if self.add_blank {
            fn intersperse(values: Vec<i32>) -> Vec<i32> {
                let mut out = Vec::with_capacity(values.len() * 2 + 1);
                out.push(0);
                for v in values {
                    out.push(v);
                    out.push(0);
                }
                out
            }
            phones = intersperse(phones);
            tones = intersperse(tones);
            languages = intersperse(languages);
        }
        Ok(FrontendOutput {
            phones,
            tones,
            languages,
        })
    }
}

const DIGITS: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
fn digits(text: &str) -> String {
    text.bytes().map(|c| DIGITS[(c - b'0') as usize]).collect()
}

fn integer(text: &str) -> String {
    if text.len() > 12 || (text.len() > 1 && text.starts_with('0')) {
        return digits(text);
    }
    let mut n: u64 = text.parse().unwrap_or(0);
    if n == 0 {
        return "零".to_owned();
    }
    let mut groups = Vec::new();
    while n > 0 {
        groups.push((n % 10000) as u32);
        n /= 10000;
    }
    let mut out = String::new();
    let mut skipped = false;
    for i in (0..groups.len()).rev() {
        let g = groups[i];
        if g == 0 {
            skipped = true;
            continue;
        }
        if !out.is_empty() && (skipped || g < 1000) {
            out.push('零');
        }
        let mut inner = String::new();
        let mut zero = false;
        for p in (0..4).rev() {
            let d = (g / 10u32.pow(p)) % 10;
            if d == 0 {
                if !inner.is_empty() {
                    zero = true;
                }
                continue;
            }
            if zero {
                inner.push('零');
                zero = false;
            }
            if !(d == 1 && p == 1 && inner.is_empty() && out.is_empty()) {
                inner.push(DIGITS[d as usize]);
            }
            inner.push_str(["", "十", "百", "千"][p as usize]);
        }
        out.push_str(&inner);
        out.push_str(["", "万", "亿"][i]);
        skipped = false;
    }
    out
}

/// Deliberately bounded normalization. Unsupported symbols fail during encoding,
/// rather than stripping e.g. formula operators and pretending they were spoken.
pub fn normalize_text(text: &str) -> Result<String> {
    if text.len() > 64 * 1024 {
        bail!("TTS input exceeds 64KiB; split the document first");
    }
    let mut out = String::new();
    let mut i = 0;
    while i < text.len() {
        let tail = &text[i..];
        let c = tail.chars().next().unwrap();
        if c.is_ascii_digit() {
            let count = tail.bytes().take_while(u8::is_ascii_digit).count();
            // ISO-like dates: explicit calendar date, never silently read a path.
            if count == 4 && tail.len() >= 8 && matches!(tail.as_bytes()[4], b'-' | b'/') {
                let sep = tail.as_bytes()[4];
                let rest = &tail[5..];
                let mlen = rest.bytes().take_while(u8::is_ascii_digit).count();
                if (1..=2).contains(&mlen) && rest.as_bytes().get(mlen) == Some(&sep) {
                    let day_part = &rest[mlen + 1..];
                    let dlen = day_part.bytes().take_while(u8::is_ascii_digit).count();
                    if (1..=2).contains(&dlen) {
                        let year: u32 = tail[..4].parse()?;
                        let month: u32 = rest[..mlen].parse()?;
                        let day: u32 = day_part[..dlen].parse()?;
                        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
                        let days = match month {
                            2 => {
                                if leap {
                                    29
                                } else {
                                    28
                                }
                            }
                            4 | 6 | 9 | 11 => 30,
                            1..=12 => 31,
                            _ => 0,
                        };
                        if day == 0 || day > days {
                            bail!("Invalid calendar date near {tail:?}");
                        }
                        out.push_str(&digits(&tail[..4]));
                        out.push('年');
                        out.push_str(&integer(&month.to_string()));
                        out.push('月');
                        out.push_str(&integer(&day.to_string()));
                        out.push('日');
                        i += 5 + mlen + 1 + dlen;
                        continue;
                    }
                }
            }
            let mut end = count;
            let mut spoken = if tail[count..].starts_with('年') || count >= 7 {
                digits(&tail[..count])
            } else {
                integer(&tail[..count])
            };
            if tail.as_bytes().get(end) == Some(&b'.')
                && tail.as_bytes().get(end + 1).is_some_and(u8::is_ascii_digit)
            {
                let dcount = tail[end + 1..]
                    .bytes()
                    .take_while(u8::is_ascii_digit)
                    .count();
                spoken.push('点');
                spoken.push_str(&digits(&tail[end + 1..end + 1 + dcount]));
                end += 1 + dcount;
            }
            if tail[end..].starts_with('%') {
                out.push_str("百分之");
                end += 1;
            }
            out.push_str(&spoken);
            // Numeric units only; ordinary words and identifiers are untouched.
            let remaining = &tail[end..];
            let units = [
                ("GHz", "吉赫兹"),
                ("MHz", "兆赫兹"),
                ("kHz", "千赫兹"),
                ("Hz", "赫兹"),
                ("GiB", "吉比字节"),
                ("MiB", "兆比字节"),
                ("GB", "吉字节"),
                ("MB", "兆字节"),
                ("ms", "毫秒"),
                ("μs", "微秒"),
                ("us", "微秒"),
                ("mm", "毫米"),
                ("mA", "毫安"),
                ("°C", "摄氏度"),
            ];
            for (unit, reading) in units {
                if let Some(after) = remaining.strip_prefix(unit) {
                    if !after
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_alphabetic())
                    {
                        out.push_str(reading);
                        end += unit.len();
                        break;
                    }
                }
            }
            i += end;
            continue;
        }
        let replacement = match c {
            '，' | '、' | '：' | ':' | '；' | ';' => ",",
            '。' | '\n' | '\r' => ".",
            '！' => "!",
            '？' => "?",
            '“' | '”' | '‘' | '’' | '「' | '」' | '（' | '）' | '(' | ')' | '《' | '》' | '['
            | ']' => " ",
            '—' | '–' => ",",
            '…' => "…",
            _ => "",
        };
        if c == '-' && tail.as_bytes().get(1).is_some_and(u8::is_ascii_digit) {
            out.push('负');
        } else if replacement.is_empty() {
            out.push(c);
        } else {
            out.push_str(replacement);
        }
        i += c.len_utf8();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frontend() -> Frontend {
        let tokens = [
            ("_", 0),
            ("y", 1),
            ("in", 2),
            ("h", 3),
            ("ang", 4),
            ("x", 5),
            ("ing", 6),
            (".", 7),
            ("s", 8),
            ("iy", 9),
            ("p", 10),
            ("uw", 11),
        ]
        .into_iter()
        .map(|(s, n)| (s.to_owned(), n))
        .collect();
        let mut f = Frontend {
            tokens,
            lexicon: HashMap::new(),
            max_word_chars: 1,
            language_id: 3,
            add_blank: true,
            num_tones: 16,
        };
        f.add_lexicon(
            "银行 y in h ang 2 2 2 2\n行 x ing 2 2\nc s iy 7 9\np p iy 7 9\nu y uw 7 9\n",
            false,
        )
        .unwrap();
        f
    }

    #[test]
    fn phrase_pronunciation_and_blank_language_contract() {
        let out = frontend().encode("银行。行").unwrap();
        assert_eq!(
            out.phones,
            vec![0, 1, 0, 2, 0, 3, 0, 4, 0, 7, 0, 5, 0, 6, 0]
        );
        assert_eq!(out.tones, vec![0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2, 0]);
        assert!(out
            .languages
            .iter()
            .enumerate()
            .all(|(i, x)| *x == if i % 2 == 0 { 0 } else { 3 }));
    }

    #[test]
    fn unknown_english_and_symbols_are_errors_acronyms_are_spelled() {
        let f = frontend();
        assert!(f
            .encode("vulkan")
            .unwrap_err()
            .to_string()
            .contains("custom-lexicon"));
        assert!(f.encode("🙂").is_err());
        assert!(f.encode("CPU").is_ok());
        assert!(f.encode("cpu").is_err());
    }

    #[test]
    fn custom_term_overrides_and_invalid_phone_is_rejected() {
        let mut f = frontend();
        f.add_lexicon("行 h ang 2 2", true).unwrap();
        assert_eq!(f.encode("行").unwrap().phones, vec![0, 3, 0, 4, 0]);
        assert!(f.add_lexicon("gpu nonexistent 2", true).is_err());
    }

    #[test]
    fn bounded_number_date_unit_normalization() {
        assert_eq!(
            normalize_text("2026-11-04，耗时3.14ms，提升12.5%。").unwrap(),
            "二零二六年十一月四日,耗时三点一四毫秒,提升百分之十二点五."
        );
        assert_eq!(
            normalize_text("2024年，10001，010，110，-20").unwrap(),
            "二零二四年,一万零一,零一零,一百一十,负二十"
        );
        assert_eq!(
            normalize_text("0.5GB，128MHz").unwrap(),
            "零点五吉字节,一百二十八兆赫兹"
        );
        assert!(normalize_text("2025-02-29").is_err());
        assert!(normalize_text("2024-02-29").is_ok());
    }

    #[test]
    fn sentence_boundaries_keep_punctuation_and_blank_once() {
        let f = frontend();
        let parts = f.encode_sentences("银行。行。").unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].phones.len(), 11);
        assert_eq!(parts[1].phones.len(), 7);
        assert!(f.encode_sentences("  。").is_err());
    }
}
