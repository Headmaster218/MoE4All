use anyhow::{anyhow, bail, Context};
use base64::Engine as _;
use flate2::{bufread::GzDecoder, write::GzEncoder, Compression};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

pub const ARMOR_VERSION: &str = "MOE4ALL-SUBMISSION/1";
const PAYLOAD_BEGIN: &str = "-----BEGIN MOE4ALL PAYLOAD-----";
const PAYLOAD_END: &str = "-----END MOE4ALL PAYLOAD-----";
pub const MAX_ARMOR_BYTES: usize = 128 * 1024;
pub const MAX_DECODED_BYTES: usize = 128 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Benchmark,
    Error,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Benchmark => "benchmark",
            Self::Error => "error",
        }
    }

    pub fn schema(self) -> &'static str {
        match self {
            Self::Benchmark => "benchmark/1",
            Self::Error => "error/1",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Benchmark => "m4b",
            Self::Error => "m4e",
        }
    }

    fn parse(raw: &str) -> anyhow::Result<Self> {
        match raw {
            "benchmark" => Ok(Self::Benchmark),
            "error" => Ok(Self::Error),
            _ => bail!("unsupported MoE4All submission kind `{raw}`"),
        }
    }
}

#[derive(Debug)]
pub struct Decoded {
    pub kind: Kind,
    pub schema: String,
    pub id: String,
    pub created_at: String,
    pub payload: Value,
}

fn one_line(value: impl AsRef<str>, max_chars: usize) -> String {
    let mut output = String::new();
    let mut previous_space = false;
    for ch in value.as_ref().chars() {
        let ch = if ch.is_control() { ' ' } else { ch };
        if ch.is_whitespace() {
            if previous_space {
                continue;
            }
            previous_space = true;
            output.push(' ');
        } else {
            previous_space = false;
            output.push(ch);
        }
        if output.chars().count() >= max_chars {
            output.push_str("...");
            break;
        }
    }
    output.trim().to_string()
}

fn value_str<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str()
}

fn value_u64(value: &Value, path: &[&str]) -> Option<u64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current
        .as_f64()
        .filter(|v| (0.0..=9_007_199_254_740_991.0).contains(v) && v.fract() == 0.0)
        .map(|v| v as u64)
}

fn value_f64(value: &Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_f64()
}

fn human_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
}

pub fn summary(kind: Kind, payload: &Value) -> Vec<String> {
    match kind {
        Kind::Benchmark => benchmark_summary(payload),
        Kind::Error => error_summary(payload),
    }
}

fn benchmark_summary(payload: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    let model = value_str(payload, &["model", "name"]).unwrap_or("unknown model");
    let architecture = value_str(payload, &["model", "architecture"]).unwrap_or("unknown");
    let model_size = value_u64(payload, &["model", "storage_bytes"])
        .map(human_gib)
        .unwrap_or_else(|| "unknown size".into());
    lines.push(format!("model: {model} | {architecture} | {model_size}"));

    let gpu = value_str(payload, &["hardware", "gpu", "name"]).unwrap_or("unknown device");
    let vram = value_u64(payload, &["hardware", "gpu", "vram_total_bytes"])
        .map(human_gib)
        .unwrap_or_else(|| "unknown VRAM".into());
    lines.push(format!("device: {gpu} | {vram}"));
    if let Some(profile) = payload.get("profile").and_then(Value::as_str) {
        lines.push(format!("profile: {profile}"));
    }

    if let Some(contexts) = payload.get("contexts").and_then(Value::as_array) {
        for context in contexts {
            let tokens = value_u64(context, &["requested_context_tokens"]).unwrap_or(0);
            let label = if tokens > 0 && tokens.is_multiple_of(1024) {
                format!("{}K", tokens / 1024)
            } else {
                tokens.to_string()
            };
            let mut result = Vec::new();
            if let Some(rate) = value_f64(context, &["prefill", "tokens_per_second"]) {
                result.push(format!("prefill={rate:.2}"));
            }
            if let Some(cases) = context.get("decode").and_then(Value::as_array) {
                for case in cases {
                    let parallel = value_u64(case, &["parallel"]).unwrap_or(0);
                    if let Some(rate) = value_f64(case, &["overall_tokens_per_second"]) {
                        result.push(format!("x{parallel}={rate:.2}"));
                    }
                }
            }
            if let Some(manual) = context.get("manual_result") {
                if let Some(rate) = value_f64(manual, &["average_tokens_per_second"]) {
                    result.push(format!("manual={rate:.2}"));
                }
            }
            if !result.is_empty() {
                lines.push(format!("result {label}: {} tok/s", result.join(" | ")));
            }
        }
    }
    lines
}

fn error_summary(payload: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    let failure_kind = value_str(payload, &["failure_kind"]).unwrap_or("unknown");
    lines.push(format!("failure: {failure_kind}"));
    if let Some(model) = value_str(payload, &["model", "name"]) {
        lines.push(format!("model: {model}"));
    }
    if let Some(device) = value_str(payload, &["hardware", "gpu", "name"]) {
        lines.push(format!("device: {device}"));
    }
    if let Some(description) = payload.get("description").and_then(Value::as_str) {
        let description = one_line(description, 240);
        if !description.is_empty() {
            lines.push(format!("description: {description}"));
        }
    }
    if let Some(message) = value_str(payload, &["error", "message"]) {
        lines.push(format!("error: {}", one_line(message, 240)));
    }
    lines
}

fn required_string<'a>(payload: &'a Value, field: &str) -> anyhow::Result<&'a str> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("submission payload is missing string field `{field}`"))
}

pub fn encode(kind: Kind, payload: &Value) -> anyhow::Result<String> {
    let id = required_string(payload, "id")?;
    let created_at = required_string(payload, "created_at_utc")?;
    let engine = payload
        .get("engine_version")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let decoded = serde_json::to_vec(payload).context("serialize compact submission JSON")?;
    if decoded.len() > MAX_DECODED_BYTES {
        bail!(
            "submission payload is {} bytes; limit is {} bytes",
            decoded.len(),
            MAX_DECODED_BYTES
        );
    }
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&decoded)?;
    let compressed = encoder.finish()?;
    let digest = Sha256::digest(&decoded);
    let mut sha256 = String::with_capacity(64);
    for byte in digest {
        write!(&mut sha256, "{byte:02x}").expect("writing to String cannot fail");
    }
    let encoded = base64::engine::general_purpose::STANDARD.encode(&compressed);
    let summary = summary(kind, payload);

    let mut output = String::new();
    writeln!(&mut output, "{ARMOR_VERSION}")?;
    writeln!(&mut output, "kind: {}", kind.as_str())?;
    writeln!(&mut output, "schema: {}", kind.schema())?;
    writeln!(&mut output, "id: {}", one_line(id, 160))?;
    writeln!(&mut output, "created-at: {}", one_line(created_at, 80))?;
    writeln!(&mut output, "engine: {}", one_line(engine, 80))?;
    writeln!(&mut output, "encoding: gzip+base64")?;
    writeln!(&mut output, "decoded-bytes: {}", decoded.len())?;
    writeln!(&mut output, "compressed-bytes: {}", compressed.len())?;
    writeln!(&mut output, "sha256: {sha256}")?;
    writeln!(&mut output, "summary-count: {}", summary.len())?;
    for (index, line) in summary.iter().enumerate() {
        writeln!(
            &mut output,
            "summary-{:03}: {}",
            index + 1,
            one_line(line, 500)
        )?;
    }
    writeln!(&mut output)?;
    writeln!(&mut output, "{PAYLOAD_BEGIN}")?;
    for chunk in encoded.as_bytes().chunks(76) {
        output.push_str(std::str::from_utf8(chunk).expect("base64 is ASCII"));
        output.push('\n');
    }
    writeln!(&mut output, "{PAYLOAD_END}")?;
    if output.len() > MAX_ARMOR_BYTES {
        bail!(
            "submission text is {} bytes; limit is {} bytes",
            output.len(),
            MAX_ARMOR_BYTES
        );
    }
    Ok(output)
}

fn parse_usize(headers: &BTreeMap<String, String>, name: &str) -> anyhow::Result<usize> {
    headers
        .get(name)
        .ok_or_else(|| anyhow!("submission header is missing `{name}`"))?
        .parse::<usize>()
        .with_context(|| format!("submission header `{name}` is not an integer"))
}

pub fn decode(text: &str) -> anyhow::Result<Decoded> {
    if text.len() > MAX_ARMOR_BYTES {
        bail!("submission text exceeds the {} byte limit", MAX_ARMOR_BYTES);
    }
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = normalized.lines();
    if lines.next() != Some(ARMOR_VERSION) {
        bail!("not a {ARMOR_VERSION} document");
    }
    let mut headers = BTreeMap::new();
    let mut found_begin = false;
    for line in &mut lines {
        if line == PAYLOAD_BEGIN {
            found_begin = true;
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| anyhow!("invalid submission header line `{line}`"))?;
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            bail!("invalid submission header name `{name}`");
        }
        if headers
            .insert(name.clone(), value.trim().to_string())
            .is_some()
        {
            bail!("duplicate submission header `{name}`");
        }
    }
    if !found_begin {
        bail!("submission payload start marker is missing");
    }
    let mut encoded = String::new();
    let mut found_end = false;
    for line in &mut lines {
        if line == PAYLOAD_END {
            found_end = true;
            break;
        }
        if !line.trim().is_empty() {
            encoded.push_str(line.trim());
        }
    }
    if !found_end {
        bail!("submission payload end marker is missing");
    }
    if lines.any(|line| !line.trim().is_empty()) {
        bail!("unexpected text after submission payload");
    }
    if headers.get("encoding").map(String::as_str) != Some("gzip+base64") {
        bail!("submission encoding must be gzip+base64");
    }
    let kind = Kind::parse(
        headers
            .get("kind")
            .ok_or_else(|| anyhow!("submission header is missing `kind`"))?,
    )?;
    let schema = headers
        .get("schema")
        .cloned()
        .ok_or_else(|| anyhow!("submission header is missing `schema`"))?;
    if schema != kind.schema() {
        bail!(
            "submission schema `{schema}` does not match kind `{}`",
            kind.as_str()
        );
    }
    let compressed = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .context("decode submission Base64 payload")?;
    let expected_compressed = parse_usize(&headers, "compressed-bytes")?;
    if compressed.len() != expected_compressed {
        bail!(
            "compressed payload length {} does not match header {}",
            compressed.len(),
            expected_compressed
        );
    }
    let mut decoder = GzDecoder::new(Cursor::new(compressed.as_slice()));
    let mut decoded = Vec::new();
    decoder
        .by_ref()
        .take((MAX_DECODED_BYTES + 1) as u64)
        .read_to_end(&mut decoded)
        .context("decompress submission gzip payload")?;
    if decoder.get_ref().position() != compressed.len() as u64 {
        bail!("submission gzip payload must contain exactly one member and no trailing bytes");
    }
    if decoded.len() > MAX_DECODED_BYTES {
        bail!(
            "decoded submission exceeds the {} byte limit",
            MAX_DECODED_BYTES
        );
    }
    let expected_decoded = parse_usize(&headers, "decoded-bytes")?;
    if decoded.len() != expected_decoded {
        bail!(
            "decoded payload length {} does not match header {}",
            decoded.len(),
            expected_decoded
        );
    }
    let digest = Sha256::digest(&decoded);
    let actual_sha = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let expected_sha = headers
        .get("sha256")
        .ok_or_else(|| anyhow!("submission header is missing `sha256`"))?;
    if actual_sha != *expected_sha {
        bail!("submission SHA-256 mismatch");
    }
    let payload = crate::submission_validation::parse_json(&decoded)
        .context("parse submission JSON payload")?;
    let id = required_string(&payload, "id")?.to_string();
    let created_at = required_string(&payload, "created_at_utc")?.to_string();
    if headers.get("id") != Some(&id) {
        bail!("submission header id does not match payload id");
    }
    if headers.get("created-at") != Some(&created_at) {
        bail!("submission header created-at does not match payload timestamp");
    }
    let engine = payload
        .get("engine_version")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if headers.get("engine").map(String::as_str) != Some(one_line(engine, 80).as_str()) {
        bail!("submission header engine does not match payload engine_version");
    }
    let declared_summary = parse_usize(&headers, "summary-count")?;
    let expected_summary = summary(kind, &payload);
    if declared_summary != expected_summary.len() {
        bail!("submission summary count does not match decoded payload");
    }
    for (index, expected) in expected_summary.iter().enumerate() {
        let name = format!("summary-{:03}", index + 1);
        if headers.get(&name).map(String::as_str) != Some(one_line(expected, 500).as_str()) {
            bail!("submission header `{name}` does not match decoded payload");
        }
    }
    let summary_headers = headers
        .keys()
        .filter(|name| name.starts_with("summary-") && name.as_str() != "summary-count")
        .count();
    if summary_headers != declared_summary {
        bail!("submission contains unexpected summary headers");
    }
    Ok(Decoded {
        kind,
        schema,
        id,
        created_at,
        payload,
    })
}

pub fn read(path: &Path) -> anyhow::Result<Decoded> {
    if std::fs::metadata(path)?.len() > MAX_ARMOR_BYTES as u64 {
        bail!("submission file exceeds the {} byte limit", MAX_ARMOR_BYTES);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read MoE4All submission {}", path.display()))?;
    decode(&text).with_context(|| format!("decode MoE4All submission {}", path.display()))
}

pub fn write_new(directory: &Path, kind: Kind, payload: &Value) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(directory)
        .with_context(|| format!("create submission directory {}", directory.display()))?;
    let id = required_string(payload, "id")?;
    let path = directory.join(format!("{id}.{}", kind.extension()));
    let text = encode(kind, payload)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("create submission {}", path.display()))?;
    file.write_all(text.as_bytes())?;
    file.flush()?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_summary_vectors_and_envelopes() {
        let vectors: Value = serde_json::from_str(include_str!(
            "../../../scripts/fixtures/submission-vectors.json"
        ))
        .unwrap();
        for case in vectors["one_line"].as_array().unwrap() {
            let input = case["text"].as_str().map(str::to_owned).unwrap_or_else(|| {
                case["unit"]
                    .as_str()
                    .unwrap()
                    .repeat(case["count"].as_u64().unwrap() as usize)
            });
            let expected = case["expected"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| {
                    case["unit"]
                        .as_str()
                        .unwrap()
                        .repeat(case["expected_count"].as_u64().unwrap() as usize)
                        + case["suffix"].as_str().unwrap()
                });
            assert_eq!(
                one_line(input, case["limit"].as_u64().unwrap() as usize),
                expected,
                "{}",
                case["name"]
            );
        }
        for kind in [Kind::Benchmark, Kind::Error] {
            let text = encode(kind, &vectors[kind.as_str()]).unwrap();
            assert_eq!(decode(&text).unwrap().payload, vectors[kind.as_str()]);
        }
    }

    fn sample() -> Value {
        serde_json::json!({
            "schema_version": 1,
            "id": "20261019T010203.004Z-00112233445566778899aabbccddeeff-000",
            "created_at_utc": "2026-10-19T01:02:03.004Z",
            "engine_version": "1.0.3",
            "profile": "quick",
            "model": {"name": "model.gguf", "architecture": "qwen35moe", "storage_bytes": 1234},
            "hardware": {"gpu": {"name": "Example GPU", "vram_total_bytes": 5678}},
            "contexts": [{
                "requested_context_tokens": 6144,
                "prefill": {"tokens_per_second": 1234.567},
                "decode": [{"parallel": 1, "overall_tokens_per_second": 98.765}],
                "failures": []
            }],
            "failures": []
        })
    }

    #[test]
    fn armor_round_trip_is_text_and_lossless() {
        let value = sample();
        let text = encode(Kind::Benchmark, &value).unwrap();
        assert!(text.starts_with("MOE4ALL-SUBMISSION/1\nkind: benchmark\n"));
        assert!(text.contains("summary-001: model: model.gguf"));
        assert_eq!(decode(&text).unwrap().payload, value);
    }

    #[test]
    fn armor_rejects_header_or_payload_tampering() {
        let text = encode(Kind::Benchmark, &sample()).unwrap();
        let bad_summary = text.replace("prefill=1234.57", "prefill=9999.99");
        assert!(decode(&bad_summary)
            .unwrap_err()
            .to_string()
            .contains("summary"));
        let bad_payload = text.replacen('H', "I", 1);
        assert!(decode(&bad_payload).is_err());
    }

    #[test]
    fn armor_accepts_crlf_for_copy_paste() {
        let text = encode(Kind::Benchmark, &sample())
            .unwrap()
            .replace('\n', "\r\n");
        assert_eq!(decode(&text).unwrap().id, sample()["id"].as_str().unwrap());
    }

    #[test]
    fn armor_rejects_trailing_gzip_bytes() {
        let text = encode(Kind::Benchmark, &sample()).unwrap();
        let (header, payload) = text.split_once(PAYLOAD_BEGIN).unwrap();
        let (encoded, _) = payload.split_once(PAYLOAD_END).unwrap();
        let mut compressed = base64::engine::general_purpose::STANDARD
            .decode(encoded.lines().collect::<String>())
            .unwrap();
        let old_len = compressed.len();
        compressed.push(0);
        let header = header.replace(
            &format!("compressed-bytes: {old_len}"),
            &format!("compressed-bytes: {}", compressed.len()),
        );
        let encoded = base64::engine::general_purpose::STANDARD.encode(compressed);
        let mut tampered = format!("{header}{PAYLOAD_BEGIN}\n");
        for chunk in encoded.as_bytes().chunks(76) {
            tampered.push_str(std::str::from_utf8(chunk).unwrap());
            tampered.push('\n');
        }
        tampered.push_str(PAYLOAD_END);
        tampered.push('\n');
        assert!(decode(&tampered)
            .unwrap_err()
            .to_string()
            .contains("exactly one member"));
    }

    #[test]
    fn armor_rejects_decoded_or_encoded_payloads_above_128_kib() {
        let mut decoded_too_large = sample();
        decoded_too_large["padding"] = Value::String("x".repeat(MAX_DECODED_BYTES));
        assert!(encode(Kind::Benchmark, &decoded_too_large)
            .unwrap_err()
            .to_string()
            .contains("payload"));

        let mut armor_too_large = sample();
        let mut bytes = vec![0u8; 120_000];
        getrandom::fill(&mut bytes).unwrap();
        let noisy = bytes
            .into_iter()
            .map(|byte| char::from(32 + byte % 95))
            .collect::<String>();
        armor_too_large["padding"] = Value::String(noisy);
        let error = encode(Kind::Benchmark, &armor_too_large).unwrap_err();
        assert!(error.to_string().contains("submission text"));
    }
}
