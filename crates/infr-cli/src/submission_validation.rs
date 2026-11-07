use anyhow::{bail, ensure, Context, Result};
use regex::Regex;
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::LazyLock;

// The wire format must not silently discard duplicate keys before validation.
pub fn parse_json(bytes: &[u8]) -> Result<Value> {
    struct Strict(Value);
    impl<'de> Deserialize<'de> for Strict {
        fn deserialize<D: Deserializer<'de>>(
            deserializer: D,
        ) -> std::result::Result<Self, D::Error> {
            struct JsonVisitor;
            impl<'de> Visitor<'de> for JsonVisitor {
                type Value = Strict;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("interoperable JSON")
                }
                fn visit_bool<E: serde::de::Error>(
                    self,
                    v: bool,
                ) -> std::result::Result<Strict, E> {
                    Ok(Strict(v.into()))
                }
                fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<Strict, E> {
                    Ok(Strict(v.into()))
                }
                fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<Strict, E> {
                    Ok(Strict(v.into()))
                }
                fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<Strict, E> {
                    serde_json::Number::from_f64(v)
                        .map(|v| Strict(Value::Number(v)))
                        .ok_or_else(|| E::custom("invalid number"))
                }
                fn visit_str<E: serde::de::Error>(self, v: &str) -> std::result::Result<Strict, E> {
                    Ok(Strict(v.into()))
                }
                fn visit_string<E: serde::de::Error>(
                    self,
                    v: String,
                ) -> std::result::Result<Strict, E> {
                    Ok(Strict(v.into()))
                }
                fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Strict, E> {
                    Ok(Strict(Value::Null))
                }
                fn visit_seq<A: SeqAccess<'de>>(
                    self,
                    mut access: A,
                ) -> std::result::Result<Strict, A::Error> {
                    let mut values = Vec::new();
                    while let Some(Strict(value)) = access.next_element()? {
                        values.push(value);
                    }
                    Ok(Strict(Value::Array(values)))
                }
                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut access: A,
                ) -> std::result::Result<Strict, A::Error> {
                    let mut values = serde_json::Map::new();
                    while let Some(key) = access.next_key::<String>()? {
                        if values.contains_key(&key)
                            || matches!(key.as_str(), "__proto__" | "constructor" | "prototype")
                        {
                            return Err(serde::de::Error::custom(
                                "duplicate or reserved JSON field",
                            ));
                        }
                        let Strict(value) = access.next_value()?;
                        values.insert(key, value);
                    }
                    Ok(Strict(Value::Object(values)))
                }
            }
            deserializer.deserialize_any(JsonVisitor)
        }
    }
    fn visit(value: &Value, depth: usize) -> Result<()> {
        ensure!(depth <= 48, "JSON nesting exceeds 48 levels");
        if let Some(number) = value.as_f64() {
            ensure!(
                number.is_finite()
                    && (number.fract() != 0.0 || number.abs() <= 9_007_199_254_740_991.0),
                "JSON number exceeds the interoperable range"
            );
        }
        match value {
            Value::Array(values) => {
                for value in values {
                    visit(value, depth + 1)?;
                }
            }
            Value::Object(values) => {
                for value in values.values() {
                    visit(value, depth + 1)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let Strict(value) = Strict::deserialize(&mut decoder)?;
    decoder.end()?;
    visit(&value, 0)?;
    Ok(value)
}

fn uint(v: &Value) -> bool {
    v.as_f64()
        .is_some_and(|v| (0.0..=9_007_199_254_740_991.0).contains(&v) && v.fract() == 0.0)
}
fn rate(v: &Value) -> Result<()> {
    ensure!(
        v.as_f64()
            .is_some_and(|v| v.is_finite() && (0.0..=1e8).contains(&v)),
        "invalid throughput"
    );
    Ok(())
}
fn optional_text(v: &Value) -> Result<()> {
    ensure!(
        v.is_null() || v.as_str().is_some_and(|s| s.chars().count() <= 1024),
        "invalid text field"
    );
    Ok(())
}
fn optional_capacity(v: &Value) -> Result<()> {
    ensure!(v.is_null() || uint(v), "invalid capacity");
    Ok(())
}
fn array(v: &Value) -> Result<&Vec<Value>> {
    v.as_array().context("expected array")
}

pub fn validate(kind: &str, p: &Value) -> Result<()> {
    static ID: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$").unwrap());
    ensure!(
        p.is_object() && p["schema_version"].as_f64() == Some(1.0),
        "schema_version must be 1"
    );
    ensure!(
        p["id"].as_str().is_some_and(|v| ID.is_match(v)),
        "invalid report id"
    );
    let created = p["created_at_utc"].as_str().context("missing timestamp")?;
    ensure!(
        created.ends_with('Z') || created.ends_with("+00:00"),
        "timestamp must be UTC"
    );
    ensure!(
        chrono::DateTime::parse_from_rfc3339(created)?
            .offset()
            .local_minus_utc()
            == 0,
        "timestamp must be UTC"
    );
    ensure!(
        p["engine_version"]
            .as_str()
            .is_some_and(|s| !s.is_empty() && s.chars().count() <= 128),
        "invalid engine version"
    );
    match kind {
        "benchmark" => {
            ensure!(
                matches!(
                    p["profile"].as_str(),
                    Some("quick" | "standard" | "full" | "manual")
                ),
                "invalid benchmark profile"
            );
            for field in ["model", "hardware", "configuration"] {
                ensure!(p[field].is_object(), "missing {field}");
            }
            ensure!(
                p["model"]["name"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty() && s.chars().count() <= 1024),
                "invalid model name"
            );
            for field in ["architecture", "quant", "quantization"] {
                optional_text(&p["model"][field])?;
            }
            optional_capacity(&p["model"]["storage_bytes"])?;
            optional_text(&p["hardware"]["os"])?;
            for field in ["gpu", "cpu", "ram"] {
                ensure!(
                    p["hardware"][field].is_null() || p["hardware"][field].is_object(),
                    "invalid hardware object"
                );
            }
            for (device, field) in [
                ("gpu", "name"),
                ("gpu", "configured_device"),
                ("cpu", "name"),
            ] {
                optional_text(&p["hardware"][device][field])?;
            }
            optional_capacity(&p["hardware"]["gpu"]["vram_total_bytes"])?;
            optional_capacity(&p["hardware"]["ram"]["total_bytes"])?;
            let configuration = &p["configuration"];
            for field in ["backend", "device", "kv_type_k", "kv_type_v"] {
                optional_text(&configuration[field])?;
            }
            ensure!(
                configuration["mtp"].is_null() || configuration["mtp"].is_boolean(),
                "invalid MTP flag"
            );
            for field in ["ubatch", "repetitions"] {
                optional_capacity(&configuration[field])?;
            }
            let contexts = array(&p["contexts"])?;
            ensure!(
                !contexts.is_empty() && contexts.len() <= 64,
                "invalid contexts count"
            );
            let mut seen_contexts = BTreeSet::new();
            for context in contexts {
                let tokens = &context["requested_context_tokens"];
                ensure!(
                    context.is_object()
                        && uint(tokens)
                        && seen_contexts.insert(tokens.as_f64().unwrap() as u64),
                    "invalid or duplicate context"
                );
                let failures = array(&context["failures"])?;
                let decode = array(&context["decode"])?;
                ensure!(decode.len() <= 64, "too many decode cases");
                if !context["prefill"].is_null() {
                    ensure!(context["prefill"].is_object(), "invalid prefill");
                    rate(&context["prefill"]["tokens_per_second"])?;
                }
                if p["profile"] == "manual" {
                    let manual = &context["manual_result"];
                    ensure!(
                        manual.is_object()
                            && matches!(manual["kind"].as_str(), Some("prefill" | "decode")),
                        "invalid manual result kind"
                    );
                    rate(&manual["average_tokens_per_second"])?;
                } else {
                    ensure!(
                        !context["prefill"].is_null() || !failures.is_empty(),
                        "missing prefill result or failure"
                    );
                }
                let mut slots = BTreeSet::new();
                for case in decode {
                    let parallel = &case["parallel"];
                    ensure!(
                        case.is_object()
                            && uint(parallel)
                            && (1.0..=64.0).contains(&parallel.as_f64().unwrap())
                            && slots.insert(parallel.as_f64().unwrap() as u64),
                        "invalid or duplicate parallel"
                    );
                    rate(&case["overall_tokens_per_second"])?;
                    let means = array(&case["section_mean_tokens_per_second"])?;
                    ensure!(means.len() == 3, "expected three section means");
                    for value in means {
                        rate(value)?;
                    }
                    let repetitions = array(&case["repetitions"])?;
                    ensure!(
                        !repetitions.is_empty() && repetitions.len() <= 100,
                        "invalid repetitions"
                    );
                    for rep in repetitions {
                        let sections = array(&rep["sections"])?;
                        ensure!(
                            sections.len() == 3 && rep["overall"].is_object(),
                            "invalid measurement sections"
                        );
                        for section in sections.iter().chain(std::iter::once(&rep["overall"])) {
                            ensure!(
                                section.is_object()
                                    && uint(&section["tokens"])
                                    && section["seconds"]
                                        .as_f64()
                                        .is_some_and(|v| v.is_finite() && v >= 0.0),
                                "invalid section tokens or seconds"
                            );
                            rate(&section["tokens_per_second"])?;
                        }
                        let lanes = array(&rep["lane_tokens_per_second"])?;
                        ensure!(lanes.len() == 3, "expected three lane groups");
                        for lane in lanes {
                            let lane = array(lane)?;
                            ensure!(
                                lane.len() == parallel.as_f64().unwrap() as usize,
                                "lane count differs from parallel"
                            );
                            for value in lane {
                                rate(value)?;
                            }
                        }
                    }
                }
            }
        }
        "error" => {
            const ALLOWED: &[&str] = &[
                "schema_version",
                "id",
                "created_at_utc",
                "engine_version",
                "engine_commit",
                "failure_kind",
                "description",
                "engine",
                "hardware",
                "model",
                "launch",
                "error",
                "logs",
                "privacy",
            ];
            ensure!(
                p.as_object()
                    .unwrap()
                    .keys()
                    .all(|k| ALLOWED.contains(&k.as_str())),
                "unrecognized error field"
            );
            ensure!(
                p["failure_kind"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty() && s.chars().count() <= 80),
                "invalid failure kind"
            );
            ensure!(
                p["description"]
                    .as_str()
                    .is_some_and(|s| !s.trim().is_empty() && s.chars().count() <= 4096),
                "description must contain 1..4096 code points"
            );
            for field in [
                "engine", "hardware", "model", "launch", "error", "logs", "privacy",
            ] {
                ensure!(p[field].is_object(), "missing {field}");
            }
            ensure!(
                p["error"]["message"].is_string(),
                "error message must be text"
            );
            let chain = array(&p["error"]["chain"])?;
            let startup = array(&p["logs"]["startup"])?;
            let tail = array(&p["logs"]["tail"])?;
            ensure!(
                startup.len() <= 220 && tail.len() <= 12,
                "diagnostic line count exceeds limit"
            );
            ensure!(
                startup
                    .iter()
                    .chain(tail)
                    .chain(chain)
                    .all(Value::is_string),
                "diagnostics must be text lines"
            );
            ensure!(
                startup
                    .iter()
                    .chain(tail)
                    .map(|v| v.as_str().unwrap().len())
                    .sum::<usize>()
                    <= 32768,
                "diagnostics exceed 32 KiB"
            );
            ensure!(
                p["logs"]["speed_statistics_included"] == false
                    && p["privacy"]["prompt_or_response_included"] == false
                    && p["privacy"]["api_keys_included"] == false
                    && p["privacy"]["local_paths_redacted"] == true,
                "invalid privacy flags"
            );
        }
        _ => bail!("unknown submission kind"),
    }
    validate_privacy(p)
}

fn validate_privacy(value: &Value) -> Result<()> {
    static CREDENTIAL: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"(?i)\bBearer\s+([A-Za-z0-9._~+/-]{8,})|\bsk-[A-Za-z0-9_-]{12,}|\b(?:api[_-]?key|password|secret)\s*[=:]\s*["']?([A-Za-z0-9._~+/-]{8,})"#).unwrap()
    });
    static PATH: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)[A-Za-z]:[\\/]|/(?:home|Users)/").unwrap());
    match value {
        Value::String(text) => {
            ensure!(
                !CREDENTIAL.is_match(text),
                "submission contains a suspected credential"
            );
            for hit in PATH.find_iter(text) {
                ensure!(
                    text.as_bytes().get(hit.end()) == Some(&b'['),
                    "submission contains a local path"
                );
            }
        }
        Value::Array(values) => {
            for value in values {
                validate_privacy(value)?;
            }
        }
        Value::Object(values) => {
            for (key, value) in values {
                ensure!(
                    !matches!(
                        key.to_ascii_lowercase().as_str(),
                        "hostname"
                            | "serial_number"
                            | "serial"
                            | "username"
                            | "user_name"
                            | "prompt"
                            | "response"
                            | "messages"
                            | "api_key"
                            | "password"
                            | "private_redactions"
                    ),
                    "submission contains a private field"
                );
                validate_privacy(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_json_rejects_duplicate_reserved_and_unsafe_fields() {
        for text in [
            r#"{"x":1,"x":2}"#,
            r#"{"x":1,"\u0078":2}"#,
            r#"{"__proto__":{}}"#,
            r#"{"x":9007199254740992}"#,
            r#"{"x":1e400}"#,
        ] {
            assert!(parse_json(text.as_bytes()).is_err());
        }
    }
    #[test]
    fn shared_payload_vectors_match_upload_rules() {
        let vectors: Value = serde_json::from_str(include_str!(
            "../../../scripts/fixtures/submission-vectors.json"
        ))
        .unwrap();
        for kind in ["benchmark", "error"] {
            validate(kind, &vectors[kind]).unwrap();
        }
        for invalid in vectors["invalid"].as_array().unwrap() {
            let kind = invalid["kind"].as_str().unwrap();
            let mut payload = vectors[kind].clone();
            *payload
                .pointer_mut(invalid["path"].as_str().unwrap())
                .unwrap() = invalid["value"].clone();
            assert!(validate(kind, &payload).is_err(), "{invalid}");
        }
    }
}
