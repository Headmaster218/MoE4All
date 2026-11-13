//! Structured-output requests: ONE parser and ONE JSON Schema subset gate for both APIs.
//!
//! `/v1/chat/completions`'s `response_format` and `/v1/responses`'s `text.format` both normalise
//! into [`ResponseFormat`]; the handlers and the generator (`infr-cli`'s `run_chat` →
//! `infr_llama::grammar::output_constraint_for`) consume that one value, so the two wire shapes
//! cannot drift into two grammars.
//!
//! The constraint is applied from the FIRST generated token. That is a deliberate trade-off: the
//! model cannot think first and print JSON afterwards, because a reasoning prefix is not valid
//! JSON. `GenParams::from_request` therefore pins `chat_template_options.enable_thinking` to
//! `false` for structured requests (and rejects an explicit `true`), and `run_chat` refuses a
//! template that prefills a thinking block anyway.
//!
//! # Supported JSON Schema subset
//!
//! `type` (string or array of type names), `enum`, `const`, `properties`, `required`,
//! `additionalProperties` (bool or schema), `items` (one schema), `anyOf`, `allOf`,
//! `minLength`/`maxLength`, `minItems`/`maxItems`,
//! `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`, plus the annotations
//! `title`/`description`/`default`/`examples`/`$comment`/`$schema`/`deprecated`/`readOnly`/
//! `writeOnly` (ignored by decoding).
//!
//! Everything else is rejected with an explicit 400 that names the keyword — never a silent
//! downgrade to free text. In particular: `$ref`/`$defs`/`definitions` (inline instead),
//! `oneOf` (a grammar cannot enforce exactly-one), `not`, `if`/`then`/`else`, `pattern`/
//! `patternProperties`/`propertyNames`, `format`, `contains`, `uniqueItems`, `multipleOf`,
//! `dependent*`, `unevaluated*`, `minProperties`/`maxProperties`.
//!
//! The same subset drives [`ResponseFormat::verify`], the post-generation check: a backend that
//! ignored the constraint (or a decode truncated at `max_tokens`) can therefore never hand a
//! client JSON that merely LOOKS structured.

use serde_json::{Map, Value};

use crate::ParamError;

/// Upper bounds on a schema we are willing to compile per request. Kept small enough that grammar
/// construction cannot become a request-level denial of service.
const MAX_SCHEMA_BYTES: usize = 64 * 1024;
const MAX_SCHEMA_DEPTH: usize = 32;
const MAX_SCHEMA_NODES: usize = 4096;

/// Normalised `response_format` / `text.format`.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: Option<String>,
        schema: Value,
        strict: bool,
    },
}

impl ResponseFormat {
    pub fn is_structured(&self) -> bool {
        !matches!(self, Self::Text)
    }

    /// The schema the decode must enforce; `None` for plain text. `json_object` is the loosest
    /// schema that still forces a JSON object.
    pub fn schema(&self) -> Option<Value> {
        match self {
            Self::Text => None,
            Self::JsonObject => Some(serde_json::json!({ "type": "object" })),
            Self::JsonSchema { schema, .. } => Some(schema.clone()),
        }
    }

    /// Verify a completed reply against the requested format. `Text` always passes (returning
    /// null); a structured format must parse as ONE JSON value and satisfy the schema subset.
    /// The error string is the message shown to the client — it must say what failed, not just
    /// that something did.
    pub fn verify(&self, text: &str) -> Result<Value, String> {
        if !self.is_structured() {
            return Ok(Value::Null);
        }
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(
                "structured output produced no content: the constrained decode ended without \
                 emitting JSON"
                    .into(),
            );
        }
        let value: Value = serde_json::from_str(trimmed).map_err(|error| {
            format!(
                "structured output is not valid JSON ({error}); the constrained decode did not \
                 complete (max_tokens too small?)"
            )
        })?;
        match self {
            Self::JsonObject => {
                if !value.is_object() {
                    return Err(format!(
                        "structured output is not a JSON object (got {})",
                        type_name(&value)
                    ));
                }
            }
            Self::JsonSchema { schema, .. } => validate_instance(schema, &value, "$")?,
            Self::Text => unreachable!("handled above"),
        }
        Ok(value)
    }
}

/// Parse `/v1/chat/completions`'s `response_format` (the nested `{"type":"json_schema",
/// "json_schema":{...}}` shape).
pub fn parse_response_format(
    value: Option<&Value>,
    param: &'static str,
) -> Result<ResponseFormat, ParamError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(ResponseFormat::Text);
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid(param, "response_format must be an object"))?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(param, "response_format requires a string \"type\""))?;
    match kind {
        "text" => {
            reject_extra_keys(object, &["type"], param, "response_format")?;
            Ok(ResponseFormat::Text)
        }
        "json_object" => {
            reject_extra_keys(object, &["type"], param, "response_format")?;
            Ok(ResponseFormat::JsonObject)
        }
        "json_schema" => {
            reject_extra_keys(object, &["type", "json_schema"], param, "response_format")?;
            let spec = object
                .get("json_schema")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    invalid(
                        param,
                        "response_format.json_schema must be an object with name/schema",
                    )
                })?;
            parse_json_schema_spec(spec, param, "response_format.json_schema")
        }
        other => Err(invalid(
            param,
            format!("response_format.type {other:?} is not supported; use text, json_object or json_schema"),
        )),
    }
}

/// Parse `/v1/responses`'s `text.format` (the FLAT shape: `{"type":"json_schema","name":...,
/// "schema":{...},"strict":true}` — OpenAI puts the fields at the top level there, unlike chat).
pub fn parse_text_format(
    value: Option<&Value>,
    param: &'static str,
) -> Result<ResponseFormat, ParamError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(ResponseFormat::Text);
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid(param, "text.format must be an object"))?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(param, "text.format requires a string \"type\""))?;
    match kind {
        "text" => {
            reject_extra_keys(object, &["type"], param, "text.format")?;
            Ok(ResponseFormat::Text)
        }
        "json_object" => {
            reject_extra_keys(object, &["type"], param, "text.format")?;
            Ok(ResponseFormat::JsonObject)
        }
        "json_schema" => parse_json_schema_spec(object, param, "text.format"),
        other => Err(invalid(
            param,
            format!(
                "text.format.type {other:?} is not supported; use text, json_object or json_schema"
            ),
        )),
    }
}

fn parse_json_schema_spec(
    spec: &Map<String, Value>,
    param: &'static str,
    prefix: &str,
) -> Result<ResponseFormat, ParamError> {
    reject_extra_keys(
        spec,
        &["type", "name", "schema", "strict", "description"],
        param,
        prefix,
    )?;
    let name = match spec.get("name").filter(|value| !value.is_null()) {
        Some(Value::String(name)) if !name.is_empty() && name.len() <= 64 => Some(name.clone()),
        Some(Value::String(_)) => {
            return Err(invalid(
                param,
                format!("{prefix}.name must be 1-64 characters"),
            ))
        }
        Some(_) => return Err(invalid(param, format!("{prefix}.name must be a string"))),
        // OpenAI requires a name; keeping it optional locally would hide a client bug only on
        // OpenAI, so mirror the requirement instead.
        None => {
            return Err(invalid(
                param,
                format!("{prefix}.name is required for json_schema"),
            ))
        }
    };
    let strict = match spec.get("strict").filter(|value| !value.is_null()) {
        None => true,
        Some(Value::Bool(strict)) => *strict,
        Some(_) => return Err(invalid(param, format!("{prefix}.strict must be a boolean"))),
    };
    if let Some(Value::String(description)) = spec.get("description") {
        if description.len() > 1024 {
            return Err(invalid(
                param,
                format!("{prefix}.description must be at most 1024 bytes"),
            ));
        }
    } else if spec
        .get("description")
        .is_some_and(|value| !value.is_null())
    {
        return Err(invalid(
            param,
            format!("{prefix}.description must be a string"),
        ));
    }
    let schema = spec
        .get("schema")
        .ok_or_else(|| invalid(param, format!("{prefix}.schema is required")))?;
    if !schema.is_object() {
        return Err(invalid(
            param,
            format!("{prefix}.schema must be a JSON Schema object"),
        ));
    }
    if serde_json::to_vec(schema).is_ok_and(|bytes| bytes.len() > MAX_SCHEMA_BYTES) {
        return Err(invalid(
            param,
            format!("{prefix}.schema exceeds {MAX_SCHEMA_BYTES} bytes"),
        ));
    }
    let mut nodes = 0usize;
    validate_schema(schema, param, prefix, 0, &mut nodes)?;
    Ok(ResponseFormat::JsonSchema {
        name,
        schema: schema.clone(),
        strict,
    })
}

fn invalid(param: &'static str, message: impl Into<String>) -> ParamError {
    ParamError {
        param,
        message: message.into(),
    }
}

fn reject_extra_keys(
    object: &Map<String, Value>,
    allowed: &[&str],
    param: &'static str,
    prefix: &str,
) -> Result<(), ParamError> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(
            param,
            format!(
                "{prefix}.{key} is not supported (allowed: {})",
                allowed.join(", ")
            ),
        ));
    }
    Ok(())
}

/// Explicit reason a keyword outside the supported subset is refused. `None` = not a keyword we
/// know at all, which gets its own "unsupported keyword" message.
fn rejected_keyword(keyword: &str) -> Option<&'static str> {
    Some(match keyword {
        "$ref" | "$defs" | "definitions" | "$id" | "$anchor" | "$dynamicRef" | "$dynamicAnchor" => {
            "references are not supported; inline the referenced schema"
        }
        "oneOf" => "oneOf is not supported (a grammar cannot enforce \"exactly one\"); use anyOf",
        "not" => "not is not supported",
        "if" | "then" | "else" => "conditional schemas (if/then/else) are not supported",
        "pattern" | "patternProperties" | "propertyNames" => "regex constraints are not supported",
        "format" => "format is not enforced by this server; remove it or validate client-side",
        "contains" | "minContains" | "maxContains" => {
            "array contains constraints are not supported"
        }
        "uniqueItems" => "uniqueItems is not supported",
        "multipleOf" => "multipleOf is not supported",
        "dependentRequired" | "dependentSchemas" => "dependent constraints are not supported",
        "unevaluatedProperties" | "unevaluatedItems" => "unevaluated constraints are not supported",
        "minProperties" | "maxProperties" => "object size constraints are not supported",
        _ => return None,
    })
}

const ANNOTATIONS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "$comment",
    "$schema",
    "deprecated",
    "readOnly",
    "writeOnly",
];

const STRUCTURAL: &[&str] = &[
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "anyOf",
    "allOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
];

fn validate_schema(
    schema: &Value,
    param: &'static str,
    prefix: &str,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), ParamError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(invalid(
            param,
            format!("{prefix}.schema nests deeper than {MAX_SCHEMA_DEPTH} levels"),
        ));
    }
    *nodes += 1;
    if *nodes > MAX_SCHEMA_NODES {
        return Err(invalid(
            param,
            format!("{prefix}.schema has more than {MAX_SCHEMA_NODES} nodes"),
        ));
    }
    let object = schema
        .as_object()
        .ok_or_else(|| invalid(param, format!("{prefix}.schema must be an object")))?;
    for keyword in object.keys() {
        if STRUCTURAL.contains(&keyword.as_str()) || ANNOTATIONS.contains(&keyword.as_str()) {
            continue;
        }
        return Err(match rejected_keyword(keyword) {
            Some(reason) => invalid(param, format!("{prefix}.schema.{keyword}: {reason}")),
            None => invalid(
                param,
                format!("{prefix}.schema.{keyword} is not a supported JSON Schema keyword"),
            ),
        });
    }
    if let Some(value) = object.get("type") {
        validate_type_keyword(value, param, prefix)?;
    }
    if let Some(value) = object.get("enum") {
        match value.as_array() {
            Some(values) if !values.is_empty() => {}
            _ => {
                return Err(invalid(
                    param,
                    format!("{prefix}.schema.enum must be a non-empty array"),
                ))
            }
        }
    }
    if let Some(properties) = object.get("properties") {
        let properties = properties.as_object().ok_or_else(|| {
            invalid(
                param,
                format!("{prefix}.schema.properties must be an object"),
            )
        })?;
        for (name, child) in properties {
            validate_schema(
                child,
                param,
                &format!("{prefix}.schema.properties.{name}"),
                depth + 1,
                nodes,
            )?;
        }
    }
    if let Some(required) = object.get("required") {
        let required = required
            .as_array()
            .ok_or_else(|| invalid(param, format!("{prefix}.schema.required must be an array")))?;
        let properties = object.get("properties").and_then(Value::as_object);
        for name in required {
            let Some(name) = name.as_str().filter(|name| !name.is_empty()) else {
                return Err(invalid(
                    param,
                    format!("{prefix}.schema.required must contain non-empty strings"),
                ));
            };
            if properties.is_none_or(|properties| !properties.contains_key(name)) {
                return Err(invalid(
                    param,
                    format!(
                        "{prefix}.schema.required names {name:?}, which is not in properties; \
                         a required property needs a schema to constrain it"
                    ),
                ));
            }
        }
    }
    if let Some(additional) = object.get("additionalProperties") {
        match additional {
            Value::Bool(_) => {}
            Value::Object(_) => validate_schema(
                additional,
                param,
                &format!("{prefix}.schema.additionalProperties"),
                depth + 1,
                nodes,
            )?,
            _ => {
                return Err(invalid(
                    param,
                    format!(
                        "{prefix}.schema.additionalProperties must be a boolean or a schema object"
                    ),
                ))
            }
        }
    }
    if let Some(items) = object.get("items") {
        if !items.is_object() {
            return Err(invalid(
                param,
                format!(
                    "{prefix}.schema.items must be one schema object (tuple arrays are not \
                     supported)"
                ),
            ));
        }
        validate_schema(
            items,
            param,
            &format!("{prefix}.schema.items"),
            depth + 1,
            nodes,
        )?;
    }
    for keyword in ["anyOf", "allOf"] {
        if let Some(value) = object.get(keyword) {
            let alternatives = value.as_array().ok_or_else(|| {
                invalid(param, format!("{prefix}.schema.{keyword} must be an array"))
            })?;
            if alternatives.is_empty() {
                return Err(invalid(
                    param,
                    format!("{prefix}.schema.{keyword} must not be empty"),
                ));
            }
            for (index, alternative) in alternatives.iter().enumerate() {
                validate_schema(
                    alternative,
                    param,
                    &format!("{prefix}.schema.{keyword}[{index}]"),
                    depth + 1,
                    nodes,
                )?;
            }
        }
    }
    for keyword in ["minLength", "maxLength", "minItems", "maxItems"] {
        if let Some(value) = object.get(keyword) {
            if value.as_u64().is_none() {
                return Err(invalid(
                    param,
                    format!("{prefix}.schema.{keyword} must be a non-negative integer"),
                ));
            }
        }
    }
    for keyword in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
        if let Some(value) = object.get(keyword) {
            if !value.is_number() {
                return Err(invalid(
                    param,
                    format!("{prefix}.schema.{keyword} must be a number"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_type_keyword(
    value: &Value,
    param: &'static str,
    prefix: &str,
) -> Result<(), ParamError> {
    let names = match value {
        Value::String(name) => std::slice::from_ref(name),
        Value::Array(names) => {
            let mut out = Vec::with_capacity(names.len());
            for name in names {
                let Some(name) = name.as_str() else {
                    return Err(invalid(
                        param,
                        format!("{prefix}.schema.type entries must be strings"),
                    ));
                };
                out.push(name.to_owned());
            }
            if out.is_empty() {
                return Err(invalid(
                    param,
                    format!("{prefix}.schema.type must not be empty"),
                ));
            }
            // The slice borrows `out`; validate below through the owned copy instead.
            for name in &out {
                if !is_type_name(name) {
                    return Err(invalid(
                        param,
                        format!(
                            "{prefix}.schema.type {name:?} is not one of object, array, string, \
                             number, integer, boolean, null"
                        ),
                    ));
                }
            }
            return Ok(());
        }
        _ => {
            return Err(invalid(
                param,
                format!("{prefix}.schema.type must be a string or an array of strings"),
            ))
        }
    };
    for name in names {
        if !is_type_name(name) {
            return Err(invalid(
                param,
                format!(
                    "{prefix}.schema.type {name:?} is not one of object, array, string, number, \
                     integer, boolean, null"
                ),
            ));
        }
    }
    Ok(())
}

fn is_type_name(name: &str) -> bool {
    matches!(
        name,
        "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
    )
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.as_i64().is_some() || number.as_u64().is_some() => {
            "integer"
        }
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn type_matches(value: &Value, expected: &str) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => {
            value.as_i64().is_some()
                || value.as_u64().is_some()
                || value.as_f64().is_some_and(|number| number.fract() == 0.0)
        }
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}

/// Validate one generated value against the same subset [`validate_schema`] accepts. Paths are
/// JSONPath-ish (`$`, `$.answer`, `$[0]`) so a client can point at the offending node.
fn validate_instance(schema: &Value, value: &Value, path: &str) -> Result<(), String> {
    let Some(object) = schema.as_object() else {
        return Ok(());
    };
    if let Some(expected) = object.get("type") {
        let matches = match expected {
            Value::String(name) => type_matches(value, name),
            Value::Array(names) => names
                .iter()
                .filter_map(Value::as_str)
                .any(|name| type_matches(value, name)),
            _ => true,
        };
        if !matches {
            return Err(format!(
                "{path}: expected {}, got {}",
                expected
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| expected.to_string()),
                type_name(value)
            ));
        }
    }
    if let Some(Value::Array(allowed)) = object.get("enum") {
        if !allowed.iter().any(|candidate| candidate == value) {
            return Err(format!(
                "{path}: value is not one of the schema's enum values"
            ));
        }
    }
    if let Some(constant) = object.get("const") {
        if constant != value {
            return Err(format!("{path}: value does not equal the schema's const"));
        }
    }
    if let Some(Value::Array(alternatives)) = object.get("anyOf") {
        if !alternatives
            .iter()
            .any(|alternative| validate_instance(alternative, value, path).is_ok())
        {
            return Err(format!("{path}: value matches no anyOf alternative"));
        }
    }
    if let Some(Value::Array(conjuncts)) = object.get("allOf") {
        for conjunct in conjuncts {
            validate_instance(conjunct, value, path)?;
        }
    }
    if let Some(map) = value.as_object() {
        let properties = object.get("properties").and_then(Value::as_object);
        if let Some(required) = object.get("required").and_then(Value::as_array) {
            for name in required.iter().filter_map(Value::as_str) {
                if !map.contains_key(name) {
                    return Err(format!("{path}: missing required property {name:?}"));
                }
            }
        }
        for (name, child) in map {
            let child_path = format!("{path}.{name}");
            match properties.and_then(|properties| properties.get(name)) {
                Some(schema) => validate_instance(schema, child, &child_path)?,
                None => match object.get("additionalProperties") {
                    Some(Value::Bool(false)) => {
                        return Err(format!(
                            "{child_path}: property is not allowed (additionalProperties=false)"
                        ))
                    }
                    Some(additional @ Value::Object(_)) => {
                        validate_instance(additional, child, &child_path)?
                    }
                    _ => {}
                },
            }
        }
    }
    if let Some(items) = value.as_array() {
        if let Some(min) = object.get("minItems").and_then(Value::as_u64) {
            if (items.len() as u64) < min {
                return Err(format!("{path}: expected at least {min} items"));
            }
        }
        if let Some(max) = object.get("maxItems").and_then(Value::as_u64) {
            if (items.len() as u64) > max {
                return Err(format!("{path}: expected at most {max} items"));
            }
        }
        if let Some(schema) = object.get("items").filter(|schema| schema.is_object()) {
            for (index, item) in items.iter().enumerate() {
                validate_instance(schema, item, &format!("{path}[{index}]"))?;
            }
        }
    }
    if let Some(text) = value.as_str() {
        let length = text.chars().count() as u64;
        if let Some(min) = object.get("minLength").and_then(Value::as_u64) {
            if length < min {
                return Err(format!("{path}: string is shorter than minLength {min}"));
            }
        }
        if let Some(max) = object.get("maxLength").and_then(Value::as_u64) {
            if length > max {
                return Err(format!("{path}: string is longer than maxLength {max}"));
            }
        }
    }
    if let Some(number) = value.as_f64() {
        if let Some(min) = object.get("minimum").and_then(Value::as_f64) {
            if number < min {
                return Err(format!("{path}: number is below minimum {min}"));
            }
        }
        if let Some(max) = object.get("maximum").and_then(Value::as_f64) {
            if number > max {
                return Err(format!("{path}: number is above maximum {max}"));
            }
        }
        if let Some(min) = object.get("exclusiveMinimum").and_then(Value::as_f64) {
            if number <= min {
                return Err(format!(
                    "{path}: number must be greater than exclusiveMinimum {min}"
                ));
            }
        }
        if let Some(max) = object.get("exclusiveMaximum").and_then(Value::as_f64) {
            if number >= max {
                return Err(format!(
                    "{path}: number must be less than exclusiveMaximum {max}"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(value: Value) -> Result<ResponseFormat, ParamError> {
        parse_response_format(Some(&value), "response_format")
    }

    #[test]
    fn absent_and_text_are_plain_text() {
        assert_eq!(
            parse_response_format(None, "response_format").unwrap(),
            ResponseFormat::Text
        );
        assert_eq!(chat(json!({"type":"text"})).unwrap(), ResponseFormat::Text);
    }

    #[test]
    fn json_object_and_json_schema_normalise() {
        assert_eq!(
            chat(json!({"type":"json_object"})).unwrap(),
            ResponseFormat::JsonObject
        );
        let format = chat(json!({
            "type":"json_schema",
            "json_schema":{
                "name":"answer",
                "strict":true,
                "schema":{
                    "type":"object",
                    "properties":{"answer":{"type":"string","maxLength":8}},
                    "required":["answer"],
                    "additionalProperties":false
                }
            }
        }))
        .unwrap();
        match &format {
            ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => {
                assert_eq!(name.as_deref(), Some("answer"));
                assert!(*strict);
                assert_eq!(schema["properties"]["answer"]["type"], "string");
            }
            other => panic!("expected json_schema, got {other:?}"),
        }
        assert_eq!(format.schema().unwrap()["type"], "object");
    }

    #[test]
    fn responses_flat_shape_parses_and_nests_correctly() {
        let format = parse_text_format(
            Some(&json!({
                "type":"json_schema",
                "name":"answer",
                "schema":{"type":"object","properties":{"answer":{"type":"string"}},
                          "required":["answer"],"additionalProperties":false},
                "strict":true
            })),
            "text.format",
        )
        .unwrap();
        assert!(format.is_structured());
    }

    #[test]
    fn unsupported_constructs_are_named_in_the_error() {
        for (schema, needle) in [
            (
                json!({"type":"object","properties":{"a":{"$ref":"#/x"}}}),
                "$ref",
            ),
            (
                json!({"type":"object","oneOf":[{"type":"object"}]}),
                "oneOf",
            ),
            (json!({"type":"string","format":"date-time"}), "format"),
            (
                json!({"type":"array","items":{"type":"string"},"uniqueItems":true}),
                "uniqueItems",
            ),
            (
                json!({"type":"object","properties":{"a":{"type":"string"}},"patternProperties":{}}),
                "patternProperties",
            ),
        ] {
            let error =
                chat(json!({"type":"json_schema","json_schema":{"name":"x","schema":schema}}))
                    .unwrap_err();
            assert_eq!(error.param, "response_format");
            assert!(error.message.contains(needle), "{}", error.message);
        }
    }

    #[test]
    fn malformed_envelopes_are_rejected() {
        for value in [
            json!("json"),
            json!({"type":"yaml"}),
            json!({"type":"json_schema"}),
            json!({"type":"json_schema","json_schema":{"schema":{"type":"object"}}}),
            json!({"type":"json_schema","json_schema":{"name":"x"}}),
            json!({"type":"json_object","extra":1}),
        ] {
            assert!(chat(value).is_err());
        }
    }

    #[test]
    fn required_must_name_a_property() {
        let error = chat(json!({"type":"json_schema","json_schema":{
            "name":"x",
            "schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["b"]}
        }}))
        .unwrap_err();
        assert!(error.message.contains("required"), "{}", error.message);
    }

    #[test]
    fn verify_accepts_matching_values_and_rejects_mismatches() {
        let format = chat(json!({"type":"json_schema","json_schema":{
            "name":"x",
            "schema":{
                "type":"object",
                "properties":{
                    "answer":{"type":"string"},
                    "score":{"type":"number","minimum":0,"maximum":1}
                },
                "required":["answer"],
                "additionalProperties":false
            }
        }}))
        .unwrap();
        assert!(format.verify(r#"{"answer":"ok","score":0.5}"#).is_ok());
        assert!(format
            .verify(r#"{"answer":"ok","score":2}"#)
            .unwrap_err()
            .contains("maximum"));
        assert!(format
            .verify(r#"{"answer":1}"#)
            .unwrap_err()
            .contains("$.answer"));
        assert!(format
            .verify(r#"{"answer":"ok","extra":1}"#)
            .unwrap_err()
            .contains("additionalProperties"));
        assert!(format
            .verify("not json")
            .unwrap_err()
            .contains("not valid JSON"));
        assert!(format.verify("").unwrap_err().contains("no content"));
    }

    #[test]
    fn json_object_verifies_only_objects() {
        let format = ResponseFormat::JsonObject;
        assert!(format.verify("{\"a\":1}").is_ok());
        assert!(format.verify("[1]").unwrap_err().contains("JSON object"));
        assert!(format
            .verify("text")
            .unwrap_err()
            .contains("not valid JSON"));
    }
}
