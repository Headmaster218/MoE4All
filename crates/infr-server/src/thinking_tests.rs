//! Request DTO -> validated controls -> model template -> JSON/SSE. No model weights required.
use super::*;
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;

const QWEN38: &str = include_str!("../../infr-chat/tests/fixtures/qwen38_chat_template.jinja");

struct TemplateGen;
impl ChatGenerator for TemplateGen {
    fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
        _tool_choice: Option<&str>,
        params: &GenParams,
        _cancel: &Arc<AtomicBool>,
        _progress: Option<infr_core::GenerationProgressCallback>,
        on_delta: &mut dyn FnMut(Delta),
    ) -> anyhow::Result<ChatOutcome> {
        // The production generator uses OaiRenderer over a GGUF. This adapter feeds the same raw
        // template seam so the HTTP tests can run without downloading a 100+ GB model.
        let messages = messages
            .iter()
            .map(|m| {
                json!({
                    "role": m.role, "content": m.content, "reasoning_content": m.reasoning_content,
                })
            })
            .collect();
        let prompt = infr_chat::render_template_with_options(
            QWEN38,
            messages,
            tools.cloned().unwrap_or(serde_json::Value::Null),
            "",
            "",
            true,
            &params.chat_template_options.resolve(&Config::default()),
        )
        .map_err(infr_engine::TemplateError::Render)?;
        on_delta(Delta::Content(prompt));
        Ok(ChatOutcome {
            finish: Finish::Stop,
            prompt_tokens: 1,
            cached_prompt_tokens: 0,
            completion_tokens: 1,
        })
    }
}

async fn post(body: serde_json::Value) -> (StatusCode, String) {
    let router = build_router(AppState::new(
        Arc::new(TemplateGen),
        "m",
        2,
        Arc::new(Config::default()),
    ));
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn body() -> serde_json::Value {
    json!({"model":"m","messages":[{"role":"user","content":"Hello"}]})
}

#[tokio::test]
async fn effort_changes_the_prompt_through_both_http_modes() {
    for stream in [false, true] {
        for level in ["low", "medium", "xhigh"] {
            let mut input = body();
            input["stream"] = stream.into();
            input["reasoning_effort"] = level.into();
            let (status, output) = post(input).await;
            assert_eq!(status, StatusCode::OK, "{output}");
            assert!(!output.contains("invalid_request_error"), "{output}");
            assert_eq!(output.contains("set to low"), level == "low", "{output}");
            assert_eq!(
                output.contains("set to xhigh"),
                level == "xhigh",
                "{output}"
            );
            if stream {
                assert_eq!(output.matches("[DONE]").count(), 1);
            }
        }
    }
}

#[tokio::test]
async fn nested_options_and_reasoning_history_reach_the_template() {
    let mut input = body();
    input["chat_template_kwargs"] = json!({"reasoning_effort":"medium", "preserve_thinking":true});
    input["messages"] = json!([
        {"role":"user","content":"First"},
        {"role":"assistant","content":"Answer", "reasoning":"alias trace"},
        {"role":"user","content":"Next"}
    ]);
    let (status, output) = post(input.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(output.contains("alias trace"), "{output}");
    assert!(!output.contains("set to xhigh"));
    input["chat_template_kwargs"]["preserve_thinking"] = false.into();
    let (status, output) = post(input).await;
    assert_eq!(status, StatusCode::OK, "{output}");
    assert!(!output.contains("alias trace"), "{output}");
}

#[tokio::test]
async fn request_can_disable_thinking() {
    let mut input = body();
    input["chat_template_kwargs"] = json!({"enable_thinking":false});
    let (status, output) = post(input).await;
    assert_eq!(status, StatusCode::OK);
    let output: serde_json::Value = serde_json::from_str(&output).unwrap();
    let prompt = output["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(prompt.ends_with("<think>\n\n</think>\n\n"));
    assert!(!prompt.contains("Reasoning effort is set"));
}

#[tokio::test]
async fn invalid_types_typos_and_reserved_context_keys_are_400() {
    for fields in [
        json!({"reasoning_effort":"midium"}),
        json!({"reasoning_effort":"height"}),
        json!({"reasoning_effort":12}),
        json!({"chat_template_kwargs":[]}),
        json!({"chat_template_kwargs":{"enable_thinking":"false"}}),
        json!({"chat_template_kwargs":{"preserve_thinking":1}}),
        json!({"chat_template_kwargs":{"messages":[]}}),
        json!({"chat_template_kwargs":{"tools":[]}}),
        json!({"chat_template_kwargs":{"bos_token":"injected"}}),
        json!({"chat_template_kwargs":{"add_generation_prompt":false}}),
        json!({"reasoning_effort":"low","chat_template_kwargs":{"reasoning_effort":"xhigh"}}),
    ] {
        let mut input = body();
        input
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let (status, output) = post(input).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{fields}: {output}");
        assert!(output.contains("invalid_request_error"));
    }
}

#[tokio::test]
async fn model_rejected_level_is_an_input_error_and_closes_stream() {
    for stream in [false, true] {
        let mut input = body();
        input["stream"] = stream.into();
        input["reasoning_effort"] = "high".into();
        let (status, output) = post(input).await;
        assert_eq!(
            status,
            if stream {
                StatusCode::OK
            } else {
                StatusCode::BAD_REQUEST
            }
        );
        assert!(
            output.contains("Unexpected reasoning effort high"),
            "{output}"
        );
        assert!(output.contains("invalid_request_error"), "{output}");
        assert!(!output.contains("server_error"));
        if stream {
            assert_eq!(output.matches("[DONE]").count(), 1);
        }
    }
}

#[test]
fn identical_duplicate_effort_is_accepted_and_null_inherits() {
    let mut input = body();
    input["reasoning_effort"] = "medium".into();
    input["chat_template_kwargs"] = json!({"reasoning_effort":"medium"});
    let request: ChatRequest = serde_json::from_value(input).unwrap();
    assert_eq!(
        GenParams::from_request(&request)
            .unwrap()
            .chat_template_options
            .reasoning_effort,
        Some(infr_core::config::ReasoningEffort::Medium)
    );
    let mut input = body();
    input["reasoning_effort"] = serde_json::Value::Null;
    input["chat_template_kwargs"] = serde_json::Value::Null;
    let request: ChatRequest = serde_json::from_value(input).unwrap();
    assert_eq!(
        GenParams::from_request(&request).unwrap(),
        GenParams::default()
    );
}

#[test]
fn canonical_reasoning_takes_precedence_without_rejecting_both_fields() {
    let dto: ChatMessageDto = serde_json::from_value(json!({
        "role":"assistant", "content":"Answer", "reasoning_content":"canonical", "reasoning":"alias",
    }))
    .unwrap();
    assert_eq!(
        dto_to_engine(&dto).reasoning_content.as_deref(),
        Some("canonical")
    );
}
