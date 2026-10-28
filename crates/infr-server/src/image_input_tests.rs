use super::*;
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use tower::ServiceExt;

struct ImageGen {
    typed: bool,
}

impl ChatGenerator for ImageGen {
    fn chat(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&serde_json::Value>,
        _tool_choice: Option<&str>,
        _params: &GenParams,
        _cancel: &Arc<AtomicBool>,
        _progress: Option<infr_core::GenerationProgressCallback>,
        on_delta: &mut dyn FnMut(Delta),
    ) -> anyhow::Result<ChatOutcome> {
        if messages.iter().any(|m| !m.images.is_empty()) {
            let error = if self.typed {
                anyhow::Error::new(infr_core::error::InvalidInput("invalid image bytes".into()))
            } else {
                anyhow::anyhow!("invalid image: backend allocation failed")
            };
            return Err(error.context("preparing image"));
        }
        on_delta(Delta::Content("ok".into()));
        Ok(ChatOutcome {
            finish: Finish::Stop,
            prompt_tokens: 3,
            cached_prompt_tokens: 0,
            completion_tokens: 1,
        })
    }
}

fn request(responses: bool, stream: bool, image: bool) -> Request<Body> {
    let body = if responses {
        serde_json::json!({"model":"m", "stream":stream, "input":[{
            "role":"user", "content": if image {
                serde_json::json!([{"type":"input_image", "image_url":"data:image/png;base64,bm90LWFuLWltYWdl"}])
            } else {
                serde_json::json!([{"type":"input_text", "text":"hello"}])
            }
        }]})
    } else {
        serde_json::json!({"model":"m", "stream":stream, "messages":[{
            "role":"user", "content": if image {
                serde_json::json!([{"type":"image_url", "image_url":{"url":"data:image/png;base64,bm90LWFuLWltYWdl"}}])
            } else { serde_json::json!("hello") }
        }]})
    };
    Request::builder()
        .method("POST")
        .uri(if responses {
            "/v1/responses"
        } else {
            "/v1/chat/completions"
        })
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn invalid_image_classification_and_slot_recovery() {
    for responses in [false, true] {
        for stream in [false, true] {
            for typed in [false, true] {
                let state = AppState::new(
                    Arc::new(ImageGen { typed }),
                    "m",
                    1,
                    Arc::new(Config::default()),
                );
                let stats = state.stats.clone();
                let app = build_router(state);
                let response = app
                    .clone()
                    .oneshot(request(responses, stream, true))
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    if stream {
                        StatusCode::OK
                    } else if typed {
                        StatusCode::BAD_REQUEST
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                );
                let bytes = response.into_body().collect().await.unwrap().to_bytes();
                let body = String::from_utf8(bytes.to_vec()).unwrap();
                let expected = if typed {
                    "invalid_request_error"
                } else {
                    "server_error"
                };
                if stream {
                    let events: Vec<serde_json::Value> = body
                        .lines()
                        .filter_map(|s| s.strip_prefix("data: "))
                        .filter_map(|s| serde_json::from_str(s).ok())
                        .collect();
                    if responses {
                        let failed = events
                            .iter()
                            .find(|e| e["type"] == "response.failed")
                            .expect(&body);
                        assert_eq!(failed["response"]["error"]["code"], expected, "{body}");
                        assert!(!events.iter().any(|e| e["type"] == "response.completed"));
                    } else {
                        let failed = events.iter().find(|e| !e["error"].is_null()).expect(&body);
                        assert_eq!(failed["error"]["type"], expected, "{body}");
                        assert_eq!(body.matches("data: [DONE]").count(), 1);
                        assert!(!events
                            .iter()
                            .any(|e| e["choices"][0]["finish_reason"] == "stop"));
                    }
                } else {
                    let body: serde_json::Value = serde_json::from_str(&body).unwrap();
                    assert_eq!(body["error"]["type"], expected);
                }
                // A failed image must release the only slot and count as one failure.
                let response = tokio::time::timeout(
                    Duration::from_secs(2),
                    app.oneshot(request(responses, false, false)),
                )
                .await
                .expect("slot leaked")
                .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let bytes = response.into_body().collect().await.unwrap().to_bytes();
                assert!(String::from_utf8_lossy(&bytes).contains("ok"));
                let window = stats.drain(Duration::from_secs(1));
                assert_eq!(window.failed, 1);
                assert_eq!(window.completed, 2);
                assert_eq!(window.active, 0);
            }
        }
    }
}
