//! Capability composition uses mocks only; no model, device, upload files or listener needed.

use super::*;
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use std::path::Path;
use tower::ServiceExt;

struct Chat;
impl ChatGenerator for Chat {
    fn chat(
        &self,
        _: &[ChatMessage],
        _: Option<&serde_json::Value>,
        _: Option<&str>,
        _: &GenParams,
        _: &Arc<AtomicBool>,
        _: Option<infr_core::GenerationProgressCallback>,
        emit: &mut dyn FnMut(Delta),
    ) -> anyhow::Result<ChatOutcome> {
        emit(Delta::Content("chat reply".into()));
        Ok(ChatOutcome {
            finish: Finish::Stop,
            prompt_tokens: 1,
            cached_prompt_tokens: 0,
            completion_tokens: 1,
        })
    }
}

struct Embedding;
impl EmbeddingGenerator for Embedding {
    fn embed(&self, inputs: &[String]) -> anyhow::Result<EmbeddingOutcome> {
        Ok(EmbeddingOutcome {
            embeddings: inputs.iter().map(|_| vec![1.0, 0.0]).collect(),
            prompt_tokens: inputs.len() as u32,
        })
    }
}

struct Audio;
impl infr_audio::Transcriber for Audio {
    fn model_id(&self) -> &str {
        "audio"
    }

    fn transcribe_file(
        &self,
        _: &Path,
        _: &infr_audio::TranscriptionOptions,
        _: &AtomicBool,
        _: &mut dyn FnMut(infr_audio::TranscriptionSegment) -> anyhow::Result<()>,
    ) -> anyhow::Result<infr_audio::TranscriptionResult> {
        panic!("composition tests must not reach audio inference");
    }
}

fn composed(modules: u8, cfg: Arc<Config>) -> AppState {
    let mut state = if modules & 1 != 0 {
        AppState::new(Arc::new(Chat), "chat", 2, cfg)
    } else {
        AppState::empty(cfg)
    };
    if modules & 2 != 0 {
        state = state.with_embedding(Arc::new(Embedding), "embedding", 3);
    }
    if modules & 4 != 0 {
        state = state.with_transcription(Arc::new(Audio));
    }
    state
}

fn json_request(uri: &str, body: &str) -> Request<Body> {
    Request::post(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

#[test]
fn empty_state_has_no_placeholder_models_slots_or_dashboard_rows() {
    let cfg = Arc::new(Config::default());
    let state = AppState::empty(cfg.clone());
    assert!(Arc::ptr_eq(&state.cfg, &cfg));
    assert!(state.models.is_empty());
    assert!(state.embeddings.is_empty());
    assert!(state.transcription.is_none());
    assert_eq!(state.slot_occupancy(), (0, 0));
    assert!(state.dashboard.slots.lock().unwrap().is_empty());

    let audio = AppState::transcription(Arc::new(Audio), cfg.clone());
    assert!(audio.models.is_empty());
    assert_eq!(audio.slot_occupancy(), (0, 0));
    assert!(audio.dashboard.slots.lock().unwrap().is_empty());
    let embedding = AppState::embedding(Arc::new(Embedding), "embedding", 3, cfg);
    assert!(embedding.models.is_empty());
    assert_eq!(embedding.slot_occupancy(), (0, 3));
}

#[tokio::test]
async fn serving_empty_state_fails_before_binding_or_spawning_reporter() {
    let error = serve_state(
        AppState::empty(Arc::new(Config::default())),
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("at least one hosted capability"));
}

#[tokio::test]
async fn seven_combinations_list_only_real_capabilities_and_route_independently() {
    for modules in 1..8 {
        let state = composed(modules, Arc::new(Config::default()));
        let capacity = if modules & 1 != 0 { 2 } else { 0 } + if modules & 2 != 0 { 3 } else { 0 };
        assert_eq!(state.slot_occupancy(), (0, capacity));
        assert_eq!(
            state.dashboard.slots.lock().unwrap().len(),
            capacity as usize
        );
        let router = build_router(state);
        let response = router
            .clone()
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let ids: Vec<_> = json["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|card| card["id"].as_str().unwrap())
            .collect();
        let expected: Vec<_> = [(1, "chat"), (2, "embedding"), (4, "audio")]
            .into_iter()
            .filter_map(|(bit, name)| (modules & bit != 0).then_some(name))
            .collect();
        assert_eq!(ids, expected, "combination {modules}");

        let chat = router
            .clone()
            .oneshot(json_request(
                "/v1/chat/completions",
                r#"{"model":"chat","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .await
            .unwrap();
        assert_eq!(
            chat.status(),
            if modules & 1 != 0 {
                StatusCode::OK
            } else {
                StatusCode::NOT_FOUND
            }
        );
        let responses = router
            .clone()
            .oneshot(json_request(
                "/v1/responses",
                r#"{"model":"chat","input":"hi"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(
            responses.status(),
            if modules & 1 != 0 {
                StatusCode::OK
            } else {
                StatusCode::NOT_FOUND
            }
        );
        let embedding = router
            .clone()
            .oneshot(json_request(
                "/v1/embeddings",
                r#"{"model":"embedding","input":"hi"}"#,
            ))
            .await
            .unwrap();
        assert_eq!(
            embedding.status(),
            if modules & 2 != 0 {
                StatusCode::OK
            } else {
                StatusCode::NOT_FOUND
            }
        );
        // Invalid multipart metadata stops before creating upload files or calling a backend.
        let audio = router
            .oneshot(json_request("/v1/audio/transcriptions", "{}"))
            .await
            .unwrap();
        assert_eq!(
            audio.status(),
            if modules & 4 != 0 {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            }
        );
    }
}

#[tokio::test]
async fn auxiliary_combinations_preserve_auth_without_a_chat_placeholder() {
    for modules in [2, 4, 6] {
        let mut cfg = Config::default();
        cfg.serve.api_key = Some("secret".into());
        let cfg = Arc::new(cfg);
        let state = composed(modules, cfg.clone());
        assert!(Arc::ptr_eq(&state.cfg, &cfg));
        let router = build_router(state);
        for uri in [
            "/v1/embeddings",
            "/v1/audio/transcriptions",
            "/v1/chat/completions",
            "/v1/responses",
        ] {
            let response = router
                .clone()
                .oneshot(json_request(uri, "{}"))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }
        let response = router
            .clone()
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let health = router
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
    }
}
