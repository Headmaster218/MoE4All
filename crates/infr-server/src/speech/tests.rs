use super::*;
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use std::sync::atomic::AtomicUsize;
use tower::ServiceExt;

#[derive(Default)]
struct Mock {
    calls: AtomicUsize,
    speaker: AtomicUsize,
    hold: AtomicBool,
    cancelled: AtomicBool,
    error: Option<bool>,
}
impl Synthesizer for Mock {
    fn model_id(&self) -> &str {
        "melo-test"
    }
    fn synthesize(
        &self,
        _text: &str,
        options: &SynthesisOptions,
        cancel: &AtomicBool,
    ) -> anyhow::Result<SpeechResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.speaker.store(options.speaker_id, Ordering::Relaxed);
        if let Some(invalid) = self.error {
            if invalid {
                return Err(anyhow::Error::new(infr_tts::InvalidInput::new(
                    "unsupported speaker ID",
                ))
                .context("native request validation"));
            }
            anyhow::bail!(
                "unsupported speaker ID; private model path and internal compute failure"
            );
        }
        let until = Instant::now() + Duration::from_secs(3);
        while self.hold.load(Ordering::Relaxed) && Instant::now() < until {
            if cancel.load(Ordering::Relaxed) {
                self.cancelled.store(true, Ordering::Relaxed);
                anyhow::bail!("cancelled");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        Ok(SpeechResult {
            samples: vec![0.0, 0.5, -1.0],
            sample_rate: 44100,
        })
    }
}

#[tokio::test]
async fn native_invalid_input_is_400_while_loader_and_compute_errors_remain_private_500() {
    for (invalid, expected, kind) in [
        (true, StatusCode::BAD_REQUEST, "invalid_request_error"),
        (false, StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
    ] {
        let mock = Arc::new(Mock {
            error: Some(invalid),
            ..Default::default()
        });
        let state = AppState::empty(Arc::new(Config::default())).with_speech(mock.clone());
        let slots = state.speech.as_ref().unwrap().slots.clone();
        let admission = state.speech.as_ref().unwrap().admission.clone();
        let response = build_router(state)
            .oneshot(json(
                serde_json::json!({ "input": "request text must not leak", "voice": "0" }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
        let body: serde_json::Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["error"]["type"], kind);
        if invalid {
            assert_eq!(body["error"]["message"], "unsupported speaker ID");
        } else {
            assert!(!body.to_string().contains("private model path"));
        }
        assert!(!body.to_string().contains("request text must not leak"));
        assert_eq!(mock.calls.load(Ordering::Relaxed), 1);
        assert_eq!(slots.available_permits(), 1);
        assert_eq!(admission.available_permits(), 4);
    }
}
fn json(value: serde_json::Value) -> Request<Body> {
    Request::post("/v1/audio/speech")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(value.to_string()))
        .unwrap()
}
async fn until(mut condition: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(2);
    while !condition() {
        assert!(Instant::now() < end, "condition timed out");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[test]
fn pcm16_and_wav_preserve_rate_and_finite_bounded_samples() {
    let samples = SpeechResult {
        samples: vec![0.0, 0.5, -1.0],
        sample_rate: 44100,
    };
    let pcm = encode_audio(&samples, AudioFormat::Pcm).unwrap();
    assert_eq!(pcm, [0, 0, 255, 63, 0, 128]);
    let wav = encode_audio(&samples, AudioFormat::Wav).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 44100);
    assert_eq!(&wav[44..], pcm);
    assert!(encode_audio(
        &SpeechResult {
            samples: vec![f32::NAN],
            sample_rate: 44100
        },
        AudioFormat::Wav
    )
    .is_err());
}

#[tokio::test]
async fn speech_routes_are_optional_authenticated_and_advertised() {
    let empty = build_router(AppState::empty(Arc::new(Config::default())));
    assert_eq!(
        empty
            .oneshot(json(serde_json::json!({"input":"hello"})))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    let mock = Arc::new(Mock::default());
    let mut cfg = Config::default();
    cfg.serve.api_key = Some("test-key".into());
    let app = build_router(AppState::empty(Arc::new(cfg)).with_speech(mock.clone()));
    assert_eq!(
        app.clone()
            .oneshot(json(serde_json::json!({"input":"hello"})))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(mock.calls.load(Ordering::Relaxed), 0);
    let response = app
        .oneshot(
            Request::get("/v1/models")
                .header(header::AUTHORIZATION, "Bearer test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert_eq!(body["data"][0]["id"], "melo-test");
}

#[tokio::test]
async fn speech_validates_before_model_execution_and_returns_native_pcm_or_wav() {
    let mock = Arc::new(Mock::default());
    let app = build_router(AppState::empty(Arc::new(Config::default())).with_speech(mock.clone()));
    for value in [
        serde_json::json!({"input":""}),
        serde_json::json!({"input":"hello","model":"other"}),
        serde_json::json!({"input":"hello","speed":0}),
        serde_json::json!({"input":"hello","voice":"unknown"}),
        serde_json::json!({"input":"hello","stream":true}),
    ] {
        assert_eq!(
            app.clone().oneshot(json(value)).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }
    let wrong_type = Request::post("/v1/audio/speech")
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from("hello"))
        .unwrap();
    assert_eq!(
        app.clone().oneshot(wrong_type).await.unwrap().status(),
        StatusCode::UNSUPPORTED_MEDIA_TYPE
    );
    let oversized = json(serde_json::json!({ "input": "x".repeat(BODY_LIMIT + 1) }));
    assert_eq!(
        app.clone().oneshot(oversized).await.unwrap().status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    assert_eq!(mock.calls.load(Ordering::Relaxed), 0);
    for (format, content, size) in [("wav", "audio/wav", 50), ("pcm", "audio/pcm", 6)] {
        let response = app
            .clone()
            .oneshot(json(
                serde_json::json!({"input":"hello", "response_format":format}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            mock.speaker.load(Ordering::Relaxed),
            1,
            "default voice must preserve the native package speaker"
        );
        assert_eq!(response.headers()[header::CONTENT_TYPE], content);
        assert_eq!(response.headers()["x-audio-sample-rate"], "44100");
        assert_eq!(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .len(),
            size
        );
    }
}

#[tokio::test]
async fn cancelled_handlers_keep_capacity_until_worker_returns_and_queues_are_bounded() {
    let mock = Arc::new(Mock::default());
    mock.hold.store(true, Ordering::Relaxed);
    let state = AppState::empty(Arc::new(Config::default())).with_speech(mock.clone());
    let admission = state.speech.as_ref().unwrap().admission.clone();
    let slots = state.speech.as_ref().unwrap().slots.clone();
    let app = build_router(state);
    let mut requests = Vec::new();
    for _ in 0..4 {
        let app = app.clone();
        requests.push(tokio::spawn(async move {
            app.oneshot(json(serde_json::json!({"input":"hello"})))
                .await
        }));
    }
    until(|| admission.available_permits() == 0 && mock.calls.load(Ordering::Relaxed) == 1).await;
    assert_eq!(
        app.oneshot(json(serde_json::json!({"input":"overflow"})))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    for task in requests {
        task.abort();
        let _ = task.await;
    }
    until(|| {
        mock.cancelled.load(Ordering::Relaxed)
            && admission.available_permits() == 4
            && slots.available_permits() == 1
    })
    .await;
    assert_eq!(mock.calls.load(Ordering::Relaxed), 1);
}
