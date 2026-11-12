//! Console behaviour that must hold without a GPU, a model or a running engine.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use crate::launch::{self, LaunchRequest};
use crate::state::{now_ms, Console, DeltaKind, Phase, Round, Status};
use crate::ui;

fn console() -> Arc<Console> {
    let (events, _) = tokio::sync::broadcast::channel(64);
    Arc::new(Console {
        upstream: Mutex::new("http://127.0.0.1:1".into()),
        upstream_key: Mutex::new(None),
        binary: std::path::PathBuf::from("infr-test-does-not-exist"),
        state_path: std::env::temp_dir().join(format!("infr-console-test-{}.json", now_ms())),
        http: reqwest::Client::new(),
        keys: Mutex::new(Vec::new()),
        rounds: Mutex::new(Default::default()),
        series: Mutex::new(Default::default()),
        uncensor: Mutex::new(Default::default()),
        engine: Mutex::new(Default::default()),
        engine_log: Mutex::new(Default::default()),
        child: Mutex::new(None),
        shutdown_file: Mutex::new(None),
        schema: Mutex::new(None),
        conversations: Mutex::new(HashMap::new()),
        events,
        subscribers: AtomicUsize::new(0),
        next_round: AtomicU64::new(1),
        console_log: Mutex::new(Default::default()),
    })
}

fn known(paths: &[&str]) -> std::collections::HashSet<String> {
    paths.iter().map(|p| p.to_string()).collect()
}

fn request(preset: &str, set: &[(&str, &str)]) -> LaunchRequest {
    LaunchRequest {
        preset: preset.into(),
        set: set.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect::<BTreeMap<_, _>>(),
        consent: false,
    }
}

#[test]
fn presets_map_to_the_wizards_auto_profile() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    let conservative = launch::build_args(&request("conservative", &[]), shutdown, &known(&[])).unwrap();
    assert!(conservative.windows(2).any(|w| w == ["--set", "device.auto_profile=conservative"]));
    let aggressive = launch::build_args(&request("aggressive", &[]), shutdown, &known(&[])).unwrap();
    assert!(aggressive.windows(2).any(|w| w == ["--set", "device.auto_profile=aggressive"]));
    // 全手动 means the operator owns every value; an auto profile would silently override them.
    let manual = launch::build_args(&request("manual", &[]), shutdown, &known(&[])).unwrap();
    assert!(!manual.iter().any(|a| a.starts_with("device.auto_profile=")));
    assert!(launch::build_args(&request("turbo", &[]), shutdown, &known(&[])).is_err());
}

#[test]
fn every_launch_carries_the_graceful_shutdown_file() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    let args = launch::build_args(&request("manual", &[]), shutdown, &known(&[])).unwrap();
    assert!(args.ends_with(&["--set".to_string(), format!("serve.shutdown_file={}", shutdown.display())]));
    assert!(args.iter().any(|a| a == "serve.shutdown_file=/tmp/engine.shutdown"));
    // The console owns this knob; a form value must not replace the path it knows how to write.
    let overridden = launch::build_args(
        &request("manual", &[("serve.shutdown_file", "/somewhere/else")]),
        shutdown,
        &known(&["serve.shutdown_file"]),
    )
    .unwrap();
    assert!(!overridden.iter().any(|a| a.contains("/somewhere/else")));
}

#[test]
fn cli_only_options_become_flags_not_set_paths() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    let args = launch::build_args(
        &request(
            "conservative",
            &[
                ("cli:model", "C:/models/ad.gguf"),
                ("cli:addr", "127.0.0.1:9000"),
                ("cli:parallel", "2"),
                ("cli:mmproj", "C:/models/mmproj.gguf"),
                ("device.ctx", "262144"),
            ],
        ),
        shutdown,
        &known(&["device.ctx"]),
    )
    .unwrap();
    assert!(args.windows(2).any(|w| w == ["--addr", "127.0.0.1:9000"]));
    assert!(args.windows(2).any(|w| w == ["--parallel", "2"]));
    assert!(args.windows(2).any(|w| w == ["--mmproj", "C:/models/mmproj.gguf"]));
    assert!(args.iter().any(|a| a == "C:/models/ad.gguf"));
    assert!(args.windows(2).any(|w| w == ["--set", "device.ctx=262144"]));
    assert!(!args.iter().any(|a| a.starts_with("cli:")));
    // A flag left at its default is not emitted, so the engine's own default stays in force.
    let defaults = launch::build_args(
        &request("conservative", &[("cli:addr", "127.0.0.1:8080"), ("cli:parallel", "")]),
        shutdown,
        &known(&[]),
    )
    .unwrap();
    assert!(!defaults.iter().any(|a| a == "--addr" || a == "--parallel"));
}

#[test]
fn unknown_paths_and_control_characters_are_refused_before_spawn() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    assert!(launch::build_args(&request("manual", &[("kv.not_a_knob", "1")]), shutdown, &known(&["kv.paged"])).is_err());
    assert!(launch::build_args(&request("manual", &[("device.ctx", "8192\n--dev")]), shutdown, &known(&["device.ctx"])).is_err());
}

#[tokio::test]
async fn uncensor_toggle_injects_both_directions() {
    let console = console();
    let mut body = serde_json::json!({ "model": "m", "messages": [] });
    // No vector loaded: nothing is injected, and the round is recorded as off.
    assert!(!crate::proxy::apply_uncensor(&console, &mut body));
    assert!(body.get("uncensor").is_none());

    console.uncensor.lock().unwrap().vector_loaded = true;
    // The engine defaults to ON once a vector exists, so "off" has to be stated explicitly.
    assert!(!crate::proxy::apply_uncensor(&console, &mut body));
    assert_eq!(body["uncensor"], serde_json::Value::Bool(false));

    console.uncensor.lock().unwrap().consented = true;
    console.uncensor.lock().unwrap().default_on = true;
    let mut body = serde_json::json!({ "model": "m" });
    assert!(crate::proxy::apply_uncensor(&console, &mut body));
    assert_eq!(body["uncensor"], serde_json::Value::Bool(true));

    // An explicit client answer wins over the console default, either way.
    let mut body = serde_json::json!({ "uncensor": false });
    assert!(!crate::proxy::apply_uncensor(&console, &mut body));
    assert_eq!(body["uncensor"], serde_json::Value::Bool(false));
}

#[test]
fn prefix_reuse_counts_the_shared_opening_messages() {
    let console = console();
    let first = vec!["system\u{1}你是助手".to_string(), "user\u{1}你好".to_string()];
    assert!(console.estimate_reuse("conv", &first).is_none(), "the first turn has nothing to reuse");
    let second = vec![
        "system\u{1}你是助手".to_string(),
        "user\u{1}你好".to_string(),
        "assistant\u{1}你好！".to_string(),
        "user\u{1}再见".to_string(),
    ];
    let est = console.estimate_reuse("conv", &second).unwrap();
    assert_eq!(est.messages, 2);
    assert!(est.ratio > 0.0 && est.ratio < 1.0);
    // A different conversation shares nothing, even with the same key.
    let other = console.estimate_reuse("other", &second);
    assert!(other.is_none());
}

#[test]
fn decode_tri_state_and_ttft_come_from_the_deltas() {
    let mut round = Round {
        id: 1,
        key_id: "k".into(),
        key_label: "k".into(),
        model: "m".into(),
        api: "chat",
        started_at_ms: now_ms() - 500,
        ended_at_ms: None,
        status: Status::Streaming,
        phase: Phase::Prefill,
        tools: Vec::new(),
        uncensor: false,
        prompt_tokens: 100,
        cached_prompt_tokens: 40,
        completion_tokens: 0,
        reuse_est: None,
        ttft_ms: None,
        live_tps: None,
        prefill_tps: None,
        decode_tps: None,
        duration_ms: None,
        frames_truncated: false,
        request: serde_json::Value::Null,
        frames: Vec::new(),
        first_delta_ms: None,
        prefill_ms: Some(200),
    };
    round.observe(DeltaKind::Reasoning, None, 1);
    assert_eq!(round.phase, Phase::Thinking);
    assert!(round.ttft_ms.unwrap() >= 500);
    round.observe(DeltaKind::Tool, Some("tap"), 1);
    assert_eq!(round.phase, Phase::ToolCall);
    round.observe(DeltaKind::Tool, Some("tap"), 1);
    assert_eq!(round.tools, vec!["tap".to_string()], "a tool is named once, not once per fragment");
    round.observe(DeltaKind::Content, None, 1);
    assert_eq!(round.phase, Phase::Output);
    round.finish(Status::Done);
    assert_eq!(round.status, Status::Done);
    // 60 of 100 prompt tokens were evaluated in 200 ms.
    assert_eq!(round.prefill_tps, Some(300.0));
    assert!(round.duration_ms.unwrap() >= 500);
}

#[tokio::test]
async fn key_lifecycle_and_the_consent_gate() {
    let console = console();
    let app = ui::router(Arc::clone(&console));

    let added = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/keys")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"label":"android","token":"sk-console-fixed"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(added.status(), StatusCode::OK);
    assert_eq!(console.keys.lock().unwrap().len(), 1);

    // Turning the projection on without a vector, or without consent, must not succeed.
    for body in [r#"{"on":true,"consent":true}"#, r#"{"on":true}"#] {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/ui/uncensor")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{body}");
    }
    console.uncensor.lock().unwrap().vector_loaded = true;
    let refused = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/uncensor")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"on":true,"consent":false}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    let accepted = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/uncensor")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"on":true,"consent":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(accepted.status(), StatusCode::OK);
    assert!(console.uncensor.lock().unwrap().default_on);

    let id = console.keys.lock().unwrap()[0].id.clone();
    let deleted = app
        .oneshot(Request::builder().method("DELETE").uri(format!("/ui/keys/{id}")).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::OK);
    assert!(console.keys.lock().unwrap().is_empty());
    let _ = std::fs::remove_file(&console.state_path);
}

#[tokio::test]
async fn per_key_counters_fold_finished_rounds_only() {
    let console = console();
    console.keys.lock().unwrap().push(crate::state::ApiKey {
        id: "k1".into(),
        label: "one".into(),
        token: "t".into(),
        requests: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        cached_tokens: 0,
        ttft_sum_ms: 0,
        ttft_count: 0,
        prefill_sum: 0.0,
        prefill_count: 0,
        decode_sum: 0.0,
        decode_count: 0,
    });
    let mut round = Round {
        id: 7,
        key_id: "k1".into(),
        key_label: "one".into(),
        model: "m".into(),
        api: "chat",
        started_at_ms: now_ms(),
        ended_at_ms: None,
        status: Status::Streaming,
        phase: Phase::Output,
        tools: Vec::new(),
        uncensor: true,
        prompt_tokens: 1000,
        cached_prompt_tokens: 900,
        completion_tokens: 50,
        reuse_est: None,
        ttft_ms: Some(2400),
        live_tps: Some(40.0),
        prefill_tps: Some(2000.0),
        decode_tps: None,
        duration_ms: None,
        frames_truncated: false,
        request: serde_json::Value::Null,
        frames: Vec::new(),
        first_delta_ms: None,
        prefill_ms: None,
    };
    round.finish(Status::Done);
    console.push_round(round);
    console.fold_key(7);
    let keys = console.keys.lock().unwrap();
    let view = keys[0].view();
    assert_eq!(view.requests, 1);
    assert_eq!(view.prompt_tokens, 1000);
    assert_eq!(view.cached_tokens, 900);
    assert_eq!(view.completion_tokens, 50);
    assert_eq!(view.ttft_avg_ms, Some(2400));
    assert_eq!(view.decode_avg_tps, Some(40.0));
    // A key with no samples renders as absent, not as a plausible zero.
    assert_eq!(view.prefill_avg_tps, Some(2000.0));
}

#[test]
fn base64_payloads_are_elided_from_the_retained_request() {
    let mut body = serde_json::json!({
        "messages": [{ "role": "user", "content": [
            { "type": "text", "text": "点哪里" },
            { "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{}", "A".repeat(4096)) } },
        ]}],
    });
    crate::state::redact_request(&mut body);
    let text = body.to_string();
    assert!(!text.contains(&"A".repeat(256)));
    assert!(text.contains("点哪里"));
    assert!(text.contains("elided"));
}