//! Console behaviour that must hold without a GPU, a model or a running engine.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use crate::auth::{AuthState, PasswordRecord, PersistentAuth};
use crate::crypto::SecureStore;
use crate::launch::{self, LaunchRequest};
use crate::state::{now_ms, Console, DeltaKind, Phase, Round, Status};
use crate::ui;

fn console() -> Arc<Console> {
    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
    let (events, _) = tokio::sync::broadcast::channel(64);
    let (state_dirty, _) = tokio::sync::broadcast::channel(8);
    // One directory per console: the secure store keeps fixed sibling names (`auth.json`,
    // `keys.json`, `state.key`), so sharing the temp root would let parallel tests clobber
    // each other's state.
    let dir = std::env::temp_dir().join(format!(
        "infr-console-test-{}-{}-{}",
        now_ms(),
        std::process::id(),
        NEXT_DIR.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let state_path = dir.join("state.json");
    let store = Arc::new(SecureStore::new(&state_path).unwrap());
    let auth = AuthState::from_persistent(PersistentAuth {
        admin: Some(PasswordRecord::new("test-admin")),
        users: Vec::new(),
    });
    Arc::new(Console {
        upstream: Mutex::new("http://127.0.0.1:1".into()),
        upstream_key: Mutex::new(None),
        binary: std::path::PathBuf::from("infr-test-does-not-exist"),
        state_path: state_path.clone(),
        store,
        auth: Mutex::new(auth),
        tls: false,
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
        conversations: Mutex::new(Default::default()),
        events,
        state_dirty,
        subscribers: AtomicUsize::new(0),
        next_round: AtomicU64::new(1),
        console_log: Mutex::new(Default::default()),
    })
}

/// A config field as the engine schema advertises it: `path` + shipped `default`. An unset Option
/// defaults to `none`, which `build_args` emits verbatim so the engine parses it back to `None`.
fn cfg(path: &str, default: &str) -> serde_json::Value {
    serde_json::json!({ "path": path, "default": default })
}

/// A CLI-only field: it carries a `flag` spelling (empty = the positional model) instead of a `--set`
/// path. `default` is what the engine uses when the operator leaves it blank.
fn cli(path: &str, flag: &str, default: &str) -> serde_json::Value {
    serde_json::json!({ "path": path, "flag": flag, "default": default })
}

fn fields(paths: &[&str]) -> Vec<serde_json::Value> {
    paths.iter().map(|p| cfg(p, "none")).collect()
}

fn request(preset: &str, set: &[(&str, &str)]) -> LaunchRequest {
    LaunchRequest {
        preset: preset.into(),
        set: set
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<BTreeMap<_, _>>(),
        consent: false,
    }
}

#[test]
fn presets_map_to_the_wizards_auto_profile() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    let conservative =
        launch::build_args(&request("conservative", &[]), &fields(&[]), shutdown).unwrap();
    assert!(conservative
        .windows(2)
        .any(|w| w == ["--set", "device.auto_profile=conservative"]));
    let aggressive =
        launch::build_args(&request("aggressive", &[]), &fields(&[]), shutdown).unwrap();
    assert!(aggressive
        .windows(2)
        .any(|w| w == ["--set", "device.auto_profile=aggressive"]));
    // 全手动 means the operator owns every value; an auto profile would silently override them.
    let manual = launch::build_args(&request("manual", &[]), &fields(&[]), shutdown).unwrap();
    assert!(!manual.iter().any(|a| a.starts_with("device.auto_profile=")));
    assert!(launch::build_args(&request("turbo", &[]), &fields(&[]), shutdown).is_err());
}

#[test]
fn every_launch_carries_the_graceful_shutdown_file() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    let args = launch::build_args(&request("manual", &[]), &fields(&[]), shutdown).unwrap();
    assert!(args.ends_with(&[
        "--set".to_string(),
        format!("serve.shutdown_file={}", shutdown.display())
    ]));
    assert!(args
        .iter()
        .any(|a| a == "serve.shutdown_file=/tmp/engine.shutdown"));
    // The console owns this knob; a form value must not replace the path it knows how to write.
    let overridden = launch::build_args(
        &request("manual", &[("serve.shutdown_file", "/somewhere/else")]),
        &fields(&["serve.shutdown_file"]),
        shutdown,
    )
    .unwrap();
    assert!(!overridden.iter().any(|a| a.contains("/somewhere/else")));
}

#[test]
fn cli_only_options_become_flags_not_set_paths() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    let schema = vec![
        cli("cli:model", "", ""),
        cli("cli:addr", "--addr", "127.0.0.1:8080"),
        cli("cli:parallel", "--parallel", ""),
        cli("cli:mmproj", "--mmproj", ""),
        cfg("device.ctx", "none"),
    ];
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
        &schema,
        shutdown,
    )
    .unwrap();
    assert!(args.windows(2).any(|w| w == ["--addr", "127.0.0.1:9000"]));
    assert!(args.windows(2).any(|w| w == ["--parallel", "2"]));
    assert!(args
        .windows(2)
        .any(|w| w == ["--mmproj", "C:/models/mmproj.gguf"]));
    assert!(args.iter().any(|a| a == "C:/models/ad.gguf"));
    assert!(args.windows(2).any(|w| w == ["--set", "device.ctx=262144"]));
    assert!(!args.iter().any(|a| a.starts_with("cli:")));
    // FULL DUMP: a non-empty CLI flag is explicit even when it equals the default; an empty one has
    // no flag spelling and is omitted, so the engine's own default stays in force for that knob.
    let defaults = launch::build_args(
        &request(
            "conservative",
            &[("cli:addr", "127.0.0.1:8080"), ("cli:parallel", "")],
        ),
        &schema,
        shutdown,
    )
    .unwrap();
    assert!(defaults
        .windows(2)
        .any(|w| w == ["--addr", "127.0.0.1:8080"]));
    assert!(!defaults.iter().any(|a| a == "--parallel"));
}

#[test]
fn unknown_paths_and_control_characters_are_refused_before_spawn() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    assert!(launch::build_args(
        &request("manual", &[("kv.not_a_knob", "1")]),
        &fields(&["kv.paged"]),
        shutdown
    )
    .is_err());
    assert!(launch::build_args(
        &request("manual", &[("device.ctx", "8192\n--dev")]),
        &fields(&["device.ctx"]),
        shutdown
    )
    .is_err());
}

/// The operator asked for EVERY setting written explicitly into the launch command. An untouched
/// Option must still appear as `=none` (so the engine keeps auto-filling it under the profile), the
/// engine-derived `*_specified` presence markers must NOT appear (the wizard's `--set` path never
/// sets them either), and the console-owned shutdown file is appended exactly once, last.
#[test]
fn full_dump_writes_every_field_and_skips_presence_markers() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    let schema = vec![
        cfg("device.ctx", "none"),
        cfg("device.auto_profile", "conservative"),
        cfg("device.ubatch_specified", "false"),
        cfg("kv.type_k", "none"),
        cfg("kv.type_k_specified", "false"),
        cfg("kv.paged", "false"),
        cfg("kernels.vulkan.gemv.variant", "none"),
        cfg("serve.shutdown_file", "none"),
    ];
    // Touch only ctx and the KV dtype; everything else must still be emitted at its default.
    let args = launch::build_args(
        &request(
            "conservative",
            &[("device.ctx", "262144"), ("kv.type_k", "q8_0")],
        ),
        &schema,
        shutdown,
    )
    .unwrap();
    assert!(args.windows(2).any(|w| w == ["--set", "device.ctx=262144"]));
    assert!(args.windows(2).any(|w| w == ["--set", "kv.type_k=q8_0"]));
    // Untouched fields are present at their shipped default, not silently omitted.
    assert!(args.windows(2).any(|w| w == ["--set", "kv.paged=false"]));
    assert!(args
        .windows(2)
        .any(|w| w == ["--set", "kernels.vulkan.gemv.variant=none"]));
    // auto_profile comes from the preset, exactly once.
    assert_eq!(
        args.iter()
            .filter(|a| a.starts_with("device.auto_profile="))
            .count(),
        1
    );
    assert!(args
        .windows(2)
        .any(|w| w == ["--set", "device.auto_profile=conservative"]));
    // Presence markers are engine-derived and must never be pinned by the dump.
    assert!(!args.iter().any(|a| a.contains("_specified=")));
    // The shutdown file is the console's, appended once and last.
    assert_eq!(
        args.iter()
            .filter(|a| a.starts_with("serve.shutdown_file="))
            .count(),
        1
    );
    assert!(args.ends_with(&[
        "--set".to_string(),
        format!("serve.shutdown_file={}", shutdown.display())
    ]));
}

/// A config file is the base layer: with `cli:config` set, the launch must emit only the touched
/// deltas (a full dump would shadow every file value via `--set` precedence). Without it, the full
/// explicit dump is unchanged.
#[test]
fn config_file_replaces_the_full_dump_with_touched_deltas() {
    let shutdown = std::path::Path::new("/tmp/engine.shutdown");
    let schema = vec![
        cli("cli:config", "--config", ""),
        cli("cli:model", "", ""),
        cfg("device.ctx", "none"),
        cfg("kv.paged", "false"),
        cfg("kv.ring", "true"),
    ];
    let args = launch::build_args(
        &request(
            "conservative",
            &[
                ("cli:config", "C:/infr.toml"),
                ("cli:model", "m.gguf"),
                ("device.ctx", "262144"),
            ],
        ),
        &schema,
        shutdown,
    )
    .unwrap();
    assert!(args.windows(2).any(|w| w == ["--config", "C:/infr.toml"]));
    assert!(args.windows(2).any(|w| w == ["--set", "device.ctx=262144"]));
    // Untouched knobs come from the file, not a shadowing full dump.
    assert!(!args.iter().any(|a| a == "kv.paged=false"));
    assert!(!args.iter().any(|a| a == "kv.ring=true"));
    assert!(args.ends_with(&[
        "--set".to_string(),
        format!("serve.shutdown_file={}", shutdown.display())
    ]));
    // No config file → the full dump still writes every field at its default.
    let full = launch::build_args(
        &request("conservative", &[("device.ctx", "262144")]),
        &schema,
        shutdown,
    )
    .unwrap();
    assert!(full.iter().any(|a| a == "kv.paged=false"));
    assert!(full.iter().any(|a| a == "kv.ring=true"));
}

/// The exported TOML speaks the engine's section-path grammar with correctly typed scalars, omits
/// unset Options / CLI-only flags / the console-owned shutdown file / the API-key secret, and parses
/// back losslessly — including a section that mixes scalars and sub-tables.
#[test]
fn export_toml_round_trips_through_the_toml_parser() {
    let schema = vec![
        serde_json::json!({"path":"device.ctx","ty":"Option<SizeSpec>","default":"none"}),
        serde_json::json!({"path":"device.auto_profile","ty":"AutoProfile","default":"conservative"}),
        serde_json::json!({"path":"kv.type_k","ty":"Option<DType>","default":"none"}),
        serde_json::json!({"path":"kv.ring","ty":"bool","default":"true"}),
        serde_json::json!({"path":"sampling.temp","ty":"f32","default":"0"}),
        serde_json::json!({"path":"kernels.qkv_fuse","ty":"bool","default":"true"}),
        serde_json::json!({"path":"kernels.vulkan.gemv.rm","ty":"u32","default":"2"}),
        serde_json::json!({"path":"serve.api_key","ty":"Option<String>","default":"none"}),
        serde_json::json!({"path":"serve.shutdown_file","ty":"Option<PathBuf>","default":"none"}),
        serde_json::json!({"path":"device.ubatch_specified","ty":"bool","default":"false"}),
        serde_json::json!({"path":"cli:model","ty":"String","default":"","flag":""}),
    ];
    let req = request(
        "conservative",
        &[
            ("device.ctx", "262144"),
            ("kv.type_k", "q8_0"),
            ("serve.api_key", "secret"),
        ],
    );
    let text = launch::build_toml(&req, &schema).unwrap();
    assert!(!text.contains("secret"));
    assert!(!text.contains("api_key"));
    assert!(!text.contains("shutdown_file"));
    assert!(!text.contains("ubatch_specified"));
    assert!(!text.contains("cli:model"));
    let doc: toml::Table = toml::from_str(&text).unwrap();
    assert_eq!(doc["device"]["ctx"].as_str(), Some("262144"));
    assert_eq!(doc["device"]["auto_profile"].as_str(), Some("conservative"));
    assert_eq!(doc["kv"]["type_k"].as_str(), Some("q8_0"));
    assert_eq!(doc["kv"]["ring"].as_bool(), Some(true));
    assert_eq!(doc["sampling"]["temp"].as_float(), Some(0.0));
    // A parent section's scalar must stay in the parent, not leak into the sub-table below it.
    assert_eq!(doc["kernels"]["qkv_fuse"].as_bool(), Some(true));
    assert_eq!(doc["kernels"]["vulkan"]["gemv"]["rm"].as_integer(), Some(2));
}

/// Importing is the inverse of exporting: section paths flatten back to `--set` keys, scalars recover
/// their string spelling, the preset returns from the header (`manual` included), and re-exporting the
/// imported form is byte-identical. A knob the live schema no longer advertises is dropped, not launched.
#[test]
fn import_toml_recovers_the_exported_form() {
    let schema = vec![
        serde_json::json!({"path":"device.ctx","ty":"Option<SizeSpec>","default":"none"}),
        serde_json::json!({"path":"device.auto_profile","ty":"AutoProfile","default":"conservative"}),
        serde_json::json!({"path":"kv.type_k","ty":"Option<DType>","default":"none"}),
        serde_json::json!({"path":"kv.ring","ty":"bool","default":"true"}),
        serde_json::json!({"path":"sampling.temp","ty":"f32","default":"0"}),
        serde_json::json!({"path":"kernels.vulkan.gemv.rm","ty":"u32","default":"2"}),
        serde_json::json!({"path":"serve.api_key","ty":"Option<String>","default":"none"}),
    ];
    let req = request(
        "conservative",
        &[
            ("device.ctx", "262144"),
            ("kv.type_k", "q8_0"),
            ("sampling.temp", "0.95"),
        ],
    );
    let text = launch::build_toml(&req, &schema).unwrap();
    let back = launch::parse_toml(&text, &schema).unwrap();
    assert_eq!(back.preset, "conservative");
    assert_eq!(
        back.set.get("device.ctx").map(String::as_str),
        Some("262144")
    );
    assert_eq!(back.set.get("kv.type_k").map(String::as_str), Some("q8_0"));
    assert_eq!(back.set.get("kv.ring").map(String::as_str), Some("true"));
    assert_eq!(
        back.set.get("sampling.temp").map(String::as_str),
        Some("0.95")
    );
    assert_eq!(
        back.set.get("kernels.vulkan.gemv.rm").map(String::as_str),
        Some("2")
    );
    // The secret never entered the file, so it cannot come back.
    assert!(!back.set.contains_key("serve.api_key"));
    // Re-exporting the imported form reproduces the file byte for byte.
    assert_eq!(launch::build_toml(&back, &schema).unwrap(), text);

    // 全手动 is recorded only in the header (auto_profile stays at its default), and must survive.
    let manual =
        launch::build_toml(&request("manual", &[("device.ctx", "8192")]), &schema).unwrap();
    assert_eq!(
        launch::parse_toml(&manual, &schema).unwrap().preset,
        "manual"
    );

    // A retired knob is dropped rather than passed to the engine as an unknown path.
    let stale = "[device]\nctx = \"262144\"\nretired_knob = \"1\"\n";
    let parsed = launch::parse_toml(stale, &schema).unwrap();
    assert_eq!(
        parsed.set.get("device.ctx").map(String::as_str),
        Some("262144")
    );
    assert!(!parsed.set.contains_key("device.retired_knob"));
}

/// Settings survive a save/load round-trip beside the keys file, so a console restart restores the
/// form. Only the preset and touched knobs are kept; defaults are re-derived from the schema.
#[test]
fn settings_persist_and_restore() {
    let console = console();
    let mut settings = crate::state::Settings::default();
    assert_eq!(settings.preset, "conservative");
    assert!(settings.set.is_empty());
    settings.preset = "manual".into();
    settings.set.insert("device.ctx".into(), "262144".into());
    settings.set.insert("kv.type_k".into(), "q8_0".into());
    console.save_settings(&settings).unwrap();
    let restored = console.load_settings();
    assert_eq!(restored.preset, "manual");
    assert_eq!(
        restored.set.get("device.ctx").map(String::as_str),
        Some("262144")
    );
    assert_eq!(
        restored.set.get("kv.type_k").map(String::as_str),
        Some("q8_0")
    );
    let _ = std::fs::remove_dir_all(console.state_path.parent().unwrap());
    let _ = std::fs::remove_file(&console.state_path);
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
    let first = vec![
        "system\u{1}你是助手".to_string(),
        "user\u{1}你好".to_string(),
    ];
    assert!(
        console.estimate_reuse("conv", &first).is_none(),
        "the first turn has nothing to reuse"
    );
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
        conv: "c".into(),
        model: "m".into(),
        api: "chat",
        started_at_ms: now_ms() - 500,
        ended_at_ms: None,
        status: Status::Streaming,
        queued: true,
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
        frames_compacted: false,
        first_delta_ms: None,
        prefill_ms: Some(200),
    };
    round.observe(DeltaKind::Reasoning, None, 1);
    assert!(
        !round.queued,
        "the first delta means the engine admitted the round"
    );
    assert_eq!(round.phase, Phase::Thinking);
    assert!(round.ttft_ms.unwrap() >= 500);
    round.observe(DeltaKind::Tool, Some("tap"), 1);
    assert_eq!(round.phase, Phase::ToolCall);
    round.observe(DeltaKind::Tool, Some("tap"), 1);
    assert_eq!(
        round.tools,
        vec!["tap".to_string()],
        "a tool is named once, not once per fragment"
    );
    round.observe(DeltaKind::Content, None, 1);
    assert_eq!(round.phase, Phase::Output);
    round.finish(Status::Done);
    assert_eq!(round.status, Status::Done);
    // 60 of 100 prompt tokens were evaluated in 200 ms.
    assert_eq!(round.prefill_tps, Some(300.0));
    assert!(round.duration_ms.unwrap() >= 500);
}

#[test]
fn a_round_that_fails_before_any_delta_is_not_left_queued() {
    let mut round = test_round(2, "c");
    round.status = Status::Streaming;
    round.queued = true;
    round.finish(Status::Error);
    assert_eq!(round.status, Status::Error);
    assert!(
        !round.queued,
        "a finished round can never keep the waiting badge"
    );
}

#[tokio::test]
async fn key_lifecycle_and_the_consent_gate() {
    let console = console();
    let app = ui::router(Arc::clone(&console));
    let unauthenticated = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/ui/state")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);
    let login = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/login")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"password":"test-admin"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(login.status(), StatusCode::OK);
    let cookie = login
        .headers()
        .get("set-cookie")
        .unwrap()
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();

    let added = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/keys")
                .header("cookie", &cookie)
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"label":"android","token":"sk-console-fixed"}"#,
                ))
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
                    .header("cookie", &cookie)
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
                .header("cookie", &cookie)
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
                .header("cookie", &cookie)
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
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/ui/keys/{id}"))
                .header("cookie", &cookie)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::OK);
    assert!(console.keys.lock().unwrap().is_empty());
    let _ = std::fs::remove_dir_all(console.state_path.parent().unwrap());
}

#[tokio::test]
async fn user_session_is_independent_from_admin() {
    let console = console();
    let app = ui::router(Arc::clone(&console));
    let admin_login = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/login")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"password":"test-admin"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let admin_cookie = admin_login
        .headers()
        .get("set-cookie")
        .unwrap()
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();

    let created = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/users")
                .header("cookie", &admin_cookie)
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"username":"alice","password":"password123"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), StatusCode::OK);

    let user_login = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/user/login")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"username":"alice","password":"password123"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(user_login.status(), StatusCode::OK);
    let user_cookie = user_login
        .headers()
        .get("set-cookie")
        .unwrap()
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();

    let me = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/ui/user/me")
                .header("cookie", &user_cookie)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(me.status(), StatusCode::OK);

    let admin_route = app
        .oneshot(
            Request::builder()
                .uri("/ui/state")
                .header("cookie", &user_cookie)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(admin_route.status(), StatusCode::UNAUTHORIZED);
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
        conv: "c".into(),
        model: "m".into(),
        api: "chat",
        started_at_ms: now_ms(),
        ended_at_ms: None,
        status: Status::Streaming,
        queued: false,
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
        frames_compacted: false,
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

#[test]
fn compact_request_keeps_only_the_opening_and_recent_messages() {
    let mut body = serde_json::json!({
        "model": "m",
        "messages": (0..40)
            .map(|i| serde_json::json!({ "role": "user", "content": format!("msg-{i}") }))
            .collect::<Vec<_>>(),
    });
    crate::state::compact_request(&mut body);
    let messages = body["messages"].as_array().unwrap();
    // Opening message + one marker + the most recent (KEEP-1) messages, never the full history.
    assert!(messages.len() <= 10, "retained {} messages", messages.len());
    assert_eq!(messages[0]["content"], "msg-0");
    let joined = body.to_string();
    assert!(joined.contains("省略"));
    assert!(!joined.contains("msg-15"), "middle history must be dropped");
    assert!(joined.contains("msg-39"), "latest message must survive");
}

#[test]
fn compact_request_caps_a_single_oversized_message() {
    let mut body = serde_json::json!({
        "messages": [{ "role": "user", "content": "x".repeat(20_000) }],
    });
    crate::state::compact_request(&mut body);
    let text = body.to_string();
    assert!(
        text.len() < 12_000,
        "retained body grew to {} bytes",
        text.len()
    );
    assert!(text.contains("截断"));
}

#[test]
fn reuse_table_is_bounded_and_stores_no_prompt_text() {
    let console = console();
    let msg = "system\u{1}你是一个有很长的系统提示的助手".repeat(20);
    // Write far more distinct conversations than the cap; the table must stay bounded.
    for i in 0..600 {
        let conv = format!("key\u{1}{msg}\u{1}turn-{i}");
        console.estimate_reuse(&conv, std::slice::from_ref(&msg));
    }
    let len = console.conversations.lock().unwrap().len();
    assert!(len <= 256, "reuse table grew to {len} conversations");
}

#[test]
fn reuse_estimate_survives_fingerprinting() {
    let console = console();
    let a = vec!["system\u{1}提示".to_string(), "user\u{1}一".to_string()];
    assert!(console.estimate_reuse("c", &a).is_none());
    let b = vec![a[0].clone(), a[1].clone(), "assistant\u{1}答".to_string()];
    let est = console.estimate_reuse("c", &b).unwrap();
    assert_eq!(est.messages, 2);
    assert!(est.ratio > 0.0 && est.ratio < 1.0);
}

#[test]
fn aged_rounds_compact_frames_but_keep_the_assembled_reply() {
    let console = console();
    let frame = |seq: u64, text: &str| crate::state::Frame {
        seq,
        data: serde_json::json!({
            "choices": [{ "delta": { "content": text } }]
        })
        .to_string(),
    };
    // One extra round past FRAME_ROUND_CAP gets pushed; a mid-list round should compact.
    for i in 0..50 {
        console.push_round(Round {
            id: i,
            key_id: "k".into(),
            key_label: "k".into(),
            conv: "c".into(),
            model: "m".into(),
            api: "chat",
            started_at_ms: now_ms(),
            ended_at_ms: None,
            status: Status::Done,
            queued: false,
            phase: Phase::Output,
            tools: Vec::new(),
            uncensor: false,
            prompt_tokens: 0,
            cached_prompt_tokens: 0,
            completion_tokens: 0,
            reuse_est: None,
            ttft_ms: None,
            live_tps: None,
            prefill_tps: None,
            decode_tps: None,
            duration_ms: None,
            frames_truncated: false,
            request: serde_json::Value::Null,
            frames: vec![frame(1, "你好"), frame(2, "世界")],
            frames_compacted: false,
            first_delta_ms: None,
            prefill_ms: None,
        });
    }
    let rounds = console.rounds.lock().unwrap();
    // The newest 48 keep raw frames; the two oldest are folded into one frame each.
    let compacted: Vec<_> = rounds.iter().filter(|r| r.frames_compacted).collect();
    assert_eq!(compacted.len(), 2, "expected 2 compacted rounds");
    for r in compacted {
        assert_eq!(
            r.frames.len(),
            1,
            "compacted round should hold a single synthetic frame"
        );
        let chunk: serde_json::Value = serde_json::from_str(&r.frames[0].data).unwrap();
        assert_eq!(chunk["choices"][0]["delta"]["content"], "你好世界");
    }
    assert!(
        rounds.front().unwrap().frames.len() == 2,
        "newest round keeps raw frames"
    );
}

#[tokio::test]
async fn health_probe_needs_no_console_api_key() {
    let console = console();
    let app = ui::router(Arc::clone(&console));
    let login = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/login")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"password":"test-admin"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let cookie = login
        .headers()
        .get("set-cookie")
        .unwrap()
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_string();
    let added = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ui/keys")
                .header("cookie", &cookie)
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"label":"probe","token":"sk-console-probe"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(added.status(), StatusCode::OK);

    // The upstream in these tests is a dead port, so a forwarded probe ends as 502. The assertion
    // that matters is that the console did not reject it for lacking a key.
    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(health.status(), StatusCode::UNAUTHORIZED);

    // Every other proxied route still requires a key once one exists.
    let models = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(models.status(), StatusCode::UNAUTHORIZED);
    let _ = std::fs::remove_dir_all(console.state_path.parent().unwrap());
}

fn test_round(id: u64, conv: &str) -> Round {
    Round {
        id,
        key_id: "k".into(),
        key_label: "k".into(),
        conv: conv.into(),
        model: "m".into(),
        api: "chat",
        started_at_ms: now_ms(),
        ended_at_ms: None,
        status: Status::Done,
        queued: false,
        phase: Phase::Output,
        tools: Vec::new(),
        uncensor: false,
        prompt_tokens: 0,
        cached_prompt_tokens: 0,
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
        frames_compacted: false,
        first_delta_ms: None,
        prefill_ms: None,
    }
}

#[test]
fn rounds_are_capped_per_conversation_not_globally() {
    let console = console();
    // One long-running agent conversation beyond the per-conversation cap, plus a short one.
    for i in 0..140 {
        console.push_round(test_round(i, "long"));
    }
    for i in 140..146 {
        console.push_round(test_round(i, "short"));
    }
    let rounds = console.rounds.lock().unwrap();
    let long: Vec<_> = rounds.iter().filter(|r| r.conv == "long").collect();
    let short: Vec<_> = rounds.iter().filter(|r| r.conv == "short").collect();
    assert_eq!(long.len(), 100, "a single conv must keep only 100 rounds");
    assert_eq!(
        short.len(),
        6,
        "a short conv must not be evicted by a long one"
    );
    // Newest-first: the retained long rounds are ids 40..139, not the first 100.
    assert_eq!(long.first().unwrap().id, 139);
    assert_eq!(long.last().unwrap().id, 40);
    assert_eq!(rounds.front().unwrap().id, 145);
}
