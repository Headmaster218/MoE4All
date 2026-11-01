use super::*;
use axum::{
    body::{Body, Bytes},
    http::Request,
};
use http_body_util::BodyExt;
use infr_audio::TranscriptionSegment;
use std::{path::Path, sync::atomic::AtomicUsize};
use tower::ServiceExt;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "20261101-000000-audio-tests-{}-{epoch}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn count(&self) -> usize {
        std::fs::read_dir(&self.0).unwrap().count()
    }
    async fn empty(&self) {
        until(|| self.count() == 0).await;
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.0);
    }
}

#[derive(Default)]
struct Mock {
    calls: AtomicUsize,
    running: AtomicUsize,
    peak: AtomicUsize,
    hold: AtomicBool,
    saw_cancel: AtomicBool,
    paths: Mutex<Vec<PathBuf>>,
    options: Mutex<Vec<TranscriptionOptions>>,
    fail: bool,
    panic: bool,
}
impl Transcriber for Mock {
    fn model_id(&self) -> &str {
        "audio-mock"
    }
    fn transcribe_file(
        &self,
        path: &Path,
        options: &TranscriptionOptions,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(TranscriptionSegment) -> anyhow::Result<()>,
    ) -> anyhow::Result<TranscriptionResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let running = self.running.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak.fetch_max(running, Ordering::Relaxed);
        struct Running<'a>(&'a AtomicUsize);
        impl Drop for Running<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        let _running = Running(&self.running);
        self.paths.lock().unwrap().push(path.to_owned());
        self.options.lock().unwrap().push(options.clone());
        assert_eq!(&std::fs::read(path)?[..12], b"RIFF\x26\0\0\0WAVE");
        let started = Instant::now();
        while self.hold.load(Ordering::Relaxed) && started.elapsed() < Duration::from_secs(3) {
            if cancel.load(Ordering::Relaxed) {
                self.saw_cancel.store(true, Ordering::Relaxed);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        assert!(!self.panic, "mock worker panic");
        anyhow::ensure!(
            !self.fail,
            "private backend details must not reach the client"
        );
        emit(TranscriptionSegment {
            text: "hello ".into(),
            start: 0.0,
            end: 0.5,
        })?;
        emit(TranscriptionSegment {
            text: "world".into(),
            start: 0.5,
            end: 1.0,
        })?;
        Ok(TranscriptionResult {
            text: if options.collect_text {
                "hello world".into()
            } else {
                String::new()
            },
            duration: 1.0,
        })
    }
}

async fn until(mut check: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while !check() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .expect("mock condition timed out");
}

fn state(mock: Arc<Mock>, dir: &Directory) -> AppState {
    AppState::transcription(mock, Arc::new(Config::default())).with_transcription_upload_dir(&dir.0)
}

fn wav() -> Vec<u8> {
    let mut bytes = b"RIFF\x26\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\x02\0\0\0".to_vec();
    bytes.extend_from_slice(&[0, 0]);
    bytes
}

fn multipart(fields: &[(&str, &str)], filename: &str, audio: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--test-boundary\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
    }
    body.extend_from_slice(format!("--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend_from_slice(audio);
    body.extend_from_slice(b"\r\n--test-boundary--\r\n");
    body
}

fn request(fields: &[(&str, &str)]) -> Request<Body> {
    raw_request(Body::from(multipart(fields, "../input.wav", &wav())))
}
fn raw_request(body: Body) -> Request<Body> {
    Request::post("/v1/audio/transcriptions")
        .header(
            header::CONTENT_TYPE,
            "multipart/form-data; boundary=test-boundary",
        )
        .body(body)
        .unwrap()
}
async fn body(response: Response) -> String {
    String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap()
}

#[tokio::test]
async fn chunked_upload_json_text_and_models() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let router = build_router(state(mock.clone(), &directory));
    let bytes = multipart(
        &[("model", "audio-mock"), ("language", "en")],
        "input.wav",
        &wav(),
    );
    let chunks = bytes
        .chunks(7)
        .map(Bytes::copy_from_slice)
        .map(Ok::<_, Infallible>)
        .collect::<Vec<_>>();
    let response = router
        .clone()
        .oneshot(raw_request(Body::from_stream(futures_util::stream::iter(
            chunks,
        ))))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: serde_json::Value = serde_json::from_str(&body(response).await).unwrap();
    assert_eq!(value["text"], "hello world");
    assert_eq!(value["duration"], 1.0);
    assert_eq!(mock.options.lock().unwrap()[0].language, "en");
    directory.empty().await;
    let response = router
        .clone()
        .oneshot(request(&[
            ("model", "audio-mock"),
            ("response_format", "text"),
        ]))
        .await
        .unwrap();
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/plain; charset=utf-8"
    );
    assert_eq!(body(response).await, "hello world");
    directory.empty().await;
    let response = router
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let value: serde_json::Value = serde_json::from_str(&body(response).await).unwrap();
    assert_eq!(value["data"].as_array().unwrap().len(), 1);
    assert_eq!(value["data"][0]["id"], "audio-mock");
}

#[tokio::test]
async fn validation_auth_and_cleanup() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let router = build_router(state(mock.clone(), &directory));
    let response = router
        .clone()
        .oneshot(
            Request::post("/v1/audio/transcriptions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let mut query = request(&[("model", "audio-mock")]);
    *query.uri_mut() = "/v1/audio/transcriptions?unknown=true".parse().unwrap();
    assert_eq!(
        router.clone().oneshot(query).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
    let missing_file = Body::from("--test-boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\naudio-mock\r\n--test-boundary--\r\n");
    assert_eq!(
        router
            .clone()
            .oneshot(raw_request(missing_file))
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    let cases: Vec<(Vec<(&str, &str)>, StatusCode)> = vec![
        (vec![], StatusCode::BAD_REQUEST),
        (vec![("model", "unknown")], StatusCode::NOT_FOUND),
        (
            vec![("model", "audio-mock"), ("temperature", "0")],
            StatusCode::BAD_REQUEST,
        ),
        (
            vec![("model", "audio-mock"), ("model", "audio-mock")],
            StatusCode::BAD_REQUEST,
        ),
        (
            vec![("model", "audio-mock"), ("language", "fr")],
            StatusCode::BAD_REQUEST,
        ),
        (
            vec![("model", "audio-mock"), ("response_format", "verbose_json")],
            StatusCode::BAD_REQUEST,
        ),
        (
            vec![("model", "audio-mock"), ("stream", "yes")],
            StatusCode::BAD_REQUEST,
        ),
        (
            vec![
                ("model", "audio-mock"),
                ("stream", "true"),
                ("response_format", "text"),
            ],
            StatusCode::BAD_REQUEST,
        ),
    ];
    for (fields, status) in cases {
        assert_eq!(
            router
                .clone()
                .oneshot(request(&fields))
                .await
                .unwrap()
                .status(),
            status
        );
        directory.empty().await;
    }
    for bytes in [
        b"garbage".to_vec(),
        multipart(&[("model", "audio-mock")], "input.mp3", &wav()),
        multipart(&[("model", "audio-mock")], "input.wav", b"not wave"),
    ] {
        assert!(router
            .clone()
            .oneshot(raw_request(Body::from(bytes)))
            .await
            .unwrap()
            .status()
            .is_client_error());
        directory.empty().await;
    }
    // Truncation after a file was created must release it too.
    let mut truncated = multipart(&[("model", "audio-mock")], "input.wav", &wav());
    truncated.truncate(truncated.len() - 12);
    assert_eq!(
        router
            .clone()
            .oneshot(raw_request(Body::from(truncated)))
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    directory.empty().await;
    let mut after_file = multipart(&[("model", "audio-mock")], "input.wav", &wav());
    after_file.truncate(after_file.len() - b"--test-boundary--\r\n".len());
    after_file.extend_from_slice(b"--test-boundary\r\nContent-Disposition: form-data; name=\"unknown\"\r\n\r\nvalue\r\n--test-boundary--\r\n");
    assert_eq!(
        router
            .clone()
            .oneshot(raw_request(Body::from(after_file)))
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    directory.empty().await;
    let mut cfg = Config::default();
    cfg.serve.api_key = Some("test-key".into());
    let secured = build_router(
        AppState::transcription(mock.clone(), Arc::new(cfg))
            .with_transcription_upload_dir(&directory.0),
    );
    // Authentication runs before even attempting malformed multipart extraction.
    assert_eq!(
        secured
            .clone()
            .oneshot(raw_request(Body::from("bad")))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let mut authenticated = request(&[("model", "audio-mock")]);
    authenticated.headers_mut().insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer test-key"),
    );
    assert_eq!(
        secured.oneshot(authenticated).await.unwrap().status(),
        StatusCode::OK
    );
    directory.empty().await;
    assert_eq!(mock.calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn interrupted_upload_cleans_up_without_invoking_transcriber() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let router = build_router(state(mock.clone(), &directory));
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(2);
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    let task = tokio::spawn(router.oneshot(raw_request(Body::from_stream(stream))));
    let mut prefix = multipart(&[("model", "audio-mock")], "input.wav", &wav());
    prefix.truncate(prefix.len() - b"\r\n--test-boundary--\r\n".len());
    prefix.extend_from_slice(&vec![0u8; 64 * 1024]);
    tx.send(Ok(Bytes::from(prefix))).await.unwrap();
    until(|| directory.count() == 1).await;
    task.abort();
    let _ = task.await;
    drop(tx);
    directory.empty().await;
    assert_eq!(mock.calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn upload_limit_is_streamed_and_route_local() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let router = build_router(state(mock.clone(), &directory));
    let mut prefix = multipart(&[("model", "audio-mock")], "input.wav", &wav());
    prefix.truncate(prefix.len() - b"\r\n--test-boundary--\r\n".len());
    // Reuse one MiB without allocating a 128 MiB input in RAM.
    let chunk = Bytes::from(vec![0u8; 1024 * 1024]);
    let chunks = futures_util::stream::iter(
        std::iter::once(Ok::<_, Infallible>(Bytes::from(prefix)))
            .chain((0..129).map(move |_| Ok(chunk.clone())))
            .chain(std::iter::once(Ok(Bytes::from_static(
                b"\r\n--test-boundary--\r\n",
            )))),
    );
    let response = router
        .clone()
        .oneshot(raw_request(Body::from_stream(chunks)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    directory.empty().await;
    assert_eq!(mock.calls.load(Ordering::Relaxed), 0);
    // The existing chat limit stays at 64 MiB; it reads an independently bounded body.
    let chunk = Bytes::from(vec![b' '; 1024 * 1024]);
    let chunks =
        futures_util::stream::iter((0..65).map(move |_| Ok::<_, Infallible>(chunk.clone())));
    let response = router
        .oneshot(
            Request::post("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from_stream(chunks))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn queued_stream_heartbeats_and_disconnect_does_not_run() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let state = state(mock.clone(), &directory);
    let slot = state
        .transcription
        .as_ref()
        .unwrap()
        .slots
        .clone()
        .acquire_owned()
        .await
        .unwrap();
    let stats = state.stats.clone();
    let router = build_router(state);
    let response = tokio::time::timeout(
        Duration::from_secs(1),
        router
            .clone()
            .oneshot(request(&[("model", "audio-mock"), ("stream", "true")])),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut response_body = response.into_body();
    let frame = tokio::time::timeout(Duration::from_secs(1), response_body.frame())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(String::from_utf8_lossy(frame.data_ref().unwrap()).contains(": keep-alive"));
    until(|| stats.queued.load(Ordering::Relaxed) == 1).await;
    drop(response_body);
    until(|| stats.queued.load(Ordering::Relaxed) == 0).await;
    directory.empty().await;
    drop(slot);
    assert_eq!(mock.calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        router
            .oneshot(request(&[("model", "audio-mock")]))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    directory.empty().await;
}

#[tokio::test]
async fn active_disconnect_retains_upload_and_slot_until_worker_exits() {
    let directory = Directory::new();
    let mock = Arc::new(Mock {
        hold: AtomicBool::new(true),
        ..Mock::default()
    });
    let state = state(mock.clone(), &directory);
    let slots = state.transcription.as_ref().unwrap().slots.clone();
    let router = build_router(state);
    let response = router
        .oneshot(request(&[("model", "audio-mock"), ("stream", "true")]))
        .await
        .unwrap();
    until(|| mock.running.load(Ordering::Relaxed) == 1).await;
    let mut response_body = response.into_body();
    let frame = tokio::time::timeout(Duration::from_secs(1), response_body.frame())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(String::from_utf8_lossy(frame.data_ref().unwrap()).contains(": keep-alive"));
    drop(response_body);
    until(|| mock.saw_cancel.load(Ordering::Relaxed)).await;
    assert_eq!(slots.available_permits(), 0);
    assert!(mock.paths.lock().unwrap()[0].exists());
    mock.hold.store(false, Ordering::Relaxed);
    until(|| slots.available_permits() == 1).await;
    directory.empty().await;
}

#[tokio::test]
async fn deadline_cancels_worker_but_waits_for_file_release() {
    let directory = Directory::new();
    let mock = Arc::new(Mock {
        hold: AtomicBool::new(true),
        ..Mock::default()
    });
    let mut cfg = Config::default();
    cfg.serve.request_timeout_secs = 1;
    let state = AppState::transcription(mock.clone(), Arc::new(cfg))
        .with_transcription_upload_dir(&directory.0);
    let slots = state.transcription.as_ref().unwrap().slots.clone();
    let task = tokio::spawn(build_router(state).oneshot(request(&[("model", "audio-mock")])));
    until(|| mock.saw_cancel.load(Ordering::Relaxed)).await;
    assert_eq!(slots.available_permits(), 0);
    assert_eq!(directory.count(), 1);
    mock.hold.store(false, Ordering::Relaxed);
    let response = task.await.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    directory.empty().await;
    assert_eq!(slots.available_permits(), 1);
}

#[tokio::test]
async fn stream_deadline_delivers_terminal_error_and_cleans_up() {
    let directory = Directory::new();
    let mock = Arc::new(Mock {
        hold: AtomicBool::new(true),
        ..Mock::default()
    });
    let mut cfg = Config::default();
    cfg.serve.request_timeout_secs = 1;
    let state = AppState::transcription(mock.clone(), Arc::new(cfg))
        .with_transcription_upload_dir(&directory.0);
    let response = build_router(state)
        .oneshot(request(&[("model", "audio-mock"), ("stream", "true")]))
        .await
        .unwrap();
    until(|| mock.saw_cancel.load(Ordering::Relaxed)).await;
    mock.hold.store(false, Ordering::Relaxed);
    let text = tokio::time::timeout(Duration::from_secs(2), body(response))
        .await
        .unwrap();
    assert!(text.contains("event: error"));
    assert!(text.contains("transcription deadline exceeded"));
    assert!(!text.contains("event: transcription.done"));
    directory.empty().await;
}

#[tokio::test]
async fn invalid_storage_and_disabled_capability_fail_without_transcribing() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let blocked = directory.0.join("not-a-directory");
    std::fs::write(&blocked, b"preserve me").unwrap();
    let state = AppState::transcription(mock.clone(), Arc::new(Config::default()))
        .with_transcription_upload_dir(&blocked);
    let response = build_router(state)
        .oneshot(request(&[("model", "audio-mock")]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(std::fs::read(&blocked).unwrap(), b"preserve me");
    std::fs::remove_file(blocked).unwrap();
    let router = build_router(
        AppState::headless("chat", Arc::new(Config::default()))
            .with_transcription_upload_dir(&directory.0),
    );
    assert_eq!(
        router
            .oneshot(request(&[("model", "audio-mock")]))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(mock.calls.load(Ordering::Relaxed), 0);
    directory.empty().await;
}

#[tokio::test]
async fn dropping_nonstream_handler_cancels_and_concurrent_requests_serialize() {
    let directory = Directory::new();
    let mock = Arc::new(Mock {
        hold: AtomicBool::new(true),
        ..Mock::default()
    });
    let state = state(mock.clone(), &directory);
    let stats = state.stats.clone();
    let router = build_router(state);
    let first = tokio::spawn(router.clone().oneshot(request(&[("model", "audio-mock")])));
    until(|| mock.running.load(Ordering::Relaxed) == 1).await;
    let second = tokio::spawn(router.oneshot(request(&[("model", "audio-mock")])));
    until(|| stats.queued.load(Ordering::Relaxed) == 1).await;
    first.abort();
    let _ = first.await;
    until(|| mock.saw_cancel.load(Ordering::Relaxed)).await;
    assert_eq!(mock.calls.load(Ordering::Relaxed), 1);
    mock.hold.store(false, Ordering::Relaxed);
    let response = tokio::time::timeout(Duration::from_secs(2), second)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(mock.calls.load(Ordering::Relaxed), 2);
    assert_eq!(mock.peak.load(Ordering::Relaxed), 1);
    directory.empty().await;
}

#[tokio::test]
async fn stream_segments_done_and_backend_failure_are_terminal() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let router = build_router(state(mock.clone(), &directory));
    let text = body(
        router
            .oneshot(request(&[("model", "audio-mock"), ("stream", "true")]))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(text.matches("event: transcription.text.delta").count(), 2);
    assert_eq!(text.matches("event: transcription.done").count(), 1);
    assert!(text.contains("\"delta\":\"hello \""));
    assert!(text.find("hello").unwrap() < text.find("world").unwrap());
    assert!(!mock.options.lock().unwrap()[0].collect_text);
    directory.empty().await;
    for panic in [false, true] {
        let mock = Arc::new(Mock {
            fail: !panic,
            panic,
            ..Mock::default()
        });
        let router = build_router(state(mock, &directory));
        let text = body(
            router
                .oneshot(request(&[("model", "audio-mock"), ("stream", "true")]))
                .await
                .unwrap(),
        )
        .await;
        assert!(text.contains("event: error"));
        assert!(!text.contains("private backend details"));
        assert!(!text.contains("event: transcription.done"));
        directory.empty().await;
    }
}

#[tokio::test]
async fn admission_counts_active_queued_and_uploading_until_cleanup() {
    let directory = Directory::new();
    let mock = Arc::new(Mock {
        hold: AtomicBool::new(true),
        ..Mock::default()
    });
    let state = state(mock.clone(), &directory);
    let uploads = state.transcription.as_ref().unwrap().uploads.clone();
    let stats = state.stats.clone();
    let router = build_router(state);
    let active = router
        .clone()
        .oneshot(request(&[("model", "audio-mock"), ("stream", "true")]))
        .await
        .unwrap();
    until(|| mock.running.load(Ordering::Relaxed) == 1).await;
    let mut queued = Vec::new();
    for _ in 0..2 {
        queued.push(
            router
                .clone()
                .oneshot(request(&[("model", "audio-mock"), ("stream", "true")]))
                .await
                .unwrap(),
        );
    }
    until(|| stats.queued.load(Ordering::Relaxed) == 2).await;

    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(2);
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    let uploading = tokio::spawn(
        router
            .clone()
            .oneshot(raw_request(Body::from_stream(stream))),
    );
    let mut prefix = multipart(&[("model", "audio-mock")], "input.wav", &wav());
    prefix.truncate(prefix.len() - b"\r\n--test-boundary--\r\n".len());
    prefix.extend_from_slice(&vec![0; 64 * 1024]);
    tx.send(Ok(Bytes::from(prefix))).await.unwrap();
    until(|| directory.count() == 4).await;
    assert_eq!(uploads.available_permits(), 0);

    // Saturation must not create a fifth file or invoke another transcriber.
    assert_eq!(
        router
            .clone()
            .oneshot(request(&[("model", "audio-mock")]))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    assert_eq!(directory.count(), 4);
    drop(active);
    until(|| mock.saw_cancel.load(Ordering::Relaxed)).await;
    assert_eq!(uploads.available_permits(), 0);
    assert_eq!(
        router
            .clone()
            .oneshot(request(&[("model", "audio-mock")]))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );

    drop(queued);
    uploading.abort();
    let _ = uploading.await;
    drop(tx);
    until(|| directory.count() == 1 && uploads.available_permits() == 3).await;
    assert_eq!(mock.calls.load(Ordering::Relaxed), 1);
    // The cancelled active worker still owns its file and admission until it really exits.
    assert!(mock.paths.lock().unwrap()[0].exists());
    mock.hold.store(false, Ordering::Relaxed);
    directory.empty().await;
    until(|| uploads.available_permits() == MAX_UPLOADS).await;
    assert_eq!(
        router
            .oneshot(request(&[("model", "audio-mock")]))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    directory.empty().await;
    assert_eq!(uploads.available_permits(), MAX_UPLOADS);
}

#[tokio::test]
async fn complete_riff_header_with_bad_structure_is_client_error_before_service() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let state = state(mock.clone(), &directory);
    let uploads = state.transcription.as_ref().unwrap().uploads.clone();
    let router = build_router(state);
    // The existing fixture is valid: 38 RIFF payload bytes + 8 = 46 total bytes.
    let valid = wav();
    assert_eq!(valid.len(), 46);
    assert_eq!(u32::from_le_bytes(valid[4..8].try_into().unwrap()), 38);
    infr_audio::wav::WavDecoder::new(io::Cursor::new(&valid), &AtomicBool::new(false)).unwrap();
    let mut bad_length = valid.clone();
    bad_length[4..8].copy_from_slice(&39u32.to_le_bytes());
    let mut missing_fmt = valid.clone();
    missing_fmt[12..16].copy_from_slice(b"JUNK");
    let mut bad_frame = valid;
    bad_frame[32..34].copy_from_slice(&0u16.to_le_bytes());
    for audio in [bad_length, missing_fmt, bad_frame] {
        for stream in [false, true] {
            let fields = [
                ("model", "audio-mock"),
                ("stream", if stream { "true" } else { "false" }),
            ];
            let response = router
                .clone()
                .oneshot(raw_request(Body::from(multipart(
                    &fields,
                    "input.wav",
                    &audio,
                ))))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                if stream {
                    StatusCode::OK
                } else {
                    StatusCode::BAD_REQUEST
                }
            );
            let text = body(response).await;
            assert!(text.contains("invalid_request_error"));
            assert!(text.contains("invalid or unsupported WAV structure"));
            if stream {
                assert!(text.contains("event: error"));
                assert!(!text.contains("event: transcription.done"));
                assert!(!text.contains("event: transcription.text.delta"));
            }
            directory.empty().await;
            assert_eq!(uploads.available_permits(), MAX_UPLOADS);
        }
    }
    assert_eq!(mock.calls.load(Ordering::Relaxed), 0);
    // Opening an unavailable file remains an internal storage error.
    let failure =
        validate_wav(&directory.0.join("missing.wav"), &AtomicBool::new(false)).unwrap_err();
    assert_eq!(failure.0, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(failure.kind(), "server_error");
}

#[tokio::test]
async fn configured_deadline_covers_incomplete_upload_and_restores_admission() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let mut cfg = Config::default();
    cfg.serve.request_timeout_secs = 1;
    let state = AppState::transcription(mock.clone(), Arc::new(cfg))
        .with_transcription_upload_dir(&directory.0);
    let uploads = state.transcription.as_ref().unwrap().uploads.clone();
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(2);
    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    let task = tokio::spawn(build_router(state).oneshot(raw_request(Body::from_stream(stream))));
    let mut prefix = multipart(&[("model", "audio-mock")], "input.wav", &wav());
    prefix.truncate(prefix.len() - b"\r\n--test-boundary--\r\n".len());
    prefix.extend_from_slice(&vec![0; 64 * 1024]);
    tx.send(Ok(Bytes::from(prefix))).await.unwrap();
    until(|| directory.count() == 1).await;
    let response = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    drop(tx);
    directory.empty().await;
    until(|| uploads.available_permits() == MAX_UPLOADS).await;
    assert_eq!(mock.calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn configured_deadline_expires_in_queue_for_json_and_sse() {
    let directory = Directory::new();
    let mock = Arc::new(Mock::default());
    let mut cfg = Config::default();
    cfg.serve.request_timeout_secs = 1;
    let state = AppState::transcription(mock.clone(), Arc::new(cfg))
        .with_transcription_upload_dir(&directory.0);
    let entry = state.transcription.as_ref().unwrap();
    let slot = entry.slots.clone().acquire_owned().await.unwrap();
    let uploads = entry.uploads.clone();
    let router = build_router(state);
    for stream in [false, true] {
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            router.clone().oneshot(request(&[
                ("model", "audio-mock"),
                ("stream", if stream { "true" } else { "false" }),
            ])),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            response.status(),
            if stream {
                StatusCode::OK
            } else {
                StatusCode::GATEWAY_TIMEOUT
            }
        );
        let text = tokio::time::timeout(Duration::from_secs(2), body(response))
            .await
            .unwrap();
        assert!(text.contains("transcription deadline exceeded"));
        assert!(text.contains("server_error"));
        if stream {
            assert!(text.contains("event: error"));
            assert!(!text.contains("event: transcription.done"));
        }
        directory.empty().await;
        assert_eq!(uploads.available_permits(), MAX_UPLOADS);
    }
    assert_eq!(mock.calls.load(Ordering::Relaxed), 0);
    drop(slot);
}
