use super::*;
use std::sync::atomic::AtomicUsize;

struct Fixture {
    root: PathBuf,
    model: PathBuf,
    wav: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "infr-speech-unit-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let model = root.join("dummy.gguf");
        let wav = root.join("public-synthetic.wav");
        std::fs::write(&model, b"unit-test-model").unwrap();
        let samples: Vec<i16> = (0..1600)
            .map(|i| ((i as f32 * 0.1).sin() * 10000.0) as i16)
            .collect();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + samples.len() as u32 * 2).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16000u32.to_le_bytes());
        bytes.extend_from_slice(&32000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        std::fs::write(&wav, bytes).unwrap();
        Self { root, model, wav }
    }
    fn config(&self) -> TranscriberConfig {
        TranscriberConfig {
            model_path: self.model.clone(),
            chunk_seconds: 0.04,
            idle_timeout: Duration::ZERO,
            ..Default::default()
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.model);
        let _ = std::fs::remove_file(&self.wav);
        let _ = std::fs::remove_dir(&self.root);
    }
}

struct FakeModel {
    front: frontend::FrontendConfig,
    live: Arc<AtomicUsize>,
}
impl AcousticModel for FakeModel {
    fn frontend_config(&self) -> &frontend::FrontendConfig {
        &self.front
    }
    fn infer(&mut self, features: &frontend::Features, _: &str, _: bool) -> Result<String> {
        ensure!(features.rows > 0 && features.cols == 560, "bad features");
        Ok("test".into())
    }
    fn weight_bytes(&self) -> u64 {
        123
    }
}
impl Drop for FakeModel {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

fn factories(
    fail_first: bool,
) -> (
    BackendFactory,
    ModelFactory,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let loaded = Arc::new(AtomicUsize::new(0));
    let live = Arc::new(AtomicUsize::new(0));
    let a = loaded.clone();
    let b = live.clone();
    let backend: BackendFactory = Arc::new(|| {
        Ok(Box::new(infr_cpu::CpuBackend::new_with(Arc::new(
            Config::default(),
        ))))
    });
    let factory: ModelFactory = Arc::new(move |_, _| {
        let n = a.fetch_add(1, Ordering::SeqCst);
        if fail_first && n == 0 {
            bail!("synthetic load failure");
        }
        b.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(FakeModel {
            front: frontend::FrontendConfig {
                sample_rate: 16000,
                num_mels: 80,
                frame_length_ms: 25.0,
                frame_shift_ms: 10.0,
                lfr_m: 7,
                lfr_n: 6,
                cmvn_add: vec![0.0; 560],
                cmvn_mul: vec![1.0; 560],
            },
            live: b.clone(),
        }))
    });
    (backend, factory, loaded, live)
}

fn run(service: &NativeTranscriber, path: &Path, collect: bool) -> Result<TranscriptionResult> {
    service.transcribe_file(
        path,
        &TranscriptionOptions {
            collect_text: collect,
            ..Default::default()
        },
        &AtomicBool::new(false),
        &mut |_| Ok(()),
    )
}

#[test]
fn lazy_reuse_release_reload_and_streaming_collection() {
    let fixture = Fixture::new();
    let (backend, factory, loaded, live) = factories(false);
    let service = NativeTranscriber::with_factories(fixture.config(), backend, factory).unwrap();
    assert_eq!(loaded.load(Ordering::SeqCst), 0);
    assert_eq!(service.resource_snapshot().resident_bytes, 0);
    assert!(!run(&service, &fixture.wav, true).unwrap().text.is_empty());
    assert!(run(&service, &fixture.wav, false).unwrap().text.is_empty());
    assert_eq!(loaded.load(Ordering::SeqCst), 1);
    assert_eq!(service.resource_snapshot().resident_bytes, 123);
    assert_eq!(service.resource_snapshot().active_requests, 0);
    assert!(service.release_idle());
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert_eq!(service.resource_snapshot().tier, MemoryTier::Ssd);
    run(&service, &fixture.wav, true).unwrap();
    assert_eq!(loaded.load(Ordering::SeqCst), 2);
    drop(service);
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn failed_load_is_retryable_and_never_published_as_resident() {
    let fixture = Fixture::new();
    let (backend, factory, loaded, _) = factories(true);
    let service = NativeTranscriber::with_factories(fixture.config(), backend, factory).unwrap();
    assert!(run(&service, &fixture.wav, true).is_err());
    assert_eq!(service.resource_snapshot().resident_bytes, 0);
    assert_eq!(service.resource_snapshot().active_requests, 0);
    run(&service, &fixture.wav, true).unwrap();
    assert_eq!(loaded.load(Ordering::SeqCst), 2);
}

#[test]
fn cancellation_and_callback_failure_release_request_lease() {
    let fixture = Fixture::new();
    let (backend, factory, loaded, _) = factories(false);
    let service = NativeTranscriber::with_factories(fixture.config(), backend, factory).unwrap();
    let cancel = AtomicBool::new(true);
    assert!(service
        .transcribe_file(&fixture.wav, &Default::default(), &cancel, &mut |_| Ok(()))
        .is_err());
    assert_eq!(loaded.load(Ordering::SeqCst), 0);
    let cancel = AtomicBool::new(false);
    assert!(service
        .transcribe_file(&fixture.wav, &Default::default(), &cancel, &mut |_| bail!(
            "closed consumer"
        ))
        .is_err());
    assert_eq!(service.resource_snapshot().active_requests, 0);
    assert!(service.release_idle());
}

#[test]
fn idle_deadline_releases_model_and_service_drop_stops_reaper() {
    let fixture = Fixture::new();
    let (backend, factory, _, live) = factories(false);
    let mut config = fixture.config();
    config.idle_timeout = Duration::from_millis(20);
    let service = NativeTranscriber::with_factories(config, backend, factory).unwrap();
    run(&service, &fixture.wav, true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while live.load(Ordering::SeqCst) != 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert_eq!(service.resource_snapshot().resident_bytes, 0);
    drop(service);
}

#[test]
fn malformed_wav_never_constructs_backend_or_loads_model() {
    let fixture = Fixture::new();
    // Correct magic and RIFF size, but no fmt/data chunks.
    std::fs::write(&fixture.wav, b"RIFF\x04\0\0\0WAVE").unwrap();
    let (_, factory, loaded, live) = factories(false);
    let backend: BackendFactory = Arc::new(|| panic!("bad WAV must not construct a backend"));
    let service = NativeTranscriber::with_factories(fixture.config(), backend, factory).unwrap();
    let error = run(&service, &fixture.wav, true).unwrap_err();
    assert!(format!("{error:#}").contains("WAV has no fmt chunk"));
    assert_eq!(loaded.load(Ordering::SeqCst), 0);
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert_eq!(service.resource_snapshot().active_requests, 0);
}

#[test]
fn callback_panic_discards_model_preserves_payload_and_allows_reload() {
    let fixture = Fixture::new();
    let (backend, factory, loaded, live) = factories(false);
    let service = NativeTranscriber::with_factories(fixture.config(), backend, factory).unwrap();
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = service.transcribe_file(
            &fixture.wav,
            &Default::default(),
            &AtomicBool::new(false),
            &mut |_| std::panic::panic_any("callback panic payload"),
        );
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"callback panic payload")
    );
    assert_eq!(live.load(Ordering::SeqCst), 0);
    let snapshot = service.resource_snapshot();
    assert_eq!(snapshot.resident_bytes, 0);
    assert_eq!(snapshot.tier, MemoryTier::Ssd);
    assert_eq!(snapshot.active_requests, 0);
    assert!(!service.inner.state.is_poisoned());
    run(&service, &fixture.wav, true).unwrap();
    assert_eq!(loaded.load(Ordering::SeqCst), 2);
    assert!(service.release_idle());
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn cancellation_during_custom_load_does_not_publish_model_and_can_retry() {
    let fixture = Fixture::new();
    let (backend, factory, loaded, live) = factories(false);
    let cancel = Arc::new(AtomicBool::new(false));
    let load_cancel = cancel.clone();
    let cancelling_factory: ModelFactory = Arc::new(move |path, backend| {
        let model = factory(path, backend)?;
        load_cancel.store(true, Ordering::Release);
        Ok(model)
    });
    let service =
        NativeTranscriber::with_factories(fixture.config(), backend, cancelling_factory).unwrap();
    let error = service
        .transcribe_file(&fixture.wav, &Default::default(), &cancel, &mut |_| Ok(()))
        .unwrap_err();
    assert!(error.downcast_ref::<wav::AudioCancelled>().is_some());
    assert_eq!(live.load(Ordering::SeqCst), 0);
    assert_eq!(service.resource_snapshot().resident_bytes, 0);
    assert_eq!(service.resource_snapshot().active_requests, 0);
    // A fresh request has its own latch; the legacy factory still works unchanged.
    run(&service, &fixture.wav, true).unwrap();
    assert_eq!(loaded.load(Ordering::SeqCst), 2);
}
