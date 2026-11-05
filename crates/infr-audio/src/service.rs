use crate::{
    frontend, model::SenseVoiceModel, validate_language, wav, AcousticModel, Transcriber,
    TranscriptionOptions, TranscriptionResult, TranscriptionSegment,
};
use anyhow::{bail, ensure, Context, Result};
use infr_core::{
    backend::Backend,
    config::Config,
    resource::{MemoryTier, ResourceKind, ResourceSnapshot, ResourceTracker},
};
use std::{
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub type BackendFactory = Arc<dyn Fn() -> Result<Box<dyn Backend>> + Send + Sync>;
pub type ModelFactory =
    Arc<dyn Fn(&Path, Box<dyn Backend>) -> Result<Box<dyn AcousticModel>> + Send + Sync>;
type CancellableModelFactory = Arc<
    dyn Fn(&Path, Box<dyn Backend>, &AtomicBool) -> Result<Box<dyn AcousticModel>> + Send + Sync,
>;

#[derive(Clone, Debug)]
pub struct TranscriberConfig {
    pub model_path: PathBuf,
    pub model_id: String,
    pub threads: usize,
    pub idle_timeout: Duration,
    pub chunk_seconds: f32,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            model_id: "sensevoice-small".into(),
            threads: 4,
            idle_timeout: Duration::from_secs(300),
            chunk_seconds: 28.0,
        }
    }
}

struct State {
    model: Option<Box<dyn AcousticModel>>,
    last_use: Instant,
}

struct Inner {
    config: TranscriberConfig,
    backend: BackendFactory,
    load_model: CancellableModelFactory,
    state: Mutex<State>,
    resource: ResourceTracker,
}

struct Reaper {
    stop: Arc<AtomicBool>,
    signal: Arc<(Mutex<()>, Condvar)>,
    worker: Option<JoinHandle<()>>,
}

impl Drop for Reaper {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.signal.1.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Lazy, independently evictable audio service. Only the CPU convenience
/// constructor chooses a device; the model/transport contracts remain generic.
pub struct NativeTranscriber {
    inner: Arc<Inner>,
    _reaper: Option<Reaper>,
}

impl NativeTranscriber {
    pub fn new_cpu(config: TranscriberConfig, engine_config: Arc<Config>) -> Result<Self> {
        let mut cpu_config = (*engine_config).clone();
        cpu_config.device.dev = Some("cpu".into());
        cpu_config.device.threads = Some(config.threads);
        let cpu_config = Arc::new(cpu_config);
        let threads = config.threads;
        let backend: BackendFactory = Arc::new(move || {
            Ok(Box::new(infr_cpu::CpuBackend::new_with_thread_limit(
                Arc::clone(&cpu_config),
                threads,
            )?))
        });
        let model: CancellableModelFactory = Arc::new(|path, backend, cancel| {
            Ok(Box::new(SenseVoiceModel::load_cancellable(
                path, backend, cancel,
            )?))
        });
        Self::with_cancellable_factories(config, backend, model)
    }

    /// Explicit backend and model factories allow other devices/models without
    /// teaching the request layer, idle policy or decoder about their details.
    pub fn with_factories(
        config: TranscriberConfig,
        backend: BackendFactory,
        load_model: ModelFactory,
    ) -> Result<Self> {
        // Preserve the public factory contract; custom loaders get boundary checks.
        Self::with_cancellable_factories(
            config,
            backend,
            Arc::new(move |path, backend, cancel| {
                crate::check_cancel(cancel)?;
                let model = load_model(path, backend)?;
                crate::check_cancel(cancel)?;
                Ok(model)
            }),
        )
    }

    fn with_cancellable_factories(
        config: TranscriberConfig,
        backend: BackendFactory,
        load_model: CancellableModelFactory,
    ) -> Result<Self> {
        ensure!(
            !config.model_id.trim().is_empty(),
            "transcription model id must not be empty"
        );
        ensure!(
            (1..=256).contains(&config.threads),
            "transcription threads must be in 1..=256"
        );
        ensure!(
            config.chunk_seconds.is_finite()
                && (0.02..=wav::MAX_SEGMENT_SECONDS).contains(&config.chunk_seconds),
            "transcription chunk length must be between 0.02 and 28 seconds"
        );
        let logical = std::fs::metadata(&config.model_path).with_context(|| {
            format!(
                "transcription model is not available: {}",
                config.model_path.display()
            )
        })?;
        ensure!(logical.is_file(), "transcription model must be a GGUF file");
        let resource = ResourceTracker::new(
            format!("audio:{}", config.model_id),
            ResourceKind::ModelWeights,
            logical.len(),
            0,
            MemoryTier::Ssd,
            logical.len(),
        );
        let inner = Arc::new(Inner {
            config,
            backend,
            load_model,
            state: Mutex::new(State {
                model: None,
                last_use: Instant::now(),
            }),
            resource,
        });
        let reaper = if inner.config.idle_timeout.is_zero() {
            None
        } else {
            let weak = Arc::downgrade(&inner);
            let stop = Arc::new(AtomicBool::new(false));
            let signal = Arc::new((Mutex::new(()), Condvar::new()));
            let thread_stop = Arc::clone(&stop);
            let thread_signal = Arc::clone(&signal);
            let interval = inner.config.idle_timeout.min(Duration::from_secs(1));
            let worker = thread::Builder::new().name("infr-audio-idle".into()).spawn(move || {
                while !thread_stop.load(Ordering::Acquire) {
                    if let Some(inner) = weak.upgrade() {
                        // A busy recognizer is never interrupted or evicted. Do not
                        // block Drop on a lock held across an in-flight inference.
                        if let Ok(mut state) = inner.state.try_lock() {
                            if state.model.is_some() && state.last_use.elapsed() >= inner.config.idle_timeout
                                && inner.resource.snapshot().active_requests == 0 {
                                state.model.take();
                                inner.resource.set_residency(MemoryTier::Ssd, 0);
                                tracing::info!(model=%inner.config.model_id, "idle transcription model released");
                            }
                        }
                    } else { break; }
                    let guard = thread_signal.0.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = thread_signal.1.wait_timeout_while(guard, interval, |_| !thread_stop.load(Ordering::Acquire));
                }
            }).context("start transcription idle reaper")?;
            Some(Reaper {
                stop,
                signal,
                worker: Some(worker),
            })
        };
        Ok(Self {
            inner,
            _reaper: reaper,
        })
    }

    pub fn resource_snapshot(&self) -> ResourceSnapshot {
        self.inner.resource.snapshot()
    }

    pub fn release_idle(&self) -> bool {
        let Ok(mut state) = self.inner.state.try_lock() else {
            return false;
        };
        if self.inner.resource.snapshot().active_requests != 0 {
            return false;
        }
        let released = state.model.take().is_some();
        if released {
            self.inner.resource.set_residency(MemoryTier::Ssd, 0);
        }
        released
    }
}

impl Transcriber for NativeTranscriber {
    fn model_id(&self) -> &str {
        &self.inner.config.model_id
    }

    fn transcribe_file(
        &self,
        path: &Path,
        options: &TranscriptionOptions,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(TranscriptionSegment) -> Result<()>,
    ) -> Result<TranscriptionResult> {
        validate_language(&options.language)?;
        ensure!(!cancel.load(Ordering::Acquire), "transcription cancelled");
        // Validate container before creating a potentially expensive acoustic model.
        let probe = std::fs::File::open(path).context("open audio upload")?;
        drop(wav::WavDecoder::new(probe, cancel).context("validate audio WAV structure")?);
        let _lease = self.inner.resource.acquire();
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("transcription model lock poisoned"))?;
        // Keep the guard outside the unwind boundary: discard damaged state, then
        // unlock before propagating the original panic so the mutex stays usable.
        let result = catch_unwind(AssertUnwindSafe(|| {
            crate::check_cancel(cancel)?;
            if state.model.is_none() {
                let start = Instant::now();
                let backend = (self.inner.backend)()?;
                crate::check_cancel(cancel)?;
                let model =
                    (self.inner.load_model)(&self.inner.config.model_path, backend, cancel)?;
                crate::check_cancel(cancel)?;
                self.inner
                    .resource
                    .set_residency(MemoryTier::Ram, model.weight_bytes());
                state.model = Some(model);
                tracing::info!(model=%self.inner.config.model_id, seconds=start.elapsed().as_secs_f64(), "transcription model loaded on demand");
            }
            let model = state.model.as_mut().expect("loaded above");
            let mut duration = 0.0f64;
            let mut text = String::new();
            let mut segments = 0usize;
            wav::for_each_wav_segment(
                path,
                self.inner.config.chunk_seconds,
                cancel,
                &mut |segment| {
                    if cancel.load(Ordering::Acquire) {
                        bail!("transcription cancelled");
                    }
                    // Exact digital silence contains no acoustic information. Avoid CTC
                    // hallucinations on silent chunks without an energy threshold that
                    // could suppress quiet speech. Keep its timing/event below.
                    let piece = if segment.samples.iter().all(|&sample| sample == 0.0) {
                        String::new()
                    } else {
                        let features =
                            model.prepare_audio(&segment.samples, segment.sample_rate)?;
                        model.infer_cancellable(
                            &features,
                            &options.language,
                            options.use_itn,
                            cancel,
                        )?
                    };
                    ensure!(!cancel.load(Ordering::Acquire), "transcription cancelled");
                    duration = segment.start_seconds
                        + segment.samples.len() as f64 / segment.sample_rate as f64;
                    if options.collect_text && !piece.is_empty() {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(&piece);
                    }
                    emit(TranscriptionSegment {
                        text: piece,
                        start: segment.start_seconds,
                        end: duration,
                    })?;
                    segments += 1;
                    Ok(())
                },
            )?;
            ensure!(segments != 0, "audio contains no decodable samples");
            Ok(TranscriptionResult { text, duration })
        }));
        state.last_use = Instant::now();
        match result {
            Ok(result) => result,
            Err(payload) => {
                let damaged = state.model.take();
                self.inner.resource.set_residency(MemoryTier::Ssd, 0);
                drop(state);
                drop(damaged);
                resume_unwind(payload)
            }
        }
    }
}

impl AcousticModel for SenseVoiceModel {
    fn frontend_config(&self) -> &frontend::FrontendConfig {
        SenseVoiceModel::frontend_config(self)
    }
    fn infer(
        &mut self,
        features: &frontend::Features,
        language: &str,
        use_itn: bool,
    ) -> Result<String> {
        SenseVoiceModel::infer(self, features, language, use_itn)
    }
    fn weight_bytes(&self) -> u64 {
        SenseVoiceModel::weight_bytes(self)
    }
    fn infer_cancellable(
        &mut self,
        features: &frontend::Features,
        language: &str,
        use_itn: bool,
        cancel: &AtomicBool,
    ) -> Result<String> {
        SenseVoiceModel::infer_cancellable(self, features, language, use_itn, cancel)
    }
}

#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;
