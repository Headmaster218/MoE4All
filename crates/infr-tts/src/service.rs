use crate::{ensure_input, frontend::Frontend, model::MeloModel, InvalidInput};
use anyhow::{ensure, Context, Result};
use infr_core::config::Config;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub struct SynthesizerConfig {
    pub model_path: PathBuf,
    /// External voice-package directory (tokens.txt, lexicon.txt, config.json).
    pub lexicon_path: PathBuf,
    pub model_id: String,
    pub threads: usize,
    pub idle_timeout: Duration,
    pub max_output_seconds: f32,
}
#[derive(Clone, Debug)]
pub struct SynthesisOptions {
    pub speaker_id: usize,
    pub speed: f32,
    pub noise_scale: f32,
    pub seed: u64,
}
impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            speaker_id: 1,
            speed: 1.,
            noise_scale: 0.667,
            seed: 0,
        }
    }
}
#[derive(Debug)]
pub struct SpeechResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}
pub trait Synthesizer: Send + Sync {
    fn model_id(&self) -> &str;
    fn synthesize(
        &self,
        text: &str,
        options: &SynthesisOptions,
        cancel: &AtomicBool,
    ) -> Result<SpeechResult>;
}
struct Loaded {
    model: MeloModel,
    frontend: Frontend,
}
struct State {
    loaded: Option<Loaded>,
    last_used: Instant,
}
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    shutdown: AtomicBool,
}
/// Lazy voice ownership, independent of LLM/ASR backend and GPU allocation.
/// One voice executes at a time. HTTP admission/cancellation lives above this
/// layer; the model checks cancellation between bounded CPU matrix tiles.
pub struct NativeSynthesizer {
    config: SynthesizerConfig,
    engine_config: Arc<Config>,
    shared: Arc<Shared>,
    reaper: Mutex<Option<JoinHandle<()>>>,
}
impl NativeSynthesizer {
    pub fn new_cpu(config: SynthesizerConfig, engine_config: Arc<Config>) -> Result<Self> {
        ensure!(
            !config.model_id.trim().is_empty() && config.model_id.len() <= 256,
            "invalid TTS model ID"
        );
        ensure!(
            config.threads > 0 && config.threads <= 32,
            "TTS CPU thread count must be 1..32"
        );
        ensure!(
            config.max_output_seconds.is_finite()
                && config.max_output_seconds > 0.
                && config.max_output_seconds <= 120.,
            "TTS output limit must be >0 and <=120 seconds"
        );
        ensure!(
            config.idle_timeout <= Duration::from_secs(86400),
            "TTS idle timeout must be <=24h (0 disables idle unloading)"
        );
        ensure!(
            config.model_path.is_file(),
            "TTS model file does not exist: {}",
            config.model_path.display()
        );
        ensure!(
            config.lexicon_path.is_dir(),
            "TTS lexicon path must be a voice-package directory"
        );
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                loaded: None,
                last_used: Instant::now(),
            }),
            wake: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        let inner = shared.clone();
        let timeout = config.idle_timeout;
        let reaper = if timeout.is_zero() {
            None
        } else {
            Some(
                std::thread::Builder::new()
                    .name("infr-tts-idle".into())
                    .spawn(move || {
                        let Ok(mut state) = inner.state.lock() else {
                            return;
                        };
                        loop {
                            if inner.shutdown.load(Ordering::Relaxed) {
                                break;
                            }
                            let wait = if state.loaded.is_some() {
                                timeout.saturating_sub(state.last_used.elapsed())
                            } else {
                                timeout
                            };
                            let Ok((next, _)) = inner
                                .wake
                                .wait_timeout(state, wait.max(Duration::from_millis(1)))
                            else {
                                return;
                            };
                            state = next;
                            if state.loaded.is_some() && state.last_used.elapsed() >= timeout {
                                state.loaded = None;
                                tracing::debug!("unloaded idle native TTS voice");
                            }
                        }
                    })
                    .context("spawn TTS idle reaper")?,
            )
        };
        Ok(Self {
            config,
            engine_config,
            shared,
            reaper: Mutex::new(reaper),
        })
    }
    /// Immediate idle release is useful when a UI closes its playback session.
    /// An active synthesis owns the mutex until its cancellation has completed.
    pub fn unload(&self) -> Result<()> {
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("TTS state poisoned"))?;
        state.loaded = None;
        Ok(())
    }
}
impl Synthesizer for NativeSynthesizer {
    fn model_id(&self) -> &str {
        &self.config.model_id
    }
    fn synthesize(
        &self,
        text: &str,
        options: &SynthesisOptions,
        cancel: &AtomicBool,
    ) -> Result<SpeechResult> {
        crate::check_cancel(cancel)?;
        ensure_input(
            !text.trim().is_empty() && text.len() <= 8192 && text.chars().count() <= 2048,
            "TTS input must contain 1..2048 characters and <=8192 bytes",
        )?;
        ensure_input(
            options.speed.is_finite() && (0.25..=4.).contains(&options.speed),
            "TTS speed must be .25..4",
        )?;
        ensure_input(
            options.noise_scale.is_finite() && (0.0..=2.).contains(&options.noise_scale),
            "TTS noise scale must be 0..2",
        )?;
        ensure_input(
            options.speaker_id <= i32::MAX as usize,
            "speaker ID out of range",
        )?;
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("TTS state poisoned"))?;
        crate::check_cancel(cancel)?;
        if state.loaded.is_none() {
            let model = MeloModel::load(
                &self.config.model_path,
                self.engine_config.clone(),
                self.config.threads,
                cancel,
            )?;
            crate::check_cancel(cancel)?;
            let frontend = Frontend::load(&self.config.lexicon_path)?;
            crate::check_cancel(cancel)?;
            state.loaded = Some(Loaded { model, frontend });
        }
        let result = (|| {
            let loaded = state.loaded.as_ref().context("TTS load missing")?;
            let sample_rate = loaded.model.sample_rate;
            let max_samples =
                (self.config.max_output_seconds * sample_rate as f32).floor() as usize;
            // Loading the frontend above is an internal/IO boundary. Encoding
            // below only interprets request text against that validated voice.
            let sentences = loaded
                .frontend
                .encode_sentences(text)
                .map_err(|error| InvalidInput::new(error.to_string()))?;
            ensure_input(!sentences.is_empty(), "TTS input has no speakable text")?;
            let mut samples = Vec::new();
            for (i, sentence) in sentences.iter().enumerate() {
                crate::check_cancel(cancel)?;
                let mut opts = options.clone();
                opts.seed = options.seed.wrapping_add(i as u64);
                let part =
                    loaded
                        .model
                        .infer(sentence, &opts, max_samples - samples.len(), cancel)?;
                ensure!(
                    part.len() <= max_samples - samples.len(),
                    "TTS output limit exceeded"
                );
                samples.extend(part);
            }
            ensure!(
                !samples.is_empty() && samples.iter().all(|s| s.is_finite()),
                "empty/nonfinite TTS output"
            );
            Ok(SpeechResult {
                samples,
                sample_rate,
            })
        })();
        state.last_used = Instant::now();
        self.shared.wake.notify_all();
        result
    }
}
impl Drop for NativeSynthesizer {
    fn drop(&mut self) {
        // Take the wait mutex to prevent a notify-before-wait shutdown race.
        {
            let _guard = self.shared.state.lock();
            self.shared.shutdown.store(true, Ordering::Relaxed);
            self.shared.wake.notify_all();
        }
        if let Ok(handle) = self.reaper.get_mut() {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires INFR_MELO_MODEL voice package"]
    fn real_voice_lazy_idle_unload_and_inflight_cancel() -> Result<()> {
        let path =
            PathBuf::from(std::env::var_os("INFR_MELO_MODEL").context("INFR_MELO_MODEL required")?);
        let assets = path.parent().context("voice package directory")?.to_owned();
        let cfg = Arc::new(Config::default());
        let mut service = NativeSynthesizer::new_cpu(
            SynthesizerConfig {
                model_path: path.clone(),
                lexicon_path: assets.clone(),
                model_id: "lifecycle-test".into(),
                threads: 2,
                idle_timeout: Duration::from_millis(50),
                max_output_seconds: 0.05,
            },
            cfg.clone(),
        )?;
        assert!(
            service.shared.state.lock().unwrap().loaded.is_none(),
            "construction must stay lazy"
        );
        assert!(service
            .synthesize(
                "你好。",
                &SynthesisOptions::default(),
                &AtomicBool::new(true)
            )
            .is_err());
        assert!(
            service.shared.state.lock().unwrap().loaded.is_none(),
            "pre-cancelled request must not load weights"
        );
        let budget_error = service
            .synthesize(
                "你好。",
                &SynthesisOptions::default(),
                &AtomicBool::new(false),
            )
            .unwrap_err();
        assert!(
            budget_error.is::<InvalidInput>(),
            "real predicted duration budget must be a client error"
        );
        let voice_error = service
            .synthesize(
                "你好。",
                &SynthesisOptions {
                    speaker_id: 0,
                    ..Default::default()
                },
                &AtomicBool::new(false),
            )
            .unwrap_err();
        assert!(
            voice_error.is::<InvalidInput>(),
            "unsupported voice must be a client error"
        );
        let frontend_error = service
            .synthesize(
                "你好🛸。",
                &SynthesisOptions::default(),
                &AtomicBool::new(false),
            )
            .unwrap_err();
        assert!(
            frontend_error.is::<InvalidInput>(),
            "unknown text symbol must be a client error"
        );
        assert!(service.shared.state.lock().unwrap().loaded.is_some());
        let deadline = Instant::now() + Duration::from_secs(3);
        while service.shared.state.lock().unwrap().loaded.is_some() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            service.shared.state.lock().unwrap().loaded.is_none(),
            "idle voice failed to unload"
        );

        // Warm the actual model without doing a full clip, then cancel while
        // another thread is executing its encoder/flow/vocoder work.
        service.config.max_output_seconds = 30.;
        let never_cancel = AtomicBool::new(false);
        let model = MeloModel::load(&path, cfg, 2, &never_cancel)?;
        let frontend = Frontend::load(&assets)?;
        {
            let mut state = service.shared.state.lock().unwrap();
            state.loaded = Some(Loaded { model, frontend });
            state.last_used = Instant::now();
        }
        let service = Arc::new(service);
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_service = service.clone();
        let worker_cancel = cancel.clone();
        let started = Instant::now();
        let worker = std::thread::spawn(move || {
            worker_service.synthesize(
                "你好，我是你的智能助手，现在可以使用本地中文语音。",
                &SynthesisOptions::default(),
                &worker_cancel,
            )
        });
        std::thread::sleep(Duration::from_millis(20));
        cancel.store(true, Ordering::Relaxed);
        let result = worker
            .join()
            .map_err(|_| anyhow::anyhow!("TTS cancellation worker panicked"))?;
        assert!(result.is_err(), "in-flight synthesis ignored cancellation");
        ensure!(
            started.elapsed() < Duration::from_secs(2),
            "in-flight cancellation took too long"
        );
        eprintln!(
            "native in-flight cancellation latency {:.3}ms",
            started.elapsed().as_secs_f64() * 1000.
        );
        drop(service);

        let no_reaper = NativeSynthesizer::new_cpu(
            SynthesizerConfig {
                model_path: path,
                lexicon_path: assets,
                model_id: "keep-loaded-test".into(),
                threads: 1,
                idle_timeout: Duration::ZERO,
                max_output_seconds: 1.,
            },
            Arc::new(Config::default()),
        )?;
        assert!(
            no_reaper.reaper.lock().unwrap().is_none(),
            "zero idle timeout must not create a polling worker"
        );
        Ok(())
    }
}
