use anyhow::{Context, Result};
use clap::Args;
use infr_audio::{NativeTranscriber, Transcriber, TranscriberConfig, TranscriptionOptions};
use infr_core::config::Config;
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

#[derive(Args, Debug)]
pub(crate) struct ServeOptions {
    /// Native transcription GGUF at /v1/audio/transcriptions (loaded on first request).
    /// File-audio API supports an OpenAI-compatible subset, using an independent CPU backend.
    #[arg(long, value_name = "GGUF")]
    pub transcription_model: Option<PathBuf>,
    /// Transcription backend. This first native implementation supports only cpu.
    #[arg(long, default_value = "cpu")]
    pub transcription_dev: String,
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(usize))]
    pub transcription_threads: usize,
    /// Seconds before native audio weights unload; 0 keeps them resident.
    #[arg(long, default_value_t = 300, value_name = "SECONDS")]
    pub transcription_idle_timeout: u64,
    /// Controlled upload directory. Default: <installation>/data/audio-upload.
    #[arg(long, value_name = "DIR")]
    pub transcription_upload_dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub(crate) struct FileOptions {
    #[arg(long, value_name = "WAV")]
    pub file: PathBuf,
    #[arg(long, default_value = "auto")]
    pub language: String,
    #[arg(long, default_value_t = 4)]
    pub threads: usize,
    #[arg(long, default_value = "cpu")]
    pub dev: String,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub use_itn: bool,
    /// Print a JSON object with text and audio duration instead of plain text.
    #[arg(long)]
    pub json: bool,
}

fn validate(dev: &str, threads: usize) -> Result<()> {
    anyhow::ensure!(
        dev.eq_ignore_ascii_case("cpu"),
        "unsupported transcription device {dev:?}; only cpu is supported"
    );
    anyhow::ensure!(
        (1..=256).contains(&threads),
        "transcription threads must be in 1..=256"
    );
    Ok(())
}

fn create(
    path: &Path,
    dev: &str,
    threads: usize,
    idle_timeout: u64,
    cfg: &Arc<Config>,
) -> Result<Arc<dyn Transcriber>> {
    validate(dev, threads)?;
    anyhow::ensure!(
        path.is_file(),
        "transcription model is not a file: {}",
        path.display()
    );
    anyhow::ensure!(
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")),
        "native transcription requires a GGUF model"
    );
    let model_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .context("transcription model requires a UTF-8 filename")?
        .to_owned();
    let config = TranscriberConfig {
        model_path: path.to_path_buf(),
        model_id,
        threads,
        idle_timeout: Duration::from_secs(idle_timeout),
        ..TranscriberConfig::default()
    };
    Ok(Arc::new(NativeTranscriber::new_cpu(config, cfg.clone())?))
}

impl ServeOptions {
    pub(crate) fn load(&self, cfg: &Arc<Config>) -> Result<Option<Arc<dyn Transcriber>>> {
        // Reject unsupported backends even before chat model resolution or GPU initialization.
        validate(&self.transcription_dev, self.transcription_threads)?;
        self.transcription_model
            .as_deref()
            .map(|path| {
                create(
                    path,
                    &self.transcription_dev,
                    self.transcription_threads,
                    self.transcription_idle_timeout,
                    cfg,
                )
            })
            .transpose()
    }

    pub(crate) fn attach(
        &self,
        mut state: infr_server::AppState,
        transcriber: Option<Arc<dyn Transcriber>>,
    ) -> infr_server::AppState {
        if let Some(transcriber) = transcriber {
            state = state.with_transcription(transcriber);
        }
        if let Some(directory) = &self.transcription_upload_dir {
            state = state.with_transcription_upload_dir(directory);
        }
        state
    }
}

pub(crate) fn transcribe(model: &str, options: FileOptions, cfg: &Arc<Config>) -> Result<()> {
    validate(&options.dev, options.threads)?;
    infr_audio::validate_language(&options.language)?;
    anyhow::ensure!(
        options
            .file
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("wav")),
        "unsupported audio format; only WAV is supported"
    );
    let transcriber = create(Path::new(model), &options.dev, options.threads, 0, cfg)?;
    let request = TranscriptionOptions {
        language: options.language,
        use_itn: options.use_itn,
        collect_text: true,
    };
    let cancel = AtomicBool::new(false);
    let finished = AtomicBool::new(false);
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            while !finished.load(Ordering::Relaxed) {
                if infr_core::shutdown::shutdown_requested() {
                    cancel.store(true, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        });
        // Also stop the watcher if a transcriber unwinds.
        struct Finish<'a>(&'a AtomicBool);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let _finish = Finish(&finished);
        transcriber.transcribe_file(&options.file, &request, &cancel, &mut |_| Ok(()))
    })?;
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "transcription cancelled");
    // Explicit command output only: never send transcript/audio to tracing or diagnostic logs.
    let mut out = std::io::stdout().lock();
    if options.json {
        serde_json::to_writer(&mut out, &result)?;
        writeln!(out)?;
    } else {
        writeln!(out, "{}", result.text)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Cmd};
    use clap::Parser;

    #[test]
    fn audio_only_and_combined_serve_flags_parse() {
        let cli = Cli::try_parse_from([
            "infr",
            "serve",
            "--transcription-model",
            "audio.gguf",
            "--transcription-dev",
            "cpu",
            "--transcription-threads",
            "4",
            "--transcription-idle-timeout",
            "300",
            "--transcription-upload-dir",
            "controlled/uploads",
        ])
        .unwrap();
        let Some(Cmd::Serve {
            model,
            transcription,
            ..
        }) = cli.cmd
        else {
            panic!("expected serve");
        };
        assert!(model.is_none());
        assert_eq!(
            transcription.transcription_model.as_deref(),
            Some(Path::new("audio.gguf"))
        );
        assert_eq!(transcription.transcription_dev, "cpu");
        assert_eq!(transcription.transcription_threads, 4);
        assert_eq!(transcription.transcription_idle_timeout, 300);
        assert_eq!(
            transcription.transcription_upload_dir.as_deref(),
            Some(Path::new("controlled/uploads"))
        );
        let cli = Cli::try_parse_from([
            "infr",
            "serve",
            "chat.gguf",
            "--transcription-model",
            "audio.gguf",
        ])
        .unwrap();
        let Some(Cmd::Serve { model, .. }) = cli.cmd else {
            panic!("expected serve");
        };
        assert_eq!(model.as_deref(), Some("chat.gguf"));
    }

    #[test]
    fn standalone_defaults_and_language_flags_parse() {
        let cli = Cli::try_parse_from([
            "infr",
            "transcribe",
            "audio.gguf",
            "--file",
            "input.wav",
            "--language",
            "zh",
            "--use-itn",
            "false",
            "--json",
        ])
        .unwrap();
        let Some(Cmd::Transcribe { model, options }) = cli.cmd else {
            panic!("expected transcribe");
        };
        assert_eq!(model, "audio.gguf");
        assert_eq!(options.file, Path::new("input.wav"));
        assert_eq!(options.language, "zh");
        assert!(!options.use_itn);
        assert!(options.json);
        assert_eq!(options.dev, "cpu");
        assert_eq!(options.threads, 4);
    }

    #[test]
    fn unsupported_devices_fail_before_model_access() {
        for device in ["vulkan", "cuda", "metal", "Vulkan0"] {
            let result = create(
                Path::new("missing.gguf"),
                device,
                4,
                300,
                &Arc::new(Config::default()),
            );
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains("unsupported transcription device"));
        }
        assert!(validate("cpu", 0).is_err());
        assert!(validate("cpu", 257).is_err());
        assert!(validate("cpu", 256).is_ok());
        assert!(validate("CPU", 4).is_ok());
    }

    #[test]
    fn combined_services_reject_gpu_transcription_before_loading_any_model() {
        for device in ["vulkan", "Vulkan1", "cuda", "metal"] {
            let cli = Cli::try_parse_from([
                "infr",
                "serve",
                "--embedding-model",
                "missing-embedding.gguf",
                "--dev",
                "Vulkan1",
                "--transcription-model",
                "missing-audio.gguf",
                "--transcription-dev",
                device,
            ])
            .unwrap();
            let cmd = cli.cmd.unwrap();
            let cfg = Arc::new(Config::load_from_layers(&[
                crate::cli_flag_layer(&cmd).unwrap()
            ]));
            let original = format!("{cfg:?}");
            let error = crate::dispatch(cmd, &cfg, &infr_core::config::PartialConfig::default())
                .unwrap_err();
            assert!(error
                .to_string()
                .contains("unsupported transcription device"));
            assert_eq!(cfg.device.dev.as_deref(), Some("Vulkan1"));
            assert_eq!(format!("{cfg:?}"), original);
        }
    }
}
