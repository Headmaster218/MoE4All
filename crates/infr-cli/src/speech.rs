//! Optional native Melo CPU speech capability; it does not select the chat/GPU backend.
#[cfg(feature = "tts")]
use anyhow::Context;
use anyhow::Result;
use clap::Args;
use infr_core::config::Config;
use std::{path::PathBuf, sync::Arc};

#[derive(Args, Debug)]
pub(crate) struct ServeOptions {
    /// Optional native Melo GGUF at /v1/audio/speech (requires a tts-enabled build).
    #[arg(long, value_name = "GGUF", requires = "speech_lexicon")]
    pub speech_model: Option<PathBuf>,
    /// External asset directory containing lexicon.txt, tokens.txt and config.json.
    #[arg(long, value_name = "DIR", requires = "speech_model")]
    pub speech_lexicon: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    pub speech_threads: usize,
    /// Seconds before idle native speech weights unload; 0 keeps them resident.
    #[arg(long, default_value_t = 300)]
    pub speech_idle_timeout: u64,
    /// Hard cap per generated clip, in seconds (1..=120).
    #[arg(long, default_value_t = 30.0)]
    pub speech_max_output_seconds: f32,
}

#[derive(Args, Debug)]
pub(crate) struct FileOptions {
    #[arg(long, value_name = "DIR")]
    pub lexicon: PathBuf,
    #[arg(long)]
    pub text: String,
    /// Explicit output path; mono signed PCM16, native sample rate.
    #[arg(long, value_name = "PATH")]
    pub output: PathBuf,
    #[arg(long, value_parser = ["wav", "pcm"], default_value = "wav")]
    pub format: String,
    #[arg(long, default_value_t = 1)]
    pub speaker_id: usize,
    #[arg(long, default_value_t = 1.0)]
    pub speed: f32,
    #[arg(long, default_value_t = 4)]
    pub threads: usize,
    #[arg(long, default_value_t = 30.0)]
    pub max_output_seconds: f32,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
}

fn validate_limits(threads: usize, seconds: f32) -> Result<()> {
    anyhow::ensure!(
        (1..=256).contains(&threads),
        "speech threads must be in 1..=256"
    );
    anyhow::ensure!(
        seconds.is_finite() && (1.0..=120.0).contains(&seconds),
        "speech max output seconds must be in 1..=120"
    );
    Ok(())
}

impl ServeOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        validate_limits(self.speech_threads, self.speech_max_output_seconds)?;
        anyhow::ensure!(
            self.speech_model.is_some() == self.speech_lexicon.is_some(),
            "--speech-model and --speech-lexicon must be supplied together"
        );
        #[cfg(not(feature = "tts"))]
        anyhow::ensure!(
            self.speech_model.is_none(),
            "native speech is unavailable in this build; rebuild infr-cli with --features tts"
        );
        if let Some(model) = &self.speech_model {
            anyhow::ensure!(
                model.is_file(),
                "speech model is not a file: {}",
                model.display()
            );
            anyhow::ensure!(
                model
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")),
                "native speech requires a GGUF model"
            );
        }
        if let Some(lexicon) = &self.speech_lexicon {
            anyhow::ensure!(
                lexicon.is_dir(),
                "speech assets are not a directory: {}",
                lexicon.display()
            );
        }
        Ok(())
    }

    pub(crate) fn attach(
        &self,
        state: infr_server::AppState,
        cfg: &Arc<Config>,
    ) -> Result<infr_server::AppState> {
        self.validate()?;
        #[cfg(feature = "tts")]
        if let (Some(model), Some(lexicon)) = (&self.speech_model, &self.speech_lexicon) {
            let synthesizer = create(
                model,
                lexicon,
                self.speech_threads,
                self.speech_idle_timeout,
                self.speech_max_output_seconds,
                cfg,
            )?;
            return Ok(state.with_speech(synthesizer));
        }
        let _ = cfg;
        Ok(state)
    }
}

#[cfg(feature = "tts")]
fn create(
    model: &std::path::Path,
    lexicon: &std::path::Path,
    threads: usize,
    idle: u64,
    seconds: f32,
    cfg: &Arc<Config>,
) -> Result<Arc<dyn infr_tts::Synthesizer>> {
    let model_id = model
        .file_stem()
        .and_then(|stem| stem.to_str())
        .context("speech model requires a UTF-8 filename")?
        .to_owned();
    Ok(Arc::new(infr_tts::NativeSynthesizer::new_cpu(
        infr_tts::SynthesizerConfig {
            model_path: model.to_path_buf(),
            lexicon_path: lexicon.to_path_buf(),
            model_id,
            threads,
            idle_timeout: std::time::Duration::from_secs(idle),
            max_output_seconds: seconds,
        },
        cfg.clone(),
    )?))
}

#[cfg(not(feature = "tts"))]
pub(crate) fn speak(_model: &str, _options: FileOptions, _cfg: &Arc<Config>) -> Result<()> {
    anyhow::bail!(
        "native speech is unavailable in this build; rebuild infr-cli with --features tts"
    )
}

#[cfg(feature = "tts")]
pub(crate) fn speak(model: &str, options: FileOptions, cfg: &Arc<Config>) -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    validate_limits(options.threads, options.max_output_seconds)?;
    anyhow::ensure!(
        !options.text.trim().is_empty()
            && options.text.len() <= 8192
            && options.text.chars().count() <= 2048,
        "speech text must contain 1..2048 characters and at most 8192 bytes"
    );
    anyhow::ensure!(
        options.speed.is_finite() && (0.25..=4.0).contains(&options.speed),
        "speech speed must be in 0.25..=4"
    );
    let synthesizer = create(
        std::path::Path::new(model),
        &options.lexicon,
        options.threads,
        0,
        options.max_output_seconds,
        cfg,
    )?;
    let cancel = AtomicBool::new(false);
    let finished = AtomicBool::new(false);
    let request = infr_tts::SynthesisOptions {
        speaker_id: options.speaker_id,
        speed: options.speed,
        seed: options.seed,
        ..Default::default()
    };
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            while !finished.load(Ordering::Relaxed) {
                if infr_core::shutdown::shutdown_requested() {
                    cancel.store(true, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
        });
        struct Finish<'a>(&'a AtomicBool);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let _finish = Finish(&finished);
        synthesizer.synthesize(&options.text, &request, &cancel)
    })?;
    anyhow::ensure!(!cancel.load(Ordering::Relaxed), "speech cancelled");
    let format = if options.format == "pcm" {
        infr_server::speech::AudioFormat::Pcm
    } else {
        infr_server::speech::AudioFormat::Wav
    };
    let bytes = infr_server::speech::encode_audio(&result, format)?;
    // create_new avoids overwriting source assets or a previously generated clip.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&options.output)
        .with_context(|| format!("cannot create speech output {}", options.output.display()))?;
    use std::io::Write;
    file.write_all(&bytes)?;
    // print-ok: explicitly requested synthesis result, never the input text.
    println!(
        "speech: {} Hz, {:.2} seconds -> {}",
        result.sample_rate,
        result.samples.len() as f64 / f64::from(result.sample_rate),
        options.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Cmd};
    use clap::Parser;

    #[test]
    fn speech_only_and_combined_modules_parse_without_selecting_a_gpu() {
        for prefix in [
            vec!["infr", "serve"],
            vec![
                "infr",
                "serve",
                "chat.gguf",
                "--embedding-model",
                "embed.gguf",
                "--transcription-model",
                "asr.gguf",
            ],
        ] {
            let mut args = prefix;
            args.extend([
                "--speech-model",
                "melo.gguf",
                "--speech-lexicon",
                "lexicon.json",
            ]);
            let cli = Cli::try_parse_from(args).unwrap();
            let Some(Cmd::Serve { speech, .. }) = cli.cmd else {
                panic!("serve expected")
            };
            assert_eq!(speech.speech_model.unwrap(), PathBuf::from("melo.gguf"));
            assert_eq!(speech.speech_threads, 4);
        }
        assert!(Cli::try_parse_from(["infr", "serve", "--speech-model", "melo.gguf"]).is_err());
        let cmd = Cli::try_parse_from([
            "infr",
            "speak",
            "melo.gguf",
            "--lexicon",
            "lexicon.json",
            "--text",
            "hello",
            "--output",
            "clip.wav",
        ])
        .unwrap()
        .cmd
        .unwrap();
        assert!(!crate::uses_llm(&cmd));
        let Cmd::Speak { options, .. } = cmd else {
            unreachable!()
        };
        assert_eq!(options.speaker_id, 1);
    }

    #[test]
    fn limits_reject_unbounded_or_invalid_work() {
        for (threads, seconds) in [(0, 30.0), (257, 30.0), (4, 0.0), (4, f32::NAN), (4, 121.0)] {
            assert!(validate_limits(threads, seconds).is_err());
        }
        assert!(validate_limits(4, 30.0).is_ok());
    }
}
