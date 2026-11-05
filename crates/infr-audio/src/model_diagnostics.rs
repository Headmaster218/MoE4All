//! Opt-in test-only CPU diagnostics. No production timers, downloads or services.
#![cfg(test)]

use super::*;
use std::{cell::RefCell, sync::Arc, time::Instant};

#[derive(Default)]
struct StageTiming {
    name: String,
    compile: f64,
    execute: f64,
    wall: f64,
}

thread_local! {
    static STAGES: RefCell<Option<Vec<StageTiming>>> = const { RefCell::new(None) };
}

pub(super) fn record(name: &str, compile: f64, execute: f64, wall: f64) {
    STAGES.with(|stages| {
        if let Some(stages) = stages.borrow_mut().as_mut() {
            stages.push(StageTiming {
                name: name.into(),
                compile,
                execute,
                wall,
            });
        }
    });
}

#[test]
#[ignore = "local public input; coordinate exclusive CPU window before running"]
fn public_model_cpu_profile() -> Result<()> {
    let model_path =
        std::env::var_os("MOE4ALL_AUDIO_TEST_MODEL").context("set MOE4ALL_AUDIO_TEST_MODEL")?;
    let wav_path =
        std::env::var_os("MOE4ALL_AUDIO_TEST_WAV").context("set MOE4ALL_AUDIO_TEST_WAV")?;
    let language = std::env::var("MOE4ALL_AUDIO_TEST_LANGUAGE").unwrap_or_else(|_| "auto".into());
    let reference = std::env::var("MOE4ALL_AUDIO_DIAG_REFERENCE").is_ok_and(|x| x == "1");
    let trials: usize = std::env::var("MOE4ALL_AUDIO_DIAG_TRIALS")
        .unwrap_or_else(|_| "3".into())
        .parse()?;
    ensure!(
        (1..=10).contains(&trials),
        "diagnostic trials must be 1..10"
    );
    let mut config = infr_core::config::Config::load_from_env();
    config.device.threads = Some(4);
    config.kernels.cpu.reference = reference;
    let operator_profile = config.prof.ops;
    let backend = infr_cpu::CpuBackend::new_with_thread_limit(Arc::new(config), 4)?;
    let started = Instant::now();
    let mut model = SenseVoiceModel::load(Path::new(&model_path), Box::new(backend))?;
    println!("ASR_LOAD {{\"seconds\":{},\"weight_bytes\":{},\"threads\":4,\"reference\":{reference},\"operator_profile\":{operator_profile}}}",
        started.elapsed().as_secs_f64(), model.weight_bytes());
    let mut expected_text: Option<String> = None;
    for trial in 0..trials {
        let request_started = Instant::now();
        let started = Instant::now();
        let mut segments = Vec::new();
        crate::wav::for_each_wav_segment(
            Path::new(&wav_path),
            28.,
            &AtomicBool::new(false),
            &mut |segment| {
                ensure!(
                    segments.is_empty(),
                    "diagnostics require one sample of at most 28 seconds"
                );
                segments.push(segment);
                Ok(())
            },
        )?;
        let decode_seconds = started.elapsed().as_secs_f64();
        ensure!(
            !segments.is_empty(),
            "diagnostic input has no audio samples"
        );
        let mut duration = 0.;
        let mut frontend_seconds = 0.;
        let mut infer_seconds = 0.;
        let mut transcript = String::new();
        STAGES.with(|stages| *stages.borrow_mut() = Some(Vec::new()));
        for segment in segments {
            duration += segment.samples.len() as f64 / segment.sample_rate as f64;
            let started = Instant::now();
            let features =
                crate::frontend::extract_features(&segment.samples, model.frontend_config())?;
            frontend_seconds += started.elapsed().as_secs_f64();
            let started = Instant::now();
            let text = model.infer(&features, &language, true)?;
            infer_seconds += started.elapsed().as_secs_f64();
            if !transcript.is_empty() {
                transcript.push('\n');
            }
            transcript.push_str(&text);
        }
        let wall = request_started.elapsed().as_secs_f64();
        let stages = STAGES.with(|stages| stages.borrow_mut().take().unwrap());
        let compile: f64 = stages.iter().map(|s| s.compile).sum();
        let execute: f64 = stages.iter().map(|s| s.execute).sum();
        let stage_wall: f64 = stages
            .iter()
            .filter(|s| s.name != "ctc.pipeline")
            .map(|s| s.wall)
            .sum();
        let ctc: f64 = stages
            .iter()
            .filter(|s| s.name == "ctc.pipeline")
            .map(|s| s.wall)
            .sum();
        for s in &stages {
            println!("ASR_STAGE {{\"trial\":{trial},\"name\":{:?},\"compile_seconds\":{},\"execute_seconds\":{},\"wall_seconds\":{}}}",
                s.name, s.compile, s.execute, s.wall);
        }
        println!("ASR_TRIAL {{\"trial\":{trial},\"duration_seconds\":{duration},\"decode_seconds\":{decode_seconds},\"frontend_seconds\":{frontend_seconds},\"infer_seconds\":{infer_seconds},\"stage_compile_seconds\":{compile},\"stage_execute_seconds\":{execute},\"stage_wall_seconds\":{stage_wall},\"ctc_stage_seconds\":{ctc},\"request_seconds_excluding_load\":{wall},\"rtf\":{},\"text\":{transcript:?}}}", wall / duration);
        if let Some(expected) = &expected_text {
            ensure!(&transcript == expected, "warm repeat changed transcript");
        } else {
            expected_text = Some(transcript);
        }
    }
    Ok(())
}
