//! Minimal CPU-only artifact probe; production HTTP/CLI lives in infr-server/cli.
use anyhow::{ensure, Context, Result};
use infr_tts::{NativeSynthesizer, SynthesisOptions, Synthesizer, SynthesizerConfig};
use std::{
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc},
    time::{Duration, Instant},
};
fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 4 || args.len() == 5,
        "usage: native-speech MODEL.gguf VOICE_DIRECTORY OUTPUT.wav TEXT [THREADS]"
    );
    let threads = if args.len() == 5 {
        args[4].to_str().context("thread count")?.parse()?
    } else {
        2
    };
    let model = NativeSynthesizer::new_cpu(
        SynthesizerConfig {
            model_path: PathBuf::from(&args[0]),
            lexicon_path: PathBuf::from(&args[1]),
            model_id: "melo-native-cpu".into(),
            threads,
            idle_timeout: Duration::from_secs(60),
            max_output_seconds: 30.,
        },
        Arc::new(infr_core::config::Config::default()),
    )?;
    let text = args[3].to_str().context("text must be Unicode")?;
    let now = Instant::now();
    let speech = model.synthesize(text, &SynthesisOptions::default(), &AtomicBool::new(false))?;
    let elapsed = now.elapsed();
    let mut warm_seconds = Vec::new();
    for _ in 0..2 {
        let now = Instant::now();
        let warm = model.synthesize(text, &SynthesisOptions::default(), &AtomicBool::new(false))?;
        ensure!(
            warm.samples == speech.samples,
            "seeded warm synthesis differs from cold synthesis"
        );
        warm_seconds.push(now.elapsed().as_secs_f64());
    }
    let mut bytes = Vec::with_capacity(44 + speech.samples.len() * 2);
    let len = (speech.samples.len() * 2) as u32;
    bytes.extend(b"RIFF");
    bytes.extend((36 + len).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(speech.sample_rate.to_le_bytes());
    bytes.extend((speech.sample_rate * 2).to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(len.to_le_bytes());
    for s in &speech.samples {
        let sample = (s.clamp(-1., 1.) * 32767.).round() as i16;
        bytes.extend(sample.to_le_bytes());
    }
    std::fs::write(&args[2], bytes)?;
    println!(
        "{}",
        serde_json::json!({"sample_rate":speech.sample_rate,"samples":speech.samples.len(),"audio_seconds":speech.samples.len() as f64/speech.sample_rate as f64,"cold_seconds":elapsed.as_secs_f64(),"warm_seconds":warm_seconds,"threads":threads,"peak":speech.samples.iter().map(|s|s.abs()).fold(0f32,f32::max)})
    );
    Ok(())
}
