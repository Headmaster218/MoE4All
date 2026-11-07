//! One-hypothesis CPU probe: Graph marshalling versus borrowed F32 activations.
//! Diagnostic counters are optional and absent in ordinary production builds.
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
        args.len() == 4,
        "usage: cpu-ab MODEL.gguf VOICE_DIRECTORY RESULT_PREFIX TEXT"
    );
    let prefix = PathBuf::from(&args[2]);
    let synthesizer = NativeSynthesizer::new_cpu(
        SynthesizerConfig {
            model_path: PathBuf::from(&args[0]),
            lexicon_path: PathBuf::from(&args[1]),
            model_id: "cpu-ab".into(),
            threads: 4,
            idle_timeout: Duration::from_secs(300),
            max_output_seconds: 30.,
        },
        Arc::new(infr_core::config::Config::default()),
    )?;
    let text = args[3].to_str().context("UTF8 text")?;
    // Export the actual native frontend output for the later, serialized
    // PyTorch comparison. This preparation is outside synthesis timing and
    // drops the temporary frontend before the first lazy model load.
    {
        let frontend = infr_tts::frontend::Frontend::load(&PathBuf::from(&args[1]))?;
        let sentences = frontend.encode_sentences(text)?;
        let tokens = sentences
            .iter()
            .map(|s| serde_json::json!({"phones":s.phones,"tones":s.tones,"languages":s.languages}))
            .collect::<Vec<_>>();
        std::fs::write(
            prefix.with_extension("tokens.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"text":text,"sentences":tokens,"speaker_id":1,"speed":1.,"noise_scale":0.,"seed":0,"sdp_ratio":0.,"bert_mode":"zero-with-projection-bias"}),
            )?,
        )?;
    }
    let options = SynthesisOptions {
        speaker_id: 1,
        speed: 1.,
        noise_scale: 0.,
        seed: 0,
    };
    let cancel = AtomicBool::new(false);
    let mut first = None;
    let mut rows = vec![];
    for repetition in 0..3 {
        let _ = infr_cpu::f32_profile::take();
        let _ = infr_tts::cpu_profile::take();
        let started = Instant::now();
        let speech = synthesizer.synthesize(text, &options, &cancel)?;
        let total = started.elapsed().as_secs_f64();
        let p = infr_cpu::f32_profile::take();
        let phases = infr_tts::cpu_profile::take();
        if let Some(first) = &first {
            ensure!(&speech.samples == first, "warm seeded waveform differs");
        } else {
            first = Some(speech.samples.clone());
        }
        let audio_seconds = speech.samples.len() as f64 / speech.sample_rate as f64;
        let api = p.linear_wall_ns as f64 / 1e9;
        let dispatch = p.dispatch_wall_ns as f64 / 1e9;
        let row = serde_json::json!({"repetition":repetition,"kind":if repetition==0{"first lazy load; OS file cache warm"}else{"warm"},
            "total_seconds":total,"audio_seconds":audio_seconds,"rtf":total/audio_seconds,
            "sample_rate":speech.sample_rate,"samples":speech.samples.len(),"linear_calls":p.linear_calls,
            "linear_api_wall_seconds":api,"cpu_dispatch_wall_seconds":dispatch,
            "outer_graph_and_marshalling_wall_seconds":(api-dispatch).max(0.),
            "non_linear_and_load_wall_seconds":(total-api).max(0.),
            "parallel_sgemm_task_wall_seconds_sum":p.parallel_sgemm_task_wall_ns_sum as f64/1e9,
            "phase_wall_seconds":{"conv_total":phases[0],"conv_pack":phases[1],"conv_linear":phases[2],"conv_output_copy":phases[3],"transpose_total":phases[4],"transpose_linear":phases[5],"transpose_scatter":phases[6],"all_signal_bias":phases[7],"all_signal_add":phases[8],"all_signal_leaky":phases[9],"conv_fused_epilogue":phases[10]}});
        println!("{row}");
        rows.push(row);
        if repetition == 0 {
            std::fs::write(
                prefix.with_extension("f32"),
                bytemuck::cast_slice(&speech.samples),
            )?;
            let data_bytes = (speech.samples.len() * 2) as u32;
            let mut wav = Vec::with_capacity(data_bytes as usize + 44);
            wav.extend(b"RIFF");
            wav.extend((data_bytes + 36).to_le_bytes());
            wav.extend(b"WAVEfmt ");
            wav.extend(16u32.to_le_bytes());
            wav.extend(1u16.to_le_bytes());
            wav.extend(1u16.to_le_bytes());
            wav.extend(speech.sample_rate.to_le_bytes());
            wav.extend((speech.sample_rate * 2).to_le_bytes());
            wav.extend(2u16.to_le_bytes());
            wav.extend(16u16.to_le_bytes());
            wav.extend(b"data");
            wav.extend(data_bytes.to_le_bytes());
            for v in &speech.samples {
                wav.extend(((v.clamp(-1., 1.) * 32767.).round() as i16).to_le_bytes());
            }
            std::fs::write(prefix.with_extension("wav"), wav)?;
        }
    }
    let result = serde_json::json!({"route":if cfg!(feature="fused-conv"){"fused-conv"}else if cfg!(feature="borrowed-f32"){"borrowed-f32"}else{"graph"},"threads":4,"speaker_id":1,"speed":1.,"noise_scale":0.,"seed":0,"text":text,
        "definition":"dispatch includes the existing SGEMM packing, CPU pool handoff and transpose; summed parallel SGEMM task wall times overlap and must not be added to wall latency",
        "measurements":rows});
    std::fs::write(
        prefix.with_extension("json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
