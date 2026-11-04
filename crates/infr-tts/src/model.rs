//! Melo Chinese zero-BERT / SDP=0 inference. The export preserves original
//! checkpoint names and folds weight normalization offline. Training-only
//! posterior/discriminators and stochastic duration flows are not executed.
use crate::{
    frontend::FrontendOutput,
    math::{self, Signal},
    SynthesisOptions,
};
use anyhow::{bail, ensure, Context, Result};
use infr_core::{config::Config, Backend, Buffer, BufferUsage, DType, WeightSource};
use infr_cpu::CpuBackend;
use infr_gguf::{Gguf, TensorBytes};
use std::{
    collections::HashMap,
    path::Path,
    sync::{atomic::AtomicBool, Arc},
};
type Trace<'a> = dyn FnMut(&str, &Signal) -> Result<()> + 'a;

struct Weight {
    shape: Vec<usize>, // original PyTorch dimensions, slowest first
    bytes: TensorBytes,
    buffer: Box<dyn Buffer>,
}
pub(crate) struct MeloModel {
    backend: CpuBackend,
    weights: HashMap<String, Weight>,
    config: serde_json::Value,
    pub sample_rate: u32,
    hop: usize,
    hidden: usize,
    inter: usize,
    heads: usize,
    layers: usize,
    flow_layers: usize,
    upsample: Vec<usize>,
    kernels: Vec<usize>,
    dilations: Vec<Vec<usize>>,
    speakers: Vec<usize>,
}

impl MeloModel {
    pub fn load(
        path: &Path,
        cfg: Arc<Config>,
        threads: usize,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        crate::check_cancel(cancel)?;
        let gguf =
            Gguf::open(path).with_context(|| format!("open Melo GGUF {}", path.display()))?;
        ensure!(
            gguf.metadata().str("general.architecture") == Some("melo_tts"),
            "expected native melo_tts GGUF export"
        );
        ensure!(
            gguf.metadata().str("melo.bert_mode") == Some("zero-with-projection-bias")
                && gguf.metadata().str("melo.sdp_ratio") == Some("0"),
            "native Melo v1 requires the explicit zero-BERT / deterministic-duration export"
        );
        let config: serde_json::Value = serde_json::from_str(
            gguf.metadata()
                .str("melo.config")
                .context("missing melo.config")?,
        )?;
        let m = &config["model"];
        let number = |v: &serde_json::Value, key: &str| -> Result<usize> {
            let n = v[key]
                .as_u64()
                .with_context(|| format!("missing Melo config {key}"))?;
            ensure!(n > 0 && n <= 8192, "Melo config {key} out of range");
            Ok(n as usize)
        };
        let hidden = number(m, "hidden_channels")?;
        let inter = number(m, "inter_channels")?;
        let heads = number(m, "n_heads")?;
        let layers = number(m, "n_layers")?;
        let flow_layers = number(m, "n_layers_trans_flow")?;
        // This first native format deliberately supports the tested Chinese
        // TransformerCoupling/ResBlock1 graph, not arbitrary VITS graphs.
        ensure!(
            hidden == 192 && inter == 192 && heads == 2 && layers == 6 && flow_layers == 3,
            "unsupported Melo architecture; expected Chinese 192/2/6/3 export"
        );
        ensure!(
            m["resblock"].as_str() == Some("1"),
            "only Melo ResBlock1 is supported"
        );
        let sample_rate = config["data"]["sampling_rate"]
            .as_u64()
            .context("missing sampling rate")?;
        ensure!(sample_rate == 44100, "unsupported Melo sample rate");
        let sample_rate = sample_rate as u32;
        let hop = number(&config["data"], "hop_length")?;
        let speakers = config["data"]["spk2id"]
            .as_object()
            .context("Melo package must declare available speakers")?
            .values()
            .map(|id| {
                id.as_u64()
                    .map(|n| n as usize)
                    .context("invalid available speaker ID")
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            !speakers.is_empty() && speakers.contains(&1),
            "Melo Chinese package must declare speaker 1"
        );
        let list = |v: &serde_json::Value| -> Result<Vec<usize>> {
            v.as_array()
                .context("missing Melo array")?
                .iter()
                .map(|n| {
                    let n = n.as_u64().context("invalid Melo array value")?;
                    ensure!(n > 0 && n <= 64, "Melo array value out of range");
                    Ok(n as usize)
                })
                .collect()
        };
        let upsample = list(&m["upsample_rates"])?;
        let kernels = list(&m["resblock_kernel_sizes"])?;
        let dilations = m["resblock_dilation_sizes"]
            .as_array()
            .context("missing resblock dilations")?
            .iter()
            .map(list)
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            upsample == [8, 8, 2, 2, 2]
                && kernels == [3, 7, 11]
                && dilations == vec![vec![1, 3, 5]; 3],
            "unsupported Melo decoder geometry"
        );
        ensure!(
            upsample.iter().product::<usize>() == hop && sample_rate == 44100,
            "inconsistent Melo audio geometry"
        );
        let backend = CpuBackend::new_with_thread_limit(cfg, threads)?.with_f32_gemm()?;
        let mut weights = HashMap::new();
        ensure!(gguf.tensors().len() <= 2048, "excessive Melo tensor count");
        for info in gguf.tensors() {
            crate::check_cancel(cancel)?;
            ensure!(
                info.dtype == DType::F32,
                "native Melo v1 requires F32 weights: {}",
                info.name
            );
            ensure!(
                !info.shape.is_empty() && info.shape.iter().all(|&n| n > 0 && n <= 1_000_000),
                "invalid tensor dimensions {}",
                info.name
            );
            let bytes = gguf.tensor_bytes_arc(&info.name)?;
            ensure!(
                bytes.len() <= 256 * 1024 * 1024,
                "excessive Melo tensor {}",
                info.name
            );
            let shape = info.shape.iter().rev().copied().collect::<Vec<_>>();
            let buffer = if info.name.starts_with("dec.ups.") && info.name.ends_with(".weight") {
                ensure!(
                    shape.len() == 3,
                    "invalid ConvTranspose weight {}",
                    info.name
                );
                let (input, output, k) = (shape[0], shape[1], shape[2]);
                let raw = read_f32(&bytes)?;
                let mut packed = vec![0.; raw.len()];
                for ic in 0..input {
                    for oc in 0..output {
                        for tap in 0..k {
                            packed[(oc * k + tap) * input + ic] = raw[(ic * output + oc) * k + tap];
                        }
                    }
                }
                let buf = backend.alloc(packed.len() * 4, BufferUsage::Weights)?;
                backend.upload(buf.as_ref(), bytemuck::cast_slice(&packed))?;
                buf
            } else if info.name.ends_with(".weight") && shape.len() == 3 && shape[2] > 1 {
                // Reorder Conv1d [out,in,tap] to [out,tap,in] so
                // runtime im2col copies complete contiguous channel rows.
                let (output, input, k) = (shape[0], shape[1], shape[2]);
                let raw = read_f32(&bytes)?;
                let mut packed = vec![0.; raw.len()];
                for oc in 0..output {
                    for tap in 0..k {
                        for ic in 0..input {
                            packed[(oc * k + tap) * input + ic] = raw[(oc * input + ic) * k + tap];
                        }
                    }
                }
                let buf = backend.alloc(packed.len() * 4, BufferUsage::Weights)?;
                backend.upload(buf.as_ref(), bytemuck::cast_slice(&packed))?;
                bytes.release_source_pages();
                buf
            } else {
                backend.map_weight(bytes.clone())
            };
            ensure!(
                weights
                    .insert(
                        info.name.clone(),
                        Weight {
                            shape,
                            bytes,
                            buffer
                        }
                    )
                    .is_none(),
                "duplicate Melo tensor {}",
                info.name
            );
        }
        Ok(Self {
            backend,
            weights,
            config,
            sample_rate,
            hop,
            hidden,
            inter,
            heads,
            layers,
            flow_layers,
            upsample,
            kernels,
            dilations,
            speakers,
        })
    }
    fn weight(&self, name: &str) -> Result<&Weight> {
        self.weights
            .get(name)
            .with_context(|| format!("missing Melo weight {name}"))
    }
    fn values(&self, name: &str) -> Result<Vec<f32>> {
        read_f32(&self.weight(name)?.bytes)
    }
    fn conv(&self, x: &Signal, name: &str, dilation: usize, cancel: &AtomicBool) -> Result<Signal> {
        let w = self.weight(&format!("{name}.weight"))?;
        ensure!(
            w.shape.len() == 3 && w.shape[1] == x.c,
            "Melo Conv shape mismatch {name}"
        );
        let bias = self
            .weights
            .get(&format!("{name}.bias"))
            .map(|w| read_f32(&w.bytes))
            .transpose()?;
        math::conv(
            &self.backend,
            x,
            w.buffer.as_ref(),
            w.shape[0],
            w.shape[2],
            dilation,
            bias.as_deref(),
            cancel,
        )
        .with_context(|| name.to_owned())
    }
    #[cfg(feature = "fused-conv")]
    fn conv_post(
        &self,
        x: &Signal,
        name: &str,
        dilation: usize,
        slope: Option<f32>,
        residual: Option<&Signal>,
        cancel: &AtomicBool,
    ) -> Result<Signal> {
        let w = self.weight(&format!("{name}.weight"))?;
        ensure!(
            w.shape.len() == 3 && w.shape[1] == x.c,
            "Melo Conv shape mismatch {name}"
        );
        let bias = self
            .weights
            .get(&format!("{name}.bias"))
            .map(|w| read_f32(&w.bytes))
            .transpose()?;
        math::conv_fused(
            &self.backend,
            x,
            w.buffer.as_ref(),
            w.shape[0],
            w.shape[2],
            dilation,
            bias.as_deref(),
            slope,
            residual,
            cancel,
        )
        .with_context(|| name.to_owned())
    }
    fn dense(&self, x: &Signal, name: &str) -> Result<Signal> {
        let w = self.weight(&format!("{name}.weight"))?;
        ensure!(
            w.shape.len() == 2 && w.shape[1] == x.c,
            "Melo Linear shape mismatch {name}"
        );
        let mut y = Signal {
            t: x.t,
            c: w.shape[0],
            v: math::linear(&self.backend, &x.v, x.t, x.c, w.buffer.as_ref(), w.shape[0])?,
        };
        if self.weights.contains_key(&format!("{name}.bias")) {
            y.bias(&self.values(&format!("{name}.bias"))?)?;
        }
        Ok(y)
    }
    fn norm(&self, x: &mut Signal, name: &str) -> Result<()> {
        math::norm(
            x,
            &self.values(&format!("{name}.gamma"))?,
            &self.values(&format!("{name}.beta"))?,
        )
    }
    fn embedding(&self, name: &str, ids: &[i32], channels: usize) -> Result<Signal> {
        let w = self.weight(name)?;
        ensure!(
            w.shape.len() == 2 && w.shape[1] == channels,
            "embedding shape mismatch {name}"
        );
        let values = read_f32(&w.bytes)?;
        let mut out = Signal::zeros(ids.len(), channels);
        for (i, &id) in ids.iter().enumerate() {
            ensure!(
                id >= 0 && (id as usize) < w.shape[0],
                "{name} ID {id} out of range"
            );
            out.v[i * channels..(i + 1) * channels]
                .copy_from_slice(&values[id as usize * channels..(id as usize + 1) * channels]);
        }
        Ok(out)
    }
    fn encoder(
        &self,
        mut x: Signal,
        name: &str,
        layers: usize,
        g: &Signal,
        cancel: &AtomicBool,
    ) -> Result<Signal> {
        let speaker = self.dense(g, &format!("{name}.spk_emb_linear"))?;
        for i in 0..layers {
            crate::check_cancel(cancel)?;
            if i == 2 {
                x.bias(&speaker.v)?;
            }
            let a = format!("{name}.attn_layers.{i}");
            let q = self.conv(&x, &format!("{a}.conv_q"), 1, cancel)?;
            let k = self.conv(&x, &format!("{a}.conv_k"), 1, cancel)?;
            let v = self.conv(&x, &format!("{a}.conv_v"), 1, cancel)?;
            let rk = self.weight(&format!("{a}.emb_rel_k"))?;
            ensure!(
                rk.shape.len() == 3 && rk.shape[2] == x.c / self.heads && rk.shape[1] % 2 == 1,
                "relative embedding dimensions {a}"
            );
            let y = math::attention(
                &self.backend,
                &q,
                &k,
                &v,
                self.heads,
                &read_f32(&rk.bytes)?,
                &self.values(&format!("{a}.emb_rel_v"))?,
                rk.shape[0],
                rk.shape[1] / 2,
                cancel,
            )?;
            let y = self.conv(&y, &format!("{a}.conv_o"), 1, cancel)?;
            x.add(&y)?;
            self.norm(&mut x, &format!("{name}.norm_layers_1.{i}"))?;
            let mut y = self.conv(&x, &format!("{name}.ffn_layers.{i}.conv_1"), 1, cancel)?;
            y.relu();
            let y = self.conv(&y, &format!("{name}.ffn_layers.{i}.conv_2"), 1, cancel)?;
            x.add(&y)?;
            self.norm(&mut x, &format!("{name}.norm_layers_2.{i}"))?;
        }
        Ok(x)
    }
    pub fn infer(
        &self,
        tokens: &FrontendOutput,
        options: &SynthesisOptions,
        max_samples: usize,
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        self.infer_traced(tokens, options, max_samples, cancel, &mut |_, _| Ok(()))
    }
    fn infer_traced(
        &self,
        tokens: &FrontendOutput,
        options: &SynthesisOptions,
        max_samples: usize,
        cancel: &AtomicBool,
        trace: &mut Trace<'_>,
    ) -> Result<Vec<f32>> {
        crate::check_cancel(cancel)?;
        let n = tokens.phones.len();
        crate::ensure_input(
            n > 0 && n <= 2048,
            "speech phone sequence must contain 1..2048 tokens; split the text into shorter clips",
        )?;
        ensure!(
            tokens.tones.len() == n && tokens.languages.len() == n,
            "Melo frontend output shape mismatch"
        );
        crate::ensure_input(
            options.speaker_id <= i32::MAX as usize,
            "speaker ID out of range",
        )?;
        crate::ensure_input(
            self.speakers.contains(&options.speaker_id),
            format!(
                "unsupported speaker ID {}; this voice package declares {:?}",
                options.speaker_id, self.speakers
            ),
        )?;
        let gin = self.config["model"]["gin_channels"]
            .as_u64()
            .context("missing gin_channels")? as usize;
        let g = self.embedding("emb_g.weight", &[options.speaker_id as i32], gin)?;
        let mut x = self.embedding("enc_p.emb.weight", &tokens.phones, self.hidden)?;
        x.add(&self.embedding("enc_p.tone_emb.weight", &tokens.tones, self.hidden)?)?;
        x.add(&self.embedding("enc_p.language_emb.weight", &tokens.languages, self.hidden)?)?;
        // A zero BERT input retains each Conv1d's learned bias.
        x.bias(&self.values("enc_p.bert_proj.bias")?)?;
        x.bias(&self.values("enc_p.ja_bert_proj.bias")?)?;
        for v in &mut x.v {
            *v *= (self.hidden as f32).sqrt();
        }
        trace("embedding", &x)?;
        x = self.encoder(x, "enc_p.encoder", self.layers, &g, cancel)?;
        trace("encoder", &x)?;
        let stats = self.conv(&x, "enc_p.proj", 1, cancel)?;
        trace("stats", &stats)?;
        ensure!(stats.c == 2 * self.inter, "text prior channels mismatch");
        let mut duration_input = x.clone();
        duration_input.bias(&self.conv(&g, "dp.cond", 1, cancel)?.v)?;
        let mut h = self.conv(&duration_input, "dp.conv_1", 1, cancel)?;
        h.relu();
        self.norm(&mut h, "dp.norm_1")?;
        h = self.conv(&h, "dp.conv_2", 1, cancel)?;
        h.relu();
        self.norm(&mut h, "dp.norm_2")?;
        let logw = self.conv(&h, "dp.proj", 1, cancel)?;
        trace("logw", &logw)?;
        ensure!(logw.c == 1, "duration predictor channel mismatch");
        let durations = durations(&logw.v, options.speed, max_samples / self.hop)?;
        let frames = durations.iter().sum::<usize>();
        ensure!(
            frames > 0 && frames <= max_samples / self.hop,
            "speech exceeds configured output duration"
        );
        let mut rng = math::Gaussian::new(options.seed);
        let mut z = Signal::zeros(frames, self.inter);
        let mut row = 0;
        for (i, &duration) in durations.iter().enumerate() {
            for _ in 0..duration {
                for c in 0..self.inter {
                    let m = stats.v[i * stats.c + c];
                    let logs = stats.v[i * stats.c + self.inter + c];
                    let noise = if options.noise_scale == 0. {
                        0.
                    } else {
                        rng.next() * logs.exp() * options.noise_scale
                    };
                    let value = m + noise;
                    ensure!(value.is_finite(), "nonfinite text prior");
                    z.v[row * self.inter + c] = value;
                }
                row += 1;
            }
        }
        trace("prior", &z)?;
        // Reverse [coupling, flip] pairs. Flip reverses the whole channel axis.
        for i in (0..4).rev() {
            crate::check_cancel(cancel)?;
            for row in z.v.chunks_exact_mut(z.c) {
                row.reverse();
            }
            let half = z.c / 2;
            let mut x0 = Signal::zeros(z.t, half);
            for t in 0..z.t {
                x0.v[t * half..(t + 1) * half].copy_from_slice(&z.v[t * z.c..t * z.c + half]);
            }
            let prefix = format!("flow.flows.{}", i * 2);
            let h = self.conv(&x0, &format!("{prefix}.pre"), 1, cancel)?;
            let h = self.encoder(h, &format!("{prefix}.enc"), self.flow_layers, &g, cancel)?;
            let means = self.conv(&h, &format!("{prefix}.post"), 1, cancel)?;
            ensure!(means.c == half, "flow must use mean_only=true");
            for t in 0..z.t {
                for c in 0..half {
                    z.v[t * z.c + half + c] -= means.v[t * half + c];
                }
            }
            trace(&format!("flow.{i}"), &z)?;
        }
        self.decode(z, &g, max_samples, cancel, trace)
    }
    fn decode(
        &self,
        z: Signal,
        g: &Signal,
        max_samples: usize,
        cancel: &AtomicBool,
        trace: &mut Trace<'_>,
    ) -> Result<Vec<f32>> {
        let mut x = self.conv(&z, "dec.conv_pre", 1, cancel)?;
        x.bias(&self.conv(g, "dec.cond", 1, cancel)?.v)?;
        trace("dec.pre", &x)?;
        for (i, &stride) in self.upsample.iter().enumerate() {
            x.leaky(0.1);
            let name = format!("dec.ups.{i}");
            let w = self.weight(&format!("{name}.weight"))?;
            ensure!(
                w.shape.len() == 3 && w.shape[0] == x.c,
                "transpose convolution shape mismatch {name}"
            );
            x = math::transpose_conv(
                &self.backend,
                &x,
                w.buffer.as_ref(),
                w.shape[1],
                w.shape[2],
                stride,
                &self.values(&format!("{name}.bias"))?,
                cancel,
            )?;
            ensure!(x.t <= max_samples, "speech exceeds output sample bound");
            let mut sum: Option<Signal> = None;
            for j in 0..self.kernels.len() {
                let mut h = x.clone();
                for (p, &dilation) in self.dilations[j].iter().enumerate() {
                    let prefix = format!("dec.resblocks.{}", i * self.kernels.len() + j);
                    #[cfg(feature = "fused-conv")]
                    {
                        let y = h.clone_leaky(0.1);
                        let y = self.conv_post(
                            &y,
                            &format!("{prefix}.convs1.{p}"),
                            dilation,
                            Some(0.1),
                            None,
                            cancel,
                        )?;
                        h = self.conv_post(
                            &y,
                            &format!("{prefix}.convs2.{p}"),
                            1,
                            None,
                            Some(&h),
                            cancel,
                        )?;
                    }
                    #[cfg(not(feature = "fused-conv"))]
                    {
                        let mut y = h.clone();
                        y.leaky(0.1);
                        y = self.conv(&y, &format!("{prefix}.convs1.{p}"), dilation, cancel)?;
                        y.leaky(0.1);
                        y = self.conv(&y, &format!("{prefix}.convs2.{p}"), 1, cancel)?;
                        h.add(&y)?;
                    }
                }
                if let Some(sum) = sum.as_mut() {
                    sum.add(&h)?;
                } else {
                    sum = Some(h);
                }
            }
            x = sum.context("empty decoder residual block")?;
            for v in &mut x.v {
                *v /= self.kernels.len() as f32;
            }
            trace(&format!("dec.{i}"), &x)?;
        }
        // Upstream uses default leaky_relu slope here, unlike the blocks.
        x.leaky(0.01);
        x = self.conv(&x, "dec.conv_post", 1, cancel)?;
        ensure!(
            x.c == 1 && x.t <= max_samples,
            "invalid decoder waveform shape"
        );
        for v in &mut x.v {
            ensure!(v.is_finite(), "nonfinite decoder waveform");
            *v = v.tanh();
        }
        trace("waveform", &x)?;
        crate::check_cancel(cancel)?;
        Ok(x.v)
    }
}

fn read_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(bytes.len() % 4 == 0, "invalid F32 tensor length");
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite model weight"
    );
    Ok(values)
}
fn durations(logw: &[f32], speed: f32, max_frames: usize) -> Result<Vec<usize>> {
    let mut remaining = max_frames;
    let mut out = Vec::with_capacity(logw.len());
    for &log in logw {
        let w = (log.exp() / speed).ceil();
        ensure!(w.is_finite() && w >= 0., "nonfinite duration prediction");
        crate::ensure_input(w<=remaining as f32,"speech exceeds configured output duration; shorten the input or increase the clip limit")?;
        let w = w as usize;
        remaining -= w;
        out.push(w);
    }
    if out.iter().all(|&n| n == 0) {
        bail!("empty predicted speech");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duration_ceil_and_budget_are_checked_before_allocation() {
        assert_eq!(
            durations(&[0., 2f32.ln(), 0.5f32.ln()], 1., 4).unwrap(),
            [1, 2, 1]
        );
        assert!(durations(&[100.], 1., 2000).is_err());
        let nonfinite = durations(&[f32::NAN], 1., 2000).unwrap_err();
        assert!(
            !nonfinite.is::<crate::InvalidInput>(),
            "numerical failure must remain internal"
        );
        let over_budget = durations(&[0., 0.], 0.25, 7).unwrap_err();
        assert!(
            over_budget.is::<crate::InvalidInput>(),
            "finite prediction exceeding the user clip budget is a client error"
        );
    }

    /// Explicit opt-in: original PyTorch inference fixture, never downloaded or
    /// executed by normal unit tests. The fixture uses zero noise for parity.
    #[test]
    #[ignore = "requires INFR_MELO_MODEL and INFR_MELO_REFERENCE"]
    fn real_chinese_reference() -> Result<()> {
        let path = std::env::var_os("INFR_MELO_MODEL").context("INFR_MELO_MODEL required")?;
        let fixture = std::path::PathBuf::from(
            std::env::var_os("INFR_MELO_REFERENCE").context("INFR_MELO_REFERENCE required")?,
        );
        let root = fixture.parent().context("fixture parent")?;
        let f: serde_json::Value = serde_json::from_slice(&std::fs::read(&fixture)?)?;
        let ids = |name: &str| -> Result<Vec<i32>> {
            f[name]
                .as_array()
                .context("fixture IDs")?
                .iter()
                .map(|v| v.as_i64().map(|v| v as i32).context("fixture ID"))
                .collect()
        };
        let tokens = FrontendOutput {
            phones: ids("phones")?,
            tones: ids("tones")?,
            languages: ids("languages")?,
        };
        let cancel = AtomicBool::new(false);
        let model = MeloModel::load(Path::new(&path), Arc::new(Config::default()), 2, &cancel)?;
        assert!(
            model
                .infer(
                    &tokens,
                    &SynthesisOptions {
                        speaker_id: 0,
                        ..Default::default()
                    },
                    44_100,
                    &cancel
                )
                .is_err(),
            "embedding-table capacity is not an available speaker declaration"
        );
        assert!(model
            .infer(
                &tokens,
                &SynthesisOptions::default(),
                44_100,
                &AtomicBool::new(true)
            )
            .is_err());
        assert!(
            model
                .infer(&tokens, &SynthesisOptions::default(), 1, &cancel)
                .is_err(),
            "duration limit must reject before latent/vocoder allocation"
        );
        let stages = f["stages"].as_array().context("fixture stages")?;
        let mut seen = 0;
        let mut reports = vec![];
        let output = std::env::var_os("INFR_MELO_NATIVE_OUTPUT").map(std::path::PathBuf::from);
        if let Some(dir) = &output {
            std::fs::create_dir_all(dir)?;
        }
        let _ = crate::math::take_conv_times();
        let mut last_trace = std::time::Instant::now();
        let mut trace = |name: &str, x: &Signal| -> Result<()> {
            let compute_ms = last_trace.elapsed().as_secs_f64() * 1000.;
            let (conv_packing_ms, conv_linear_ms) = crate::math::take_conv_times();
            let stage = stages
                .iter()
                .find(|s| s["name"].as_str() == Some(name))
                .with_context(|| format!("missing fixture stage {name}"))?;
            ensure!(
                stage["t"].as_u64() == Some(x.t as u64) && stage["c"].as_u64() == Some(x.c as u64),
                "{name} reference shape differs: native [{},{}], reference [{},{}]",
                x.t,
                x.c,
                stage["t"],
                stage["c"]
            );
            let expected = read_f32(&std::fs::read(
                root.join(stage["file"].as_str().context("stage file")?),
            )?)?;
            ensure!(expected.len() == x.v.len(), "{name} element count differs");
            let mut max_abs = 0f32;
            let mut squared = 0f64;
            let mut ref_squared = 0f64;
            let mut max_ref = 0f32;
            for (&a, &b) in x.v.iter().zip(&expected) {
                max_abs = max_abs.max((a - b).abs());
                max_ref = max_ref.max(b.abs());
                squared += (a as f64 - b as f64).powi(2);
                ref_squared += (b as f64).powi(2);
            }
            let relative_l2 = (squared / ref_squared.max(1e-20)).sqrt();
            let report = serde_json::json!({"name":name,"t":x.t,"c":x.c,"max_abs":max_abs,"relative_l2":relative_l2,"compute_ms_since_previous_stage":compute_ms,"conv_packing_ms":conv_packing_ms,"conv_linear_ms":conv_linear_ms});
            eprintln!("{report}");
            reports.push(report);
            seen += 1;
            if let Some(dir) = &output {
                std::fs::write(dir.join(format!("{name}.f32")), bytemuck::cast_slice(&x.v))?;
            }
            ensure!(
                relative_l2 < 0.003 && max_abs < 0.03 * max_ref.max(1.),
                "{name} reference tolerance failed: rel_l2={relative_l2}, max_abs={max_abs}"
            );
            last_trace = std::time::Instant::now();
            Ok(())
        };
        let waveform = model.infer_traced(
            &tokens,
            &SynthesisOptions {
                noise_scale: 0.,
                ..Default::default()
            },
            model.sample_rate as usize * 30,
            &cancel,
            &mut trace,
        )?;
        drop(trace);
        ensure!(seen == stages.len(), "not all reference stages compared");
        if let Some(dir) = &output {
            std::fs::write(
                dir.join("comparison.json"),
                serde_json::to_vec_pretty(&reports)?,
            )?;
        }
        ensure!(
            waveform.iter().any(|x| x.abs() > 0.001),
            "reference waveform is silent"
        );
        Ok(())
    }
}
