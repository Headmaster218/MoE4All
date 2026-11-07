//! Native deterministic Kaldi/FunASR fbank, LFR and CMVN (no inference runtime).
//!
//! Input is normalized mono PCM, as returned by `wav`; conversion to PCM16
//! magnitude happens here, exactly once. Defaults: 16 kHz, 400/160 sample
//! frames, symmetric Hamming, 512-point unnormalized FFT, power, 80 mel bins,
//! 20 Hz to Nyquist, natural log with f32 epsilon floor, no dither/energy/VTLN.
//! Complete frames follow Kaldi `snip_edges=true`. A nonempty utterance shorter
//! than one frame is explicitly extended by repeating its last sample; empty
//! input is an error. This short-input extension is outside Kaldi's contract.
//!
//! Mathematical references (read-only; implementation is self-contained):
//! https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.cc
//! https://github.com/kaldi-asr/kaldi/blob/master/src/feat/mel-computations.cc
//! https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-fbank.cc
//! https://github.com/modelscope/FunASR/blob/main/funasr/frontends/wav_frontend.py
//!
//! Reference content SHA-256, read 2026-11-01 (no generated window/mel table):
//! Kaldi feature-window.cc:
//! 250bce4313355b7f42904aa7ac8a0a5661bc9833f63b94c6394ef2d04ff944dc
//! Kaldi mel-computations.cc:
//! 189e5c7a50e29dc21744955098a964bd440360b3b33e09af950ec26ff747fb42
//! Kaldi feature-fbank.cc:
//! 550f3a47a3c62e36b18c16b86dfd3be8f0606011ee9e1933d7052e7459790a22
//! FunASR wav_frontend.py:
//! 39749ad998b1b432fcf7c30654c01e0547f215f98db8d25d3049ac05c6d5648b
//! SenseVoice.cpp log-mel-filter-bank.h (only three test coefficients quoted):
//! bad535fbcc4672a1a7b403597a4e757dd10fd79d9d33a6a8b0ce63cfb285e67c
//! Local evidence upstream-sense-voice-frontend.cc:
//! 01be972aa4332c9554bbcf0940831d166efdb45055788b3d01bf2fe2ee5dd0e5
//!
//! Kaldi reference copyrights: 2009-2012 Karel Vesely; 2009-2011 Phonexia
//! s.r.o., Petr Motlicek, Microsoft Corporation; 2013-2016 Johns Hopkins
//! University (Daniel Povey); 2014 IMSL, PKU-HKUST (Wei Shi). Those sources are
//! Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0 . Their mathematical
//! definitions are independently implemented here; no Kaldi code/table is copied.
//! FunASR reference copyright: Alibaba, Inc. and its affiliates (MIT).
//!
//! SenseVoice.cpp's `sense-voice-frontend.cc` was also consulted. Its periodic
//! Hamming (N denominator), rounded 1.19e-7 floor and packed RDFT treatment of
//! DC/Nyquist differ from Kaldi. We intentionally use Kaldi's N-1 denominator,
//! f32::EPSILON and separate complex bins. Thus this is not bit-compatible with
//! that C++ file. LFR copies edge frames safely even for fewer than four frames.
//!
//! Attribution for the consulted SenseVoice.cpp/LFR implementation:
//! MIT License
//! Copyright (c) 2024 lovemefan
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//! THE SOFTWARE.

use anyhow::{ensure, Context, Result};
use std::f64::consts::TAU;

/// Model-owned frontend parameters; CMVN operates on the stacked LFR columns.
/// Both CMVN arrays empty means identity. Otherwise both must have exactly
/// `num_mels * lfr_m` finite entries. `cmvn_add` is AddShift, not a mean to subtract.
#[derive(Clone, Debug)]
pub struct FrontendConfig {
    pub sample_rate: u32,
    pub num_mels: usize,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
    pub lfr_m: usize,
    pub lfr_n: usize,
    pub cmvn_add: Vec<f32>,
    pub cmvn_mul: Vec<f32>,
}

impl Default for FrontendConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            num_mels: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            lfr_m: 7,
            lfr_n: 6,
            cmvn_add: Vec::new(),
            cmvn_mul: Vec::new(),
        }
    }
}

impl FrontendConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.sample_rate == 16_000,
            "frontend requires 16000 Hz mono PCM"
        );
        ensure!(
            self.frame_length_ms.is_finite() && self.frame_length_ms > 0.0,
            "frame_length_ms must be finite and positive"
        );
        ensure!(
            self.frame_shift_ms.is_finite() && self.frame_shift_ms > 0.0,
            "frame_shift_ms must be finite and positive"
        );
        let (length, shift) = self.frame_sizes();
        ensure!(
            (2..=8192).contains(&length),
            "frame length must be 2..8192 samples"
        );
        ensure!(
            shift > 0 && shift <= length,
            "frame shift must be 1..frame_length samples"
        );
        ensure!(
            (3..=length.next_power_of_two() / 2).contains(&self.num_mels),
            "invalid number of mel bins"
        );
        ensure!(
            (1..=128).contains(&self.lfr_m) && self.lfr_n > 0,
            "LFR requires 1 <= m <= 128 and n > 0"
        );
        let cols = self
            .num_mels
            .checked_mul(self.lfr_m)
            .context("LFR width overflow")?;
        ensure!(cols <= 65_536, "LFR width exceeds 65536 columns");
        ensure!(
            (self.cmvn_add.is_empty() && self.cmvn_mul.is_empty())
                || (self.cmvn_add.len() == cols && self.cmvn_mul.len() == cols),
            "CMVN add/mul must both be empty or each have {cols} entries"
        );
        ensure!(
            self.cmvn_add
                .iter()
                .chain(&self.cmvn_mul)
                .all(|x| x.is_finite()),
            "CMVN contains nonfinite values"
        );
        Ok(())
    }

    fn frame_sizes(&self) -> (usize, usize) {
        // Kaldi truncates milliseconds * sample frequency, rather than rounding.
        (
            (self.sample_rate as f32 * 0.001 * self.frame_length_ms) as usize,
            (self.sample_rate as f32 * 0.001 * self.frame_shift_ms) as usize,
        )
    }
}

/// Contiguous row-major features, with no positional encoding applied.
#[derive(Clone, Debug)]
pub struct Features {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub fn extract_features(samples: &[f32], cfg: &FrontendConfig) -> Result<Features> {
    cfg.validate()?;
    ensure!(
        !samples.is_empty(),
        "cannot extract features from empty audio"
    );
    ensure!(
        samples.iter().all(|x| x.is_finite()),
        "audio contains nonfinite samples"
    );
    let (length, shift) = cfg.frame_sizes();
    let fft_size = length.next_power_of_two();
    let frame_count = if samples.len() < length {
        1
    } else {
        1 + (samples.len() - length) / shift
    };
    let mel = mel_weights(cfg.num_mels, fft_size, cfg.sample_rate)?;
    let window: Vec<f32> = (0..length)
        .map(|i| (0.54 - 0.46 * (TAU * i as f64 / (length - 1) as f64).cos()) as f32)
        .collect();
    let mut fbanks = float_buffer(
        frame_count
            .checked_mul(cfg.num_mels)
            .context("fbank size overflow")?,
    )?;
    let mut real = vec![0.0f64; fft_size];
    let mut imag = vec![0.0f64; fft_size];
    let mut frame = vec![0.0f32; length];
    let mut power = vec![0.0f32; fft_size / 2];
    for row in 0..frame_count {
        let start = row * shift;
        for (i, value) in frame.iter_mut().enumerate() {
            *value = samples[(start + i).min(samples.len() - 1)] * 32768.0;
            ensure!(value.is_finite(), "sample magnitude overflows PCM16 scale");
        }
        // Keep windowing and mel accumulation in f32, like Kaldi BaseFloat.
        let mean = (frame.iter().map(|&x| x as f64).sum::<f64>() / length as f64) as f32;
        for x in &mut frame {
            *x -= mean;
        }
        for i in (1..length).rev() {
            let previous = frame[i - 1];
            frame[i] -= 0.97 * previous;
        }
        frame[0] = frame[0] - 0.97 * frame[0];
        real.fill(0.0);
        imag.fill(0.0);
        for i in 0..length {
            real[i] = (frame[i] * window[i]) as f64;
        }
        fft(&mut real, &mut imag);
        for k in 0..power.len() {
            power[k] = (real[k] * real[k] + imag[k] * imag[k]) as f32;
        }
        for (bin, weights) in mel.iter().enumerate() {
            let energy = weights
                .values
                .iter()
                .enumerate()
                .fold(0.0f32, |sum, (k, &w)| sum + power[weights.start + k] * w);
            ensure!(
                energy.is_finite() && energy >= 0.0,
                "nonfinite or negative mel energy in frame {row}, bin {bin}"
            );
            let value = energy.max(f32::EPSILON).ln();
            ensure!(
                value.is_finite(),
                "nonfinite fbank energy in frame {row}, bin {bin}"
            );
            fbanks[row * cfg.num_mels + bin] = value;
        }
    }
    stack_and_normalize(&fbanks, frame_count, cfg)
}

fn float_buffer(size: usize) -> Result<Vec<f32>> {
    let mut data = Vec::new();
    data.try_reserve_exact(size)
        .context("unable to allocate audio features")?;
    data.resize(size, 0.0);
    Ok(data)
}

struct MelWeights {
    start: usize,
    values: Vec<f32>,
}

fn mel_scale(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_weights(num_mels: usize, fft_size: usize, sample_rate: u32) -> Result<Vec<MelWeights>> {
    let low = mel_scale(20.0);
    let delta = (mel_scale(sample_rate as f32 * 0.5) - low) / (num_mels + 1) as f32;
    let hz_per_bin = sample_rate as f32 / fft_size as f32;
    let mut filters = Vec::with_capacity(num_mels);
    for bin in 0..num_mels {
        let left = low + bin as f32 * delta;
        let center = low + (bin + 1) as f32 * delta;
        let right = low + (bin + 2) as f32 * delta;
        let mut start = None;
        let mut values = Vec::new();
        // Nyquist is outside the open upper support; do not double interior bins
        // or mix the packed real FFT's DC and Nyquist elements.
        for k in 0..fft_size / 2 {
            let mel = mel_scale(k as f32 * hz_per_bin);
            if mel > left && mel < right {
                start.get_or_insert(k);
                values.push(if mel <= center {
                    (mel - left) / (center - left)
                } else {
                    (right - mel) / (right - center)
                });
            }
        }
        filters.push(MelWeights {
            start: start
                .with_context(|| format!("mel bin {bin} has no FFT support; reduce num_mels"))?,
            values,
        });
    }
    Ok(filters)
}

fn stack_and_normalize(fbank: &[f32], frames: usize, cfg: &FrontendConfig) -> Result<Features> {
    let rows = frames / cfg.lfr_n + usize::from(!frames.is_multiple_of(cfg.lfr_n));
    let cols = cfg.num_mels * cfg.lfr_m;
    let mut data = float_buffer(rows.checked_mul(cols).context("LFR size overflow")?)?;
    let left_pad = (cfg.lfr_m - 1) / 2;
    for row in 0..rows {
        for stacked in 0..cfg.lfr_m {
            let source = (row * cfg.lfr_n)
                .saturating_add(stacked)
                .saturating_sub(left_pad)
                .min(frames - 1);
            for mel in 0..cfg.num_mels {
                let col = stacked * cfg.num_mels + mel;
                let mut value = fbank[source * cfg.num_mels + mel];
                if !cfg.cmvn_add.is_empty() {
                    value = (value + cfg.cmvn_add[col]) * cfg.cmvn_mul[col];
                }
                ensure!(
                    value.is_finite(),
                    "CMVN overflow at row {row}, column {col}"
                );
                data[row * cols + col] = value;
            }
        }
    }
    Ok(Features { data, rows, cols })
}

/// Iterative radix-2 complex FFT, forward sign and no normalization. Local scratch
/// only: no global tables, plans, locks or extra runtime dependencies.
fn fft(real: &mut [f64], imag: &mut [f64]) {
    let n = real.len();
    debug_assert!(n.is_power_of_two() && imag.len() == n);
    let mut reversed = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if i < reversed {
            real.swap(i, reversed);
            imag.swap(i, reversed);
        }
    }
    let mut width = 2;
    while width <= n {
        let (step_im, step_re) = (-TAU / width as f64).sin_cos();
        for base in (0..n).step_by(width) {
            let (mut wr, mut wi) = (1.0, 0.0);
            for j in 0..width / 2 {
                let a = base + j;
                let b = a + width / 2;
                let tr = wr * real[b] - wi * imag[b];
                let ti = wr * imag[b] + wi * real[b];
                real[b] = real[a] - tr;
                imag[b] = imag[a] - ti;
                real[a] += tr;
                imag[a] += ti;
                let next_wr = wr * step_re - wi * step_im;
                wi = wr * step_im + wi * step_re;
                wr = next_wr;
            }
        }
        width *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}, tolerance {tol}");
    }

    #[test]
    fn fft_matches_direct_complex_dft_including_dc_and_nyquist() {
        let input: Vec<f64> = (0..512)
            .map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0)
            .collect();
        let mut real = input.clone();
        let mut imag = vec![0.0; 512];
        fft(&mut real, &mut imag);
        for k in 0..512 {
            let (mut re, mut im) = (0.0, 0.0);
            for (i, &x) in input.iter().enumerate() {
                let (s, c) = (-TAU * (k * i) as f64 / 512.0).sin_cos();
                re += x * c;
                im += x * s;
            }
            close(real[k], re, 1e-9);
            close(imag[k], im, 1e-9);
        }
    }

    #[test]
    fn mel_weights_match_upstream_80_by_256_reference() {
        // SenseVoice.cpp/log-mel-filter-bank.h, first two sparse rows.
        let mel = mel_weights(80, 512, 16_000).unwrap();
        assert_eq!(mel[0].start, 1);
        assert_eq!(mel[0].values.len(), 2);
        close(mel[0].values[0] as f64, 0.50398138, 3e-6);
        close(mel[0].values[1] as f64, 0.13572533, 3e-6);
        assert_eq!(mel[1].start, 2);
        close(mel[1].values[0] as f64, 0.86427467, 3e-6);
    }

    #[test]
    fn fbank_matches_independent_direct_mathematical_reference() {
        let cfg = FrontendConfig {
            lfr_m: 1,
            lfr_n: 1,
            ..Default::default()
        };
        // Broad-spectrum exact PCM values avoid measuring f32 cancellation in
        // almost-empty bins of a pure tone instead of the mathematical formula.
        let samples: Vec<f32> = (0..400)
            .map(|i| ((i * 37 % 101 - 50) * 32 + 4096) as f32 / 32768.0)
            .collect();
        let actual = extract_features(&samples, &cfg).unwrap();
        // Independent slow DFT oracle, not a second call through FFT/window/mel helpers.
        let pcm: Vec<f64> = samples.iter().map(|&x| x as f64 * 32768.0).collect();
        let mean = pcm.iter().sum::<f64>() / 400.0;
        let windowed: Vec<f64> = (0..400)
            .map(|i| {
                let x = pcm[i] - mean;
                let prev = pcm[i.saturating_sub(1)] - mean;
                (x - 0.97 * prev) * (0.54 - 0.46 * (TAU * i as f64 / 399.0).cos())
            })
            .collect();
        let spectrum: Vec<f64> = (0..256)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for (i, &x) in windowed.iter().enumerate() {
                    let (s, c) = (TAU * (k * i) as f64 / 512.0).sin_cos();
                    re += x * c;
                    im += x * s;
                }
                re * re + im * im
            })
            .collect();
        let to_mel = |hz: f64| 1127.0 * (1.0 + hz / 700.0).ln();
        let low = to_mel(20.0);
        let delta = (to_mel(8000.0) - low) / 81.0;
        for j in 0..80 {
            let left = low + j as f64 * delta;
            let center = left + delta;
            let right = center + delta;
            let energy: f64 = spectrum
                .iter()
                .enumerate()
                .map(|(k, &p)| {
                    let m = to_mel(k as f64 * 31.25);
                    p * ((m - left) / (center - left))
                        .min((right - m) / (right - center))
                        .max(0.0)
                })
                .sum();
            close(
                actual.data[j] as f64,
                energy.max(f32::EPSILON as f64).ln(),
                2e-3,
            );
        }
    }

    #[test]
    fn silence_dc_floor_frame_count_and_short_audio() {
        let cfg = FrontendConfig::default();
        for length in [1, 399, 400, 559, 560, 1360] {
            let out = extract_features(&vec![0.25; length], &cfg).unwrap();
            let frames = if length < 400 {
                1
            } else {
                1 + (length - 400) / 160
            };
            assert_eq!(out.rows, frames.div_ceil(6));
            assert_eq!(out.cols, 560);
            assert!(out.data.iter().all(|&x| x == f32::EPSILON.ln()));
        }
        assert!(extract_features(&[], &cfg).is_err());
    }

    #[test]
    fn lfr_seven_six_repeats_both_edges_then_applies_column_cmvn() {
        let cfg = FrontendConfig {
            num_mels: 3,
            lfr_m: 7,
            lfr_n: 6,
            cmvn_add: (0..21).map(|i| i as f32).collect(),
            cmvn_mul: vec![2.0; 21],
            ..Default::default()
        };
        let raw: Vec<f32> = (0..8)
            .flat_map(|r| (0..3).map(move |c| (r * 10 + c) as f32))
            .collect();
        let out = stack_and_normalize(&raw, 8, &cfg).unwrap();
        assert_eq!((out.rows, out.cols), (2, 21));
        for (row, indices) in [[0, 0, 0, 0, 1, 2, 3], [3, 4, 5, 6, 7, 7, 7]]
            .iter()
            .enumerate()
        {
            for (j, &source) in indices.iter().enumerate() {
                for c in 0..3 {
                    let col = j * 3 + c;
                    assert_eq!(
                        out.data[row * 21 + col],
                        ((source * 10 + c) as f32 + col as f32) * 2.0
                    );
                }
            }
        }
        for frames in 1..=3 {
            assert_eq!(
                stack_and_normalize(&raw[..frames * 3], frames, &cfg)
                    .unwrap()
                    .data
                    .len(),
                21
            );
        }
    }

    #[test]
    fn rejects_bad_parameters_nonfinite_audio_and_cmvn() {
        let mut cfg = FrontendConfig {
            cmvn_add: vec![0.0; 80],
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
        cfg.cmvn_add.clear();
        cfg.frame_shift_ms = f32::NAN;
        assert!(cfg.validate().is_err());
        cfg.frame_shift_ms = 10.0;
        cfg.lfr_n = 0;
        assert!(cfg.validate().is_err());
        cfg.lfr_n = 6;
        assert!(extract_features(&[f32::NAN; 400], &cfg).is_err());
        cfg.sample_rate = 44_100;
        assert!(cfg.validate().is_err());
    }
}
