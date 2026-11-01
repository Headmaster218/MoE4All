//! Streaming WAV decoder and bounded 16 kHz mono segment delivery.
//!
//! No codec/runtime dependencies. RIFF/WAVE PCM8/16/24/32, IEEE float32/64 and
//! WAVE_FORMAT_EXTENSIBLE PCM/float are accepted. Big-endian RIFX, RF64, encoded
//! WAV, multiple data chunks and malformed/truncated input return explicit errors.
//! There is no application duration limit (RIFF itself has 32-bit chunk sizes).
//! Unknown chunks are skipped by seeking; their odd-byte padding is validated.
//!
//! Decoder output is normalized PCM (PCM16 / 32768). Downmix is the arithmetic
//! mean of all channels. Resampling uses a DC-normalized, Hann-windowed sinc
//! with a reduced cutoff when downsampling, constant edge extension and integer
//! rational phase. Its phase/history persist across segment boundaries. No
//! feature-alignment or model state is implied by these hard audio segments.

use anyhow::{ensure, Context, Result};
use std::collections::VecDeque;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub const OUTPUT_SAMPLE_RATE: u32 = 16_000;
pub const MAX_SEGMENT_SECONDS: f32 = 28.0;
const READ_BUFFER_BYTES: usize = 16_384;
const MAX_INPUT_RATE: u32 = 384_000;
const MAX_CHANNELS: u16 = 64;
const SINC_LOBES: usize = 16;

#[derive(Debug)]
pub struct AudioSegment {
    pub samples: Vec<f32>,
    /// Original stream position, in output-sample units / 16000. No accumulated
    /// floating point duration drift, even when the last segment is very short.
    pub start_seconds: f64,
    pub sample_rate: u32,
}

/// Cancellation is a distinguishable error, never a successful partial decode.
#[derive(Debug)]
pub struct AudioCancelled;

impl fmt::Display for AudioCancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("audio decoding cancelled")
    }
}
impl std::error::Error for AudioCancelled {}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        return Err(AudioCancelled.into());
    }
    Ok(())
}

/// Future codecs can implement this small pull interface without changing the
/// resampler, cancellation or segmenter. Samples must be finite normalized mono
/// PCM; `None` means clean EOF, whereas truncated packets must return an error.
pub trait AudioDecoder {
    fn sample_rate(&self) -> u32;
    fn next_mono_sample(&mut self) -> Result<Option<f32>>;
}

/// Visit bounded segments synchronously. At most floor(min(max_seconds, 28) *
/// 16000) samples per callback; every nonempty tail is delivered, even one sample.
/// `max_seconds` must be finite, positive and at least one output sample long.
/// Callback errors stop decoding immediately. Already visited segments cannot be
/// rolled back if a subsequent read fails; consumers must propagate the result.
/// For segments of at least four seconds a quiet 20 ms window is sought within
/// the final two seconds (or final quarter, whichever is shorter). A selected
/// cut is near a zero crossing; the entire remaining tail becomes the start of
/// the next segment. Decoder-owned audio storage stays below two segments.
pub fn for_each_wav_segment(
    path: &Path,
    max_seconds: f32,
    cancel: &AtomicBool,
    visit: &mut dyn FnMut(AudioSegment) -> Result<()>,
) -> Result<()> {
    segment_capacity(max_seconds)?;
    check_cancel(cancel)?;
    let file = File::open(path).with_context(|| format!("open WAV {}", path.display()))?;
    let decoder =
        WavDecoder::new(file, cancel).with_context(|| format!("parse WAV {}", path.display()))?;
    for_each_decoder_segment(decoder, max_seconds, cancel, visit)
}

/// Codec-independent entry point. The decoder and resampler are consumed once
/// per complete stream, rather than being restarted for each model segment.
pub fn for_each_decoder_segment<D: AudioDecoder>(
    decoder: D,
    max_seconds: f32,
    cancel: &AtomicBool,
    visit: &mut dyn FnMut(AudioSegment) -> Result<()>,
) -> Result<()> {
    let capacity = segment_capacity(max_seconds)?;
    check_cancel(cancel)?;
    let mut stream = Resampler::new(decoder)?;
    let mut start = 0u64;
    let mut pending = Vec::new();
    let mut eof = false;
    loop {
        check_cancel(cancel)?;
        pending
            .try_reserve_exact(capacity - pending.len())
            .context("allocate bounded audio segment")?;
        while pending.len() < capacity && !eof {
            // Poll before each output, including the first. One resampler step
            // reads at most a bounded filter window (<= 771 input frames).
            check_cancel(cancel)?;
            match stream.next_sample()? {
                Some(value) => pending.push(value),
                None => {
                    eof = true;
                    break;
                }
            }
        }
        check_cancel(cancel)?;
        if pending.is_empty() {
            return Ok(());
        }
        let count = if eof {
            pending.len()
        } else {
            quiet_cut(&pending)
        };
        let mut tail = Vec::new();
        tail.try_reserve_exact(pending.len() - count)
            .context("allocate bounded audio tail")?;
        tail.extend_from_slice(&pending[count..]);
        pending.truncate(count);
        let samples = std::mem::replace(&mut pending, tail);
        check_cancel(cancel)?;
        visit(AudioSegment {
            samples,
            start_seconds: start as f64 / OUTPUT_SAMPLE_RATE as f64,
            sample_rate: OUTPUT_SAMPLE_RATE,
        })?;
        start = start
            .checked_add(count as u64)
            .context("audio stream position overflow")?;
        check_cancel(cancel)?;
        if eof {
            return Ok(());
        }
    }
}

fn quiet_cut(samples: &[f32]) -> usize {
    let rate = OUTPUT_SAMPLE_RATE as usize;
    if samples.len() < 4 * rate {
        return samples.len();
    }
    let window = rate / 50; // 20 ms power windows, 10 ms search hop.
    let hop = window / 2;
    let search = (2 * rate).min(samples.len() / 4);
    let mean_power =
        samples.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / samples.len() as f64;
    // Fully silent audio can use the full hard segment; do not create a small
    // extra tail just because every candidate window has the same zero energy.
    if mean_power <= 1e-10 {
        return samples.len();
    }
    let threshold = (mean_power * 0.1).min(1e-4); // <= -40 dBFS, and 10 dB below segment mean.
    let mut best_power = threshold;
    let mut best = None;
    for start in (samples.len() - search..=samples.len() - window).step_by(hop) {
        let power = samples[start..start + window]
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            / window as f64;
        if power <= best_power {
            best_power = power;
            best = Some(start + window / 2);
        }
    }
    let Some(center) = best else {
        return samples.len();
    };
    // Keep the cut inside the quiet window and prefer low-amplitude crossings.
    let mut cut = center;
    for at in center - hop / 2..=center + hop / 2 {
        if samples[at].abs() < samples[cut].abs() {
            cut = at;
        }
    }
    cut
}

fn segment_capacity(seconds: f32) -> Result<usize> {
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "max_seconds must be finite and positive"
    );
    let samples =
        (seconds.min(MAX_SEGMENT_SECONDS) as f64 * OUTPUT_SAMPLE_RATE as f64).floor() as usize;
    ensure!(
        samples > 0,
        "max_seconds must contain at least one 16000 Hz sample"
    );
    Ok(samples)
}

#[derive(Clone, Copy)]
struct WaveFormat {
    tag: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
    block_align: usize,
}

/// A fixed byte buffer holds complete interleaved frames; no allocation depends
/// on the WAV data size, chunk count, metadata size or total audio duration.
pub struct WavDecoder<R: Read + Seek> {
    reader: R,
    format: WaveFormat,
    remaining_bytes: u64,
    buffer: [u8; READ_BUFFER_BYTES],
    cursor: usize,
    buffered: usize,
}

impl<R: Read + Seek> WavDecoder<R> {
    /// Validate RIFF structure/metadata by seeking, then position at data. Data
    /// need not follow fmt immediately; data-before-fmt is also supported.
    pub fn new(mut reader: R, cancel: &AtomicBool) -> Result<Self> {
        check_cancel(cancel)?;
        let file_length = reader.seek(SeekFrom::End(0)).context("seek WAV length")?;
        reader.seek(SeekFrom::Start(0))?;
        let mut header = [0u8; 12];
        reader
            .read_exact(&mut header)
            .context("truncated RIFF/WAVE header")?;
        ensure!(
            &header[..4] == b"RIFF",
            "expected little-endian RIFF WAV (RIFX/RF64 unsupported)"
        );
        ensure!(&header[8..] == b"WAVE", "RIFF type is not WAVE");
        let riff_end = u32_at(&header, 4) as u64 + 8;
        ensure!(riff_end >= 12, "invalid RIFF size");
        ensure!(riff_end == file_length,
            "RIFF declared length {riff_end} differs from file length {file_length} (truncated or trailing data)");
        let mut position = 12u64;
        let mut format = None;
        let mut data = None;
        while position < riff_end {
            check_cancel(cancel)?;
            ensure!(
                riff_end - position >= 8,
                "incomplete WAV chunk header at {position}"
            );
            reader.seek(SeekFrom::Start(position))?;
            let mut chunk = [0u8; 8];
            reader
                .read_exact(&mut chunk)
                .context("truncated WAV chunk header")?;
            let length = u32_at(&chunk, 4) as u64;
            let payload = position + 8;
            let end = payload
                .checked_add(length)
                .context("WAV chunk extent overflow")?;
            let padded_end = end
                .checked_add(length & 1)
                .context("WAV chunk padding overflow")?;
            ensure!(
                padded_end <= riff_end,
                "WAV chunk at {position} exceeds RIFF bounds or is missing odd-byte padding"
            );
            match &chunk[..4] {
                b"fmt " => {
                    ensure!(format.is_none(), "duplicate WAV fmt chunk");
                    ensure!(length >= 16, "WAV fmt chunk is shorter than 16 bytes");
                    ensure!(length != 17, "incomplete WAV fmt extension length");
                    let mut bytes = [0u8; 40];
                    let size = length.min(bytes.len() as u64) as usize;
                    reader
                        .read_exact(&mut bytes[..size])
                        .context("truncated WAV fmt payload")?;
                    format = Some(parse_format(&bytes, length)?);
                }
                b"data" => {
                    ensure!(data.is_none(), "multiple WAV data chunks are unsupported");
                    data = Some((payload, length));
                }
                _ => {} // Checked extent; skip metadata, without allocating its payload.
            }
            // Reading padding catches a concurrently truncated file too. Padding
            // contents are unspecified and are not interpreted as PCM bytes.
            if length & 1 != 0 {
                reader.seek(SeekFrom::Start(end))?;
                let mut padding = [0u8; 1];
                reader
                    .read_exact(&mut padding)
                    .context("missing WAV chunk padding byte")?;
            }
            position = padded_end;
        }
        let format = format.context("WAV has no fmt chunk")?;
        let (offset, length) = data.context("WAV has no data chunk")?;
        ensure!(length > 0, "WAV data is empty");
        ensure!(
            length % format.block_align as u64 == 0,
            "WAV data ends with an incomplete interleaved sample frame"
        );
        reader.seek(SeekFrom::Start(offset))?;
        check_cancel(cancel)?;
        Ok(Self {
            reader,
            format,
            remaining_bytes: length,
            buffer: [0; READ_BUFFER_BYTES],
            cursor: 0,
            buffered: 0,
        })
    }
}

impl<R: Read + Seek> AudioDecoder for WavDecoder<R> {
    fn sample_rate(&self) -> u32 {
        self.format.sample_rate
    }

    fn next_mono_sample(&mut self) -> Result<Option<f32>> {
        if self.cursor == self.buffered {
            if self.remaining_bytes == 0 {
                return Ok(None);
            }
            let frames = READ_BUFFER_BYTES / self.format.block_align;
            let size = self
                .remaining_bytes
                .min((frames * self.format.block_align) as u64) as usize;
            self.reader
                .read_exact(&mut self.buffer[..size])
                .context("truncated WAV audio payload")?;
            self.remaining_bytes -= size as u64;
            self.cursor = 0;
            self.buffered = size;
        }
        let sample_bytes = (self.format.bits / 8) as usize;
        let mut sum = 0.0f64;
        for channel in 0..self.format.channels as usize {
            let offset = self.cursor + channel * sample_bytes;
            let bytes = &self.buffer[offset..offset + sample_bytes];
            let value = match (self.format.tag, self.format.bits) {
                (1, 8) => (bytes[0] as f64 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([bytes[0], bytes[1]]) as f64 / 32768.0,
                (1, 24) => {
                    let value =
                        ((bytes[0] as i32) | ((bytes[1] as i32) << 8) | ((bytes[2] as i32) << 16))
                            << 8
                            >> 8;
                    value as f64 / 8_388_608.0
                }
                (1, 32) => {
                    i32::from_le_bytes(bytes.try_into().expect("validated 32-bit sample")) as f64
                        / 2_147_483_648.0
                }
                (3, 32) => {
                    f32::from_le_bytes(bytes.try_into().expect("validated 32-bit sample")) as f64
                }
                (3, 64) => f64::from_le_bytes(bytes.try_into().expect("validated 64-bit sample")),
                _ => unreachable!("format was validated"),
            };
            ensure!(
                value.is_finite(),
                "WAV contains a nonfinite floating point sample"
            );
            sum += value;
        }
        self.cursor += self.format.block_align;
        let value = (sum / self.format.channels as f64) as f32;
        ensure!(
            value.is_finite(),
            "WAV sample cannot be represented as finite f32"
        );
        Ok(Some(value))
    }
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn parse_format(bytes: &[u8; 40], length: u64) -> Result<WaveFormat> {
    let mut tag = u16_at(bytes, 0);
    let channels = u16_at(bytes, 2);
    let sample_rate = u32_at(bytes, 4);
    let byte_rate = u32_at(bytes, 8);
    let block_align = u16_at(bytes, 12) as usize;
    let bits = u16_at(bytes, 14);
    ensure!(
        (1..=MAX_CHANNELS).contains(&channels),
        "WAV channel count must be 1..{MAX_CHANNELS}"
    );
    ensure!(
        (1..=MAX_INPUT_RATE).contains(&sample_rate),
        "WAV sample rate must be 1..{MAX_INPUT_RATE} Hz"
    );
    if length >= 18 {
        ensure!(
            18 + u16_at(bytes, 16) as u64 <= length,
            "truncated WAV fmt extension"
        );
    }
    if tag == 0xfffe {
        ensure!(
            length >= 40 && u16_at(bytes, 16) >= 22,
            "WAVE_FORMAT_EXTENSIBLE requires 22 extension bytes"
        );
        // GUID: 00000001/00000003-0000-0010-8000-00aa00389b71.
        ensure!(
            bytes[26..40] == [0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113],
            "unsupported WAVE_FORMAT_EXTENSIBLE subformat GUID"
        );
        tag = u16_at(bytes, 24);
        let valid_bits = u16_at(bytes, 18);
        ensure!(
            valid_bits > 0 && valid_bits <= bits,
            "invalid WAV valid-bits count"
        );
        if tag == 3 {
            ensure!(
                valid_bits == bits,
                "float WAV valid bits must equal container bits"
            );
        }
        // PCM valid bits are left-aligned, so division by the container scale
        // also handles e.g. 24 valid bits in a 32-bit container.
        let mask = u32_at(bytes, 20);
        ensure!(
            mask == 0 || mask.count_ones() == channels as u32,
            "WAV channel mask does not match channel count"
        );
    }
    ensure!(
        matches!((tag, bits), (1, 8 | 16 | 24 | 32) | (3, 32 | 64)),
        "unsupported WAV format tag {tag}, bits {bits}; expected PCM8/16/24/32 or float32/64"
    );
    ensure!(
        block_align == channels as usize * (bits / 8) as usize,
        "invalid WAV block_align"
    );
    ensure!(
        byte_rate as u64 == sample_rate as u64 * block_align as u64,
        "invalid WAV byte_rate"
    );
    Ok(WaveFormat {
        tag,
        channels,
        sample_rate,
        bits,
        block_align,
    })
}

/// The ring is at most 2*radius+2 mono samples. At 384 kHz radius is 384,
/// so decoded history never grows with duration. The one-frame 16 kHz fast
/// path is exact and introduces neither filtering nor a resampling delay.
struct Resampler<D: AudioDecoder> {
    decoder: D,
    rate: u32,
    radius: usize,
    cutoff: f64,
    ring: VecDeque<f32>,
    ring_start: u64,
    loaded: u64,
    eof: bool,
    position: u64,
    phase: u32,
}

impl<D: AudioDecoder> Resampler<D> {
    fn new(decoder: D) -> Result<Self> {
        let rate = decoder.sample_rate();
        ensure!(
            (1..=MAX_INPUT_RATE).contains(&rate),
            "decoder sample rate must be 1..{MAX_INPUT_RATE} Hz"
        );
        let ratio = (rate as f64 / OUTPUT_SAMPLE_RATE as f64).max(1.0);
        let radius = (SINC_LOBES as f64 * ratio).ceil() as usize;
        Ok(Self {
            decoder,
            rate,
            radius,
            cutoff: 0.94 / ratio,
            ring: VecDeque::with_capacity(2 * radius + 2),
            ring_start: 0,
            loaded: 0,
            eof: false,
            position: 0,
            phase: 0,
        })
    }

    fn next_sample(&mut self) -> Result<Option<f32>> {
        if self.rate == OUTPUT_SAMPLE_RATE {
            let value = self.decoder.next_mono_sample()?;
            if let Some(x) = value {
                ensure!(x.is_finite(), "decoder returned nonfinite PCM");
            }
            return Ok(value);
        }
        let keep_from = self.position.saturating_sub(self.radius as u64);
        while self.ring_start < keep_from && !self.ring.is_empty() {
            self.ring.pop_front();
            self.ring_start += 1;
        }
        let needed = self
            .position
            .checked_add(self.radius as u64)
            .context("resampler position overflow")?;
        while !self.eof && self.loaded <= needed {
            match self.decoder.next_mono_sample()? {
                Some(value) => {
                    ensure!(value.is_finite(), "decoder returned nonfinite PCM");
                    if self.loaded >= keep_from {
                        self.ring.push_back(value);
                    } else {
                        self.ring_start = self.loaded + 1;
                    }
                    self.loaded = self
                        .loaded
                        .checked_add(1)
                        .context("input frame count overflow")?;
                }
                None => self.eof = true,
            }
        }
        if self.eof && self.position >= self.loaded {
            return Ok(None);
        }
        let fraction = self.phase as f64 / OUTPUT_SAMPLE_RATE as f64;
        let (mut sum, mut weights) = (0.0f64, 0.0f64);
        for delta in -(self.radius as i64)..=self.radius as i64 {
            let x = delta as f64 - fraction;
            if x.abs() >= self.radius as f64 {
                continue;
            }
            let argument = std::f64::consts::PI * self.cutoff * x;
            let sinc = if argument.abs() < 1e-12 {
                1.0
            } else {
                argument.sin() / argument
            };
            let weight = self.cutoff
                * sinc
                * 0.5
                * (1.0 + (std::f64::consts::PI * x / self.radius as f64).cos());
            let index = if delta < 0 {
                self.position.saturating_sub((-delta) as u64)
            } else {
                self.position
                    .checked_add(delta as u64)
                    .context("filter position overflow")?
            };
            let index = index.min(self.loaded - 1);
            let sample = self.ring[(index - self.ring_start) as usize];
            sum += sample as f64 * weight;
            weights += weight;
        }
        let output = (sum / weights) as f32;
        ensure!(output.is_finite(), "resampler produced nonfinite PCM");
        let advance = self.phase as u64 + self.rate as u64;
        self.position = self
            .position
            .checked_add(advance / OUTPUT_SAMPLE_RATE as u64)
            .context("resampler position overflow")?;
        self.phase = (advance % OUTPUT_SAMPLE_RATE as u64) as u32;
        debug_assert!(self.ring.len() <= 2 * self.radius + 2);
        Ok(Some(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::bail;
    use std::io::Cursor;
    use std::sync::Arc;

    fn chunk(id: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() & 1 != 0 {
            out.push(0);
        }
        out
    }

    fn format(tag: u16, bits: u16, channels: u16, rate: u32) -> Vec<u8> {
        let align = channels * (bits / 8);
        let mut out = Vec::new();
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * align as u32).to_le_bytes());
        out.extend_from_slice(&align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out
    }

    fn riff(chunks: &[Vec<u8>]) -> Vec<u8> {
        let mut out = b"RIFF".to_vec();
        let size = 4 + chunks.iter().map(Vec::len).sum::<usize>();
        out.extend_from_slice(&(size as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        for c in chunks {
            out.extend_from_slice(c);
        }
        out
    }

    fn decode(bytes: Vec<u8>, seconds: f32) -> Result<Vec<AudioSegment>> {
        let cancel = AtomicBool::new(false);
        let decoder = WavDecoder::new(Cursor::new(bytes), &cancel)?;
        let mut output = Vec::new();
        for_each_decoder_segment(decoder, seconds, &cancel, &mut |s| {
            output.push(s);
            Ok(())
        })?;
        Ok(output)
    }

    #[test]
    fn pcm_formats_float_and_stereo_downmix_have_expected_scale() {
        for (tag, bits, data) in [
            (1, 8, vec![0, 128, 128, 255]),
            (
                1,
                16,
                [-32768i16, 0, 0, 32767]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            ),
            (1, 24, vec![0, 0, 128, 0, 0, 0, 0, 0, 0, 255, 255, 127]),
            (
                1,
                32,
                [i32::MIN, 0, 0, i32::MAX]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            ),
            (
                3,
                32,
                [-1.0f32, 0.0, 0.0, 0.5]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            ),
            (
                3,
                64,
                [-1.0f64, 0.0, 0.0, 0.5]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
            ),
        ] {
            let bytes = riff(&[
                chunk(b"fmt ", &format(tag, bits, 2, 16_000)),
                chunk(b"data", &data),
            ]);
            let segments = decode(bytes, 1.0).unwrap();
            assert_eq!(segments.len(), 1);
            let values = &segments[0].samples;
            assert_eq!(values.len(), 2);
            assert_eq!(values[0], -0.5);
            let expected = if tag == 3 {
                0.25
            } else {
                0.5 - 0.5 / (2.0f32).powi(bits as i32 - 1)
            };
            assert_eq!(values[1], expected);
        }
    }

    #[test]
    fn odd_metadata_and_odd_data_padding_are_not_audio() {
        let bytes = riff(&[
            chunk(b"JUNK", &[9, 8, 7]),
            chunk(b"data", &[128, 129, 130]),
            chunk(b"fmt ", &format(1, 8, 1, 16_000)),
            chunk(b"LIST", &[1]),
        ]);
        let segments = decode(bytes, 1.0).unwrap();
        assert_eq!(segments[0].samples, vec![0.0, 1.0 / 128.0, 2.0 / 128.0]);
    }

    #[test]
    fn pcm24_three_channels_cross_fixed_byte_buffer_without_losing_alignment() {
        let mut pcm = Vec::new();
        for frame in 0..2001i32 {
            for _ in 0..3 {
                pcm.extend_from_slice(&frame.to_le_bytes()[..3]);
            }
        }
        let bytes = riff(&[
            chunk(b"fmt ", &format(1, 24, 3, 16_000)),
            chunk(b"data", &pcm),
        ]);
        let decoded = decode(bytes, 1.0).unwrap();
        assert_eq!(decoded[0].samples.len(), 2001);
        for (i, &sample) in decoded[0].samples.iter().enumerate() {
            assert_eq!(sample, i as f32 / 8_388_608.0);
        }
    }

    #[test]
    fn extensible_24_valid_bits_in_32_bit_container() {
        let mut fmt = format(0xfffe, 32, 1, 16_000);
        fmt.extend_from_slice(&22u16.to_le_bytes());
        fmt.extend_from_slice(&24u16.to_le_bytes());
        fmt.extend_from_slice(&4u32.to_le_bytes());
        fmt.extend_from_slice(&[1, 0, 0, 0, 0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]);
        let bytes = riff(&[
            chunk(b"fmt ", &fmt),
            chunk(b"data", &0x40000000i32.to_le_bytes()),
        ]);
        assert_eq!(decode(bytes, 1.0).unwrap()[0].samples, vec![0.5]);
        fmt[26] = 1;
        assert!(decode(riff(&[chunk(b"fmt ", &fmt), chunk(b"data", &[0; 4])]), 1.0).is_err());
    }

    struct Synthetic {
        rate: u32,
        count: u64,
        index: u64,
        hz: f64,
    }
    impl AudioDecoder for Synthetic {
        fn sample_rate(&self) -> u32 {
            self.rate
        }
        fn next_mono_sample(&mut self) -> Result<Option<f32>> {
            if self.index == self.count {
                return Ok(None);
            }
            let x = if self.hz == 0.0 {
                0.25
            } else {
                (std::f64::consts::TAU * self.hz * self.index as f64 / self.rate as f64).sin()
                    as f32
            };
            self.index += 1;
            Ok(Some(x))
        }
    }

    fn synthetic(rate: u32, count: u64, hz: f64) -> Synthetic {
        Synthetic {
            rate,
            count,
            index: 0,
            hz,
        }
    }

    #[test]
    fn resampling_has_exact_rational_length_and_preserves_dc_including_tails() {
        for rate in [
            8000, 11025, 16000, 22050, 44100, 48000, 96000, 192000, 384000,
        ] {
            for count in [1, 7, 137] {
                let mut resampler = Resampler::new(synthetic(rate, count, 0.0)).unwrap();
                let mut emitted = 0;
                while let Some(x) = resampler.next_sample().unwrap() {
                    assert!((x - 0.25).abs() < 1e-6);
                    emitted += 1;
                }
                assert_eq!(emitted, (count * 16_000 + rate as u64 - 1) / rate as u64);
            }
        }
    }

    #[test]
    fn segment_boundaries_preserve_samples_phase_timestamps_and_one_sample_tail() {
        let cancel = AtomicBool::new(false);
        for rate in [8000, 16000, 44100, 48000] {
            let count = rate as u64 / 5 + 1;
            let mut whole = Vec::new();
            for_each_decoder_segment(synthetic(rate, count, 731.0), 28.0, &cancel, &mut |s| {
                whole.extend(s.samples);
                Ok(())
            })
            .unwrap();
            let mut parts = Vec::new();
            for_each_decoder_segment(synthetic(rate, count, 731.0), 0.031, &cancel, &mut |s| {
                parts.push(s);
                Ok(())
            })
            .unwrap();
            let mut position = 0;
            for s in &parts {
                assert_eq!(s.start_seconds, position as f64 / 16_000.0);
                assert_eq!(s.sample_rate, 16_000);
                assert!(
                    !s.samples.is_empty() && s.samples.len() <= segment_capacity(0.031).unwrap()
                );
                position += s.samples.len();
            }
            assert_eq!(
                whole,
                parts
                    .into_iter()
                    .flat_map(|s| s.samples)
                    .collect::<Vec<_>>()
            );
        }
        let mut sizes = Vec::new();
        for_each_decoder_segment(synthetic(16_000, 448_001, 0.0), 100.0, &cancel, &mut |s| {
            sizes.push(s.samples.len());
            Ok(())
        })
        .unwrap();
        assert_eq!(sizes, vec![448_000, 1]);
    }

    #[test]
    fn downsampling_rejects_above_nyquist_tone() {
        for (hz, minimum, maximum) in [(1000.0, 0.65, 0.75), (12_000.0, 0.0, 0.01)] {
            let mut r = Resampler::new(synthetic(48_000, 4800, hz)).unwrap();
            let mut values = Vec::new();
            while let Some(x) = r.next_sample().unwrap() {
                values.push(x);
            }
            let interior = &values[100..values.len() - 100];
            let rms = (interior.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()
                / interior.len() as f64)
                .sqrt();
            assert!(rms >= minimum && rms <= maximum, "{hz} Hz RMS {rms}");
        }
    }

    #[test]
    fn long_synthetic_stream_uses_constant_history_capacity() {
        // Declare 24 hours lazily, inspect a bounded prefix. Allocation cannot
        // depend on total duration; ring wrap/phase are exercised for 10 seconds.
        let mut r = Resampler::new(synthetic(16_001, 16_001 * 86_400, 0.0)).unwrap();
        let initial = r.ring.capacity();
        for _ in 0..160_000 {
            assert!(r.next_sample().unwrap().is_some());
            assert!(r.ring.len() <= 2 * r.radius + 2);
            assert_eq!(r.ring.capacity(), initial);
        }
        assert!(r.decoder.index < r.decoder.count);
        assert_eq!(READ_BUFFER_BYTES, 16_384);
    }

    struct MemoryAudio {
        samples: Vec<f32>,
        cursor: usize,
    }
    impl AudioDecoder for MemoryAudio {
        fn sample_rate(&self) -> u32 {
            16_000
        }
        fn next_mono_sample(&mut self) -> Result<Option<f32>> {
            let value = self.samples.get(self.cursor).copied();
            self.cursor += usize::from(value.is_some());
            Ok(value)
        }
    }

    #[test]
    fn quiet_cut_retains_exact_tail_and_never_swallows_sub_frame_remainder() {
        let mut samples = vec![0.25; 5 * 16_000 + 7];
        samples[55_000..58_000].fill(0.0);
        let original = samples.clone();
        let cancel = AtomicBool::new(false);
        let mut decoded = Vec::new();
        let mut lengths = Vec::new();
        let mut next_start = 0;
        for_each_decoder_segment(
            MemoryAudio { samples, cursor: 0 },
            4.0,
            &cancel,
            &mut |segment| {
                assert_eq!(segment.start_seconds, next_start as f64 / 16_000.0);
                next_start += segment.samples.len();
                lengths.push(segment.samples.len());
                decoded.extend(segment.samples);
                Ok(())
            },
        )
        .unwrap();
        assert!((55_000..58_000).contains(&lengths[0]));
        assert_eq!(decoded, original);
        assert_eq!(quiet_cut(&vec![0.0; 64_000]), 64_000);
        assert_eq!(quiet_cut(&vec![0.25; 64_000]), 64_000);
        let mut sizes = Vec::new();
        for_each_decoder_segment(
            synthetic(16_000, 64_007, 0.0),
            4.0,
            &cancel,
            &mut |segment| {
                sizes.push(segment.samples.len());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(sizes, vec![64_000, 7]);
    }

    #[test]
    fn cancellation_and_visitor_failure_stop_without_success_or_extra_callbacks() {
        let cancel = Arc::new(AtomicBool::new(true));
        let mut calls = 0;
        let err = for_each_decoder_segment(synthetic(16_000, 100, 0.0), 1.0, &cancel, &mut |_| {
            calls += 1;
            Ok(())
        })
        .unwrap_err();
        assert!(err.downcast_ref::<AudioCancelled>().is_some());
        assert_eq!(calls, 0);
        cancel.store(false, Ordering::Relaxed);
        let signal = cancel.clone();
        let err =
            for_each_decoder_segment(synthetic(16_000, 100_000, 0.0), 0.01, &cancel, &mut |_| {
                calls += 1;
                signal.store(true, Ordering::Relaxed);
                Ok(())
            })
            .unwrap_err();
        assert!(err.downcast_ref::<AudioCancelled>().is_some());
        assert_eq!(calls, 1);
        cancel.store(false, Ordering::Relaxed);
        let err =
            for_each_decoder_segment(synthetic(16_000, 100_000, 0.0), 0.01, &cancel, &mut |_| {
                bail!("visitor stopped")
            })
            .unwrap_err();
        assert!(err.to_string().contains("visitor stopped"));
    }

    #[test]
    fn malformed_and_truncated_wav_is_never_reported_as_clean_eof() {
        let valid = riff(&[
            chunk(b"fmt ", &format(1, 16, 1, 16_000)),
            chunk(b"data", &[0; 8]),
        ]);
        for length in 0..valid.len() {
            assert!(
                decode(valid[..length].to_vec(), 1.0).is_err(),
                "accepted truncation at {length}"
            );
        }
        let mut bad = valid.clone();
        bad[32] = 3; // block align disagrees with sample width.
        assert!(decode(bad, 1.0).is_err());
        let mut bad = valid.clone();
        bad.extend_from_slice(&[1]);
        assert!(decode(bad, 1.0).is_err());
        for chunks in [
            vec![chunk(b"data", &[0; 2])],
            vec![chunk(b"fmt ", &format(1, 16, 1, 16000))],
            vec![
                chunk(b"fmt ", &format(1, 16, 1, 16000)),
                chunk(b"data", &[]),
            ],
            vec![
                chunk(b"fmt ", &format(1, 24, 2, 16000)),
                chunk(b"data", &[0; 3]),
            ],
            vec![
                chunk(b"fmt ", &format(6, 8, 1, 16000)),
                chunk(b"data", &[0; 2]),
            ],
            vec![
                chunk(b"fmt ", &format(3, 32, 1, 16000)),
                chunk(b"data", &f32::NAN.to_le_bytes()),
            ],
            vec![
                chunk(b"fmt ", &format(1, 16, 1, 16000)),
                chunk(b"data", &[0; 2]),
                chunk(b"data", &[0; 2]),
            ],
        ] {
            assert!(decode(riff(&chunks), 1.0).is_err());
        }
        let mut missing_pad = riff(&[
            chunk(b"fmt ", &format(1, 8, 1, 16000)),
            chunk(b"data", &[128]),
        ]);
        missing_pad.pop();
        let size = (missing_pad.len() - 8) as u32;
        missing_pad[4..8].copy_from_slice(&size.to_le_bytes());
        assert!(decode(missing_pad, 1.0).is_err());
        for seconds in [0.0, -1.0, f32::NAN, f32::INFINITY, 0.000001] {
            assert!(decode(valid.clone(), seconds).is_err());
        }
    }

    /// A multi-gigabyte PCM payload generated lazily, with no on-disk fixture or
    /// allocation proportional to total length. Reads record their largest size.
    struct LazyWave {
        header: Vec<u8>,
        position: u64,
        length: u64,
        largest_read: usize,
    }
    impl Read for LazyWave {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            self.largest_read = self.largest_read.max(output.len());
            let count = output
                .len()
                .min(self.length.saturating_sub(self.position) as usize);
            output[..count].fill(0);
            if self.position < self.header.len() as u64 {
                let start = self.position as usize;
                let copied = count.min(self.header.len() - start);
                output[..copied].copy_from_slice(&self.header[start..start + copied]);
            }
            self.position += count as u64;
            Ok(count)
        }
    }
    impl Seek for LazyWave {
        fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
            let position = match from {
                SeekFrom::Start(x) => x as i128,
                SeekFrom::Current(x) => self.position as i128 + x as i128,
                SeekFrom::End(x) => self.length as i128 + x as i128,
            };
            if position < 0 || position > u64::MAX as i128 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "invalid seek",
                ));
            }
            self.position = position as u64;
            Ok(self.position)
        }
    }

    #[test]
    fn multi_gigabyte_wav_reads_only_fixed_byte_blocks() {
        let data_bytes = 3_200_000_000u64;
        let mut header = riff(&[
            chunk(b"fmt ", &format(1, 16, 1, 16_000)),
            chunk(b"data", &[]),
        ]);
        let length = header.len() as u64 + data_bytes;
        header[4..8].copy_from_slice(&((length - 8) as u32).to_le_bytes());
        header[40..44].copy_from_slice(&(data_bytes as u32).to_le_bytes());
        let reader = LazyWave {
            header,
            position: 0,
            length,
            largest_read: 0,
        };
        let mut decoder = WavDecoder::new(reader, &AtomicBool::new(false)).unwrap();
        for _ in 0..20_000 {
            assert_eq!(decoder.next_mono_sample().unwrap(), Some(0.0));
        }
        assert_eq!(decoder.buffer.len(), READ_BUFFER_BYTES);
        assert!(decoder.reader.largest_read <= READ_BUFFER_BYTES);
        assert!(decoder.remaining_bytes > 3_000_000_000);
    }

    /// Reader advertises a complete RIFF but faults during audio reads, modeling
    /// late I/O failure/file truncation after metadata has already been checked.
    struct FailingReader {
        inner: Cursor<Vec<u8>>,
        fail_at: u64,
    }
    impl Read for FailingReader {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            if self.inner.position() >= self.fail_at {
                return Ok(0);
            }
            let n = output
                .len()
                .min((self.fail_at - self.inner.position()) as usize);
            self.inner.read(&mut output[..n])
        }
    }
    impl Seek for FailingReader {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(position)
        }
    }

    #[test]
    fn late_short_read_is_error_not_a_short_successful_segment() {
        let bytes = riff(&[
            chunk(b"fmt ", &format(1, 16, 1, 16000)),
            chunk(b"data", &[0; 100]),
        ]);
        let cancel = AtomicBool::new(false);
        let reader = FailingReader {
            inner: Cursor::new(bytes),
            fail_at: 48,
        };
        let mut decoder = WavDecoder::new(reader, &cancel).unwrap();
        assert!(decoder.next_mono_sample().is_err());
    }
}
