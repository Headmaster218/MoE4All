//! Slow, test-only CPU oracle for the Qwen vision tower.
//!
//! The implementation intentionally streams one transformer block at a time so a real projector
//! parity test does not keep a second, fully-dequantized copy of the tower in RAM. It follows the
//! official Qwen4Exp vision forward and is not linked into production builds.

use crate::{merge_major_pos, ClipConfig, PreparedImage, VisionWeights};
use anyhow::{bail, Context, Result};
use half::f16;
use infr_core::{loader::TensorInfo, WeightSource};
use infr_gguf::Gguf;
use rayon::prelude::*;
use std::path::Path;

const VIT_THETA: f32 = 10_000.0;

fn dequant(gguf: &Gguf, info: &TensorInfo) -> Result<Vec<f32>> {
    let bytes = gguf
        .tensor_bytes(&info.name)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    infr_gguf::dequant::dequant_block(info.dtype, bytes)
        .with_context(|| format!("dequantize {}", info.name))
}

fn linear(
    x: &[f32],
    rows: usize,
    in_f: usize,
    weight: &[f32],
    bias: &[f32],
    out_f: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * in_f);
    assert_eq!(weight.len(), in_f * out_f);
    assert_eq!(bias.len(), out_f);
    let mut out = vec![0.0f32; rows * out_f];
    out.par_chunks_mut(out_f)
        .zip(x.par_chunks(in_f))
        .for_each(|(dst, src)| {
            for (o, value) in dst.iter_mut().enumerate() {
                let row = &weight[o * in_f..(o + 1) * in_f];
                *value = row
                    .iter()
                    .zip(src)
                    .fold(bias[o], |sum, (&w, &v)| sum + w * v);
            }
        });
    out
}

fn layer_norm(
    x: &[f32],
    rows: usize,
    dim: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    assert_eq!(x.len(), rows * dim);
    let mut out = vec![0.0f32; x.len()];
    out.par_chunks_mut(dim)
        .zip(x.par_chunks(dim))
        .for_each(|(dst, src)| {
            let mean = src.iter().sum::<f32>() / dim as f32;
            let variance = src
                .iter()
                .map(|value| {
                    let centered = value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / dim as f32;
            let inv = (variance + eps).sqrt().recip();
            for i in 0..dim {
                dst[i] = (src[i] - mean) * inv * weight[i] + bias[i];
            }
        });
    out
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
}

fn rope_2d(q: &mut [f32], k: &mut [f32], positions: &[i32], heads: usize, head_dim: usize) {
    let rows = positions.len() / 2;
    let half = head_dim / 2;
    let axis_pairs = half / 2;
    for row in 0..rows {
        let y = positions[row * 2] as f32;
        let x = positions[row * 2 + 1] as f32;
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for pair in 0..half {
                let axis_pair = pair % axis_pairs;
                let position = if pair < axis_pairs { y } else { x };
                let angle = position * VIT_THETA.powf(-2.0 * axis_pair as f32 / half as f32);
                let (sin, cos) = angle.sin_cos();
                let a = pair;
                let b = pair + half;
                for values in [&mut *q, &mut *k] {
                    let va = values[base + a];
                    let vb = values[base + b];
                    values[base + a] = va * cos - vb * sin;
                    values[base + b] = va * sin + vb * cos;
                }
            }
        }
    }
}

fn round_attention_inputs(values: &mut [f32]) {
    values
        .par_iter_mut()
        .for_each(|value| *value = f16::from_f32(*value).to_f32());
}

fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let width = heads * head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0f32; rows * width];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, dst)| {
            for head in 0..heads {
                let q_head = &q[row * width + head * head_dim..row * width + (head + 1) * head_dim];
                let mut scores = vec![0.0f32; rows];
                for (key_row, score) in scores.iter_mut().enumerate() {
                    let k_head = &k[key_row * width + head * head_dim
                        ..key_row * width + (head + 1) * head_dim];
                    *score = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum = scores.iter().map(|score| (score - max).exp()).sum::<f32>();
                let dst_head = &mut dst[head * head_dim..(head + 1) * head_dim];
                for (key_row, score) in scores.into_iter().enumerate() {
                    let probability = (score - max).exp() / sum;
                    let v_head = &v[key_row * width + head * head_dim
                        ..key_row * width + (head + 1) * head_dim];
                    for (target, value) in dst_head.iter_mut().zip(v_head) {
                        *target += probability * value;
                    }
                }
            }
        });
    out
}

pub(crate) fn encode(path: &Path, image: &PreparedImage) -> Result<Vec<f32>> {
    let gguf = Gguf::open(path).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let cfg = ClipConfig::from_gguf(&gguf)?;
    let weights = VisionWeights::load(&gguf)?;
    let rows = image.n_patches();
    let tokens = image.grid_nx * image.grid_ny;
    let dim = cfg.embedding_length;
    let patch_in = 3 * cfg.patch_size * cfg.patch_size;

    let patch_bias = dequant(&gguf, &weights.patch_embd_bias)?;
    let patch_weight = dequant(&gguf, &weights.patch_embd_weight)?;
    let mut state = linear(
        &image.patches,
        rows,
        patch_in,
        &patch_weight,
        &patch_bias,
        dim,
    );
    if let Some(second) = &weights.patch_embd_weight_video {
        let second = dequant(&gguf, second)?;
        let zeros = vec![0.0f32; dim];
        let temporal = linear(&image.patches, rows, patch_in, &second, &zeros, dim);
        for (value, extra) in state.iter_mut().zip(temporal) {
            *value += extra;
        }
    }
    for (value, position) in state.iter_mut().zip(&image.pos_embed) {
        *value += position;
    }

    let patch_grid_x = image.grid_nx * cfg.spatial_merge_size;
    let mut positions = vec![0i32; rows * 2];
    for (index, pair) in positions.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let (y, x) = merge_major_pos(index, patch_grid_x, cfg.spatial_merge_size);
        pair.copy_from_slice(&[y as i32, x as i32]);
    }

    for block in &weights.blocks {
        let ln1 = layer_norm(
            &state,
            rows,
            dim,
            &dequant(&gguf, &block.ln1_weight)?,
            &dequant(&gguf, &block.ln1_bias)?,
            cfg.layer_norm_epsilon,
        );
        let qkv = linear(
            &ln1,
            rows,
            dim,
            &dequant(&gguf, &block.attn_qkv_weight)?,
            &dequant(&gguf, &block.attn_qkv_bias)?,
            3 * dim,
        );
        let mut q = vec![0.0f32; rows * dim];
        let mut k = vec![0.0f32; rows * dim];
        let mut v = vec![0.0f32; rows * dim];
        for row in 0..rows {
            let source = &qkv[row * 3 * dim..(row + 1) * 3 * dim];
            q[row * dim..(row + 1) * dim].copy_from_slice(&source[..dim]);
            k[row * dim..(row + 1) * dim].copy_from_slice(&source[dim..2 * dim]);
            v[row * dim..(row + 1) * dim].copy_from_slice(&source[2 * dim..]);
        }
        rope_2d(&mut q, &mut k, &positions, cfg.head_count, cfg.head_dim);
        round_attention_inputs(&mut q);
        round_attention_inputs(&mut k);
        round_attention_inputs(&mut v);
        let attended = attention(&q, &k, &v, rows, cfg.head_count, cfg.head_dim);
        let projected = linear(
            &attended,
            rows,
            dim,
            &dequant(&gguf, &block.attn_out_weight)?,
            &dequant(&gguf, &block.attn_out_bias)?,
            dim,
        );
        for (value, residual) in state.iter_mut().zip(projected) {
            *value += residual;
        }

        let ln2 = layer_norm(
            &state,
            rows,
            dim,
            &dequant(&gguf, &block.ln2_weight)?,
            &dequant(&gguf, &block.ln2_bias)?,
            cfg.layer_norm_epsilon,
        );
        let mut up = linear(
            &ln2,
            rows,
            dim,
            &dequant(&gguf, &block.ffn_up_weight)?,
            &dequant(&gguf, &block.ffn_up_bias)?,
            cfg.feed_forward_length,
        );
        up.par_iter_mut().for_each(|value| *value = gelu(*value));
        let down = linear(
            &up,
            rows,
            cfg.feed_forward_length,
            &dequant(&gguf, &block.ffn_down_weight)?,
            &dequant(&gguf, &block.ffn_down_bias)?,
            dim,
        );
        for (value, residual) in state.iter_mut().zip(down) {
            *value += residual;
        }
    }

    let post = layer_norm(
        &state,
        rows,
        dim,
        &dequant(&gguf, &weights.post_ln_weight)?,
        &dequant(&gguf, &weights.post_ln_bias)?,
        cfg.layer_norm_epsilon,
    );
    let merged_dim = dim * cfg.spatial_merge_size * cfg.spatial_merge_size;
    if post.len() != tokens * merged_dim {
        bail!("vision merge shape does not cover all patches");
    }
    let mut merged = linear(
        &post,
        tokens,
        merged_dim,
        &dequant(&gguf, &weights.mm0_weight)?,
        &dequant(&gguf, &weights.mm0_bias)?,
        merged_dim,
    );
    merged
        .par_iter_mut()
        .for_each(|value| *value = gelu(*value));
    let output = linear(
        &merged,
        tokens,
        merged_dim,
        &dequant(&gguf, &weights.mm2_weight)?,
        &dequant(&gguf, &weights.mm2_bias)?,
        cfg.projection_dim,
    );
    if output.iter().any(|value| !value.is_finite()) {
        bail!("CPU vision reference produced non-finite values");
    }
    Ok(output)
}
