//! The qwen3vl_merger ViT forward (stage V3) — CPU reference implementation.
//!
//! Mirrors llama.cpp `tools/mtmd/models/qwen3vl.cpp` op-for-op:
//! patch-embed Linear → +position-embed → 27 × (LN → biased fused QKV → 2D RoPE (VISION
//! mode) → bidirectional attention → out → residual → LN → GELU MLP → residual) → post-LN
//! → 2×2 spatial-merge reshape → mm.0 Linear → GELU → mm.2 Linear → `[n_tokens, 2048]`.
//!
//! v1 runs on the CPU (f32 weights dequantized once at load, rayon-parallel matmuls) — the
//! correctness oracle a Vulkan port must match. Deepstack is rejected loudly when the mmproj
//! actually ships it (Ornith's is all-zero/inert).

use crate::preprocess::prepare_image_bytes;
use crate::{ClipConfig, PreparedImage, VisionEngine, VisionWeights};
use anyhow::{bail, Context, Result};
use infr_core::WeightSource;
use infr_gguf::Gguf;
use rayon::prelude::*;

/// The ViT's 2D-RoPE base frequency — llama.cpp's qwen3vl vision encoder hardcodes
/// theta 10000 (distinct from the LM's `rope.freq_base`).
pub(crate) const VIT_THETA: f32 = 10_000.0;

/// One dequantized transformer block. Weight matrices are GGUF `[in, out]` ne0-first,
/// held row-major `[in][out]` — `y[o] = Σ_i x[i]·w[i*out + o] + b[o]`.
struct Block {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    qkv_w: Vec<f32>,
    qkv_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    up_w: Vec<f32>,
    up_b: Vec<f32>,
    down_w: Vec<f32>,
    down_b: Vec<f32>,
}

/// A CPU vision tower. `encode` is [`VisionEngine::encode`]; `pos_table` feeds the
/// preprocessor (position-embed interpolation needs the base table).
struct CpuTower {
    cfg: ClipConfig,
    patch_embd_w: Vec<f32>,
    patch_embd_b: Vec<f32>,
    pos_table: Vec<f32>,
    blocks: Vec<Block>,
    post_ln_w: Vec<f32>,
    post_ln_b: Vec<f32>,
    mm0_w: Vec<f32>,
    mm0_b: Vec<f32>,
    mm2_w: Vec<f32>,
    mm2_b: Vec<f32>,
}

/// A vision tower: either the CPU reference implementation (f32-dequantized weights,
/// rayon-parallel) or the Vulkan port (native-f16 weights, GPU graph). Construction goes
/// through [`VitEngine::new_cpu`] / [`VitEngine::new_vulkan`] / [`VitEngine::new`].
pub struct VitEngine {
    inner: Inner,
}

enum Inner {
    Cpu(CpuTower),
    Vulkan(crate::vulkan::VkVit),
}

/// Load a tensor and dequantize to host f32 (the mtp `load_tensor_dequant` pattern, local
/// because that one is `pub(crate)` to infr-llama).
fn dequant(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .with_context(|| format!("mmproj tensor not found: {name}"))?
        .clone();
    let bytes = g
        .tensor_bytes(&info.name)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    infr_gguf::dequant::dequant_block(info.dtype, &bytes).with_context(|| format!("tensor {name}"))
}

impl VitEngine {
    /// Open an mmproj GGUF, dequantize every weight to f32 (~0.9 GB host RAM for the 447M
    /// tower), and build the CPU engine.
    pub fn new_cpu(mmproj_path: &std::path::Path) -> Result<Self> {
        Ok(Self {
            inner: Inner::Cpu(Self::load_cpu(mmproj_path)?),
        })
    }

    /// Open an mmproj GGUF and build the VULKAN engine (native-f16 weights, per-shape GPU graph
    /// plans). Fails — without touching any CPU state — when Vulkan is unavailable or the mmproj
    /// carries a dtype the native path can't upload; callers fall back to [`Self::new_cpu`].
    pub fn new_vulkan(mmproj_path: &std::path::Path) -> Result<Self> {
        Self::new_vulkan_on(mmproj_path, None)
    }

    /// [`new_vulkan`](Self::new_vulkan) pinned to physical device `dev` (`Some(idx)` = `VulkanN`):
    /// the caller's model device, so the vision tower shares the GPU the chat backend selected
    /// (PR #21 review fix — the tower used to always open the default device). `None` = default.
    pub fn new_vulkan_on(mmproj_path: &std::path::Path, dev: Option<usize>) -> Result<Self> {
        Ok(Self {
            inner: Inner::Vulkan(crate::vulkan::VkVit::load_on(mmproj_path, dev)?),
        })
    }

    /// Preferred-construction entry point: try Vulkan when `prefer_gpu`, fall back to the CPU
    /// engine on ANY error (logged with the reason — the CPU path is the correctness oracle and
    /// always works).
    pub fn new(mmproj_path: &std::path::Path, prefer_gpu: bool) -> Result<Self> {
        if prefer_gpu {
            match Self::new_vulkan(mmproj_path) {
                Ok(e) => return Ok(e),
                Err(e) => {
                    tracing::warn!("Vulkan vision tower unavailable, falling back to CPU: {e:#}");
                }
            }
        }
        Self::new_cpu(mmproj_path)
    }

    /// The heavy CPU load: dequantize every weight to f32. The CPU forward is the correctness
    /// oracle every other backend must match.
    fn load_cpu(mmproj_path: &std::path::Path) -> Result<CpuTower> {
        let g = Gguf::open(mmproj_path)?;
        let w = VisionWeights::load(&g)?;
        let cfg = ClipConfig::from_gguf(&g)?;
        if cfg.is_deepstack_layers.iter().any(|&d| d) {
            bail!(
                "mmproj ships deepstack tensors (is_deepstack_layers nonzero) — deepstack \
                 injection is not implemented yet"
            );
        }
        let blocks: Vec<Block> = w
            .blocks
            .iter()
            .map(|b| {
                Ok(Block {
                    ln1_w: dequant(&g, &b.ln1_weight.name)?,
                    ln1_b: dequant(&g, &b.ln1_bias.name)?,
                    qkv_w: dequant(&g, &b.attn_qkv_weight.name)?,
                    qkv_b: dequant(&g, &b.attn_qkv_bias.name)?,
                    out_w: dequant(&g, &b.attn_out_weight.name)?,
                    out_b: dequant(&g, &b.attn_out_bias.name)?,
                    ln2_w: dequant(&g, &b.ln2_weight.name)?,
                    ln2_b: dequant(&g, &b.ln2_bias.name)?,
                    up_w: dequant(&g, &b.ffn_up_weight.name)?,
                    up_b: dequant(&g, &b.ffn_up_bias.name)?,
                    down_w: dequant(&g, &b.ffn_down_weight.name)?,
                    down_b: dequant(&g, &b.ffn_down_bias.name)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(CpuTower {
            cfg,
            patch_embd_w: dequant(&g, &w.patch_embd_weight.name)?,
            patch_embd_b: dequant(&g, &w.patch_embd_bias.name)?,
            pos_table: dequant(&g, &w.position_embd_weight.name)?,
            blocks,
            post_ln_w: dequant(&g, &w.post_ln_weight.name)?,
            post_ln_b: dequant(&g, &w.post_ln_bias.name)?,
            mm0_w: dequant(&g, &w.mm0_weight.name)?,
            mm0_b: dequant(&g, &w.mm0_bias.name)?,
            mm2_w: dequant(&g, &w.mm2_weight.name)?,
            mm2_b: dequant(&g, &w.mm2_bias.name)?,
        })
    }

    /// The dequantized base position-embed table, for [`prepare_image_bytes`].
    pub fn pos_table(&self) -> &[f32] {
        match &self.inner {
            Inner::Cpu(c) => &c.pos_table,
            Inner::Vulkan(v) => v.pos_table(),
        }
    }

    /// Decode + preprocess raw image bytes and encode them in one call.
    pub fn encode_image_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        let prep = prepare_image_bytes(bytes, &self.cfg(), self.pos_table())?;
        self.encode(&prep)
    }

    /// [`encode_image_bytes`](Self::encode_image_bytes) plus the merged-token grid. The chat
    /// layer's expansion pass (stage V5) needs `(grid_nx, grid_ny)` to fan the prompt's single
    /// `<|image_pad|>` marker out to `nx*ny` pad tokens and to build the span's 2D mrope
    /// positions — one return value instead of a prepare/encode split at every call site.
    pub fn encode_image_bytes_with_grid(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize, usize)> {
        let prep = prepare_image_bytes(bytes, &self.cfg(), self.pos_table())?;
        let (nx, ny) = (prep.grid_nx, prep.grid_ny);
        let embeds = self.encode(&prep)?;
        Ok((embeds, nx, ny))
    }

    fn cfg(&self) -> &ClipConfig {
        match &self.inner {
            Inner::Cpu(c) => &c.cfg,
            Inner::Vulkan(v) => v.config(),
        }
    }
}

impl VisionEngine for VitEngine {
    fn encode(&self, img: &PreparedImage) -> Result<Vec<f32>> {
        match &self.inner {
            Inner::Cpu(c) => c.encode(img),
            Inner::Vulkan(v) => v.encode(img),
        }
    }
}

impl VisionEngine for CpuTower {
    fn encode(&self, img: &PreparedImage) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let d = cfg.embedding_length; // 1152
        let n = img.n_patches();
        let n_tok = img.grid_nx * img.grid_ny;
        let proj = cfg.projection_dim; // 2048
        let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size; // 4
        let nh = cfg.head_count; // 16
        let hd = cfg.head_dim; // 72
        let _n_pairs = hd / 2; // 36 — ggml rope n_dims = d_head/2

        // ── patch embed: [n, 768] @ [768, d] + bias ────────────────────────────
        let mut x = linear(
            &img.patches,
            n,
            3 * cfg.patch_size * cfg.patch_size,
            &self.patch_embd_w,
            &self.patch_embd_b,
            d,
        );
        // ── + position embed (already merge-major, same order as patches) ─────
        for (xr, pr) in x.chunks_mut(d).zip(img.pos_embed.chunks(d)) {
            for (a, b) in xr.iter_mut().zip(pr) {
                *a += b;
            }
        }

        // ── 2D rope positions: (y, x) per patch, merge-major order ────────────
        // The patch grid (pre-merge) is grid_nx*merge × grid_ny*merge; seq i's patch
        // coordinate comes from the merge-major permutation (preprocess.rs).
        let nx = img.grid_nx * cfg.spatial_merge_size;
        let mut pos_hw = vec![0i32; n * 2];
        for (i, slot) in pos_hw.chunks_mut(2).enumerate() {
            let (py, px) = crate::preprocess::merge_major_pos(i, nx, cfg.spatial_merge_size);
            slot[0] = py as i32;
            slot[1] = px as i32;
        }

        // ── 27 transformer blocks ─────────────────────────────────────────────
        // DEBUG dumps for the numpy cross-check (stage bisect).
        if std::env::var("VIT_DUMP").is_ok() {
            let d1: Vec<String> = x.iter().take(16 * 1152).map(|v| format!("{v}")).collect();
            std::fs::write(
                std::env::temp_dir().join("vit_stage1_patchemb.txt"),
                d1.join("\n"),
            )
            .unwrap();
        }
        for (bi, blk) in self.blocks.iter().enumerate() {
            let ln1 = layernorm(&x, n, d, &blk.ln1_w, &blk.ln1_b, cfg.layer_norm_epsilon);
            // fused biased QKV → [n, 3d]; split q=[0..d], k=[d..2d], v=[2d..3d]
            let qkv = linear(&ln1, n, d, &blk.qkv_w, &blk.qkv_b, 3 * d);
            let mut q = vec![0f32; n * d];
            let mut k = vec![0f32; n * d];
            let mut v = vec![0f32; n * d];
            for r in 0..n {
                q[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d..r * 3 * d + d]);
                k[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + d..r * 3 * d + 2 * d]);
                v[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + 2 * d..r * 3 * d + 3 * d]);
            }
            // 2D RoPE (VISION): Q and K in place
            rope2d(&mut q, &mut k, &pos_hw, nh, hd, VIT_THETA);
            // bidirectional attention over all patches, scale 1/sqrt(hd)
            let attn_out = attention(&q, &k, &v, n, nh, hd);
            // out proj + residual
            let o = linear(&attn_out, n, d, &blk.out_w, &blk.out_b, d);
            for i in 0..n * d {
                x[i] += o[i];
            }
            // MLP: LN2 → up → GELU → down → residual
            let ln2 = layernorm(&x, n, d, &blk.ln2_w, &blk.ln2_b, cfg.layer_norm_epsilon);
            let mut up = linear(&ln2, n, d, &blk.up_w, &blk.up_b, cfg.feed_forward_length);
            for v in up.iter_mut() {
                *v = gelu(*v);
            }
            let down = linear(&up, n, cfg.feed_forward_length, &blk.down_w, &blk.down_b, d);
            for i in 0..n * d {
                x[i] += down[i];
            }
            let _ = bi;
            if std::env::var("VIT_DUMP").is_ok() && bi == 0 {
                let d2: Vec<String> = x.iter().take(16 * 1152).map(|v| format!("{v}")).collect();
                std::fs::write(
                    std::env::temp_dir().join("vit_stage2_block0.txt"),
                    d2.join("\n"),
                )
                .unwrap();
            }
        }

        // ── post-LN ───────────────────────────────────────────────────────────
        let x = layernorm(
            &x,
            n,
            d,
            &self.post_ln_w,
            &self.post_ln_b,
            cfg.layer_norm_epsilon,
        );

        // ── 2×2 merge: token t's 4 patches ARE merge-major seq 4t..4t+3 (contiguous) ──
        let mut merged = vec![0f32; n_tok * d * merge2];
        for (t, dst) in merged.chunks_mut(d * merge2).enumerate() {
            for s in 0..merge2 {
                dst[s * d..(s + 1) * d]
                    .copy_from_slice(&x[(t * merge2 + s) * d..(t * merge2 + s + 1) * d]);
            }
        }

        // ── merger MLP: mm.0 (4608→4608) + GELU + mm.2 (4608→2048) ────────────
        let m0 = linear(
            &merged,
            n_tok,
            d * merge2,
            &self.mm0_w,
            &self.mm0_b,
            d * merge2,
        );
        let mut m0g = m0;
        for v in m0g.iter_mut() {
            *v = gelu(*v);
        }
        let out = linear(&m0g, n_tok, d * merge2, &self.mm2_w, &self.mm2_b, proj);
        if out.iter().any(|v| !v.is_finite()) {
            bail!("ViT forward produced non-finite values");
        }
        Ok(out)
    }
}

// ── primitives (rayon-parallel where it pays) ──────────────────────────────────

fn gelu(v: f32) -> f32 {
    let c = 0.797_884_6_f32; // sqrt(2/pi)
    let inner = c * (v + 0.044_715 * v * v * v);
    0.5 * v * (1.0 + inner.tanh())
}

/// `y = x @ W + b` with `W` row-major `[in][out]`, one rayon task per row.
fn linear(x: &[f32], rows: usize, inp: usize, w: &[f32], b: &[f32], out: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * out];
    y.par_chunks_mut(out)
        .zip(x.par_chunks(inp))
        .for_each(|(yr, xr)| {
            for (o, yv) in yr.iter_mut().enumerate() {
                let row = &w[o * inp..o * inp + inp];
                let mut acc = b[o];
                for (i, &xi) in xr.iter().enumerate() {
                    acc += xi * row[i];
                }
                *yv = acc;
            }
        });
    y
}

/// Mean-centred LayerNorm with weight+bias (ggml `NORM_TYPE_NORMAL`).
fn layernorm(x: &[f32], rows: usize, d: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; rows * d];
    y.par_chunks_mut(d)
        .zip(x.par_chunks(d))
        .for_each(|(yr, xr)| {
            let mean = xr.iter().sum::<f32>() / d as f32;
            let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let s = 1.0 / (var + eps).sqrt();
            for (oi, o) in yr.iter_mut().enumerate() {
                let v = xr[oi];
                *o = (v - mean) * s * w[oi % d] + b[oi % d];
            }
        });
    y
}

/// VISION-mode 2D RoPE on Q and K in place (same semantics as `Op::Rope2D`'s CPU arm):
/// NEOX split-half pairing over `head_dim/2` pairs; sections (llama.cpp `mrope_sections`
/// verbatim — numerically pair counts here) with position streams [y, x, y, x]; theta
/// ramps reset per section with `theta_scale = theta^(-2/(head_dim/2))`.
fn rope2d(q: &mut [f32], k: &mut [f32], pos_hw: &[i32], nh: usize, hd: usize, theta: f32) {
    let n = pos_hw.len() / 2;
    let n_pairs = hd / 2;
    let mut sect_start = [0usize; 4];
    let mut acc = 0usize;
    let sections = [hd as u32 / 4; 4]; // {d/4}×4 — llama.cpp's VISION call
    for s in 0..4 {
        sect_start[s] = acc;
        acc += sections[s] as usize;
    }
    let theta_scale = theta.powf(-2.0 / n_pairs as f32);
    for r in 0..n {
        let (py, px) = (pos_hw[r * 2] as f32, pos_hw[r * 2 + 1] as f32);
        for h in 0..nh {
            let b = (r * nh + h) * hd;
            for p in 0..n_pairs {
                let sect = (0..4)
                    .find(|&s| p < sect_start[s] + sections[s] as usize)
                    .unwrap_or(0);
                let l = p - sect_start[sect];
                let pos_val = if sect % 2 == 0 { py } else { px };
                let ang = pos_val * theta_scale.powi(l as i32);
                let (s_, c_) = (ang.sin(), ang.cos());
                let (i0, i1) = (p, p + n_pairs);
                let qa = q[b + i0];
                let qb = q[b + i1];
                q[b + i0] = qa * c_ - qb * s_;
                q[b + i1] = qa * s_ + qb * c_;
                let ka = k[b + i0];
                let kb = k[b + i1];
                k[b + i0] = ka * c_ - kb * s_;
                k[b + i1] = ka * s_ + kb * c_;
            }
        }
    }
}

/// Bidirectional attention (no mask): `out[r] = softmax(Q·Kᵀ/√hd) @ V` per head.
fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize, nh: usize, hd: usize) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let out = vec![0f32; n * nh * hd];
    let out = std::sync::Mutex::new(out);
    (0..n).into_par_iter().for_each(|r| {
        for h in 0..nh {
            let qb = &q[r * nh * hd + h * hd..r * nh * hd + (h + 1) * hd];
            // scores over all keys
            let mut scores = vec![0f32; n];
            for (kk, s) in scores.iter_mut().enumerate() {
                let kb = &k[kk * nh * hd + h * hd..kk * nh * hd + (h + 1) * hd];
                *s = qb.iter().zip(kb).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
            // weighted sum of V rows
            let mut ob = vec![0f32; hd];
            for (kk, s) in scores.iter().enumerate() {
                let w = (s - mx).exp() / sum;
                let vb = &v[kk * nh * hd + h * hd..kk * nh * hd + (h + 1) * hd];
                for (o, &vv) in ob.iter_mut().zip(vb) {
                    *o += w * vv;
                }
            }
            let mut guard = out.lock().unwrap();
            let dst = &mut guard[r * nh * hd + h * hd..r * nh * hd + (h + 1) * hd];
            dst.copy_from_slice(&ob);
        }
    });
    let out = out.into_inner().unwrap();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_values() {
        // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715x³)))
        assert!(gelu(0.0).abs() < 1e-7);
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-5);
        assert!((gelu(-1.0) - (-0.158_808)).abs() < 1e-5);
        assert!(gelu(-10.0).abs() < 1e-6);
        assert!((gelu(2.0) - 1.954_598).abs() < 1e-5);
    }

    #[test]
    fn rope2d_zero_position_is_identity() {
        // (0,0) positions: angle 0 everywhere → rope is identity.
        let nh = 2;
        let hd = 8;
        let q: Vec<f32> = (0..3 * nh * hd).map(|v| v as f32 * 0.1).collect();
        let k = q.clone();
        let pos = vec![0i32; 6];
        let mut q2 = q.clone();
        let mut k2 = k.clone();
        rope2d(&mut q2, &mut k2, &pos, nh, hd, 10_000.0);
        assert_eq!(q, q2);
        assert_eq!(k, k2);
    }

    #[test]
    fn rope2d_rotates_unit_circle() {
        // head_dim 8 → 4 pairs, sections {2,2,2,2}: pairs 0..1 → y, 2..3 → x.
        // position (0, 1): section-0 pairs get angle 0 (y=0) — identity; section-1 pairs
        // (p=2,3) get x=1 with theta reset (l=0,1): angles 1 rad, then 0.01 rad.
        // Split-half pairs: (q[p], q[p+4]); unit pairs need q = [1,1,1,1, 0,0,0,0].
        let nh = 1;
        let hd = 8;
        let q = vec![1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut q2 = q.clone();
        let mut k2 = q.clone();
        let pos = vec![0, 1]; // y=0, x=1
        rope2d(&mut q2, &mut k2, &pos, nh, hd, 10_000.0);
        // Pair 0 (y=0): identity. Pair 1 (y=0): identity. Pair 2 (x, l=0): rot by 1 rad.
        let a = 1.0f32;
        let (s, c) = (a.sin(), a.cos());
        assert!((q2[0] - 1.0).abs() < 1e-6 && q2[4].abs() < 1e-6); // p0: (i0=0,i1=4)
        assert!((q2[1] - 1.0).abs() < 1e-6 && q2[5].abs() < 1e-6); // p1: y=0 identity
        assert!((q2[2] - c).abs() < 1e-5 && (q2[6] - s).abs() < 1e-5); // p2: (i0=2,i1=6)
                                                                       // Pair 3 (x, l=1): angle 10000^(-2/4) = 0.01 rad.
        let a3 = 10_000f32.powf(-2.0 / 4.0);
        assert!((q2[3] - a3.cos()).abs() < 1e-4);
        assert!((q2[7] - a3.sin()).abs() < 1e-4);
    }

    #[test]
    fn linear_matches_naive() {
        let x = vec![1.0, 2.0, 3.0]; // rows=1, in=3
        let w = vec![1.0, 0.5, 0.0, 2.0, 1.0, 1.0]; // [3, 2] row-major [in][out]
        let b = vec![0.1, -0.1];
        let y = linear(&x, 1, 3, &w, &b, 2);
        // y0 = 1*1 + 2*0.5 + 3*0 + 0.1 = 2.1; y1 = 1*2 + 2*1 + 3*1 - 0.1 = 6.9
        assert!((y[0] - 2.1).abs() < 1e-6);
        assert!((y[1] - 6.9).abs() < 1e-6);
    }

    #[test]
    fn layernorm_normalizes() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows × 2
        let w = vec![1.0, 1.0];
        let b = vec![0.0, 0.0];
        let y = layernorm(&x, 2, 2, &w, &b, 1e-6);
        // row [1,2]: mean 1.5, var 0.25 → [-1, 1]
        assert!((y[0] - (-1.0)).abs() < 1e-4 && (y[1] - 1.0).abs() < 1e-4);
    }

    /// REAL-mmproj smoke test (ignored by default): load a 447M qwen3vl_merger tower on
    /// BOTH backends, encode a small synthetic image, and assert CPU↔Vulkan parity
    /// (max|cpu−vk| < 0.05 per element — f16 weight rounding makes exact equality impossible)
    /// plus, when the old f32-CPU dump exists, that the Vulkan output still matches it.
    ///
    /// Machine-independent by design (PR #21 review): the mmproj path comes from the
    /// `VIT_MMPROJ` env var; unset (or pointing at nothing) SKIPS the test instead of panicking,
    /// so contributors without a local mmproj never see a machine-specific failure.
    #[test]
    #[ignore]
    fn vit_smoke_real_mmproj() {
        let Some(mmproj) = std::env::var_os("VIT_MMPROJ").map(std::path::PathBuf::from) else {
            eprintln!("VIT_MMPROJ not set — skipping the real-mmproj smoke test");
            return;
        };
        if !mmproj.exists() {
            eprintln!("VIT_MMPROJ={mmproj:?} does not exist — skipping");
            return;
        }
        // 256×256 gradient image — deterministic, same bytes every run (pixel (x,y) =
        // [(x%256), (y%256), 128]) → 8×8 merged tokens, 1024 patches.
        let png = {
            let img = image::RgbImage::from_fn(256, 256, |x, y| {
                image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
            });
            let mut buf = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(img)
                .write_to(&mut buf, image::ImageFormat::Png)
                .unwrap();
            buf.into_inner()
        };

        // ── CPU reference (f32 weights) ────────────────────────────────────────
        let t0 = std::time::Instant::now();
        let cpu = VitEngine::new_cpu(&mmproj).expect("load vit (cpu)");
        eprintln!("cpu vit loaded in {:?}", t0.elapsed());
        let t1 = std::time::Instant::now();
        let emb = cpu.encode_image_bytes(&png).expect("cpu encode");
        eprintln!("cpu encoded in {:?} → {} floats", t1.elapsed(), emb.len());
        assert_eq!(emb.len(), 64 * 2048, "8×8 tokens × 2048 dims");
        assert!(emb.iter().all(|v| v.is_finite()));
        let mean: f32 = emb.iter().sum::<f32>() / emb.len() as f32;
        let var: f32 = emb.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / emb.len() as f32;
        eprintln!("cpu mean={mean:.4} std={:.4}", var.sqrt());
        assert!(var > 1e-6, "output is constant — encoder is degenerate");

        // Dump the fresh CPU output for the numpy cross-check (first 64 rows, full precision).
        let dump: Vec<String> = emb.iter().take(64 * 2048).map(|v| format!("{v}")).collect();
        let dump_path = std::env::temp_dir().join("vit_rust_dump.txt");
        std::fs::write(&dump_path, dump.join("\n")).unwrap();
        eprintln!("dumped to {}", dump_path.display());

        // ── Vulkan port (native-f16 weights) ──────────────────────────────────
        let t2 = std::time::Instant::now();
        let vk = VitEngine::new_vulkan(&mmproj).expect("load vit (vulkan)");
        eprintln!("vulkan vit loaded in {:?}", t2.elapsed());
        let t3 = std::time::Instant::now();
        let emb_vk = vk.encode_image_bytes(&png).expect("vulkan encode");
        eprintln!(
            "vulkan encoded (incl. plan build) in {:?} → {} floats",
            t3.elapsed(),
            emb_vk.len()
        );
        let t4 = std::time::Instant::now();
        let emb_vk2 = vk.encode_image_bytes(&png).expect("vulkan encode #2");
        eprintln!("vulkan re-encode (warm plan) in {:?}", t4.elapsed());
        assert_eq!(emb_vk.len(), emb.len());
        assert_eq!(emb_vk2, emb_vk, "warm-plan re-encode must be deterministic");

        // CPU vs Vulkan per-element parity (f16 weights → small rounding, tolerance 0.05).
        let max_err = emb
            .iter()
            .zip(&emb_vk)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("max|cpu−vulkan| = {max_err:.6}");
        assert!(max_err < 0.05, "cpu↔vulkan parity blown: {max_err}");

        // Old-dump check: the prior dump came from the same f32 CPU forward, so the CPU output
        // matches it ~exactly and the Vulkan output within the f16 rounding budget.
        if let Ok(old) = std::fs::read_to_string(&dump_path) {
            let old: Vec<f32> = old
                .lines()
                .filter_map(|l| l.trim().parse::<f32>().ok())
                .collect();
            if old.len() == emb_vk.len() {
                let d_cpu = emb
                    .iter()
                    .zip(&old)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let d_vk = emb_vk
                    .iter()
                    .zip(&old)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("max|cpu−old_dump| = {d_cpu:.6}; max|vulkan−old_dump| = {d_vk:.6}");
                assert!(d_vk < 0.05, "vulkan vs old CPU dump parity blown: {d_vk}");
            } else {
                eprintln!("old dump length mismatch ({} lines) — skipped", old.len());
            }
        }
    }
}
