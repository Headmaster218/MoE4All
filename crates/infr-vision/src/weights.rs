//! `VisionWeights` — a catalog of every mmproj tensor the vision tower needs, with shapes
//! validated against [`ClipConfig`]. Holds `TensorInfo` descriptors only; no bytes are read here
//! (the V3 forward will map/upload them).
//!
//! Tensor names follow llama.cpp's clip converter (`v.*` for the tower, `mm.*` for the merger).

use crate::ClipConfig;
use anyhow::{bail, Context, Result};
use infr_core::{loader::TensorInfo, WeightSource};
use infr_gguf::Gguf;

/// One transformer block's tensors. Deepstack companions are optional and collected by name
/// (deepstack is inert for qwen3vl_merger mmproj files whose `is_deepstack_layers` is all zero).
#[derive(Clone, Debug)]
pub struct BlockWeights {
    pub ln1_weight: TensorInfo,
    pub ln1_bias: TensorInfo,
    pub attn_qkv_weight: TensorInfo,
    pub attn_qkv_bias: TensorInfo,
    pub attn_out_weight: TensorInfo,
    pub attn_out_bias: TensorInfo,
    pub ln2_weight: TensorInfo,
    pub ln2_bias: TensorInfo,
    pub ffn_up_weight: TensorInfo,
    pub ffn_up_bias: TensorInfo,
    pub ffn_down_weight: TensorInfo,
    pub ffn_down_bias: TensorInfo,
    /// Any `v.blk.N.*deepstack*` tensors, empty when the block has none. Tolerated, unused in V1.
    pub deepstack: Vec<TensorInfo>,
}

/// Catalog of all tensors in a `qwen3vl_merger` mmproj file.
#[derive(Clone, Debug)]
pub struct VisionWeights {
    /// `v.patch_embd.weight`, `[patch, patch, 3, embd]` (GGUF ne0-first), F16.
    pub patch_embd_weight: TensorInfo,
    /// `v.patch_embd.bias`, `[embd]`.
    pub patch_embd_bias: TensorInfo,
    /// `v.patch_embd.weight.1` — the video-twin patch embedding. Parsed, unused in V1.
    pub patch_embd_weight_video: Option<TensorInfo>,
    /// `v.position_embd.weight`, `[embd, base_grid²]`, F32.
    pub position_embd_weight: TensorInfo,
    pub blocks: Vec<BlockWeights>,
    /// `v.post_ln.weight` / `v.post_ln.bias`, `[embd]`.
    pub post_ln_weight: TensorInfo,
    pub post_ln_bias: TensorInfo,
    /// `mm.0.weight` / `mm.0.bias`: merger MLP first layer, `[embd*merge², embd*merge²]`.
    pub mm0_weight: TensorInfo,
    pub mm0_bias: TensorInfo,
    /// `mm.2.weight` / `mm.2.bias`: merger MLP output layer, `[embd*merge², projection_dim]`.
    pub mm2_weight: TensorInfo,
    pub mm2_bias: TensorInfo,
}

impl VisionWeights {
    /// Build the catalog, validating every required tensor's shape against the config parsed
    /// from the same file. Optional tensors (video twin, deepstack) are `None`/empty when absent.
    pub fn load(gguf: &Gguf) -> Result<Self> {
        let cfg = ClipConfig::from_gguf(gguf)?;
        let find = |name: &str, shape: &[usize]| -> Result<TensorInfo> {
            let info = gguf
                .tensors()
                .iter()
                .find(|t| t.name == name)
                .with_context(|| format!("GGUF missing tensor {name}"))?;
            if info.shape != shape {
                bail!(
                    "GGUF tensor {name} has shape {:?}; expected {shape:?}",
                    info.shape
                );
            }
            Ok(info.clone())
        };
        let optional = |name: &str| -> Option<TensorInfo> {
            gguf.tensors().iter().find(|t| t.name == name).cloned()
        };

        let d = cfg.embedding_length;
        let ff = cfg.feed_forward_length;
        let merge2 = cfg.spatial_merge_size * cfg.spatial_merge_size;

        let mut blocks = Vec::with_capacity(cfg.block_count);
        for n in 0..cfg.block_count {
            let name = |suffix: &str| format!("v.blk.{n}.{suffix}");
            let deepstack = gguf
                .tensors()
                .iter()
                .filter(|t| {
                    t.name.starts_with(&format!("v.blk.{n}.")) && t.name.contains("deepstack")
                })
                .cloned()
                .collect();
            blocks.push(BlockWeights {
                ln1_weight: find(&name("ln1.weight"), &[d])?,
                ln1_bias: find(&name("ln1.bias"), &[d])?,
                attn_qkv_weight: find(&name("attn_qkv.weight"), &[d, 3 * d])?,
                attn_qkv_bias: find(&name("attn_qkv.bias"), &[3 * d])?,
                attn_out_weight: find(&name("attn_out.weight"), &[d, d])?,
                attn_out_bias: find(&name("attn_out.bias"), &[d])?,
                ln2_weight: find(&name("ln2.weight"), &[d])?,
                ln2_bias: find(&name("ln2.bias"), &[d])?,
                ffn_up_weight: find(&name("ffn_up.weight"), &[d, ff])?,
                ffn_up_bias: find(&name("ffn_up.bias"), &[ff])?,
                ffn_down_weight: find(&name("ffn_down.weight"), &[ff, d])?,
                ffn_down_bias: find(&name("ffn_down.bias"), &[d])?,
                deepstack,
            });
        }

        Ok(Self {
            patch_embd_weight: find(
                "v.patch_embd.weight",
                &[cfg.patch_size, cfg.patch_size, 3, d],
            )?,
            patch_embd_bias: find("v.patch_embd.bias", &[d])?,
            patch_embd_weight_video: optional("v.patch_embd.weight.1"),
            position_embd_weight: find(
                "v.position_embd.weight",
                &[d, cfg.base_grid * cfg.base_grid],
            )?,
            blocks,
            post_ln_weight: find("v.post_ln.weight", &[d])?,
            post_ln_bias: find("v.post_ln.bias", &[d])?,
            mm0_weight: find("mm.0.weight", &[d * merge2, d * merge2])?,
            mm0_bias: find("mm.0.bias", &[d * merge2])?,
            mm2_weight: find("mm.2.weight", &[d * merge2, cfg.projection_dim])?,
            mm2_bias: find("mm.2.bias", &[cfg.projection_dim])?,
        })
    }
}
