//! `ClipConfig` — vision-tower hyper-parameters parsed from mmproj GGUF metadata.
//!
//! Key layout follows llama.cpp's clip GGUF converter (`clip.vision.*`, `clip.projector_type`).
//! Ground truth verified against Ornith-1.5-35B-A3B's mmproj.gguf.

use anyhow::{bail, Context, Result};
use infr_core::{loader::MetaValue, WeightSource};
use infr_gguf::Gguf;

/// Vision-tower (mmproj) configuration.
#[derive(Clone, Debug)]
pub struct ClipConfig {
    /// Number of transformer blocks (`clip.vision.block_count`, 27).
    pub block_count: usize,
    /// Token/embedding width (`clip.vision.embedding_length`, 1152).
    pub embedding_length: usize,
    /// FFN inner width (`clip.vision.feed_forward_length`, 4304).
    pub feed_forward_length: usize,
    /// Attention heads (`clip.vision.attention.head_count`, 16).
    pub head_count: usize,
    /// `embedding_length / head_count` (72).
    pub head_dim: usize,
    /// Nominal training image size in pixels (`clip.vision.image_size`, 768). Informational —
    /// actual inputs are smart-resized to multiples of [`Self::merge_factor`].
    pub image_size: usize,
    /// Patch edge in pixels (`clip.vision.patch_size`, 16).
    pub patch_size: usize,
    /// Spatial merge edge (`clip.vision.spatial_merge_size`, 2): each LLM token covers a
    /// merge² block of patches.
    pub spatial_merge_size: usize,
    /// Output width fed to the LLM (`clip.vision.projection_dim`, 2048).
    pub projection_dim: usize,
    /// `clip.use_gelu` — GELU FFN activation (qwen3vl) rather than QuickGELU.
    pub use_gelu: bool,
    /// Per-channel normalization mean (`clip.vision.image_mean`, `[0.5; 3]`).
    pub image_mean: [f32; 3],
    /// Per-channel normalization std (`clip.vision.image_std`, `[0.5; 3]`).
    pub image_std: [f32; 3],
    /// LayerNorm epsilon (`clip.vision.attention.layer_norm_epsilon`, 1e-6).
    pub layer_norm_epsilon: f32,
    /// Per-block deepstack flags (`clip.vision.is_deepstack_layers`). INERT: parsed so V3+ can
    /// warn on a model that actually uses deepstack; an all-zero vector (Ornith) means no
    /// deepstack tensors participate in the forward.
    pub is_deepstack_layers: Vec<bool>,
    /// Edge of the base position-embedding grid, `sqrt(v.position_embd.weight rows)` = 48.
    pub base_grid: usize,
}

impl ClipConfig {
    /// Parse and validate the mmproj metadata. Bails on a non-CLIP file or a projector this
    /// stage does not implement.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let md = gguf.metadata();
        let arch = md
            .str("general.architecture")
            .context("GGUF missing general.architecture")?;
        if arch != "clip" {
            bail!("infr-vision requires general.architecture=\"clip\"; got {arch:?}");
        }
        let projector = md
            .str("clip.projector_type")
            .context("GGUF missing clip.projector_type")?;
        if projector != "qwen3vl_merger" {
            bail!(
                "infr-vision V1 supports clip.projector_type=\"qwen3vl_merger\"; got {projector:?}"
            );
        }
        let integer = |suffix: &str| -> Result<usize> {
            let key = format!("clip.vision.{suffix}");
            usize::try_from(
                md.u64(&key)
                    .with_context(|| format!("GGUF missing {key}"))?,
            )
            .with_context(|| format!("GGUF {key} is too large"))
        };
        let float3 = |key: &str| -> Result<[f32; 3]> {
            let arr = md
                .get(key)
                .and_then(MetaValue::as_arr)
                .with_context(|| format!("GGUF missing {key}"))?;
            if arr.len() != 3 {
                bail!("GGUF {key} must have 3 elements, got {}", arr.len());
            }
            let mut out = [0.0f32; 3];
            for (i, v) in arr.iter().enumerate() {
                out[i] = v
                    .as_f64()
                    .with_context(|| format!("GGUF {key}[{i}] is not a number"))?
                    as f32;
            }
            Ok(out)
        };

        let embedding_length = integer("embedding_length")?;
        let head_count = integer("attention.head_count")?;
        if head_count == 0 || embedding_length % head_count != 0 {
            bail!(
                "invalid CLIP head geometry: embedding_length={embedding_length}, \
                 head_count={head_count}"
            );
        }
        let patch_size = integer("patch_size")?;
        let spatial_merge_size = integer("spatial_merge_size")?;
        if patch_size == 0 || spatial_merge_size == 0 {
            bail!("clip.vision.patch_size and spatial_merge_size must be nonzero");
        }
        let block_count = integer("block_count")?;
        let use_gelu = md.u64("clip.use_gelu").unwrap_or(0) != 0
            || matches!(md.get("clip.use_gelu"), Some(MetaValue::Bool(true)));
        let layer_norm_epsilon = md
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(MetaValue::as_f64)
            .unwrap_or(1e-5) as f32;
        let is_deepstack_layers = md
            .get("clip.vision.is_deepstack_layers")
            .and_then(MetaValue::as_arr)
            .map(|arr| {
                arr.iter()
                    .map(|v| match v {
                        MetaValue::Bool(b) => *b,
                        other => other.as_u64().unwrap_or(0) != 0,
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![false; block_count]);

        // Base position grid comes from the tensor itself, not a metadata key: the GGUF carries
        // `v.position_embd.weight` as [embd, base²] (ne0-first), so base = sqrt(shape[1]).
        let pos = gguf
            .tensors()
            .iter()
            .find(|t| t.name == "v.position_embd.weight")
            .context("GGUF missing tensor v.position_embd.weight")?;
        if pos.shape.len() != 2 || pos.shape[0] != embedding_length {
            bail!(
                "GGUF tensor v.position_embd.weight has shape {:?}; expected [{embedding_length}, base²]",
                pos.shape
            );
        }
        let base_grid = (pos.shape[1] as f64).sqrt() as usize;
        if base_grid * base_grid != pos.shape[1] || base_grid == 0 {
            bail!(
                "GGUF tensor v.position_embd.weight rows {} is not a perfect square",
                pos.shape[1]
            );
        }

        Ok(Self {
            block_count,
            embedding_length,
            feed_forward_length: integer("feed_forward_length")?,
            head_count,
            head_dim: embedding_length / head_count,
            image_size: integer("image_size")?,
            patch_size,
            spatial_merge_size,
            projection_dim: integer("projection_dim")?,
            use_gelu,
            image_mean: float3("clip.vision.image_mean")?,
            image_std: float3("clip.vision.image_std")?,
            layer_norm_epsilon,
            is_deepstack_layers,
            base_grid,
        })
    }

    /// Pixel granularity every input dimension is snapped to: `patch_size * spatial_merge_size`
    /// (32 for qwen3vl), so both the patch grid and the merged grid divide evenly.
    pub fn merge_factor(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }

    /// LLM tokens produced for an `img_w`×`img_h` (post-resize) image: one per merge block,
    /// `(img_w/32) * (img_h/32)`.
    pub fn n_tokens(&self, img_w: usize, img_h: usize) -> usize {
        (img_w / self.merge_factor()) * (img_h / self.merge_factor())
    }

    /// Number of 16px patches, `n_tokens * merge²`.
    pub fn n_patches(&self, img_w: usize, img_h: usize) -> usize {
        (img_w / self.patch_size) * (img_h / self.patch_size)
    }
}
