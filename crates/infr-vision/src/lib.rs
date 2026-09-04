//! Vision input support (stage V1+V2): mmproj GGUF parsing and image preprocessing for the
//! Qwen3-VL-style CLIP vision tower (`general.architecture == "clip"`,
//! `clip.projector_type == "qwen3vl_merger"`).
//!
//! Reference implementation: llama.cpp `tools/mtmd/clip.cpp` and `tools/mtmd/models/qwen3vl.cpp`.
//! Where this crate pins observable behavior (patch ordering, position-embedding interpolation),
//! the doc comment on the pinning item cites the reference.
//!
//! # Supported scope (PR #21 review)
//!
//! Vision support is NARROW and gated at every layer; anything outside this matrix fails with an
//! explicit error rather than silently degrading:
//!
//! * **Text trunk**: `qwen35` / `qwen35moe` only. `DenseSeamChat::generate_mm_impl` bails for any
//!   other arch — the vision turn's 2D-mrope positioning rides qwen35's `Op::QkNormMrope` IMROPE
//!   emission, which no other arch emits.
//! * **Projector**: `clip.projector_type == "qwen3vl_merger"` only (both the CPU tower and the
//!   Vulkan tower reject other projector types at load).
//! * **Deepstack**: `clip.is_deepstack_layers` must be ALL ZERO. Any nonzero entry bails at load
//!   (`VitEngine::load_cpu` / `VkVit::load_on`) — deepstack multi-layer feature injection into the
//!   trunk is NOT implemented.
//! * **Image inputs**: `data:` base64 URIs and bare base64 strings only
//!   ([`decode_image_input`]). Plain `http(s)://` URLs are NOT fetched — they bail with a clear
//!   error (a fetcher would add network + SSRF surface to the server).
//! * **Backends**: the Vulkan tower (native-f16 weights, per-shape graph plans) is tried first and
//!   pinned to the CALLER's `VulkanN` device so it shares the model's GPU; any failure falls back
//!   to the CPU f32 tower, which is the correctness oracle.

mod config;
mod preprocess;
mod vit;
mod vulkan;
mod weights;

pub use config::ClipConfig;
pub use preprocess::{
    decode_image_input, prepare_image_bytes, smart_resize, PreparedImage, MAX_PATCHES,
};
pub use vit::VitEngine;
pub use weights::{BlockWeights, VisionWeights};

use anyhow::Result;

/// A loaded vision tower that turns one preprocessed image into per-token embeddings for the LLM.
///
/// `encode` returns `n_tokens * projection_dim` floats in row-major `[n_tokens, projection_dim]`
/// order, where `n_tokens = (img_w/32) * (img_h/32)` — one token per 2x2 spatial-merge block
/// ([`ClipConfig::n_tokens`]).
///
// Implemented by the CPU/Vulkan ViT forward (patch-embed conv, 27 blocks, post-LN, mm.0/mm.2
// merger MLP); `infr-server` (V5) calls through it.
pub trait VisionEngine: Send + Sync {
    /// Encode one prepared image into `[n_tokens, projection_dim]` row-major f32 embeddings.
    fn encode(&self, image: &PreparedImage) -> Result<Vec<f32>>;
}
