//! Vision input support: mmproj GGUF parsing and image preprocessing for the
//! Qwen3-VL-style CLIP vision tower (`general.architecture == "clip"`,
//! `clip.projector_type == "qwen3vl_merger"`).
//!
//! Reference implementation: llama.cpp `tools/mtmd/clip.cpp` and `tools/mtmd/models/qwen3vl.cpp`.
//! Where this crate pins observable behavior (patch ordering, position-embedding interpolation),
//! the doc comment on the pinning item cites the reference.
//!
//! # Supported scope
//!
//! Vision support is NARROW and gated at every layer; anything outside this matrix fails with an
//! explicit error rather than silently degrading:
//!
//! * **Projector**: `clip.projector_type == "qwen3vl_merger"` only.
//! * **Deepstack**: `clip.is_deepstack_layers` must be ALL ZERO. Any nonzero entry bails at load
//!   because deepstack multi-layer feature injection into the trunk is not implemented.
//! * **Image inputs**: `data:` base64 URIs, bare base64 strings, and `http(s)://` URLs
//!   ([`decode_image_input`]). A URL fetch is the one place the engine goes out on a caller's
//!   behalf, so it is bounded by size, by wall clock, and by an address rule that refuses
//!   loopback / link-local / private ranges (`vision.image_url_allow_private` lifts it for a NAS or
//!   a LAN camera). Any other scheme — `file://` included — bails with a clear error.
//! * **Execution**: Vulkan only. The tower derives a client from the running LLM backend, admits
//!   native GGUF weights from SSD for one image batch, and returns all temporary VRAM afterwards.

mod config;
mod engine;
mod heif;
mod preprocess;
mod weights;

#[cfg(test)]
mod reference;

pub use config::ClipConfig;
pub use engine::{NativeVisionEngine, VisionEmbedding};
pub use preprocess::{
    decode_image_input, merge_major_pos, prepare_image_bytes, smart_resize, PreparedImage,
    MAX_PATCHES,
};
pub use weights::{BlockWeights, VisionWeights};
