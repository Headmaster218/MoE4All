//! Model acquisition: resolve an `hf:org/repo[:quant]` reference (or a plain path) to a local GGUF,
//! pulling from HuggingFace over plain HTTP (no external CLI) with resume + a progress bar.
//!
//! Models live in the **standard HF Hub cache** (`~/.cache/huggingface/hub`), shared with llama.cpp
//! and `huggingface_hub`, so `infr run hf:org/repo:Q4_K_M` and `llama-cli -hf org/repo:Q4_K_M` use
//! the same files — see [`store`] for the layout.

mod download;
mod http;
mod model_ref;
mod parts;
mod pull;
mod ranged;
mod store;
#[cfg(test)]
mod testhttp;

pub use model_ref::ModelRef;
pub use pull::{pull, pull_latest};
pub use store::Store;

use infr_core::config::Config;
use infr_core::error::Result;
use std::path::PathBuf;

/// Resolve from the store if present, otherwise pull. Cache-first — does NOT check HF for updates, so
/// it stays fast and works offline. Used by `infr run` / `infr serve`.
///
/// - `Path(p)` → returned immediately.
/// - Everything else → [`Store::discover`] + [`Store::resolve`]; if not cached → [`pull`].
pub fn ensure(r: &ModelRef, cfg: &Config) -> Result<PathBuf> {
    if let ModelRef::Path(p) = r {
        return Ok(p.clone());
    }
    let store = Store::discover()?;
    if let Some(p) = store.resolve(r)? {
        return Ok(p);
    }
    pull(r, cfg)
}

/// Like [`ensure`] but checks HF for the repo's latest commit and updates the cache when it's stale
/// (falling back to the cached copy when offline). Used by `infr pull` so a re-pull picks up repo
/// updates instead of silently serving the first-pulled snapshot forever.
pub fn ensure_latest(r: &ModelRef, cfg: &Config) -> Result<PathBuf> {
    if let ModelRef::Path(p) = r {
        return Ok(p.clone());
    }
    pull_latest(r, cfg)
}
