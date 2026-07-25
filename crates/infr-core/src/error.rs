//! Shared error/result types for the whole engine.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("backend: {0}")]
    Backend(String),
    #[error("loader: {0}")]
    Loader(String),
    #[error("model: {0}")]
    Model(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The process-wide shutdown latch (`crate::shutdown`) was set — a `SIGINT`/`SIGTERM` arrived
    /// and the backend stopped issuing NEW GPU work at the next submit boundary. Work already
    /// submitted was drained before this was returned; the caller should unwind (NOT
    /// `process::exit`) so the device is destroyed properly.
    #[error("aborted: shutdown requested")]
    Aborted,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Error {
    /// Construct a backend error from anything Display — the constructor each backend used to
    /// reinvent (Vulkan's local `fn be`).
    pub fn backend(msg: impl std::fmt::Display) -> Self {
        Error::Backend(msg.to_string())
    }
}

/// Free-function form of [`Error::backend`] — the terse `be(msg)` shorthand every backend crate
/// used to re-declare locally (rocm ×5, metal, vulkan). Import it under the short name instead of
/// defining a wrapper: `use infr_core::error::backend as be;`. Same error variant, same text.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn backend(msg: impl std::fmt::Display) -> Error {
    Error::backend(msg)
}

pub type Result<T> = std::result::Result<T, Error>;
