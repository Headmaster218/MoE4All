//! Optional native CPU MeloTTS: GGUF weights, INFR CPU matrix kernels, no Python/ORT.
//! The supported export uses zero BERT inputs and deterministic duration prediction.
#[cfg(feature = "cpu-ab-profile")]
pub mod cpu_profile;
pub mod frontend;
mod math;
mod model;
mod service;

pub use service::{
    NativeSynthesizer, SpeechResult, SynthesisOptions, Synthesizer, SynthesizerConfig,
};

/// A synthesis request the caller can correct (text, voice or output budget).
/// Transport adapters may downcast this error to return a client error while
/// retaining model-loading, IO and numerical failures as internal errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidInput {
    message: String,
}
impl InvalidInput {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
impl std::fmt::Display for InvalidInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for InvalidInput {}

pub(crate) fn ensure_input(condition: bool, message: impl Into<String>) -> anyhow::Result<()> {
    if !condition {
        return Err(InvalidInput::new(message).into());
    }
    Ok(())
}

pub(crate) fn check_cancel(cancel: &std::sync::atomic::AtomicBool) -> anyhow::Result<()> {
    anyhow::ensure!(
        !cancel.load(std::sync::atomic::Ordering::Relaxed),
        "speech synthesis cancelled"
    );
    Ok(())
}
