//! Native audio models built on INFR's device-neutral graph/backend seam.
//! Audio decoding and transcription are independent from chat/KV scheduling.
pub mod frontend;
mod model;
mod service;
pub mod wav;

use anyhow::Result;
use serde::Serialize;
use std::{path::Path, sync::atomic::AtomicBool};

pub use model::SenseVoiceModel;
pub use service::{BackendFactory, ModelFactory, NativeTranscriber, TranscriberConfig};

pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(std::sync::atomic::Ordering::Acquire) {
        return Err(wav::AudioCancelled.into());
    }
    Ok(())
}

/// A loaded acoustic model. Graph construction belongs to the model, while
/// execution and allocations belong to its selected INFR Backend.
pub trait AcousticModel: Send {
    fn frontend_config(&self) -> &frontend::FrontendConfig;
    /// Models may replace the frontend without changing transport or scheduling.
    fn prepare_audio(&self, samples: &[f32], sample_rate: u32) -> Result<frontend::Features> {
        anyhow::ensure!(
            sample_rate == self.frontend_config().sample_rate,
            "audio/frontend sample-rate mismatch"
        );
        frontend::extract_features(samples, self.frontend_config())
    }
    fn infer(
        &mut self,
        features: &frontend::Features,
        language: &str,
        use_itn: bool,
    ) -> Result<String>;
    /// Compatibility fallback for models without internal cancellation boundaries.
    /// Native models should override this and poll between bounded compute stages.
    fn infer_cancellable(
        &mut self,
        features: &frontend::Features,
        language: &str,
        use_itn: bool,
        cancel: &AtomicBool,
    ) -> Result<String> {
        check_cancel(cancel)?;
        let text = self.infer(features, language, use_itn)?;
        check_cancel(cancel)?;
        Ok(text)
    }
    fn weight_bytes(&self) -> u64;
}

#[derive(Clone, Debug)]
pub struct TranscriptionOptions {
    pub language: String,
    pub use_itn: bool,
    /// Streaming callers can avoid retaining the full transcript in memory.
    pub collect_text: bool,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            language: "auto".into(),
            use_itn: true,
            collect_text: true,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TranscriptionSegment {
    pub text: String,
    /// Audio chunk boundaries, not estimated word alignments.
    pub start: f64,
    pub end: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub duration: f64,
}

/// Service-level capability. Implementations may use any compute backend;
/// request transport and resource lifetime do not depend on a model family.
pub trait Transcriber: Send + Sync {
    fn model_id(&self) -> &str;
    fn transcribe_file(
        &self,
        path: &Path,
        options: &TranscriptionOptions,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(TranscriptionSegment) -> Result<()>,
    ) -> Result<TranscriptionResult>;
}

pub fn validate_language(language: &str) -> Result<()> {
    anyhow::ensure!(
        matches!(language, "auto" | "zh" | "en" | "yue" | "ja" | "ko"),
        "unsupported transcription language {language:?}; expected auto, zh, en, yue, ja, or ko"
    );
    Ok(())
}
