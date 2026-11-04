//! Bounded, non-streaming native CPU speech transport. No files or subprocesses.
use super::*;
use axum::extract::{FromRequest, Request};
use infr_tts::{SpeechResult, SynthesisOptions, Synthesizer};

const BODY_LIMIT: usize = 16 * 1024;
const MAX_SAMPLES: usize = 192_000 * 120;

#[derive(Clone)]
pub(super) struct Entry {
    pub(super) synthesizer: Arc<dyn Synthesizer>,
    slots: Arc<Semaphore>,
    admission: Arc<Semaphore>,
}
impl Entry {
    pub(super) fn new(synthesizer: Arc<dyn Synthesizer>) -> Self {
        Self {
            synthesizer,
            slots: Arc::new(Semaphore::new(1)),
            admission: Arc::new(Semaphore::new(4)),
        }
    }
}

pub(super) fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/audio/speech", post(handler))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    #[default]
    Wav,
    Pcm,
}

/// Mono PCM16, little endian; WAV includes its native sample rate in the RIFF header.
pub fn encode_audio(result: &SpeechResult, format: AudioFormat) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        (8_000..=192_000).contains(&result.sample_rate),
        "invalid speech sample rate"
    );
    anyhow::ensure!(
        !result.samples.is_empty()
            && result.samples.len() <= MAX_SAMPLES
            && result.samples.len() as u64 <= u64::from(result.sample_rate) * 120,
        "invalid speech duration"
    );
    anyhow::ensure!(
        result.samples.iter().all(|sample| sample.is_finite()),
        "invalid speech samples"
    );
    let data_len = (result.samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(data_len as usize + 44);
    if matches!(format, AudioFormat::Wav) {
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&result.sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(result.sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
    }
    for sample in &result.samples {
        let value = sample.clamp(-1.0, 1.0);
        let pcm = if value < 0.0 {
            value * 32768.0
        } else {
            value * 32767.0
        } as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    Ok(bytes)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    model: Option<String>,
    input: String,
    voice: Option<String>,
    #[serde(default = "default_speed")]
    speed: f32,
    #[serde(default)]
    response_format: AudioFormat,
    noise_scale: Option<f32>,
    seed: Option<u64>,
}
fn default_speed() -> f32 {
    1.0
}

fn options(input: &Input, model: &str) -> Result<SynthesisOptions, &'static str> {
    if input.input.trim().is_empty()
        || input.input.len() > 8192
        || input.input.chars().count() > 2048
    {
        return Err("input must contain 1..2048 characters and at most 8192 bytes");
    }
    if input.model.as_ref().is_some_and(|id| id != model) {
        return Err("unknown speech model");
    }
    if !input.speed.is_finite() || !(0.25..=4.0).contains(&input.speed) {
        return Err("speed must be in 0.25..=4");
    }
    let mut result = SynthesisOptions::default();
    result.speaker_id = match input.voice.as_deref().unwrap_or("default") {
        "default" => result.speaker_id,
        id => id
            .parse()
            .map_err(|_| "voice must be default or a numeric speaker ID")?,
    };
    result.speed = input.speed;
    if let Some(noise) = input.noise_scale {
        if !noise.is_finite() || !(0.0..=2.0).contains(&noise) {
            return Err("noise_scale must be in 0..=2");
        }
        result.noise_scale = noise;
    }
    if let Some(seed) = input.seed {
        result.seed = seed;
    }
    Ok(result)
}

fn failure(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(error_body(
            message,
            if status.is_client_error() {
                "invalid_request_error"
            } else {
                "server_error"
            },
        )),
    )
        .into_response()
}
fn synthesis_failure(error: anyhow::Error) -> (StatusCode, String) {
    if let Some(input) = error.downcast_ref::<infr_tts::InvalidInput>() {
        (StatusCode::BAD_REQUEST, input.to_string())
    } else {
        // Model paths, loader details and private backend errors stay server-side.
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "speech synthesis failed; check model and input compatibility".to_owned(),
        )
    }
}
struct CancelGuard(Arc<AtomicBool>);
impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
async fn cancelled(cancel: &AtomicBool) {
    while !cancel.load(Ordering::Relaxed) && !infr_core::shutdown::shutdown_requested() {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    cancel.store(true, Ordering::Relaxed);
}

async fn handler(State(state): State<AppState>, request: Request) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, request.headers()) {
        return denied;
    }
    let Some(entry) = state.speech.clone() else {
        return failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "no speech model configured",
        );
    };
    if request.uri().query().is_some_and(|query| !query.is_empty()) {
        return failure(
            StatusCode::BAD_REQUEST,
            "speech parameters must be JSON fields",
        );
    }
    let admission = match entry.admission.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return failure(
                StatusCode::TOO_MANY_REQUESTS,
                "speech request capacity exhausted",
            )
        }
    };
    // Bound wall clock even when the chat service has no timeout configured.
    let seconds = match state.cfg.serve.request_timeout_secs {
        0 => 120,
        seconds => seconds.min(600),
    };
    let deadline = tokio::time::Instant::now() + Duration::from_secs(seconds);
    let input = match tokio::time::timeout_at(
        deadline,
        Json::<Input>::from_request(request, &state),
    )
    .await
    {
        Ok(Ok(Json(input))) => input,
        Ok(Err(error)) => {
            // Parameter schema errors use 400; preserve 413/415 transport failures.
            let status = match error.status() {
                StatusCode::UNPROCESSABLE_ENTITY => StatusCode::BAD_REQUEST,
                status => status,
            };
            return failure(status, "invalid speech JSON (input, optional model/voice/speed/response_format/noise_scale/seed)");
        }
        Err(_) => return failure(StatusCode::GATEWAY_TIMEOUT, "speech deadline exceeded"),
    };
    let options = match options(&input, entry.synthesizer.model_id()) {
        Ok(options) => options,
        Err(message) => return failure(StatusCode::BAD_REQUEST, message),
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let _guard = CancelGuard(cancel.clone());
    // This supervisor survives handler cancellation, keeps both permits until the synchronous
    // service really returns, and joins it after requesting cancellation on deadline/shutdown.
    let job = tokio::spawn(async move {
        let queued = QueuedGuard::new(state.stats.clone());
        let permit = tokio::select! {
            biased;
            _ = cancelled(&cancel) => return Err((StatusCode::SERVICE_UNAVAILABLE, "speech cancelled".to_owned())),
            _ = tokio::time::sleep_until(deadline) => return Err((StatusCode::GATEWAY_TIMEOUT, "speech deadline exceeded".to_owned())),
            permit = entry.slots.acquire_owned() => permit.map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "speech unavailable".to_owned()))?,
        };
        drop(queued);
        let active = ActiveGuard::new(state.stats.clone());
        let task_cancel = cancel.clone();
        let mut worker = tokio::task::spawn_blocking(move || {
            let (_permit, _admission, _active) = (permit, admission, active);
            anyhow::ensure!(!task_cancel.load(Ordering::Relaxed), "speech cancelled");
            let result = entry
                .synthesizer
                .synthesize(&input.input, &options, &task_cancel)?;
            anyhow::ensure!(!task_cancel.load(Ordering::Relaxed), "speech cancelled");
            let rate = result.sample_rate;
            Ok::<_, anyhow::Error>((
                encode_audio(&result, input.response_format)?,
                rate,
                input.response_format,
            ))
        });
        tokio::select! {
            result = &mut worker => result.map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "speech worker failed".to_owned()))?.map_err(synthesis_failure),
            _ = tokio::time::sleep_until(deadline) => { cancel.store(true, Ordering::Relaxed); let _ = worker.await; Err((StatusCode::GATEWAY_TIMEOUT, "speech deadline exceeded".to_owned())) },
            _ = cancelled(&cancel) => { let _ = worker.await; Err((StatusCode::SERVICE_UNAVAILABLE, "speech cancelled".to_owned())) },
        }
    });
    match job.await {
        Ok(Ok((bytes, rate, format))) => (
            [
                (
                    header::CONTENT_TYPE,
                    match format {
                        AudioFormat::Wav => "audio/wav",
                        AudioFormat::Pcm => "audio/pcm",
                    },
                ),
                (
                    header::HeaderName::from_static("x-audio-sample-rate"),
                    &rate.to_string(),
                ),
                (header::HeaderName::from_static("x-audio-channels"), "1"),
            ],
            bytes,
        )
            .into_response(),
        Ok(Err((status, message))) => failure(status, &message),
        Err(_) => failure(
            StatusCode::INTERNAL_SERVER_ERROR,
            "speech supervisor failed",
        ),
    }
}

#[cfg(test)]
mod tests;
