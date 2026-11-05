//! Multipart WAV transport. Segments are offline audio chunks, never token/word timestamps.
use super::*;
use axum::extract::{FromRequest, Multipart, Request};
use infr_audio::{Transcriber, TranscriptionOptions, TranscriptionResult};
use std::{fs::File, path::PathBuf};
use tokio::sync::{mpsc, OwnedSemaphorePermit};

const UPLOAD_LIMIT: usize = 128 * 1024 * 1024;
const FIELD_LIMIT: usize = 1024;
const MAX_UPLOADS: usize = 4;

#[derive(Clone)]
pub(super) struct Entry {
    pub(super) transcriber: Arc<dyn Transcriber>,
    slots: Arc<Semaphore>,
    uploads: Arc<Semaphore>,
}

impl Entry {
    pub(super) fn new(transcriber: Arc<dyn Transcriber>) -> Self {
        Self {
            transcriber,
            slots: Arc::new(Semaphore::new(1)),
            uploads: Arc::new(Semaphore::new(MAX_UPLOADS)),
        }
    }
}

pub(super) fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/audio/transcriptions", post(handler))
        .layer(DefaultBodyLimit::max(UPLOAD_LIMIT))
}

/// Every blocking file operation owns an Arc. Dropping a cancelled async upload cannot unlink
/// a file while a write or transcription is still using it, including on Windows.
struct Upload {
    path: PathBuf,
    file: Mutex<Option<File>>,
    admission: Option<OwnedSemaphorePermit>,
}

impl Drop for Upload {
    fn drop(&mut self) {
        let file = self
            .file
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        drop(file.take());
        // Only this create_new-owned path is removed. Never use client filenames for paths.
        if let Err(error) = std::fs::remove_file(&self.path) {
            if error.kind() != io::ErrorKind::NotFound {
                // Do not admit another disk allocation while this one remains on disk.
                if let Some(permit) = self.admission.take() {
                    permit.forget();
                }
                tracing::warn!("audio upload cleanup failed; admission capacity retained");
            }
        }
    }
}

impl Upload {
    async fn create(
        directory: PathBuf,
        admission: OwnedSemaphorePermit,
    ) -> Result<Arc<Self>, Failure> {
        tokio::task::spawn_blocking(move || {
            std::fs::create_dir_all(&directory).map_err(|_| Failure::storage())?;
            let directory = directory.canonicalize().map_err(|_| Failure::storage())?;
            static NEXT: AtomicU64 = AtomicU64::new(0);
            for _ in 0..128 {
                let serial = NEXT.fetch_add(1, Ordering::Relaxed);
                let epoch = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                let path =
                    directory.join(format!("audio-{}-{epoch}-{serial}.wav", std::process::id()));
                let mut options = File::options();
                options.write(true).create_new(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    options.mode(0o600);
                }
                match options.open(&path) {
                    Ok(file) => {
                        return Ok(Arc::new(Self {
                            path,
                            file: Mutex::new(Some(file)),
                            admission: Some(admission),
                        }))
                    }
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(_) => return Err(Failure::storage()),
                }
            }
            Err(Failure::storage())
        })
        .await
        .map_err(|_| Failure::storage())?
    }

    async fn write(self: &Arc<Self>, bytes: axum::body::Bytes) -> Result<(), Failure> {
        let upload = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut file = upload.file.lock().map_err(|_| Failure::storage())?;
            file.as_mut()
                .ok_or_else(Failure::storage)?
                .write_all(&bytes)
                .map_err(|_| Failure::storage())
        })
        .await
        .map_err(|_| Failure::storage())?
    }

    async fn close(self: &Arc<Self>) -> Result<(), Failure> {
        let upload = self.clone();
        tokio::task::spawn_blocking(move || {
            let mut file = upload.file.lock().map_err(|_| Failure::storage())?;
            drop(file.take());
            Ok(())
        })
        .await
        .map_err(|_| Failure::storage())?
    }
}

#[derive(Clone, Copy, Debug)]
struct Failure(StatusCode, &'static str);

impl Failure {
    fn bad(message: &'static str) -> Self {
        Self(StatusCode::BAD_REQUEST, message)
    }
    fn storage() -> Self {
        Self(
            StatusCode::INTERNAL_SERVER_ERROR,
            "audio upload storage unavailable",
        )
    }
    fn cancelled() -> Self {
        Self(StatusCode::SERVICE_UNAVAILABLE, "transcription cancelled")
    }
    fn deadline() -> Self {
        Self(
            StatusCode::GATEWAY_TIMEOUT,
            "transcription deadline exceeded",
        )
    }
    fn kind(self) -> &'static str {
        if self.0.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        }
    }
    fn multipart(error: axum::extract::multipart::MultipartError) -> Self {
        if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
            Self(
                StatusCode::PAYLOAD_TOO_LARGE,
                "multipart upload exceeds 128 MiB",
            )
        } else {
            Self::bad("invalid multipart upload")
        }
    }
    fn response(self) -> Response {
        (self.0, Json(error_body(self.1, self.kind()))).into_response()
    }
}

struct Input {
    upload: Arc<Upload>,
    options: TranscriptionOptions,
    stream: bool,
    text: bool,
}

async fn parse(
    mut multipart: Multipart,
    directory: PathBuf,
    model_id: &str,
    admission: OwnedSemaphorePermit,
) -> Result<Input, Failure> {
    let mut admission = Some(admission);
    let mut seen = std::collections::HashSet::new();
    let mut upload = None;
    let mut options = TranscriptionOptions::default();
    let mut stream = false;
    let mut text = false;
    let mut model = None;
    let mut total = 0usize;
    while let Some(mut field) = multipart.next_field().await.map_err(Failure::multipart)? {
        let name = field
            .name()
            .ok_or_else(|| Failure::bad("multipart field requires a name"))?
            .to_owned();
        if !matches!(
            name.as_str(),
            "file" | "model" | "language" | "response_format" | "stream"
        ) {
            return Err(Failure::bad("unknown transcription parameter"));
        }
        if !seen.insert(name.clone()) {
            return Err(Failure::bad("duplicate transcription parameter"));
        }
        if name == "file" {
            if let Some(filename) = field.file_name() {
                if !filename.to_ascii_lowercase().ends_with(".wav") {
                    return Err(Failure(
                        StatusCode::UNSUPPORTED_MEDIA_TYPE,
                        "only WAV audio is supported",
                    ));
                }
            }
            let file = Upload::create(
                directory.clone(),
                admission.take().expect("unique file field"),
            )
            .await?;
            let mut prefix = Vec::with_capacity(12);
            while let Some(chunk) = field.chunk().await.map_err(Failure::multipart)? {
                total = total.saturating_add(chunk.len());
                if total > UPLOAD_LIMIT {
                    return Err(Failure(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "multipart upload exceeds 128 MiB",
                    ));
                }
                let needed = (12 - prefix.len()).min(chunk.len());
                prefix.extend_from_slice(&chunk[..needed]);
                file.write(chunk).await?;
            }
            if prefix.len() != 12 || &prefix[..4] != b"RIFF" || &prefix[8..12] != b"WAVE" {
                return Err(Failure(
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "only RIFF/WAVE audio is supported",
                ));
            }
            file.close().await?;
            upload = Some(file);
        } else {
            let mut value = Vec::new();
            while let Some(chunk) = field.chunk().await.map_err(Failure::multipart)? {
                total = total.saturating_add(chunk.len());
                if total > UPLOAD_LIMIT {
                    return Err(Failure(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "multipart upload exceeds 128 MiB",
                    ));
                }
                if value.len() + chunk.len() > FIELD_LIMIT {
                    return Err(Failure::bad("multipart parameter too long"));
                }
                value.extend_from_slice(&chunk);
            }
            let value = std::str::from_utf8(&value)
                .map_err(|_| Failure::bad("multipart parameter must be UTF-8"))?;
            match name.as_str() {
                "model" => model = Some(value.to_owned()),
                "language" => {
                    infr_audio::validate_language(value)
                        .map_err(|_| Failure::bad("unsupported transcription language"))?;
                    options.language = value.to_owned();
                }
                "response_format" => match value {
                    "json" => text = false,
                    "text" => text = true,
                    _ => return Err(Failure::bad("response_format must be json or text")),
                },
                "stream" => match value {
                    "true" => stream = true,
                    "false" => stream = false,
                    _ => return Err(Failure::bad("stream must be true or false")),
                },
                _ => unreachable!(),
            }
        }
    }
    let model = model.ok_or_else(|| Failure::bad("model is required"))?;
    if model != model_id {
        return Err(Failure(
            StatusCode::NOT_FOUND,
            "unknown transcription model",
        ));
    }
    if stream && text {
        return Err(Failure::bad("stream=true requires response_format=json"));
    }
    options.collect_text = !stream;
    Ok(Input {
        upload: upload.ok_or_else(|| Failure::bad("file is required"))?,
        options,
        stream,
        text,
    })
}

/// Latches cancellation even if the HTTP handler is dropped while awaiting its blocking worker.
struct CancelGuard(Arc<AtomicBool>);
impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

async fn cancelled(cancel: &AtomicBool) {
    loop {
        if cancel.load(Ordering::Relaxed) || infr_core::shutdown::shutdown_requested() {
            cancel.store(true, Ordering::Relaxed);
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn deadline_elapsed(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending::<()>().await,
    }
}

/// Validate metadata by seeking over data chunks, before touching the model service.
fn validate_wav(path: &std::path::Path, cancel: &AtomicBool) -> Result<(), Failure> {
    let file = File::open(path).map_err(|_| Failure::storage())?;
    infr_audio::wav::WavDecoder::new(file, cancel)
        .map(|_| ())
        .map_err(|error| {
            if cancel.load(Ordering::Relaxed) {
                Failure::cancelled()
            } else if error.chain().any(|cause| {
                cause
                    .downcast_ref::<io::Error>()
                    .is_some_and(|error| error.kind() != io::ErrorKind::UnexpectedEof)
            }) {
                Failure::storage()
            } else {
                Failure::bad("invalid or unsupported WAV structure")
            }
        })
}

/// Bounded SSE buffering. Backpressure is local to this CPU audio worker, never a chat/GPU baton.
fn send_segment(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    mut event: Result<Event, Infallible>,
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    loop {
        anyhow::ensure!(
            !cancel.load(Ordering::Relaxed) && !infr_core::shutdown::shutdown_requested(),
            "transcription cancelled"
        );
        match tx.try_send(event) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Full(value)) => {
                event = value;
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                cancel.store(true, Ordering::Relaxed);
                anyhow::bail!("transcription cancelled");
            }
        }
    }
}

fn start_job(
    state: AppState,
    entry: Entry,
    input: Input,
    cancel: Arc<AtomicBool>,
    events: Option<mpsc::Sender<Result<Event, Infallible>>>,
    deadline: Option<tokio::time::Instant>,
) -> tokio::task::JoinHandle<Result<TranscriptionResult, Failure>> {
    tokio::spawn(async move {
        let queued = QueuedGuard::new(state.stats.clone());
        let permit = tokio::select! {
            biased;
            _ = cancelled(&cancel) => return Err(Failure::cancelled()),
            _ = deadline_elapsed(deadline) => return Err(Failure::deadline()),
            permit = entry.slots.acquire_owned() => permit.map_err(|_| Failure::cancelled())?,
        };
        drop(queued);
        let active = ActiveGuard::new(state.stats.clone());
        let task_cancel = cancel.clone();
        let mut worker = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let _active = active;
            let upload = input.upload;
            if task_cancel.load(Ordering::Relaxed) {
                return Err(Failure::cancelled());
            }
            validate_wav(&upload.path, &task_cancel)?;
            entry.transcriber.transcribe_file(&upload.path, &input.options, &task_cancel, &mut |segment| {
                if let Some(tx) = &events {
                    let data = serde_json::json!({"type": "transcription.text.delta", "delta": segment.text, "start": segment.start, "end": segment.end});
                    send_segment(tx, Ok(Event::default().event("transcription.text.delta").data(data.to_string())), &task_cancel)?;
                }
                Ok(())
            }).map_err(|_| Failure(StatusCode::INTERNAL_SERVER_ERROR, "transcription failed; check WAV data and model compatibility"))
            // `upload` remains owned until transcribe_file has actually returned or unwound.
        });
        tokio::select! {
            result = &mut worker => {
                if cancel.load(Ordering::Relaxed) { return Err(Failure::cancelled()); }
                result.map_err(|_| Failure(StatusCode::INTERNAL_SERVER_ERROR, "transcription worker failed"))?
            }
            _ = deadline_elapsed(deadline) => {
                cancel.store(true, Ordering::Relaxed);
                let _ = worker.await;
                Err(Failure::deadline())
            }
            _ = cancelled(&cancel) => {
                let _ = worker.await;
                Err(Failure::cancelled())
            }
        }
    })
}

struct AudioStream {
    rx: mpsc::Receiver<Result<Event, Infallible>>,
    _cancel: CancelGuard,
}
impl futures_util::Stream for AudioStream {
    type Item = Result<Event, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_recv(cx)
    }
}

async fn handler(State(state): State<AppState>, request: Request) -> Response {
    if let Some(denied) = auth_gate(&state.cfg, request.headers()) {
        return denied;
    }
    if request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .is_some_and(|length| length > UPLOAD_LIMIT as u64)
    {
        return Failure(
            StatusCode::PAYLOAD_TOO_LARGE,
            "multipart upload exceeds 128 MiB",
        )
        .response();
    }
    if request.uri().query().is_some_and(|query| !query.is_empty()) {
        return Failure::bad("transcription parameters must be multipart fields").response();
    }
    let Some(entry) = state.transcription.clone() else {
        return Failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "no transcription model configured",
        )
        .response();
    };
    let directory = match state.transcription_upload_dir.clone() {
        Some(path) => path,
        None => match std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("data/audio-upload")))
        {
            Some(path) => path,
            None => return Failure::storage().response(),
        },
    };
    let multipart = match Multipart::from_request(request, &state).await {
        Ok(multipart) => multipart,
        Err(_) => return Failure::bad("expected multipart/form-data with a boundary").response(),
    };
    let admission = match entry.uploads.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return Failure(
                StatusCode::TOO_MANY_REQUESTS,
                "transcription upload capacity exhausted",
            )
            .response()
        }
    };
    // One configured wall-clock budget covers upload, queueing and transcription.
    let deadline =
        request_timeout(&state.cfg).map(|duration| tokio::time::Instant::now() + duration);
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = CancelGuard(cancel.clone());
    let parsed = tokio::select! {
        result = parse(multipart, directory, entry.transcriber.model_id(), admission) => result,
        _ = cancelled(&cancel) => return Failure::cancelled().response(),
        _ = deadline_elapsed(deadline) => return Failure::deadline().response(),
    };
    let input = match parsed {
        Ok(input) => input,
        Err(error) => return error.response(),
    };
    if input.stream {
        let (tx, rx) = mpsc::channel(32);
        let job = start_job(
            state,
            entry,
            input,
            cancel.clone(),
            Some(tx.clone()),
            deadline,
        );
        tokio::spawn(async move {
            let event = match job.await {
                Ok(Ok(result)) => Event::default().event("transcription.done").data(
                    serde_json::json!({"type": "transcription.done", "duration": result.duration})
                        .to_string(),
                ),
                Ok(Err(failure)) => sse_error_event(failure.1, failure.kind()).event("error"),
                Err(_) => {
                    sse_error_event("transcription worker failed", "server_error").event("error")
                }
            };
            // A deadline sets `cancel` too, but its terminal error must still reach a connected
            // client. Receiver drop cancels this send without retaining the upload or CPU slot.
            let _ = tx.send(Ok(event)).await;
        });
        Sse::new(AudioStream { rx, _cancel: guard })
            .keep_alive(
                KeepAlive::new()
                    .interval(SSE_KEEP_ALIVE_INTERVAL)
                    .text("keep-alive"),
            )
            .into_response()
    } else {
        let text = input.text;
        let result = start_job(state, entry, input, cancel, None, deadline).await;
        drop(guard);
        match result {
            Ok(Ok(result)) if text => (
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                result.text,
            )
                .into_response(),
            Ok(Ok(result)) => Json(result).into_response(),
            Ok(Err(error)) => error.response(),
            Err(_) => Failure(
                StatusCode::INTERNAL_SERVER_ERROR,
                "transcription worker failed",
            )
            .response(),
        }
    }
}

#[cfg(test)]
mod tests;
