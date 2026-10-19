use crate::submission::{self, Kind};
use anyhow::{anyhow, Context};
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    mpsc::{self, Receiver, SyncSender, TrySendError},
    Mutex, OnceLock,
};
use std::thread::JoinHandle;
use std::time::Duration;

const STARTUP_LIMIT: u64 = 2 * 1024 * 1024;
const TAIL_SEGMENT_LIMIT: u64 = 16 * 1024 * 1024;
const REPORT_STARTUP_LINES: usize = 220;
const REPORT_TAIL_LINES: usize = 12;
const REPORT_LOG_BYTES: usize = 32 * 1024;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionFiles {
    pub id: String,
    pub created_at_utc: String,
    pub metadata: PathBuf,
    pub startup: PathBuf,
    pub tail: PathBuf,
    pub tail_previous: PathBuf,
    pub reports_root: PathBuf,
    pub redactions: Vec<String>,
}

enum Message {
    Log(Vec<u8>),
    Runtime,
    Flush(mpsc::Sender<()>),
    Stop(mpsc::Sender<()>),
}

static SENDER: OnceLock<SyncSender<Message>> = OnceLock::new();
static CURRENT: OnceLock<SessionFiles> = OnceLock::new();
static DROPPED: OnceLock<Mutex<u64>> = OnceLock::new();

pub struct Guard {
    sender: SyncSender<Message>,
    join: Option<JoinHandle<()>>,
    files: SessionFiles,
}

impl Guard {
    pub fn files(&self) -> &SessionFiles {
        &self.files
    }

    pub fn flush(&self) {
        let (tx, rx) = mpsc::channel();
        if self.sender.send(Message::Flush(tx)).is_ok() {
            let _ = rx.recv_timeout(Duration::from_secs(3));
        }
    }

    pub fn finish(mut self) {
        self.flush();
        let (tx, rx) = mpsc::channel();
        let _ = self.sender.send(Message::Stop(tx));
        let _ = rx.recv_timeout(Duration::from_secs(3));
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.join.is_none() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let _ = self.sender.send(Message::Stop(tx));
        let _ = rx.recv_timeout(Duration::from_secs(1));
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub struct LogWriter {
    bytes: Vec<u8>,
}

impl Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for LogWriter {
    fn drop(&mut self) {
        if self.bytes.is_empty() {
            return;
        }
        if let Some(sender) = SENDER.get() {
            match sender.try_send(Message::Log(std::mem::take(&mut self.bytes))) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    let dropped = DROPPED.get_or_init(|| Mutex::new(0));
                    *dropped.lock().unwrap_or_else(|error| error.into_inner()) += 1;
                }
                Err(TrySendError::Disconnected(_)) => {}
            }
        }
    }
}

pub fn log_writer() -> LogWriter {
    LogWriter { bytes: Vec::new() }
}

fn unique_id() -> anyhow::Result<String> {
    let now = chrono::Utc::now();
    let mut entropy = [0u8; 8];
    getrandom::fill(&mut entropy).context("obtain OS entropy for diagnostic session ID")?;
    let random = entropy
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("{}-{random}", now.format("%Y%m%dT%H%M%S%.3fZ")))
}

pub fn default_root() -> PathBuf {
    if let Some(path) = infr_core::config::env::diagnostic_directory(&|key| std::env::var_os(key)) {
        return path;
    }
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("diagnostics")
}

pub fn initialize(
    root: &Path,
    mut metadata: Value,
    redactions: Vec<String>,
) -> anyhow::Result<Guard> {
    let id = unique_id()?;
    let created_at = chrono::Utc::now().to_rfc3339();
    let sessions = root.join("sessions");
    let reports_root = root.join("reports");
    std::fs::create_dir_all(&sessions)
        .with_context(|| format!("create diagnostic directory {}", sessions.display()))?;
    std::fs::create_dir_all(&reports_root)
        .with_context(|| format!("create diagnostic directory {}", reports_root.display()))?;
    metadata["diagnostic_session_id"] = Value::String(id.clone());
    metadata["created_at_utc"] = Value::String(created_at.clone());
    metadata["pid"] = Value::from(std::process::id());
    let files = SessionFiles {
        id: id.clone(),
        created_at_utc: created_at,
        metadata: sessions.join(format!("{id}.meta.json")),
        startup: sessions.join(format!("{id}.startup.log")),
        tail: sessions.join(format!("{id}.tail.log")),
        tail_previous: sessions.join(format!("{id}.tail.1.log")),
        reports_root,
        redactions,
    };
    std::fs::write(&files.metadata, serde_json::to_vec(&metadata)?)?;
    let (sender, receiver) = mpsc::sync_channel(4096);
    let worker_files = files.clone();
    let join = std::thread::Builder::new()
        .name("infr-diagnostics".into())
        .spawn(move || worker(receiver, worker_files))
        .context("start diagnostic log worker")?;
    SENDER
        .set(sender.clone())
        .map_err(|_| anyhow!("diagnostic recorder was initialized twice"))?;
    CURRENT
        .set(files.clone())
        .map_err(|_| anyhow!("diagnostic session was initialized twice"))?;
    Ok(Guard {
        sender,
        join: Some(join),
        files,
    })
}

fn open_append(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

fn worker(receiver: Receiver<Message>, files: SessionFiles) {
    let mut startup = open_append(&files.startup).ok();
    let mut tail = open_append(&files.tail).ok();
    let mut startup_bytes = startup
        .as_ref()
        .and_then(|file| file.metadata().ok())
        .map(|meta| meta.len())
        .unwrap_or(0);
    let mut tail_bytes = tail
        .as_ref()
        .and_then(|file| file.metadata().ok())
        .map(|meta| meta.len())
        .unwrap_or(0);
    let mut runtime = false;
    while let Ok(message) = receiver.recv() {
        match message {
            Message::Log(bytes) => {
                if !runtime && startup_bytes < STARTUP_LIMIT {
                    let remaining = (STARTUP_LIMIT - startup_bytes) as usize;
                    let slice = &bytes[..bytes.len().min(remaining)];
                    if let Some(file) = startup.as_mut() {
                        if file.write_all(slice).is_ok() {
                            startup_bytes += slice.len() as u64;
                        }
                    }
                }
                if tail_bytes + bytes.len() as u64 > TAIL_SEGMENT_LIMIT {
                    if let Some(file) = tail.as_mut() {
                        let _ = file.flush();
                    }
                    drop(tail.take());
                    let _ = std::fs::remove_file(&files.tail_previous);
                    let _ = std::fs::rename(&files.tail, &files.tail_previous);
                    tail = open_append(&files.tail).ok();
                    tail_bytes = 0;
                }
                if let Some(file) = tail.as_mut() {
                    if file.write_all(&bytes).is_ok() {
                        tail_bytes += bytes.len() as u64;
                    }
                }
            }
            Message::Runtime => {
                runtime = true;
                if let Some(file) = startup.as_mut() {
                    let _ = file.flush();
                }
            }
            Message::Flush(done) => {
                if let Some(file) = startup.as_mut() {
                    let _ = file.flush();
                }
                if let Some(file) = tail.as_mut() {
                    let _ = file.flush();
                }
                let _ = done.send(());
            }
            Message::Stop(done) => {
                if let Some(file) = startup.as_mut() {
                    let _ = file.flush();
                }
                if let Some(file) = tail.as_mut() {
                    let _ = file.flush();
                }
                let _ = done.send(());
                break;
            }
        }
    }
}

pub fn mark_runtime() {
    if let Some(sender) = SENDER.get() {
        let _ = sender.try_send(Message::Runtime);
    }
}

pub fn flush() {
    let Some(sender) = SENDER.get() else {
        return;
    };
    let (tx, rx) = mpsc::channel();
    if sender.send(Message::Flush(tx)).is_ok() {
        let _ = rx.recv_timeout(Duration::from_secs(3));
    }
}

fn strip_ansi(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for code in chars.by_ref() {
                if ('@'..='~').contains(&code) {
                    break;
                }
            }
        } else if ch != '\r' {
            output.push(ch);
        }
    }
    output
}

fn is_speed_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("tok/s")
        || lower.contains("tokens/s")
        || lower.contains("infr live resources")
        || lower.contains("prefill:") && lower.contains("decode")
}

fn redact_unrecognized_paths_and_secrets(line: &str) -> String {
    static CREDENTIAL: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r#"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{8,}|\bsk-[A-Za-z0-9_-]{12,}|\b(?:api[_-]?key|password|secret)\s*[=:]\s*["']?[A-Za-z0-9._~+/-]{8,}"#).unwrap()
    });
    let line = CREDENTIAL.replace_all(line, "<redacted-setting>");
    line.split_whitespace()
        .map(|token| {
            let lower = token.to_ascii_lowercase();
            if lower.contains("api_key=")
                || lower.contains("api-key=")
                || lower.contains("token=")
                || lower.contains("password=")
                || lower.contains("secret=")
            {
                return "<redacted-setting>".to_string();
            }
            let bytes = token.as_bytes();
            let mut path_at = None;
            for index in 0..bytes.len() {
                let windows = index + 2 < bytes.len()
                    && bytes[index].is_ascii_alphabetic()
                    && bytes[index + 1] == b':'
                    && matches!(bytes[index + 2], b'\\' | b'/');
                let unix = bytes[index] == b'/'
                    && (index == 0 || matches!(bytes[index - 1], b'=' | b'\'' | b'"' | b'('));
                if windows || unix {
                    path_at = Some(index);
                    break;
                }
            }
            if let Some(index) = path_at {
                format!("{}<local-path>", &token[..index])
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn sanitize_diagnostic_text(line: &str, redactions: &[String]) -> String {
    let mut line = strip_ansi(line).trim().to_string();
    for redaction in redactions {
        if redaction.len() >= 3 {
            line = line.replace(redaction, "<local-path>");
        }
    }
    line = redact_unrecognized_paths_and_secrets(&line);
    if line.len() > 2_048 {
        line.truncate(line.floor_char_boundary(2_048));
        line.push_str("...");
    }
    line
}

fn sanitize_line(line: &str, redactions: &[String]) -> Option<String> {
    let line = sanitize_diagnostic_text(line, redactions);
    (!line.is_empty() && !is_speed_line(&line)).then_some(line)
}

fn sanitize_report_value(value: &mut Value, redactions: &[String]) {
    match value {
        Value::String(text) => *text = sanitize_diagnostic_text(text, redactions),
        Value::Array(values) => {
            for value in values {
                sanitize_report_value(value, redactions);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                sanitize_report_value(value, redactions);
            }
        }
        _ => {}
    }
}

fn read_text(path: &Path) -> String {
    let mut output = String::new();
    if let Ok(mut file) = File::open(path) {
        let _ = file.read_to_string(&mut output);
    }
    output
}

fn report_logs(files: &SessionFiles) -> (Vec<String>, Vec<String>, u64) {
    let startup = read_text(&files.startup);
    let previous = read_text(&files.tail_previous);
    let tail = read_text(&files.tail);
    let mut startup_lines = startup
        .lines()
        .filter_map(|line| sanitize_line(line, &files.redactions))
        .take(REPORT_STARTUP_LINES)
        .collect::<Vec<_>>();
    let mut tail_lines = previous
        .lines()
        .chain(tail.lines())
        .filter_map(|line| sanitize_line(line, &files.redactions))
        .collect::<Vec<_>>();
    if tail_lines.len() > REPORT_TAIL_LINES {
        tail_lines.drain(..tail_lines.len() - REPORT_TAIL_LINES);
    }
    let mut total = startup_lines.iter().map(|line| line.len()).sum::<usize>()
        + tail_lines.iter().map(|line| line.len()).sum::<usize>();
    while total > REPORT_LOG_BYTES && !startup_lines.is_empty() {
        total -= startup_lines.pop().map(|line| line.len()).unwrap_or(0);
    }
    let dropped = DROPPED
        .get()
        .map(|dropped| *dropped.lock().unwrap_or_else(|error| error.into_inner()))
        .unwrap_or(0);
    (startup_lines, tail_lines, dropped)
}

fn infer_gpu_name(startup: &str) -> Option<String> {
    startup.lines().rev().find_map(|line| {
        let marker = "[infr] GPU: ";
        let start = line.find(marker)? + marker.len();
        let name = line[start..].split(" | ").next()?.trim();
        (!name.is_empty()).then(|| name.to_string())
    })
}

fn read_metadata(files: &SessionFiles) -> Value {
    std::fs::read(&files.metadata)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_else(|| serde_json::json!({}))
}

fn report_id() -> anyhow::Result<String> {
    let now = chrono::Utc::now();
    let mut entropy = [0u8; 16];
    getrandom::fill(&mut entropy).context("obtain OS entropy for error report ID")?;
    let random = entropy
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("{}-{random}", now.format("%Y%m%dT%H%M%S%.3fZ")))
}

pub fn create_report(
    files: &SessionFiles,
    failure_kind: &str,
    description: &str,
    error_chain: &[String],
) -> anyhow::Result<PathBuf> {
    flush();
    let metadata = read_metadata(files);
    let mut hardware = metadata.get("hardware").cloned().unwrap_or(Value::Null);
    if hardware.get("gpu").is_none_or(Value::is_null) {
        if let Some(name) = infer_gpu_name(&read_text(&files.startup)) {
            if !hardware.is_object() {
                hardware = serde_json::json!({});
            }
            hardware["gpu"] = serde_json::json!({"name": name});
        }
    }
    let (startup, tail, dropped_events) = report_logs(files);
    let id = report_id()?;
    let created_at = chrono::Utc::now();
    let error_chain = error_chain
        .iter()
        .take(32)
        .map(|line| sanitize_diagnostic_text(line, &files.redactions))
        .collect::<Vec<_>>();
    let error = error_chain.first().cloned().unwrap_or_default();
    let mut launch = metadata.get("launch").cloned().unwrap_or(Value::Null);
    sanitize_report_value(&mut launch, &files.redactions);
    let payload = serde_json::json!({
        "schema_version": 1,
        "id": id,
        "created_at_utc": created_at.to_rfc3339(),
        "engine_version": env!("CARGO_PKG_VERSION"),
        "engine_commit": option_env!("INFR_GIT_COMMIT"),
        "failure_kind": failure_kind,
        "description": description,
        "engine": metadata.get("engine").cloned().unwrap_or(Value::Null),
        "hardware": hardware,
        "model": metadata.get("model").cloned().unwrap_or(Value::Null),
        "launch": launch,
        "error": {"message": error, "chain": error_chain},
        "logs": {
            "startup": startup,
            "tail": tail,
            "dropped_events": dropped_events,
            "speed_statistics_included": false,
        },
        "privacy": {
            "prompt_or_response_included": false,
            "user_description_included": !description.is_empty(),
            "local_paths_redacted": true,
            "api_keys_included": false,
        }
    });
    let directory = files
        .reports_root
        .join("v1")
        .join(created_at.format("%Y-%m").to_string());
    submission::write_new(&directory, Kind::Error, &payload)
}

pub fn describe_report(path: &Path, description: &str) -> anyhow::Result<PathBuf> {
    let mut decoded = submission::read(path)?;
    if decoded.kind != Kind::Error {
        anyhow::bail!("{} is not an error submission", path.display());
    }
    decoded.payload["description"] = Value::String(description.to_string());
    if !decoded.payload.get("privacy").is_some_and(Value::is_object) {
        decoded.payload["privacy"] = serde_json::json!({});
    }
    decoded.payload["privacy"]["user_description_included"] = Value::Bool(!description.is_empty());
    let text = submission::encode(Kind::Error, &decoded.payload)?;
    std::fs::write(path, text)?;
    Ok(path.to_path_buf())
}

pub fn find_latest_session(root: &Path) -> anyhow::Result<SessionFiles> {
    let sessions = root.join("sessions");
    let mut candidates = std::fs::read_dir(&sessions)
        .with_context(|| format!("read diagnostic sessions from {}", sessions.display()))?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| name.ends_with(".meta.json"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|entry| {
        let metadata = entry.path();
        let stem = metadata
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix(".meta.json"))
            .unwrap_or("")
            .to_string();
        [
            metadata,
            sessions.join(format!("{stem}.startup.log")),
            sessions.join(format!("{stem}.tail.1.log")),
            sessions.join(format!("{stem}.tail.log")),
        ]
        .into_iter()
        .filter_map(|path| path.metadata().and_then(|meta| meta.modified()).ok())
        .max()
    });
    let metadata = candidates
        .pop()
        .ok_or_else(|| anyhow!("no diagnostic session is available"))?
        .path();
    let name = metadata
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".meta.json"))
        .ok_or_else(|| anyhow!("invalid diagnostic session filename"))?
        .to_string();
    let value: Value = serde_json::from_slice(&std::fs::read(&metadata)?)?;
    let created_at_utc = value
        .get("created_at_utc")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let redactions = value
        .get("private_redactions")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Ok(SessionFiles {
        id: name.to_string(),
        created_at_utc,
        metadata,
        startup: sessions.join(format!("{name}.startup.log")),
        tail: sessions.join(format!("{name}.tail.log")),
        tail_previous: sessions.join(format!("{name}.tail.1.log")),
        reports_root: root.join("reports"),
        redactions,
    })
}

pub fn current_files() -> Option<&'static SessionFiles> {
    CURRENT.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_filter_excludes_speed_and_redacts_paths() {
        assert!(is_speed_line("Decode x1: 50.0 tok/s"));
        assert_eq!(
            sanitize_line("error loading D:\\Models\\x.gguf", &["D:\\Models".into()]).unwrap(),
            "error loading <local-path>\\x.gguf"
        );
        assert_eq!(
            sanitize_line("config=E:\\private\\config.toml api_key=secret", &[]).unwrap(),
            "config=<local-path> <redacted-setting>"
        );
    }

    #[test]
    fn ansi_is_removed_from_uploaded_logs() {
        assert_eq!(strip_ansi("\u{1b}[31merror\u{1b}[0m"), "error");
    }

    #[test]
    fn diagnostic_truncation_preserves_utf8_boundaries() {
        for text in ["\u{4e2d}".repeat(800), "\u{1f600}".repeat(600)] {
            let sanitized = sanitize_diagnostic_text(&text, &[]);
            assert!(sanitized.len() <= 2_051);
            assert!(sanitized.ends_with("..."));
            assert!(std::str::from_utf8(sanitized.as_bytes()).is_ok());
        }
    }

    #[test]
    fn diagnostic_error_chain_redacts_paths_and_credentials() {
        let text =
            "loading D:\\private\\model.gguf failed: Bearer abcdefghijklmnop api_key=abcdefghijk";
        let sanitized = sanitize_diagnostic_text(text, &[]);
        assert!(!sanitized.contains("D:\\"));
        assert!(!sanitized.contains("abcdefghijklmnop"));
        assert!(!sanitized.contains("abcdefghijk"));
        assert!(sanitized.contains("failed:"));
        assert!(
            sanitize_diagnostic_text("0 tok/s due to a device failure", &[])
                .contains("device failure")
        );
    }

    #[test]
    fn diagnostic_launch_redacts_paths_inside_settings_and_preserves_numbers() {
        let mut launch = serde_json::json!({
            "arguments": ["--set", "kv.session_cache_dir=C:\\private\\kv", "--ctx", "32k"],
            "process_id": 42
        });
        sanitize_report_value(&mut launch, &[]);
        assert_eq!(launch["arguments"][1], "kv.session_cache_dir=<local-path>");
        assert_eq!(launch["arguments"][3], "32k");
        assert_eq!(launch["process_id"], 42);
    }
}
