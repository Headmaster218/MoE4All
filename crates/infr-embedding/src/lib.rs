//! Embedding-model integration for INFR.
//!
//! Embedding architectures are delegated to the mature `llama.cpp` implementation instead of
//! being reimplemented as an INFR graph. INFR owns the process lifecycle and OpenAI-facing API,
//! while the whole embedding model is registered as one independently evictable resource for the
//! future unified VRAM/RAM/SSD policy.

mod tokenizer;

use anyhow::{anyhow, bail, Context, Result};
use infr_core::{MemoryTier, ResourceKind, ResourceSnapshot, ResourceTracker, WeightSource};
use infr_gguf::Gguf;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{
    ffi::OsString,
    fs::{self, File},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime},
};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingConfig {
    pub name: String,
    pub architecture: String,
    pub max_context: usize,
    pub dimensions: usize,
}

impl EmbeddingConfig {
    fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let md = gguf.metadata();
        let architecture = md
            .str("general.architecture")
            .context("GGUF missing general.architecture")?
            .to_owned();
        let integer = |suffix: &str| -> Result<usize> {
            let key = format!("{architecture}.{suffix}");
            let value = md
                .u64(&key)
                .with_context(|| format!("GGUF missing {key}"))?;
            usize::try_from(value).with_context(|| format!("GGUF {key} is too large"))
        };
        let max_context = integer("context_length")?;
        let dimensions = integer("embedding_length")?;
        Ok(Self {
            name: md
                .str("general.name")
                .unwrap_or("embedding-model")
                .to_owned(),
            architecture,
            max_context,
            dimensions,
        })
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingBatch {
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: u32,
}

/// Common execution boundary for embedding implementations.
///
/// The current llama.cpp process adapter remains the compatibility/oracle implementation. A
/// native INFR graph can implement the same contract without changing the HTTP server or GUI.
pub trait EmbeddingEngine: Send + Sync {
    fn config(&self) -> &EmbeddingConfig;
    fn resource_snapshot(&self) -> ResourceSnapshot;
    fn embed(&self, inputs: &[String]) -> Result<EmbeddingBatch>;
}

#[derive(Clone, Debug)]
enum EmbeddingDevice {
    Cpu,
    Vulkan(Option<String>),
}

impl EmbeddingDevice {
    fn wants_vulkan(&self) -> bool {
        matches!(self, Self::Vulkan(_))
    }
}

/// A `llama.cpp` embedding worker owned by INFR.
///
/// The worker binds an ephemeral loopback-only port. It is never exposed to clients: INFR's own
/// `/v1/embeddings` route remains the public endpoint and applies the same auth/admission policy as
/// chat. Dropping this object terminates the worker and releases all of its RAM/VRAM allocations.
pub struct LlamaCppEmbeddingEngine {
    cfg: EmbeddingConfig,
    model_id: String,
    endpoint: String,
    client: Client,
    child: Mutex<Child>,
    log_path: PathBuf,
    runner_path: PathBuf,
    resource: Arc<ResourceTracker>,
}

impl LlamaCppEmbeddingEngine {
    pub fn load_cpu(path: &Path, cfg: Arc<infr_core::config::Config>) -> Result<Self> {
        Self::load_cpu_with_runner(path, cfg, None, 1)
    }

    pub fn load_cpu_with_runner(
        path: &Path,
        cfg: Arc<infr_core::config::Config>,
        runner: Option<&Path>,
        parallel: usize,
    ) -> Result<Self> {
        let runner = runner.or(cfg.serve.embedding_runner.as_deref());
        Self::load(path, runner, EmbeddingDevice::Cpu, parallel)
    }

    pub fn load_vulkan(path: &Path, cfg: Arc<infr_core::config::Config>) -> Result<Self> {
        Self::load_vulkan_on(path, cfg, None, None, 1)
    }

    pub fn load_vulkan_on(
        path: &Path,
        cfg: Arc<infr_core::config::Config>,
        device: Option<String>,
        runner: Option<&Path>,
        parallel: usize,
    ) -> Result<Self> {
        let runner = runner.or(cfg.serve.embedding_runner.as_deref());
        Self::load(path, runner, EmbeddingDevice::Vulkan(device), parallel)
    }

    fn load(
        path: &Path,
        runner: Option<&Path>,
        device: EmbeddingDevice,
        parallel: usize,
    ) -> Result<Self> {
        if !path.is_file() {
            bail!("embedding model does not exist: {}", path.display());
        }

        // Read metadata only, then drop the mapping before llama.cpp opens the model. This keeps
        // INFR from retaining a second full-model mapping beside the worker.
        let (cfg, logical_bytes) = {
            let gguf = Gguf::open(path).map_err(|error| anyhow!(error.to_string()))?;
            let logical_bytes = gguf.shards().iter().map(|(_, bytes)| *bytes).sum();
            (EmbeddingConfig::from_gguf(&gguf)?, logical_bytes)
        };
        let runner_path = resolve_runner(runner, device.wants_vulkan())?;
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        drop(listener);

        let log_path =
            std::env::temp_dir().join(format!("infr-embedding-{}-{port}.log", std::process::id()));
        let log = File::create(&log_path)
            .with_context(|| format!("create embedding log {}", log_path.display()))?;
        let log_err = log.try_clone()?;
        let mut args = vec![
            OsString::from("--model"),
            path.as_os_str().to_owned(),
            OsString::from("--embedding"),
            OsString::from("--pooling"),
            OsString::from("mean"),
            OsString::from("--embd-normalize"),
            OsString::from("2"),
            OsString::from("--ctx-size"),
            OsString::from(cfg.max_context.to_string()),
            OsString::from("--parallel"),
            OsString::from(parallel.max(1).to_string()),
            OsString::from("--host"),
            OsString::from("127.0.0.1"),
            OsString::from("--port"),
            OsString::from(port.to_string()),
            OsString::from("--no-webui"),
            OsString::from("--log-disable"),
        ];
        match &device {
            EmbeddingDevice::Cpu => args.extend([
                OsString::from("--device"),
                OsString::from("none"),
                OsString::from("--gpu-layers"),
                OsString::from("0"),
            ]),
            EmbeddingDevice::Vulkan(device_name) => {
                if let Some(device_name) = device_name {
                    args.extend([OsString::from("--device"), OsString::from(device_name)]);
                }
                args.extend([OsString::from("--gpu-layers"), OsString::from("all")]);
            }
        }

        let mut command = Command::new(&runner_path);
        command
            .args(&args)
            .current_dir(runner_path.parent().unwrap_or_else(|| Path::new(".")))
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err));
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let child = command.spawn().with_context(|| {
            format!("start llama.cpp embedding runner {}", runner_path.display())
        })?;
        let client = Client::builder()
            .no_proxy()
            .timeout(REQUEST_TIMEOUT)
            .build()?;
        let tier = if device.wants_vulkan() {
            MemoryTier::Vram
        } else {
            MemoryTier::Ram
        };
        let model_id = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or(&cfg.name)
            .to_owned();
        let model = Self {
            cfg,
            model_id: model_id.clone(),
            endpoint: format!("http://127.0.0.1:{port}"),
            client,
            child: Mutex::new(child),
            log_path,
            runner_path,
            resource: Arc::new(ResourceTracker::new(
                format!("embedding:{model_id}"),
                ResourceKind::EmbeddingWeights,
                logical_bytes,
                logical_bytes,
                tier,
                logical_bytes,
            )),
        };
        if let Err(error) = model.wait_until_ready() {
            model.stop_child();
            return Err(error);
        }
        tracing::info!(
            model = %model.model_id,
            architecture = %model.cfg.architecture,
            runner = %model.runner_path.display(),
            tier = ?tier,
            weights_mib = logical_bytes as f64 / 1048576.0,
            "llama.cpp embedding worker ready"
        );
        Ok(model)
    }

    pub fn config(&self) -> &EmbeddingConfig {
        &self.cfg
    }

    pub fn resource_snapshot(&self) -> ResourceSnapshot {
        self.resource.snapshot()
    }

    pub fn embed(&self, inputs: &[String]) -> Result<EmbeddingBatch> {
        if inputs.is_empty() {
            bail!("input array must contain at least one string");
        }
        let _lease = self.resource.acquire();
        let request = LlamaEmbeddingRequest {
            model: &self.model_id,
            input: inputs,
            encoding_format: "float",
        };
        let response = self
            .client
            .post(format!("{}/v1/embeddings", self.endpoint))
            .json(&request)
            .send()
            .context("call llama.cpp embedding worker")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            bail!("llama.cpp embedding worker returned {status}: {body}");
        }
        let mut response: LlamaEmbeddingResponse = response
            .json()
            .context("decode llama.cpp embedding response")?;
        response.data.sort_unstable_by_key(|row| row.index);
        if response.data.len() != inputs.len() {
            bail!(
                "llama.cpp returned {} embeddings for {} inputs",
                response.data.len(),
                inputs.len()
            );
        }
        if let Some(wrong) = response
            .data
            .iter()
            .find(|row| row.embedding.len() != self.cfg.dimensions)
        {
            bail!(
                "llama.cpp returned embedding {} with {} dimensions; expected {}",
                wrong.index,
                wrong.embedding.len(),
                self.cfg.dimensions
            );
        }
        Ok(EmbeddingBatch {
            embeddings: response.data.into_iter().map(|row| row.embedding).collect(),
            prompt_tokens: response.usage.map_or(0, |usage| usage.prompt_tokens),
        })
    }

    fn wait_until_ready(&self) -> Result<()> {
        let started = Instant::now();
        while started.elapsed() < STARTUP_TIMEOUT {
            if let Some(status) = self
                .child
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .try_wait()?
            {
                let log = fs::read_to_string(&self.log_path).unwrap_or_default();
                bail!(
                    "llama.cpp embedding worker exited during startup ({status}): {}",
                    log.trim()
                );
            }
            if self
                .client
                .get(format!("{}/health", self.endpoint))
                .timeout(Duration::from_secs(2))
                .send()
                .is_ok_and(|response| response.status().is_success())
            {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(250));
        }
        let log = fs::read_to_string(&self.log_path).unwrap_or_default();
        bail!(
            "llama.cpp embedding worker was not ready after {}s: {}",
            STARTUP_TIMEOUT.as_secs(),
            log.trim()
        )
    }

    fn stop_child(&self) {
        let mut child = self.child.lock().unwrap_or_else(|error| error.into_inner());
        if child.try_wait().ok().flatten().is_none() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl EmbeddingEngine for LlamaCppEmbeddingEngine {
    fn config(&self) -> &EmbeddingConfig {
        LlamaCppEmbeddingEngine::config(self)
    }

    fn resource_snapshot(&self) -> ResourceSnapshot {
        LlamaCppEmbeddingEngine::resource_snapshot(self)
    }

    fn embed(&self, inputs: &[String]) -> Result<EmbeddingBatch> {
        LlamaCppEmbeddingEngine::embed(self, inputs)
    }
}

impl Drop for LlamaCppEmbeddingEngine {
    fn drop(&mut self) {
        self.stop_child();
        let _ = fs::remove_file(&self.log_path);
    }
}

/// Backward-compatible name for the external llama.cpp implementation.
///
/// New code should depend on [`EmbeddingEngine`] and select an implementation explicitly.
pub type EmbeddingModel = LlamaCppEmbeddingEngine;

#[derive(Serialize)]
struct LlamaEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'static str,
}

#[derive(Deserialize)]
struct LlamaEmbeddingResponse {
    data: Vec<LlamaEmbeddingData>,
    #[serde(default)]
    usage: Option<LlamaEmbeddingUsage>,
}

#[derive(Deserialize)]
struct LlamaEmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct LlamaEmbeddingUsage {
    prompt_tokens: u32,
}

fn resolve_runner(explicit: Option<&Path>, wants_vulkan: bool) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return validate_runner(path.to_owned());
    }
    let executable_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    if let Ok(current) = std::env::current_exe() {
        if let Some(parent) = current.parent() {
            let candidate = parent.join(executable_name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    if let Some(path) = find_on_path(executable_name) {
        return Ok(path);
    }
    if cfg!(windows) {
        if let Some(path) = find_lm_studio_runner(wants_vulkan) {
            return Ok(path);
        }
    }
    bail!("llama-server was not found; pass --embedding-runner PATH or set INFR_EMBEDDING_RUNNER")
}

fn validate_runner(path: PathBuf) -> Result<PathBuf> {
    if !path.is_file() {
        bail!("embedding runner does not exist: {}", path.display());
    }
    Ok(path)
}

fn find_on_path(executable_name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|directory| directory.join(executable_name))
            .find(|candidate| candidate.is_file())
    })
}

#[cfg(windows)]
fn find_lm_studio_runner(wants_vulkan: bool) -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("USERPROFILE")?)
        .join(".lmstudio")
        .join("extensions")
        .join("backends");
    let mut candidates = fs::read_dir(root)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if !name.starts_with("llama.cpp-") || (wants_vulkan && !name.contains("vulkan")) {
                return None;
            }
            if !wants_vulkan && name.contains("vulkan") {
                return None;
            }
            let path = entry.path().join("llama-server.exe");
            let modified = path
                .metadata()
                .ok()?
                .modified()
                .unwrap_or(SystemTime::UNIX_EPOCH);
            path.is_file().then_some((modified, path))
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(|(modified, _)| *modified);
    candidates.pop().map(|(_, path)| path)
}

#[cfg(not(windows))]
fn find_lm_studio_runner(_wants_vulkan: bool) -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_missing_explicit_runner() {
        let error =
            resolve_runner(Some(Path::new("definitely-not-a-llama-server")), false).unwrap_err();
        assert!(error.to_string().contains("does not exist"));
    }

    #[test]
    fn deserializes_openai_embedding_response() {
        let response: LlamaEmbeddingResponse = serde_json::from_str(
            r#"{"data":[{"index":0,"embedding":[0.5,-0.5]}],"usage":{"prompt_tokens":3}}"#,
        )
        .unwrap();
        assert_eq!(response.data[0].embedding, [0.5, -0.5]);
        assert_eq!(response.usage.unwrap().prompt_tokens, 3);
    }
}
