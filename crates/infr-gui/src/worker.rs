use std::{
    collections::BTreeMap,
    io::Write as _,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use infr_core::config::{ConfigLayer, ConfigOverrides};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, BufReader},
    process::{Child, Command},
    sync::{Mutex, RwLock},
};

use crate::{
    model::{DownloadRequest, DownloadStatus, ModelProfile, RuntimeStatus},
    AppState,
};

const STOP_WAIT: Duration = Duration::from_secs(660);
const LOG_LINES: usize = 240;

struct ManagedWorker {
    child: Child,
    stop_file: PathBuf,
    stop_requested: bool,
    pid: u32,
}

#[derive(Default)]
pub struct WorkerManager {
    lifecycle: Mutex<()>,
    generation: AtomicU64,
    child: Mutex<Option<ManagedWorker>>,
    status: Arc<RwLock<RuntimeStatus>>,
}

impl WorkerManager {
    pub async fn status(&self) -> RuntimeStatus {
        self.status.read().await.clone()
    }

    pub async fn switch(
        &self,
        app: &AppState,
        profile: ModelProfile,
        infr: &Path,
        workdir: &Path,
        data_dir: &Path,
    ) -> Result<(), String> {
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        let _lifecycle = self.lifecycle.lock().await;
        if self.generation.load(Ordering::Acquire) != generation {
            return Err("worker 启动已被更新的操作取消".into());
        }
        if self.is_running().await? {
            self.stop_inner(false).await?;
            self.wait_until_stopped().await?;
        }
        if self.generation.load(Ordering::Acquire) != generation {
            return Err("worker 启动已取消".into());
        }
        self.spawn(app, profile, infr, workdir, data_dir).await?;
        if self.generation.load(Ordering::Acquire) != generation {
            self.stop_inner(false).await?;
            return Err("worker 启动已取消".into());
        }
        Ok(())
    }

    async fn spawn(
        &self,
        app: &AppState,
        profile: ModelProfile,
        infr: &Path,
        workdir: &Path,
        data_dir: &Path,
    ) -> Result<(), String> {
        if infr.is_absolute() && !infr.is_file() {
            return Err(format!("找不到 infr worker：{}", infr.display()));
        }
        let service_addr: std::net::SocketAddr = profile
            .service_addr
            .parse()
            .map_err(|e| format!("服务地址无效：{e}"))?;
        let port_guard = std::net::TcpListener::bind(service_addr)
            .map_err(|e| format!("服务地址 {service_addr} 当前不可用：{e}"))?;
        drop(port_guard);
        let stamp = unix_ms();
        let stop_file = data_dir.join(format!("worker-{stamp}.stop"));
        let log_file = std::fs::File::create(data_dir.join(format!("worker-{stamp}.log")))
            .map_err(|e| format!("创建 worker 日志失败：{e}"))?;
        let log_file = Arc::new(std::sync::Mutex::new(log_file));
        let _ = std::fs::remove_file(&stop_file);

        let mut command = Command::new(infr);
        command.current_dir(workdir);
        command.env_remove("INFR_API_KEY");
        if !profile.service_api_key.is_empty() {
            command.env("INFR_API_KEY", &profile.service_api_key);
        }
        for (path, value) in profile_settings(&profile, &stop_file)? {
            command.arg("--set").arg(format!("{path}={value}"));
        }
        match profile.task.as_str() {
            "embedding" => {
                command.arg("serve-embedding").arg(&profile.model_path);
                if !profile.embedding_runner.trim().is_empty() {
                    command
                        .arg("--embedding-runner")
                        .arg(profile.embedding_runner.trim());
                }
            }
            _ => {
                command.arg("serve").arg(&profile.model_path);
                if !profile.embedding_model_path.trim().is_empty() {
                    command
                        .arg("--embedding-model")
                        .arg(profile.embedding_model_path.trim());
                    if !profile.embedding_runner.trim().is_empty() {
                        command
                            .arg("--embedding-runner")
                            .arg(profile.embedding_runner.trim());
                    }
                }
            }
        }
        command
            .arg("--addr")
            .arg(&profile.service_addr)
            .arg("--parallel")
            .arg(profile.parallel.max(1).to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        hide_child_window(&mut command);

        let mut child = command
            .spawn()
            .map_err(|e| format!("启动 infr worker 失败：{e}"))?;
        let pid = child.id().unwrap_or(0);
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        {
            let mut status = self.status.write().await;
            *status = RuntimeStatus {
                phase: "loading".into(),
                profile_id: Some(profile.id.clone()),
                model_path: Some(profile.model_path.clone()),
                service_addr: Some(profile.service_addr.clone()),
                pid: Some(pid),
                started_at_ms: Some(stamp),
                prefill_tps: None,
                decode_tps: None,
                last_error: None,
                memory: Default::default(),
                logs: Vec::new(),
            };
        }
        *self.child.lock().await = Some(ManagedWorker {
            child,
            stop_file,
            stop_requested: false,
            pid,
        });

        if let Some(stdout) = stdout {
            spawn_worker_log_reader(self.status.clone(), stdout, Arc::clone(&log_file));
        }
        if let Some(stderr) = stderr {
            spawn_worker_log_reader(self.status.clone(), stderr, log_file);
        }
        spawn_ready_probe(app.clone(), service_addr, pid);
        Ok(())
    }

    pub async fn stop(&self, force: bool) -> Result<(), String> {
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.stop_inner(force).await
    }

    async fn stop_inner(&self, force: bool) -> Result<(), String> {
        let mut slot = self.child.lock().await;
        let Some(worker) = slot.as_mut() else {
            self.status.write().await.phase = "stopped".into();
            return Ok(());
        };
        if force {
            worker.stop_requested = true;
            self.status.write().await.phase = "stopping".into();
            worker
                .child
                .kill()
                .await
                .map_err(|e| format!("强制停止 worker 失败：{e}"))?;
        } else {
            if let Some(parent) = worker.stop_file.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&worker.stop_file, b"stop\n")
                .map_err(|e| format!("写入停止触发文件失败：{e}"))?;
            worker.stop_requested = true;
            self.status.write().await.phase = "stopping".into();
        }
        Ok(())
    }

    async fn wait_until_stopped(&self) -> Result<(), String> {
        let started = tokio::time::Instant::now();
        while self.is_running().await? {
            if started.elapsed() >= STOP_WAIT {
                return Err(format!(
                    "worker 在 {} 秒内未完成 GPU 排空；请查看日志，必要时手动强制停止",
                    STOP_WAIT.as_secs()
                ));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        Ok(())
    }

    pub async fn stop_and_wait(&self) -> Result<(), String> {
        self.generation.fetch_add(1, Ordering::AcqRel);
        if self.is_running().await? {
            self.stop_inner(false).await?;
            self.wait_until_stopped().await?;
        }
        let _lifecycle = self.lifecycle.lock().await;
        if self.is_running().await? {
            self.stop_inner(false).await?;
            self.wait_until_stopped().await?;
        }
        Ok(())
    }

    async fn is_running(&self) -> Result<bool, String> {
        let mut slot = self.child.lock().await;
        let Some(worker) = slot.as_mut() else {
            return Ok(false);
        };
        match worker.child.try_wait() {
            Ok(None) => Ok(true),
            Ok(Some(_)) => {
                let stop_file = worker.stop_file.clone();
                *slot = None;
                let _ = std::fs::remove_file(stop_file);
                Ok(false)
            }
            Err(e) => Err(format!("查询 worker 状态失败：{e}")),
        }
    }

    async fn reap(&self) {
        let exited = {
            let mut slot = self.child.lock().await;
            let result = match slot.as_mut() {
                Some(worker) => match worker.child.try_wait() {
                    Ok(Some(code)) => Some((
                        worker.pid,
                        worker.stop_requested,
                        worker.stop_file.clone(),
                        Ok(code),
                    )),
                    Ok(None) => None,
                    Err(e) => Some((
                        worker.pid,
                        worker.stop_requested,
                        worker.stop_file.clone(),
                        Err(e),
                    )),
                },
                None => None,
            };
            if result.is_some() {
                *slot = None;
            }
            result
        };
        let Some((_pid, requested, stop_file, result)) = exited else {
            return;
        };
        let _ = std::fs::remove_file(stop_file);
        let mut status = self.status.write().await;
        status.pid = None;
        match result {
            Ok(code) if requested => {
                status.phase = "stopped".into();
                push_log(&mut status.logs, format!("worker stopped ({code})"));
            }
            Ok(code) if code.success() => {
                status.phase = "stopped".into();
                push_log(&mut status.logs, format!("worker exited ({code})"));
            }
            Ok(code) => {
                status.phase = "failed".into();
                let exit = format!("worker exited unexpectedly: {code}");
                if status.last_error.is_none() {
                    status.last_error = Some(exit.clone());
                }
                push_log(&mut status.logs, exit);
            }
            Err(e) => {
                status.phase = "failed".into();
                status.last_error = Some(format!("worker status failed: {e}"));
            }
        }
    }
}

#[derive(Default)]
pub struct DownloadManager {
    running: AtomicBool,
    status: Arc<RwLock<DownloadStatus>>,
}

impl DownloadManager {
    pub async fn status(&self) -> DownloadStatus {
        self.status.read().await.clone()
    }

    pub async fn start(
        &self,
        app: AppState,
        req: DownloadRequest,
        infr: &Path,
        workdir: &Path,
    ) -> Result<(), String> {
        if self
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err("已有模型正在下载".into());
        }
        if req.model_ref.trim().is_empty() {
            self.running.store(false, Ordering::Release);
            return Err("模型引用不能为空".into());
        }
        let endpoint = req.endpoint.trim().trim_end_matches('/').to_string();
        if !(endpoint.starts_with("https://") || endpoint.starts_with("http://")) {
            self.running.store(false, Ordering::Release);
            return Err("下载源必须是 http:// 或 https:// 地址".into());
        }
        if infr.is_absolute() && !infr.is_file() {
            self.running.store(false, Ordering::Release);
            return Err(format!("找不到 infr：{}", infr.display()));
        }
        {
            let mut status = self.status.write().await;
            *status = DownloadStatus {
                phase: "downloading".into(),
                model_ref: Some(req.model_ref.clone()),
                endpoint: Some(endpoint.clone()),
                downloaded_path: None,
                last_error: None,
                logs: Vec::new(),
            };
        }

        let infr = infr.to_path_buf();
        let workdir = workdir.to_path_buf();
        tokio::spawn(async move {
            let result = run_download(&app, &infr, &workdir, &req, &endpoint).await;
            app.inner.download.running.store(false, Ordering::Release);
            match result {
                Ok(path) => {
                    {
                        let mut saved = app.inner.saved.write().await;
                        let p = PathBuf::from(&path);
                        if !saved.downloaded_models.iter().any(|v| v == &p) {
                            saved.downloaded_models.push(p);
                        }
                        if let Err(e) = crate::model::save_state(&app.inner.state_file, &saved) {
                            let mut status = app.inner.download.status.write().await;
                            status.phase = "failed".into();
                            status.last_error = Some(e.to_string());
                            return;
                        }
                    }
                    let _ = crate::rescan(&app).await;
                    let mut status = app.inner.download.status.write().await;
                    status.phase = "completed".into();
                    status.downloaded_path = Some(path);
                }
                Err(e) => {
                    let mut status = app.inner.download.status.write().await;
                    status.phase = "failed".into();
                    status.last_error = Some(e);
                }
            }
        });
        Ok(())
    }
}

async fn run_download(
    app: &AppState,
    infr: &Path,
    workdir: &Path,
    req: &DownloadRequest,
    endpoint: &str,
) -> Result<String, String> {
    let mut command = Command::new(infr);
    command
        .current_dir(workdir)
        .arg("--set")
        .arg(format!("hub.endpoint={endpoint}"))
        .arg("--set")
        .arg(format!("hub.pull_jobs={}", req.jobs.max(1)))
        .arg("pull")
        .arg(req.model_ref.trim())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_child_window(&mut command);
    let mut child = command.spawn().map_err(|e| format!("启动下载失败：{e}"))?;
    let stdout = child.stdout.take().ok_or("无法捕获下载输出")?;
    let stderr = child.stderr.take().ok_or("无法捕获下载日志")?;
    let out_state = app.inner.download.status.clone();
    let err_state = out_state.clone();
    let out_task = tokio::spawn(read_download_stream(out_state, stdout, true));
    let err_task = tokio::spawn(read_download_stream(err_state, stderr, false));
    let exit = child.wait().await.map_err(|e| e.to_string())?;
    let stdout_lines = out_task.await.map_err(|e| e.to_string())?;
    let _ = err_task.await;
    if !exit.success() {
        return Err(format!("下载进程退出：{exit}"));
    }
    stdout_lines
        .iter()
        .rev()
        .find(|line| !line.trim().is_empty())
        .map(|s| s.trim().to_string())
        .ok_or_else(|| "下载完成，但 infr 没有返回本地模型路径".into())
}

async fn read_download_stream<R: AsyncRead + Unpin>(
    status: Arc<RwLock<DownloadStatus>>,
    stream: R,
    collect: bool,
) -> Vec<String> {
    let mut lines = BufReader::new(stream).lines();
    let mut captured = Vec::new();
    while let Ok(Some(line)) = lines.next_line().await {
        if collect {
            captured.push(line.clone());
        }
        push_log(&mut status.write().await.logs, line);
    }
    captured
}

pub fn validate_profile(profile: &ModelProfile) -> Result<(), String> {
    if profile.id.trim().is_empty() || profile.name.trim().is_empty() {
        return Err("配置 ID 和名称不能为空".into());
    }
    if profile.model_path.trim().is_empty() {
        return Err("请选择模型".into());
    }
    if profile.model_path.contains(['\r', '\n'])
        || profile.embedding_model_path.contains(['\r', '\n'])
        || profile.embedding_runner.contains(['\r', '\n'])
        || profile.service_api_key.contains(['\r', '\n'])
        || profile.pager_trace.contains(['\r', '\n'])
    {
        return Err("模型路径、Pager trace、Embedding runner 和 API Key 不能包含换行".into());
    }
    if !matches!(profile.task.as_str(), "chat" | "completion" | "embedding") {
        return Err(format!(
            "任务类型 `{}` 已预留，但当前推理内核尚未实现",
            profile.task
        ));
    }
    if profile.task == "embedding"
        && !profile.embedding_runner.trim().is_empty()
        && !Path::new(profile.embedding_runner.trim()).is_file()
    {
        return Err(format!(
            "找不到 Embedding runner：{}",
            profile.embedding_runner.trim()
        ));
    }
    profile
        .service_addr
        .parse::<std::net::SocketAddr>()
        .map_err(|e| format!("服务地址无效：{e}"))?;
    if profile.parallel == 0 || profile.parallel > 64 {
        return Err("并发槽必须在 1..=64".into());
    }
    let fake_stop = Path::new("worker.stop");
    let settings = profile_settings(profile, fake_stop)?;
    let overrides = ConfigOverrides {
        config_path: None,
        sets: settings
            .into_iter()
            .map(|(path, value)| format!("{path}={value}"))
            .collect(),
        flags: Default::default(),
    };
    ConfigLayer::cli(&overrides).map_err(|e| e.to_string())?;
    Ok(())
}

fn profile_settings(
    profile: &ModelProfile,
    stop_file: &Path,
) -> Result<BTreeMap<String, String>, String> {
    let mut values = BTreeMap::new();
    insert_nonempty(&mut values, "device.dev", &profile.backend);
    if matches!(profile.task.as_str(), "chat" | "completion") {
        insert_nonempty(&mut values, "device.ctx", &profile.context);
        insert_nonempty(&mut values, "device.vram_budget", &profile.vram_budget);
        insert_nonempty(&mut values, "device.vram_reserve", &profile.vram_reserve);
        if let Some(ubatch) = profile.ubatch {
            values.insert("device.ubatch".into(), ubatch.to_string());
        }
        if !profile.kv_type_k.eq_ignore_ascii_case("auto") {
            insert_nonempty(&mut values, "kv.type_k", &profile.kv_type_k);
        }
        if !profile.kv_type_v.eq_ignore_ascii_case("auto") {
            insert_nonempty(&mut values, "kv.type_v", &profile.kv_type_v);
        }
        insert_nonempty(&mut values, "paging.dram", &profile.ram_budget);
        insert_nonempty(&mut values, "paging.cache", &profile.expert_cache);
        values.insert("paging.host_dma".into(), profile.host_dma.to_string());
        values.insert("paging.dram_bypass".into(), profile.dram_bypass.to_string());
        values.insert("paging.stats".into(), profile.pager_stats.to_string());
        insert_nonempty(&mut values, "paging.trace", &profile.pager_trace);
        values.insert(
            "serve.max_tokens_cap".into(),
            profile.max_tokens_cap.max(1).to_string(),
        );
    }
    values.insert("serve.stats_interval_secs".into(), "1".into());
    values.insert(
        "serve.shutdown_file".into(),
        stop_file.to_string_lossy().into_owned(),
    );
    for (key, value) in &profile.extra {
        if matches!(key.as_str(), "serve.shutdown_file" | "serve.api_key") {
            return Err(format!("{key} 由 GUI 管理，不能在高级参数中覆盖"));
        }
        if values.contains_key(key) {
            return Err(format!("高级参数 `{key}` 与基础参数重复"));
        }
        if key.contains(['\r', '\n']) || value.contains(['\r', '\n']) {
            return Err(format!("高级参数 `{key}` 不能包含换行"));
        }
        values.insert(key.clone(), value.clone());
    }
    Ok(values)
}

fn insert_nonempty(values: &mut BTreeMap<String, String>, key: &str, value: &str) {
    if !value.trim().is_empty() {
        values.insert(key.into(), value.trim().into());
    }
}

fn spawn_worker_log_reader<R>(
    status: Arc<RwLock<RuntimeStatus>>,
    stream: R,
    log_file: Arc<std::sync::Mutex<std::fs::File>>,
) where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut reader = BufReader::new(stream);
        let mut bytes = Vec::new();
        loop {
            bytes.clear();
            match reader.read_until(b'\n', &mut bytes).await {
                Ok(0) => break,
                Ok(_) => {
                    let line = normalize_worker_log(&bytes);
                    if let Ok(mut file) = log_file.lock() {
                        let _ = writeln!(file, "{line}");
                    }
                    record_worker_log(&status, line).await;
                }
                Err(e) => {
                    record_worker_log(&status, format!("worker log read failed: {e}")).await;
                    break;
                }
            }
        }
    });
}

async fn record_worker_log(status: &RwLock<RuntimeStatus>, line: String) {
    let mut state = status.write().await;
    if let Some(v) = metric(&line, "prefill_tps=") {
        state.prefill_tps = Some(v);
    }
    if let Some(v) = metric(&line, "decode_tps=") {
        state.decode_tps = Some(v);
    }
    if line.to_ascii_lowercase().contains("error") {
        state.last_error = Some(line.clone());
    }
    update_memory_status(&mut state, &line);
    push_log(&mut state.logs, line);
}

fn update_memory_status(state: &mut RuntimeStatus, line: &str) {
    if line.contains("VRAM plan:") {
        state.memory.expert_cache_target_bytes =
            metric(line, "expert_cache_target=").map(decimal_gb_bytes);
        state.memory.elastic_pool_bytes = metric(line, "elastic_pool=").map(decimal_gb_bytes);
        state.memory.context_tokens = metric(line, "ctx=").map(|value| value as u64);
        state.memory.kv_layout = line
            .split_once('(')
            .and_then(|(_, tail)| tail.rsplit_once(')').map(|(inside, _)| inside))
            .and_then(|inside| {
                inside
                    .rsplit_once(", ctx=")
                    .map(|(layout, _)| layout.trim())
            })
            .map(str::to_string);
    }
    if line.contains("MoE host plan: bounded inclusive RAM cache") {
        state.memory.host_mode = Some("bounded".into());
        state.memory.host_cache_bytes =
            metric(line, "bounded inclusive RAM cache ").map(decimal_gb_bytes);
        state.memory.expert_payload_bytes = metric(line, " GB / ").map(decimal_gb_bytes);
    } else if line.contains("MoE host plan: full layer-contiguous RAM store") {
        state.memory.host_mode = Some("full".into());
        state.memory.host_cache_bytes =
            metric(line, "full layer-contiguous RAM store ").map(decimal_gb_bytes);
        state.memory.expert_payload_bytes = state.memory.host_cache_bytes;
    }
    if line.contains("host DMA import total:") {
        if let Some((imported, total)) = ratio_after(line, "host DMA import total:") {
            state.memory.host_dma_imported_bytes = Some(gib_bytes(imported));
            state.memory.host_dma_total_bytes = Some(gib_bytes(total));
        }
        state.memory.host_dma_arenas = line
            .split_once(" across ")
            .and_then(|(_, tail)| tail.split_once(" arena(s)").map(|(value, _)| value.trim()))
            .map(str::to_string);
    } else if line.contains("host DMA import unavailable") || line.contains("host DMA disabled") {
        state.memory.host_dma_imported_bytes = Some(0);
    }
    if let Some(bytes) = integer_after(line, "unified VRAM arena: ") {
        state.memory.unified_arena_bytes = Some(bytes);
    }
    if line.contains("unified VRAM ready") {
        if let Some(bytes) = integer_after(line, "arena_bytes=") {
            state.memory.unified_arena_bytes = Some(bytes);
        }
    }
}

fn decimal_gb_bytes(value: f64) -> u64 {
    (value * 1_000_000_000.0).round() as u64
}

fn gib_bytes(value: f64) -> u64 {
    (value * (1u64 << 30) as f64).round() as u64
}

fn ratio_after(line: &str, marker: &str) -> Option<(f64, f64)> {
    let tail = line.split_once(marker)?.1.trim_start();
    let token = tail.split_whitespace().next()?;
    let (left, right) = token.split_once('/')?;
    Some((left.parse().ok()?, right.parse().ok()?))
}

fn integer_after(line: &str, marker: &str) -> Option<u64> {
    let tail = line.split_once(marker)?.1.trim_start();
    let number: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    number.parse().ok()
}

fn normalize_worker_log(bytes: &[u8]) -> String {
    let decoded = String::from_utf8_lossy(bytes);
    let line = decoded.trim_end_matches(&['\r', '\n'][..]);
    strip_terminal_sequences(line)
}

/// Remove terminal-only control sequences before logs are rendered in the browser.
///
/// Tracing colors numeric field values independently, so leaving ANSI in a piped log is not only
/// visual noise: an escape can occur directly after `prefill_tps=` and make metric parsing fail.
fn strip_terminal_sequences(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut clean = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b {
            i += 1;
            if i == bytes.len() {
                break;
            }
            match bytes[i] {
                b'[' => {
                    // CSI: parameters/intermediates followed by one final byte in 0x40..=0x7e.
                    i += 1;
                    while i < bytes.len() {
                        let final_byte = (0x40..=0x7e).contains(&bytes[i]);
                        i += 1;
                        if final_byte {
                            break;
                        }
                    }
                }
                b']' => {
                    // OSC: terminated by BEL or the two-byte ST sequence ESC + backslash.
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == 0x07 {
                            i += 1;
                            break;
                        }
                        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'\\' {
                            i += 2;
                            break;
                        }
                        i += 1;
                    }
                }
                _ => i += 1,
            }
        } else if bytes[i] < 0x20 || bytes[i] == 0x7f {
            // A browser log has no terminal state. Keep tabs useful for alignment, drop the rest.
            if bytes[i] == b'\t' {
                clean.push(bytes[i]);
            }
            i += 1;
        } else {
            clean.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(clean).expect("removing ASCII controls preserves valid UTF-8")
}

fn metric(line: &str, marker: &str) -> Option<f64> {
    let tail = line.split_once(marker)?.1.trim_start();
    let number: String = tail
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    number.parse().ok()
}

fn spawn_ready_probe(app: AppState, service_addr: std::net::SocketAddr, pid: u32) {
    tokio::spawn(async move {
        let host = if service_addr.ip().is_unspecified() {
            match service_addr.ip() {
                std::net::IpAddr::V4(_) => "127.0.0.1".to_string(),
                std::net::IpAddr::V6(_) => "[::1]".to_string(),
            }
        } else if service_addr.is_ipv6() {
            format!("[{}]", service_addr.ip())
        } else {
            service_addr.ip().to_string()
        };
        let url = format!("http://{host}:{}/health", service_addr.port());
        let client = reqwest::Client::new();
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let status = app.inner.worker.status().await;
            if status.pid != Some(pid) || !matches!(status.phase.as_str(), "loading" | "ready") {
                break;
            }
            if client
                .get(&url)
                .timeout(Duration::from_secs(1))
                .send()
                .await
                .is_ok_and(|r| r.status().is_success())
            {
                app.inner.worker.status.write().await.phase = "ready".into();
                break;
            }
        }
    });
}

pub fn spawn_monitor(app: AppState) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            app.inner.worker.reap().await;
        }
    });
}

fn push_log(lines: &mut Vec<String>, line: String) {
    lines.push(line);
    if lines.len() > LOG_LINES {
        lines.drain(..lines.len() - LOG_LINES);
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(windows)]
fn hide_child_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.as_std_mut().creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_child_window(_command: &mut Command) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_parser_reads_tracing_fields() {
        assert_eq!(
            metric(
                "request done prefill_tps=523.4 decode_tps=56.7",
                "prefill_tps="
            ),
            Some(523.4)
        );
        assert_eq!(
            metric(
                "request done prefill_tps=523.4 decode_tps=56.7",
                "decode_tps="
            ),
            Some(56.7)
        );
    }

    #[test]
    fn worker_log_normalization_preserves_utf8_and_exposes_colored_metrics() {
        let line = normalize_worker_log(
            b"\x1b[2mrequest done\x1b[0m prefill_tps=\x1b[3m523.4\x1b[0m decode_tps=\x1b[3m56.7\x1b[0m \xe6\xa8\xa1\xe5\x9e\x8b\r\n",
        );
        assert_eq!(line, "request done prefill_tps=523.4 decode_tps=56.7 模型");
        assert_eq!(metric(&line, "prefill_tps="), Some(523.4));
        assert_eq!(metric(&line, "decode_tps="), Some(56.7));
    }

    #[test]
    fn worker_log_normalization_survives_invalid_utf8_and_osc() {
        let line = normalize_worker_log(b"ok \xff\x1b]0;title\x07 done\n");
        assert_eq!(line, "ok � done");
    }

    #[test]
    fn defaults_validate() {
        let p = ModelProfile {
            id: "default".into(),
            model_path: "model.gguf".into(),
            ..ModelProfile::default()
        };
        validate_profile(&p).unwrap();
    }

    #[test]
    fn embedding_profile_uses_only_relevant_runtime_settings() {
        let p = ModelProfile {
            id: "embedding".into(),
            model_path: "embedding.gguf".into(),
            task: "embedding".into(),
            ..ModelProfile::default()
        };
        validate_profile(&p).unwrap();
        let settings = profile_settings(&p, Path::new("worker.stop")).unwrap();
        assert_eq!(
            settings.get("device.dev").map(String::as_str),
            Some("Vulkan0")
        );
        assert!(!settings.contains_key("kv.type_k"));
        assert!(!settings.contains_key("paging.cache"));
        assert!(!settings.contains_key("serve.max_tokens_cap"));
    }

    #[test]
    fn chat_profile_carries_the_current_host_pager_controls() {
        let p = ModelProfile {
            id: "large-moe".into(),
            model_path: "model.gguf".into(),
            host_dma: false,
            dram_bypass: true,
            pager_stats: true,
            pager_trace: "pager.csv".into(),
            ..ModelProfile::default()
        };
        let settings = profile_settings(&p, Path::new("worker.stop")).unwrap();

        assert_eq!(
            settings.get("paging.host_dma").map(String::as_str),
            Some("false")
        );
        assert_eq!(
            settings.get("paging.dram_bypass").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            settings.get("paging.stats").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            settings.get("paging.trace").map(String::as_str),
            Some("pager.csv")
        );
    }

    #[test]
    fn startup_logs_populate_the_runtime_memory_summary() {
        let mut status = RuntimeStatus::default();
        update_memory_status(
            &mut status,
            "VRAM plan: total_room=19.86 GB fixed=5.20 GB state=0.16 GB \
             runtime_elastic=0.36 GB packing_margin=0.27 GB load_driver=0.00 GB \
             post_load=0.27 GB expert_cache_target=13.61 GB elastic_pool=13.97 GB \
             (k=F16, v=F16, ctx=145)",
        );
        update_memory_status(
            &mut status,
            "MoE host plan: bounded inclusive RAM cache 48.32 GB / 72.36 GB expert payload",
        );
        update_memory_status(
            &mut status,
            "[infr] host DMA import total: 28.99/45.00 GiB across 3/3 arena(s)",
        );
        update_memory_status(
            &mut status,
            "[infr] unified VRAM arena: 13966884864 bytes across 7 mapped ReBAR shard(s)",
        );

        assert_eq!(status.memory.kv_layout.as_deref(), Some("k=F16, v=F16"));
        assert_eq!(status.memory.context_tokens, Some(145));
        assert_eq!(
            status.memory.expert_cache_target_bytes,
            Some(13_610_000_000)
        );
        assert_eq!(status.memory.elastic_pool_bytes, Some(13_970_000_000));
        assert_eq!(status.memory.host_mode.as_deref(), Some("bounded"));
        assert_eq!(status.memory.host_cache_bytes, Some(48_320_000_000));
        assert_eq!(status.memory.expert_payload_bytes, Some(72_360_000_000));
        assert_eq!(status.memory.host_dma_arenas.as_deref(), Some("3/3"));
        assert_eq!(status.memory.unified_arena_bytes, Some(13_966_884_864));
    }

    #[test]
    fn secrets_stay_out_of_worker_command_line_settings() {
        let p = ModelProfile {
            id: "default".into(),
            model_path: "model.gguf".into(),
            service_api_key: "do-not-put-this-in-argv".into(),
            ..ModelProfile::default()
        };
        let settings = profile_settings(&p, Path::new("worker.stop")).unwrap();
        assert!(!settings.contains_key("serve.api_key"));
        assert_eq!(
            settings.get("serve.shutdown_file").map(String::as_str),
            Some("worker.stop")
        );
    }

    #[test]
    fn supervisor_owned_settings_cannot_be_overridden() {
        for key in ["serve.api_key", "serve.shutdown_file"] {
            let mut p = ModelProfile {
                id: "default".into(),
                model_path: "model.gguf".into(),
                ..ModelProfile::default()
            };
            p.extra.insert(key.into(), "override".into());
            assert!(validate_profile(&p).unwrap_err().contains("由 GUI 管理"));
        }
    }
}
