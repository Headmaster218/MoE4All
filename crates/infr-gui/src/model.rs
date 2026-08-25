use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GuiState {
    pub version: u32,
    pub directories: Vec<PathBuf>,
    pub downloaded_models: Vec<PathBuf>,
    pub favorites: Vec<String>,
    pub recent: Vec<String>,
    pub profiles: Vec<ModelProfile>,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            version: 2,
            directories: Vec::new(),
            downloaded_models: Vec::new(),
            favorites: Vec::new(),
            recent: Vec::new(),
            profiles: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelProfile {
    pub id: String,
    pub name: String,
    pub model_path: String,
    /// Optional native Embedding GGUF hosted beside a chat model in the same server.
    pub embedding_model_path: String,
    /// Workload hosted by the worker.
    pub task: String,
    /// Optional llama-server compatibility executable. Empty selects native Embedding.
    pub embedding_runner: String,
    pub backend: String,
    pub context: String,
    pub ubatch: Option<usize>,
    pub kv_type_k: String,
    pub kv_type_v: String,
    pub vram_budget: String,
    pub vram_reserve: String,
    pub ram_budget: String,
    pub expert_cache: String,
    pub host_dma: bool,
    pub dram_bypass: bool,
    pub pager_stats: bool,
    pub pager_trace: String,
    pub parallel: usize,
    pub service_addr: String,
    pub service_api_key: String,
    pub max_tokens_cap: u32,
    pub extra: BTreeMap<String, String>,
}

impl Default for ModelProfile {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: "Default".into(),
            model_path: String::new(),
            embedding_model_path: String::new(),
            task: "chat".into(),
            embedding_runner: String::new(),
            backend: "Vulkan0".into(),
            context: "200k".into(),
            ubatch: None,
            kv_type_k: "q8_0".into(),
            kv_type_v: "q8_0".into(),
            vram_budget: "23g".into(),
            vram_reserve: "512m".into(),
            ram_budget: String::new(),
            expert_cache: String::new(),
            host_dma: true,
            dram_bypass: false,
            pager_stats: false,
            pager_trace: String::new(),
            parallel: 1,
            service_addr: "0.0.0.0:8080".into(),
            service_api_key: String::new(),
            max_tokens_cap: 131_072,
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub path: String,
    pub architecture: Option<String>,
    pub size_bytes: u64,
    pub trained_context: Option<usize>,
    pub layers: Option<usize>,
    pub is_moe: bool,
    pub modalities: Vec<String>,
    pub tasks: Vec<String>,
    pub projector: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceView {
    pub id: String,
    pub name: String,
    pub device_type: String,
    pub vram_bytes: u64,
    pub default: bool,
}

impl From<infr_vulkan::DeviceInfo> for DeviceView {
    fn from(value: infr_vulkan::DeviceInfo) -> Self {
        Self {
            id: format!("Vulkan{}", value.index),
            name: value.name,
            device_type: value.device_type.into(),
            vram_bytes: value.vram_bytes,
            default: value.is_default_pick,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    pub phase: String,
    pub profile_id: Option<String>,
    pub model_path: Option<String>,
    pub service_addr: Option<String>,
    pub pid: Option<u32>,
    pub started_at_ms: Option<u64>,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    pub last_error: Option<String>,
    pub memory: RuntimeMemoryStatus,
    pub logs: Vec<String>,
}

impl Default for RuntimeStatus {
    fn default() -> Self {
        Self {
            phase: "stopped".into(),
            profile_id: None,
            model_path: None,
            service_addr: None,
            pid: None,
            started_at_ms: None,
            prefill_tps: None,
            decode_tps: None,
            last_error: None,
            memory: RuntimeMemoryStatus::default(),
            logs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RuntimeMemoryStatus {
    pub kv_layout: Option<String>,
    pub context_tokens: Option<u64>,
    pub expert_cache_target_bytes: Option<u64>,
    pub elastic_pool_bytes: Option<u64>,
    pub unified_arena_bytes: Option<u64>,
    pub host_mode: Option<String>,
    pub host_cache_bytes: Option<u64>,
    pub expert_payload_bytes: Option<u64>,
    pub host_dma_imported_bytes: Option<u64>,
    pub host_dma_total_bytes: Option<u64>,
    pub host_dma_arenas: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadStatus {
    pub phase: String,
    pub model_ref: Option<String>,
    pub endpoint: Option<String>,
    pub downloaded_path: Option<String>,
    pub last_error: Option<String>,
    pub logs: Vec<String>,
}

impl Default for DownloadStatus {
    fn default() -> Self {
        Self {
            phase: "idle".into(),
            model_ref: None,
            endpoint: None,
            downloaded_path: None,
            last_error: None,
            logs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigField {
    pub path: String,
    pub default_value: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Bootstrap {
    pub saved: GuiState,
    pub catalog: Vec<ModelInfo>,
    pub devices: Vec<DeviceView>,
    pub runtime: RuntimeStatus,
    pub download: DownloadStatus,
    pub config_schema: Vec<ConfigField>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StatusSnapshot {
    pub runtime: RuntimeStatus,
    pub download: DownloadStatus,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryEstimate {
    pub model_bytes: u64,
    pub fixed_vram_bytes: Option<u64>,
    pub expert_payload_bytes: Option<u64>,
    pub requested_ram_budget_bytes: Option<u64>,
    pub effective_ram_cache_bytes: Option<u64>,
    pub host_cache_mode: Option<String>,
    pub host_cache_coverage: Option<f64>,
    pub fits_ram_budget: Option<bool>,
    pub kv_bytes: Option<u64>,
    pub runtime_reserve_bytes: u64,
    pub weight_packing_margin_bytes: u64,
    pub load_driver_reserve_bytes: u64,
    pub post_load_reserve_bytes: u64,
    pub total_vram_bytes: Option<u64>,
    pub requested_vram_budget_bytes: Option<u64>,
    pub effective_vram_budget_bytes: Option<u64>,
    pub estimated_cache_room_bytes: Option<u64>,
    pub elastic_pool_bytes: Option<u64>,
    pub embedding_model_bytes: Option<u64>,
    pub trained_context: Option<usize>,
    pub architecture: Option<String>,
    pub is_moe: bool,
    pub fits_minimum: Option<bool>,
    pub confidence: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct DirectoryRequest {
    pub path: String,
}

#[derive(Debug, Deserialize)]
pub struct FavoriteRequest {
    pub path: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct StopRequest {
    pub force: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DownloadRequest {
    pub model_ref: String,
    pub endpoint: String,
    pub jobs: usize,
}

impl Default for DownloadRequest {
    fn default() -> Self {
        Self {
            model_ref: String::new(),
            endpoint: "https://huggingface.co".into(),
            jobs: 8,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ApiMessage {
    pub ok: bool,
    pub message: String,
}

impl ApiMessage {
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            ok: true,
            message: message.into(),
        }
    }
}

pub fn load_state(path: &Path) -> anyhow::Result<GuiState> {
    if !path.exists() {
        return Ok(GuiState::default());
    }
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut state: GuiState =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    if state.version < 2 {
        for profile in &mut state.profiles {
            migrate_pager_controls(profile);
        }
    }
    state.version = GuiState::default().version;
    Ok(state)
}

fn migrate_pager_controls(profile: &mut ModelProfile) {
    if let Some(value) = profile.extra.remove("paging.host_dma") {
        profile.host_dma = parse_saved_bool(&value).unwrap_or(profile.host_dma);
    }
    if let Some(value) = profile.extra.remove("paging.dram_bypass") {
        profile.dram_bypass = parse_saved_bool(&value).unwrap_or(profile.dram_bypass);
    }
    if let Some(value) = profile.extra.remove("paging.stats") {
        profile.pager_stats = parse_saved_bool(&value).unwrap_or(profile.pager_stats);
    }
    if let Some(value) = profile.extra.remove("paging.trace") {
        if profile.pager_trace.is_empty() {
            profile.pager_trace = value;
        }
    }
}

fn parse_saved_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

pub fn save_state(path: &Path, state: &GuiState) -> anyhow::Result<()> {
    let bytes = serde_json::to_vec_pretty(state)?;
    let tmp = path.with_extension("json.new");
    let backup = path.with_extension("json.bak");
    fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    if path.exists() {
        let _ = fs::copy(path, &backup);
        fs::remove_file(path).with_context(|| format!("replacing {}", path.display()))?;
    }
    fs::rename(&tmp, path).with_context(|| format!("installing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "infr-gui-state-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn state_round_trips_without_losing_profiles() {
        let root = temp_root();
        let path = root.join("state.json");
        let mut state = GuiState::default();
        state.directories.push(PathBuf::from(r"D:\Models"));
        state.favorites.push(r"D:\Models\model.gguf".into());
        let profile = ModelProfile {
            id: "balanced-q8".into(),
            model_path: r"D:\Models\model.gguf".into(),
            ..ModelProfile::default()
        };
        state.profiles.push(profile);

        save_state(&path, &state).unwrap();
        let loaded = load_state(&path).unwrap();

        assert_eq!(loaded.directories, state.directories);
        assert_eq!(loaded.favorites, state.favorites);
        assert_eq!(loaded.profiles.len(), 1);
        assert_eq!(loaded.profiles[0].id, "balanced-q8");
        assert!(loaded.profiles[0].host_dma);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn version_one_profiles_gain_current_pager_defaults() {
        let state: GuiState = serde_json::from_str(
            r#"{
                "version": 1,
                "profiles": [{"id":"old","name":"Old","model_path":"model.gguf"}]
            }"#,
        )
        .unwrap();

        assert_eq!(state.profiles.len(), 1);
        assert!(state.profiles[0].host_dma);
        assert!(!state.profiles[0].dram_bypass);
        assert!(state.profiles[0].embedding_model_path.is_empty());
    }

    #[test]
    fn load_migrates_legacy_advanced_pager_controls() {
        let root = temp_root();
        let path = root.join("state.json");
        fs::write(
            &path,
            r#"{
                "version": 1,
                "profiles": [{
                    "id":"old",
                    "name":"Old",
                    "model_path":"model.gguf",
                    "extra": {
                        "paging.host_dma":"false",
                        "paging.dram_bypass":"1",
                        "paging.stats":"true",
                        "paging.trace":"old-pager.csv"
                    }
                }]
            }"#,
        )
        .unwrap();

        let state = load_state(&path).unwrap();
        let profile = &state.profiles[0];
        assert_eq!(state.version, 2);
        assert!(!profile.host_dma);
        assert!(profile.dram_bypass);
        assert!(profile.pager_stats);
        assert_eq!(profile.pager_trace, "old-pager.csv");
        assert!(profile.extra.is_empty());
        fs::remove_dir_all(root).unwrap();
    }
}
