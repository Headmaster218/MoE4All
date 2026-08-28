use std::{collections::BTreeMap, fs, path::Path};

use infr_core::{budget::parse_kv_dtype, parse_size, SizeSpec, WeightSource};
use infr_gguf::Gguf;

use crate::model::{GuiState, MemoryEstimate, ModelInfo, ModelProfile};

const SCAN_DEPTH: usize = 8;
const VULKAN_GUARD: u64 = 256 * 1024 * 1024;
const EMBEDDING_RUNTIME_RESERVE: u64 = 512 * 1024 * 1024;

pub fn scan_state(state: &GuiState) -> Vec<ModelInfo> {
    let mut paths = Vec::new();
    for dir in &state.directories {
        collect_ggufs(dir, 0, &mut paths);
    }
    paths.extend(state.downloaded_models.iter().cloned());

    let mut unique = BTreeMap::new();
    for path in paths {
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        if lower.starts_with("mmproj") || !lower.ends_with(".gguf") {
            continue;
        }
        if infr_core::gguf_split::parse_shard(name).is_some_and(|s| s.index != 1) {
            continue;
        }
        let key = if cfg!(windows) {
            path.to_string_lossy().to_ascii_lowercase()
        } else {
            path.to_string_lossy().into_owned()
        };
        unique.entry(key).or_insert(path);
    }

    let mut out: Vec<_> = unique.into_values().map(|p| inspect(&p)).collect();
    out.sort_by(|a, b| {
        a.name
            .to_ascii_lowercase()
            .cmp(&b.name.to_ascii_lowercase())
    });
    out
}

fn collect_ggufs(dir: &Path, depth: usize, out: &mut Vec<std::path::PathBuf>) {
    if depth > SCAN_DEPTH || out.len() >= 10_000 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if out.len() >= 10_000 {
            break;
        }
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() && !kind.is_symlink() {
            collect_ggufs(&path, depth + 1, out);
        } else if path
            .extension()
            .and_then(|v| v.to_str())
            .is_some_and(|v| v.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
}

fn inspect(path: &Path) -> ModelInfo {
    let path_text = path.to_string_lossy().into_owned();
    let mut info = ModelInfo {
        id: path_text.clone(),
        name: path
            .file_stem()
            .and_then(|v| v.to_str())
            .unwrap_or("model")
            .to_string(),
        path: path_text,
        architecture: None,
        size_bytes: fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        trained_context: None,
        layers: None,
        is_moe: false,
        modalities: vec!["text".into()],
        tasks: vec!["chat".into(), "completion".into()],
        projector: find_projector(path),
        error: None,
    };
    if info.projector.is_some() {
        info.modalities.push("image".into());
    }

    match Gguf::open(path) {
        Ok(gguf) => {
            info.size_bytes = gguf.shards().iter().map(|(_, n)| *n).sum();
            info.architecture = gguf
                .metadata()
                .str("general.architecture")
                .map(str::to_string);
            if let Some(name) = gguf.metadata().str("general.name") {
                info.name = name.to_string();
            }
            if info
                .architecture
                .as_deref()
                .is_some_and(is_embedding_architecture)
            {
                info.tasks = vec!["embedding".into()];
                if let Some(architecture) = info.architecture.as_deref() {
                    info.trained_context = metadata_usize(&gguf, architecture, "context_length");
                    info.layers = metadata_usize(&gguf, architecture, "block_count");
                }
            } else {
                match infr_llama::Config::from_gguf(&gguf) {
                    Ok(cfg) => {
                        info.trained_context = Some(cfg.n_ctx_train);
                        info.layers = Some(cfg.n_layer);
                        info.is_moe = cfg.moe.is_some() || cfg.gemma4_moe;
                    }
                    Err(e) => info.error = Some(e.to_string()),
                }
            }
        }
        Err(e) => info.error = Some(e.to_string()),
    }
    info
}

fn is_embedding_architecture(architecture: &str) -> bool {
    architecture.to_ascii_lowercase().contains("bert")
}

fn metadata_usize(gguf: &Gguf, architecture: &str, suffix: &str) -> Option<usize> {
    let key = format!("{architecture}.{suffix}");
    gguf.metadata()
        .u64(&key)
        .and_then(|value| usize::try_from(value).ok())
}

fn find_projector(model: &Path) -> Option<String> {
    let dir = model.parent()?;
    fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.file_name().and_then(|v| v.to_str()).is_some_and(|n| {
                let n = n.to_ascii_lowercase();
                n.starts_with("mmproj") && n.ends_with(".gguf")
            })
        })
        .map(|p| p.to_string_lossy().into_owned())
}

pub fn estimate(
    profile: &ModelProfile,
    devices: &[infr_vulkan::DeviceInfo],
) -> Result<MemoryEstimate, String> {
    let path = Path::new(&profile.model_path);
    if !path.is_file() {
        return Err("模型尚未下载到本地，无法在加载前读取 GGUF 元数据".into());
    }
    let gguf = Gguf::open(path).map_err(|e| e.to_string())?;
    let model_bytes = gguf.shards().iter().map(|(_, n)| *n).sum::<u64>();
    let total_host_ram = infr_core::hostmem::total_bytes();
    let requested_ram_budget_bytes = parse_budget(&profile.ram_budget, total_host_ram);
    let architecture = gguf
        .metadata()
        .str("general.architecture")
        .map(str::to_string);
    if profile.task == "embedding" {
        return estimate_embedding(
            profile,
            devices,
            &gguf,
            model_bytes,
            requested_ram_budget_bytes,
            architecture,
        );
    }
    let cfg = infr_llama::Config::from_gguf(&gguf).map_err(|e| e.to_string())?;
    let is_moe = cfg.moe.is_some() || cfg.gemma4_moe;
    let footprint = infr_llama::weight_footprint(&gguf);
    let fixed_vram_bytes = footprint.dense;
    let context = parse_absolute(&profile.context);
    let planning_context = context.unwrap_or(cfg.n_ctx_train as u64) as usize;
    let planning_ubatch = profile.ubatch.unwrap_or(1024).max(1);
    let k = parse_kv_dtype(&profile.kv_type_k);
    let v = parse_kv_dtype(&profile.kv_type_v);
    let kv_bytes = match (context, k, v) {
        (Some(ctx), Some(k), Some(v)) => Some(
            infr_llama::seam::estimate_kv_bytes(&cfg, ctx as usize, true, planning_ubatch, k, v)
                * profile.parallel.max(1) as u64,
        ),
        _ => None,
    };
    let selected_device = devices
        .iter()
        .find(|d| format!("Vulkan{}", d.index).eq_ignore_ascii_case(&profile.backend));
    let total_vram_bytes = selected_device.map(|d| d.vram_bytes);
    let requested_vram_budget_bytes = parse_budget(&profile.vram_budget, total_vram_bytes);
    let reserve = parse_budget(&profile.vram_reserve, total_vram_bytes).unwrap_or(0);
    let effective_vram_budget_bytes = total_vram_bytes.map(|total| {
        requested_vram_budget_bytes
            .unwrap_or(total)
            .min(total.saturating_sub(reserve).saturating_sub(VULKAN_GUARD))
    });
    let runtime_reserve_bytes = infr_llama::seam::estimate_runtime_reserve_bytes_for_device(
        &cfg,
        planning_context,
        planning_ubatch,
        selected_device.is_some_and(|device| device.flash_attention_hd256),
    );
    let reserve_plan = infr_llama::seam::estimate_model_memory_plan(
        &cfg,
        fixed_vram_bytes,
        u64::MAX,
        kv_bytes.unwrap_or(0),
        runtime_reserve_bytes,
    )
    .expect("an unbounded control-plane plan must fit");
    let memory_plan = effective_vram_budget_bytes
        .zip(kv_bytes)
        .and_then(|(budget, state)| {
            infr_llama::seam::estimate_model_memory_plan(
                &cfg,
                fixed_vram_bytes,
                budget,
                state,
                runtime_reserve_bytes,
            )
        });
    let cache_override = parse_budget(&profile.expert_cache, total_vram_bytes);
    let estimated_cache_room_bytes = memory_plan.map(|plan| match cache_override {
        Some(requested) => requested.min(plan.expert_cache_bytes),
        None => plan.expert_cache_bytes.min(footprint.expert),
    });
    let pages_experts = is_moe
        && memory_plan.is_some_and(|plan| {
            cache_override.is_some() || plan.expert_cache_bytes < footprint.expert
        });
    let elastic_pool_bytes = memory_plan
        .zip(estimated_cache_room_bytes)
        .filter(|_| pages_experts)
        .map(|(plan, target)| plan.elastic_pool_bytes(target));
    let fits_minimum = effective_vram_budget_bytes
        .zip(kv_bytes)
        .map(|(budget, _)| memory_plan.is_some_and(|plan| plan.minimum_required_bytes() <= budget));
    let host_available = infr_core::hostmem::available_bytes();
    let process_resident = infr_core::hostmem::process_resident_bytes();
    let host = estimate_host_cache(
        profile,
        is_moe,
        footprint.expert,
        host_available,
        process_resident,
        total_host_ram,
    );
    let embedding_model_bytes = attached_embedding_bytes(profile)?;

    let mut notes = vec![
        "KV 按当前引擎的实际布局公式估算，并已乘以并发槽数。".into(),
        "固定权重、packing margin、架构驱动预留和 post-load 余量与 Vulkan 加载器使用同一套公式。"
            .into(),
    ];
    if requested_ram_budget_bytes.is_some() {
        notes.push(
            "显式 RAM 值是 infr 进程的总常驻内存预算；启动时会按工作进程的实际 Working Set 扣除非 cache 占用，当前页面只提供近似 cache 估算。"
                .into(),
        );
    }
    if is_moe {
        notes.push(match host.mode.as_deref() {
            Some("full") => "Host tier 可覆盖完整 expert payload；SSD 仅参与加载。".into(),
            Some("bounded") => {
                "Host tier 是 bounded inclusive RAM/SSD cache；启动时按层等比例预热，未命中专家从 SSD 读取。".into()
            }
            Some("bypass") => "Host RAM cache 已旁路；expert miss 直接从 SSD 进入上层。".into(),
            Some("disabled") => {
                "Host RAM cache 已禁用；expert miss 使用无独立 arena 的后备读取路径。".into()
            }
            _ => "MoE expert payload 由 VRAM/RAM/SSD 分层管理，不等于必须全部常驻显存或内存。".into(),
        });
    }
    if profile.host_dma && host.effective_bytes.unwrap_or(0) > 0 {
        notes.push(
            "Host DMA 会尝试将对齐 RAM arena 原地导入 Vulkan；实际覆盖范围以启动日志为准。".into(),
        );
    }
    if embedding_model_bytes.is_some() {
        if profile.embedding_runner.trim().is_empty() {
            notes.push("附加的原生 Embedding 权重在请求期间借用统一弹性 VRAM，完成后释放并恢复 expert slots。".into());
        } else {
            notes.push(
                "附加 Embedding 使用兼容 runner，其独立进程内存不计入此统一 VRAM 估算。".into(),
            );
        }
    }
    if context.is_none() {
        notes.push("百分比或无效上下文无法离线换算，KV 预算将在加载时确定。".into());
    }
    Ok(MemoryEstimate {
        model_bytes,
        fixed_vram_bytes: Some(fixed_vram_bytes),
        expert_payload_bytes: is_moe.then_some(footprint.expert),
        requested_ram_budget_bytes,
        effective_ram_cache_bytes: host.effective_bytes,
        host_cache_mode: host.mode,
        host_cache_coverage: host.coverage,
        fits_ram_budget: host.fits_payload,
        kv_bytes,
        runtime_reserve_bytes,
        weight_packing_margin_bytes: reserve_plan.weight_packing_margin_bytes,
        load_driver_reserve_bytes: reserve_plan.load_driver_reserve_bytes,
        post_load_reserve_bytes: reserve_plan.post_load_reserve_bytes,
        total_vram_bytes,
        requested_vram_budget_bytes,
        effective_vram_budget_bytes,
        estimated_cache_room_bytes,
        elastic_pool_bytes,
        embedding_model_bytes,
        trained_context: Some(cfg.n_ctx_train),
        architecture,
        is_moe,
        fits_minimum,
        confidence: if kv_bytes.is_some() {
            "medium".into()
        } else {
            "low".into()
        },
        notes,
    })
}

#[allow(clippy::too_many_arguments)]
fn estimate_embedding(
    profile: &ModelProfile,
    devices: &[infr_vulkan::DeviceInfo],
    gguf: &Gguf,
    model_bytes: u64,
    requested_ram_budget_bytes: Option<u64>,
    architecture: Option<String>,
) -> Result<MemoryEstimate, String> {
    let total_vram_bytes = devices
        .iter()
        .find(|device| format!("Vulkan{}", device.index).eq_ignore_ascii_case(&profile.backend))
        .map(|device| device.vram_bytes);
    let requested_vram_budget_bytes = parse_budget(&profile.vram_budget, total_vram_bytes);
    let reserve = parse_budget(&profile.vram_reserve, total_vram_bytes).unwrap_or(0);
    let effective_vram_budget_bytes = total_vram_bytes.map(|total| {
        requested_vram_budget_bytes
            .unwrap_or(total)
            .min(total.saturating_sub(reserve).saturating_sub(VULKAN_GUARD))
    });
    let fits_minimum = effective_vram_budget_bytes
        .map(|budget| model_bytes.saturating_add(EMBEDDING_RUNTIME_RESERVE) <= budget);
    let trained_context = architecture
        .as_deref()
        .and_then(|arch| metadata_usize(gguf, arch, "context_length"));
    let mut notes = vec![
        "Embedding 默认由 INFR 原生 CPU/Vulkan 图执行；当前按整模型驻留估算。".into(),
        "Embedding 没有生成式 KV cache；运行时预留按 512 MiB 保守估计。".into(),
    ];
    if profile.backend.eq_ignore_ascii_case("cpu") {
        notes.push("CPU 模式不占用 Vulkan 显存，VRAM 适配结果不适用。".into());
    }
    Ok(MemoryEstimate {
        model_bytes,
        fixed_vram_bytes: (!profile.backend.eq_ignore_ascii_case("cpu")).then_some(model_bytes),
        expert_payload_bytes: None,
        requested_ram_budget_bytes,
        effective_ram_cache_bytes: None,
        host_cache_mode: None,
        host_cache_coverage: None,
        fits_ram_budget: None,
        kv_bytes: None,
        runtime_reserve_bytes: EMBEDDING_RUNTIME_RESERVE,
        weight_packing_margin_bytes: 0,
        load_driver_reserve_bytes: 0,
        post_load_reserve_bytes: 0,
        total_vram_bytes,
        requested_vram_budget_bytes,
        effective_vram_budget_bytes,
        estimated_cache_room_bytes: effective_vram_budget_bytes.map(|budget| {
            budget
                .saturating_sub(model_bytes)
                .saturating_sub(EMBEDDING_RUNTIME_RESERVE)
        }),
        elastic_pool_bytes: None,
        embedding_model_bytes: None,
        trained_context,
        architecture,
        is_moe: false,
        fits_minimum,
        confidence: "medium".into(),
        notes,
    })
}

#[derive(Debug, Default, PartialEq)]
struct HostCacheEstimate {
    effective_bytes: Option<u64>,
    mode: Option<String>,
    coverage: Option<f64>,
    fits_payload: Option<bool>,
}

fn estimate_host_cache(
    profile: &ModelProfile,
    is_moe: bool,
    expert_payload_bytes: u64,
    available: Option<u64>,
    process_resident: Option<u64>,
    total_host_ram: Option<u64>,
) -> HostCacheEstimate {
    if !is_moe || expert_payload_bytes == 0 {
        return HostCacheEstimate::default();
    }
    let (effective_bytes, mode) = if profile.dram_bypass {
        (Some(0), "bypass")
    } else if profile.ram_budget.trim().is_empty() {
        let bytes = available
            .map(|available| {
                infr_core::hostmem::auto_cache_bytes(available, 0, expert_payload_bytes)
            })
            .unwrap_or(0);
        (
            Some(bytes),
            if bytes >= expert_payload_bytes {
                "full"
            } else {
                "bounded"
            },
        )
    } else if let Some(total_process_budget) = parse_budget(&profile.ram_budget, total_host_ram) {
        let bytes = infr_core::hostmem::cache_bytes_for_total_budget(
            total_process_budget,
            process_resident,
            expert_payload_bytes,
        );
        let mode = if bytes == 0 {
            "disabled"
        } else if bytes >= expert_payload_bytes {
            "full"
        } else {
            "bounded"
        };
        (Some(bytes), mode)
    } else {
        return HostCacheEstimate {
            mode: Some("unknown".into()),
            ..HostCacheEstimate::default()
        };
    };
    HostCacheEstimate {
        effective_bytes,
        mode: Some(mode.into()),
        coverage: effective_bytes.map(|bytes| bytes as f64 / expert_payload_bytes as f64),
        fits_payload: effective_bytes.map(|bytes| bytes >= expert_payload_bytes),
    }
}

fn attached_embedding_bytes(profile: &ModelProfile) -> Result<Option<u64>, String> {
    if profile.embedding_model_path.trim().is_empty() || profile.task == "embedding" {
        return Ok(None);
    }
    let path = Path::new(profile.embedding_model_path.trim());
    if !path.is_file() {
        return Err("附加 Embedding 模型尚未下载到本地，无法估算统一 VRAM 借用量".into());
    }
    let gguf = Gguf::open(path).map_err(|e| e.to_string())?;
    Ok(Some(infr_llama::weight_footprint(&gguf).total()))
}

fn parse_absolute(raw: &str) -> Option<u64> {
    match parse_size(raw.trim())? {
        SizeSpec::Bytes(v) => Some(v),
        SizeSpec::Percent(_) => None,
    }
}

fn parse_budget(raw: &str, base: Option<u64>) -> Option<u64> {
    if raw.trim().is_empty() {
        return None;
    }
    match parse_size(raw.trim())? {
        SizeSpec::Bytes(v) => Some(v),
        SizeSpec::Percent(p) => base.map(|b| (b as f64 * p) as u64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_and_percentage_budgets_are_distinct() {
        assert_eq!(parse_absolute("200k"), Some(200 * 1024));
        assert_eq!(parse_absolute("75%"), None);
        assert_eq!(parse_budget("25%", Some(8 * 1024)), Some(2 * 1024));
        assert_eq!(parse_budget("", Some(8 * 1024)), None);
    }

    #[test]
    fn bert_family_is_catalogued_as_embedding() {
        assert!(is_embedding_architecture("nomic-bert"));
        assert!(is_embedding_architecture("BERT"));
        assert!(!is_embedding_architecture("qwen3moe"));
    }

    #[test]
    fn host_cache_modes_match_the_moe_backing_contract() {
        let payload = 8 * 1024 * 1024 * 1024;
        let fixed = ModelProfile {
            ram_budget: "3g".into(),
            ..ModelProfile::default()
        };
        let estimated = estimate_host_cache(
            &fixed,
            true,
            payload,
            Some(64 * 1024 * 1024 * 1024),
            Some(0),
            Some(64 * 1024 * 1024 * 1024),
        );
        assert_eq!(estimated.mode.as_deref(), Some("bounded"));
        assert_eq!(estimated.effective_bytes, Some(3 * 1024 * 1024 * 1024));
        assert_eq!(estimated.fits_payload, Some(false));

        let bypass = ModelProfile {
            dram_bypass: true,
            ram_budget: "16g".into(),
            ..ModelProfile::default()
        };
        let estimated = estimate_host_cache(
            &bypass,
            true,
            payload,
            Some(64 * 1024 * 1024 * 1024),
            Some(0),
            Some(64 * 1024 * 1024 * 1024),
        );
        assert_eq!(estimated.mode.as_deref(), Some("bypass"));
        assert_eq!(estimated.effective_bytes, Some(0));

        assert_eq!(
            estimate_host_cache(
                &fixed,
                false,
                0,
                Some(64 * 1024 * 1024 * 1024),
                Some(0),
                Some(64 * 1024 * 1024 * 1024),
            ),
            HostCacheEstimate::default()
        );
    }

    #[test]
    fn host_cache_estimate_treats_an_explicit_50g_as_total_process_ram() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let profile = ModelProfile {
            ram_budget: "50g".into(),
            ..ModelProfile::default()
        };
        let estimated = estimate_host_cache(
            &profile,
            true,
            80 * GIB,
            Some(48 * GIB),
            Some(2 * GIB),
            Some(64 * GIB),
        );
        assert_eq!(estimated.mode.as_deref(), Some("bounded"));
        assert_eq!(estimated.effective_bytes, Some(48 * GIB));
        assert_eq!(estimated.fits_payload, Some(false));

        let automatic = estimate_host_cache(
            &ModelProfile::default(),
            true,
            80 * GIB,
            Some(48 * GIB),
            Some(2 * GIB),
            Some(64 * GIB),
        );
        assert_eq!(automatic.effective_bytes, Some(36 * GIB));
        assert_eq!(
            estimate_host_cache(
                &ModelProfile::default(),
                true,
                80 * GIB,
                None,
                Some(2 * GIB),
                Some(64 * GIB),
            )
            .effective_bytes,
            Some(0),
            "auto mode must not assume that all host memory is available without a probe"
        );

        let percentage = ModelProfile {
            ram_budget: "50%".into(),
            ..ModelProfile::default()
        };
        assert_eq!(
            estimate_host_cache(
                &percentage,
                true,
                80 * GIB,
                Some(48 * GIB),
                Some(2 * GIB),
                Some(64 * GIB),
            )
            .effective_bytes,
            Some(30 * GIB),
            "percentage budgets use total physical RAM before subtracting process residency"
        );
    }

    #[test]
    fn scan_uses_first_split_and_excludes_projectors() {
        let root = std::env::temp_dir().join(format!(
            "infr-gui-scan-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        for name in [
            "model-00001-of-00002.gguf",
            "model-00002-of-00002.gguf",
            "single.gguf",
            "mmproj-model.gguf",
        ] {
            fs::write(root.join(name), b"not a real GGUF").unwrap();
        }
        let mut state = GuiState::default();
        state.directories.push(root.clone());

        let models = scan_state(&state);

        assert_eq!(models.len(), 2);
        assert!(models.iter().any(|m| m.path.ends_with("single.gguf")));
        assert!(models
            .iter()
            .any(|m| m.path.ends_with("model-00001-of-00002.gguf")));
        fs::remove_dir_all(root).unwrap();
    }
}
