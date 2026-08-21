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
    let requested_ram_budget_bytes = parse_absolute(&profile.ram_budget);
    let fits_ram_budget = requested_ram_budget_bytes.map(|budget| model_bytes <= budget);
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
            fits_ram_budget,
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
            infr_llama::seam::estimate_kv_bytes(
                &cfg,
                ctx as usize,
                true,
                planning_ubatch,
                k,
                v,
            ) * profile.parallel.max(1) as u64,
        ),
        _ => None,
    };
    let total_vram_bytes = devices
        .iter()
        .find(|d| format!("Vulkan{}", d.index).eq_ignore_ascii_case(&profile.backend))
        .map(|d| d.vram_bytes);
    let requested_vram_budget_bytes = parse_budget(&profile.vram_budget, total_vram_bytes);
    let reserve = parse_budget(&profile.vram_reserve, total_vram_bytes).unwrap_or(0);
    let effective_vram_budget_bytes = total_vram_bytes.map(|total| {
        requested_vram_budget_bytes
            .unwrap_or(total)
            .min(total.saturating_sub(reserve).saturating_sub(VULKAN_GUARD))
    });
    let runtime_reserve_bytes =
        infr_llama::seam::estimate_runtime_reserve_bytes(&cfg, planning_context, planning_ubatch);
    let memory_plan = effective_vram_budget_bytes
        .zip(kv_bytes)
        .and_then(|(budget, state)| {
            infr_llama::seam::ModelMemoryPlan::new(
                budget,
                fixed_vram_bytes,
                state,
                runtime_reserve_bytes,
            )
        });
    let estimated_cache_room_bytes = memory_plan.map(|plan| plan.expert_cache_bytes);
    let fits_minimum = effective_vram_budget_bytes
        .zip(kv_bytes)
        .map(|(budget, state)| {
            infr_llama::seam::ModelMemoryPlan::new(
                budget,
                fixed_vram_bytes,
                state,
                runtime_reserve_bytes,
            )
            .is_some_and(|plan| plan.minimum_required_bytes() <= budget)
        });

    let mut notes = vec![
        "KV 按当前引擎的实际布局公式估算，并已乘以并发槽数。".into(),
        "运行时预留是加载前保守值；最终分配仍由 Vulkan budget guard 决定。".into(),
    ];
    if is_moe {
        notes.push("MoE 文件大小主要代表 RAM/磁盘权重，不等于必须全部常驻显存。".into());
    }
    if fits_ram_budget == Some(false) {
        notes.push("RAM 权重预算小于模型文件总量；未驻留部分将依赖 mmap/系统文件缓存。".into());
    }
    if context.is_none() {
        notes.push("百分比或无效上下文无法离线换算，KV 预算将在加载时确定。".into());
    }
    Ok(MemoryEstimate {
        model_bytes,
        fixed_vram_bytes: Some(fixed_vram_bytes),
        requested_ram_budget_bytes,
        fits_ram_budget,
        kv_bytes,
        runtime_reserve_bytes,
        total_vram_bytes,
        requested_vram_budget_bytes,
        effective_vram_budget_bytes,
        estimated_cache_room_bytes,
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
    fits_ram_budget: Option<bool>,
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
        "Embedding 推理由 INFR 托管的 llama.cpp worker 执行；当前按整模型驻留估算。".into(),
        "Embedding 没有生成式 KV cache；运行时预留按 512 MiB 保守估计。".into(),
    ];
    if profile.backend.eq_ignore_ascii_case("cpu") {
        notes.push("CPU 模式不占用 Vulkan 显存，VRAM 适配结果不适用。".into());
    }
    if fits_ram_budget == Some(false) {
        notes.push("RAM 权重预算小于模型文件总量。".into());
    }
    Ok(MemoryEstimate {
        model_bytes,
        fixed_vram_bytes: (!profile.backend.eq_ignore_ascii_case("cpu")).then_some(model_bytes),
        requested_ram_budget_bytes,
        fits_ram_budget,
        kv_bytes: None,
        runtime_reserve_bytes: EMBEDDING_RUNTIME_RESERVE,
        total_vram_bytes,
        requested_vram_budget_bytes,
        effective_vram_budget_bytes,
        estimated_cache_room_bytes: effective_vram_budget_bytes.map(|budget| {
            budget
                .saturating_sub(model_bytes)
                .saturating_sub(EMBEDDING_RUNTIME_RESERVE)
        }),
        trained_context,
        architecture,
        is_moe: false,
        fits_minimum,
        confidence: "medium".into(),
        notes,
    })
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
