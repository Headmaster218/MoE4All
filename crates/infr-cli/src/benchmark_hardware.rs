//! Optional platform collection runs outside measurement and never changes inference settings.
use infr_core::config::Config;
use infr_core::{SizeSpec, WeightSource};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const OUTPUT_LIMIT: usize = 64 * 1024;

fn failure(status: &str) -> Value {
    json!({"hardware": {}, "collection": {"status": status}})
}

fn read_bounded(mut reader: impl Read) -> Vec<u8> {
    let mut kept = Vec::new();
    let mut buffer = [0; 4096];
    while let Ok(count) = reader.read(&mut buffer) {
        if count == 0 {
            break;
        }
        let keep = count.min((OUTPUT_LIMIT + 1).saturating_sub(kept.len()));
        kept.extend_from_slice(&buffer[..keep]);
    }
    kept
}

fn run(mut command: Command, script: &'static [u8], timeout: Duration) -> Value {
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let Ok(mut child) = command.spawn() else {
        return failure("unavailable");
    };
    let mut input = child.stdin.take().expect("piped input");
    let output = child.stdout.take().expect("piped output");
    let error = child.stderr.take().expect("piped error");
    std::thread::spawn(move || {
        let _ = input.write_all(script);
    });
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(read_bounded(output));
    });
    std::thread::spawn(move || {
        let _ = read_bounded(error);
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(20))
            }
            _ => {
                #[cfg(unix)]
                // The collector owns its own process group, including bounded read helpers.
                unsafe {
                    libc::kill(-(child.id() as i32), libc::SIGKILL);
                }
                let _ = child.kill();
                let _ = child.wait();
                return failure("timeout");
            }
        }
    };
    if !status.success() {
        return failure("failed");
    }
    let Ok(bytes) = rx.recv_timeout(Duration::from_secs(1)) else {
        return failure("failed");
    };
    if bytes.len() > OUTPUT_LIMIT {
        return failure("output_limit");
    }
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return failure("invalid_output");
    };
    if !value.get("hardware").is_some_and(Value::is_object) {
        return failure("invalid_output");
    }
    value
}

pub fn collect(model: &Path, gpu_name: &str, pci_address: Option<&str>) -> Value {
    #[cfg(windows)]
    let (command, script) = {
        let executable = std::env::var_os("SystemRoot")
            .map(|root| Path::new(&root).join("System32/WindowsPowerShell/v1.0/powershell.exe"))
            .unwrap_or_else(|| "powershell.exe".into());
        let mut command = Command::new(executable);
        command.args(["-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
            "$s = [Console]::In.ReadToEnd(); & ([scriptblock]::Create($s)) -ModelPath $env:MOE4ALL_PROBE_MODEL -SelectedGpuName $env:MOE4ALL_PROBE_GPU -PciAddress $env:MOE4ALL_PROBE_PCI"]);
        command
            .env("MOE4ALL_PROBE_MODEL", model)
            .env("MOE4ALL_PROBE_GPU", gpu_name)
            .env("MOE4ALL_PROBE_PCI", pci_address.unwrap_or(""));
        (
            command,
            include_bytes!("../../../scripts/benchmark-hardware.ps1").as_slice(),
        )
    };
    #[cfg(not(windows))]
    let (command, script) = {
        let mut command = Command::new("/bin/sh");
        command.args(["-s", "--", "--model"]).arg(model).args([
            "--gpu-name",
            gpu_name,
            "--pci-address",
            pci_address.unwrap_or(""),
        ]);
        (
            command,
            include_bytes!("../../../scripts/benchmark-hardware.sh").as_slice(),
        )
    };
    run(command, script, Duration::from_secs(12))
}

/// Only platform hardware fields may be merged; configuration and measured results stay owned
/// by the engine. Null platform fields cannot erase a successful engine measurement.
pub fn merge(hardware: &mut Value, collected: &Value) {
    // PCI databases may name a chip differently or have no marketing name at all.
    let selected_gpu_name = hardware
        .pointer("/gpu/name")
        .filter(|value| value.as_str().is_some_and(|name| !name.is_empty()))
        .cloned();
    fn merge_object(target: &mut Value, source: &Value) {
        let (Some(target), Some(source)) = (target.as_object_mut(), source.as_object()) else {
            return;
        };
        for (key, value) in source {
            if value.is_null() {
                continue;
            }
            if value.is_object() && target.get(key).is_some_and(Value::is_object) {
                merge_object(target.get_mut(key).expect("checked key"), value);
            } else {
                target.insert(key.clone(), value.clone());
            }
        }
    }
    if let Some(source) = collected.get("hardware") {
        merge_object(hardware, source);
    }
    if let Some(name) = selected_gpu_name {
        hardware["gpu"]["name"] = name;
    }
}

pub fn size(value: Option<SizeSpec>) -> Value {
    match value {
        None => Value::Null,
        Some(SizeSpec::Bytes(bytes) | SizeSpec::FrozenAutomaticBytes(bytes)) => json!(bytes),
        Some(SizeSpec::Percent(fraction)) => json!(format!("{}%", fraction * 100.0)),
    }
}

pub fn configured(cfg: &Config, backend: &str) -> Value {
    json!({
        "backend": backend,
        "configured_device": cfg.device.dev,
        "configured_ubatch": cfg.device.ubatch,
        "configured_threads": cfg.device.threads,
        "configured_kv_type_k": cfg.kv.type_k.map(|v| format!("{v:?}")),
        "configured_kv_type_v": cfg.kv.type_v.map(|v| format!("{v:?}")),
        "configured_vram_budget": size(cfg.device.vram_budget),
        "configured_vram_reserve": size(cfg.device.vram_reserve),
        "configured_ram_budget": size(cfg.device.ram_budget),
        "auto_profile": format!("{:?}", cfg.device.auto_profile),
        "host_dma": cfg.paging.host_dma,
    })
}

pub fn enrich_manual(report: &mut Value, metadata: &Value) {
    for key in ["engine_version", "engine_commit", "hardware", "model"] {
        if let Some(value) = metadata.get(key).filter(|value| !value.is_null()) {
            report[key] = value.clone();
        }
    }
    if let Some(fields) = metadata.get("configuration").and_then(Value::as_object) {
        for (key, value) in fields {
            if key == "auto_profile" && report["configuration"].get(key).is_some() {
                continue;
            }
            report["configuration"][key] = value.clone();
        }
    }
}

pub fn model(source: &impl WeightSource, name: &str, storage_bytes: u64) -> Value {
    let mut types: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut elements = 0u64;
    let mut bytes = 0u64;
    for tensor in source.tensors() {
        let entry = types.entry(dtype_label(tensor.dtype)).or_default();
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(tensor.nbytes as u64);
        bytes = bytes.saturating_add(tensor.nbytes as u64);
        if let Some(count) = tensor
            .shape
            .iter()
            .try_fold(1u64, |a, &b| a.checked_mul(b as u64))
        {
            elements = elements.saturating_add(count);
        }
    }
    let quantization = if types.is_empty() {
        None
    } else {
        Some(types.keys().cloned().collect::<Vec<_>>().join(" + "))
    };
    json!({"name": name, "architecture": source.metadata().str("general.architecture"),
        "storage_bytes": storage_bytes, "quantization": quantization,
        "gguf_file_type": source.metadata().u64("general.file_type"),
        "quantization_source": "gguf_tensor_types",
        "weight_types": types.into_iter().map(|(dtype,(tensors,bytes))| json!({"dtype":dtype,"tensors":tensors,"bytes":bytes})).collect::<Vec<_>>(),
        "bits_per_weight": (elements != 0).then(|| bytes as f64 * 8.0 / elements as f64)})
}

fn dtype_label(dtype: infr_core::DType) -> String {
    use infr_core::DType::*;
    match dtype {
        Q2K => "Q2_K",
        Q3K => "Q3_K",
        Q4K => "Q4_K",
        Q5K => "Q5_K",
        Q6K => "Q6_K",
        Iq1S => "IQ1_S",
        Iq1M => "IQ1_M",
        Iq2Xxs => "IQ2_XXS",
        Iq2Xs => "IQ2_XS",
        Iq2S => "IQ2_S",
        Iq3Xxs => "IQ3_XXS",
        Iq3S => "IQ3_S",
        Iq4Nl => "IQ4_NL",
        Iq4Xs => "IQ4_XS",
        _ => return format!("{dtype:?}").to_ascii_uppercase(),
    }
    .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_preserves_engine_values_on_null_and_retains_partial_results() {
        let mut hardware =
            json!({"gpu":{"name":"selected", "vram_total_bytes":10},"ram":{"total_bytes":20}});
        merge(
            &mut hardware,
            &json!({"hardware":{"gpu":{"name":null,"pcie_width":8},"ram":{"memory_generation":"DDR5"}},"configuration":{"mtp":true}}),
        );
        assert_eq!(hardware["gpu"]["name"], "selected");
        assert_eq!(hardware["gpu"]["pcie_width"], 8);
        assert_eq!(hardware["ram"]["total_bytes"], 20);
        assert!(hardware.get("configuration").is_none());
        merge(
            &mut hardware,
            &json!({"hardware":{"gpu":{"name":"native PCI chip alias"}}}),
        );
        assert_eq!(hardware["gpu"]["name"], "selected");
    }

    #[test]
    fn configured_auto_is_null_not_rust_debug_text() {
        let cfg = Config::default();
        let value = configured(&cfg, "Vulkan");
        assert!(value["configured_kv_type_k"].is_null());
        assert_eq!(size(Some(SizeSpec::Bytes(1024))), 1024);
        assert_eq!(size(Some(SizeSpec::Percent(0.8))), "80%");
    }

    #[test]
    fn manual_enrichment_preserves_identity_and_all_scored_data() {
        let mut report = json!({"id":"fixture", "created_at_utc":"2026-10-25T00:00:00Z",
            "configuration":{"depth_tokens":6144,"repetitions":3,"auto_profile":"manual"},
            "contexts":[{"manual_result":{"average_tokens_per_second":123.5}}]});
        let contexts = report["contexts"].clone();
        enrich_manual(
            &mut report,
            &json!({"id":"must-not-replace", "contexts":[],
            "hardware":{"cpu":{"name":"test"}}, "configuration":{"backend":"Metal","actual_ubatch":1024,"auto_profile":"Conservative"}}),
        );
        assert_eq!(report["id"], "fixture");
        assert_eq!(report["contexts"], contexts);
        assert_eq!(report["configuration"]["depth_tokens"], 6144);
        assert_eq!(report["configuration"]["backend"], "Metal");
        assert_eq!(report["configuration"]["auto_profile"], "manual");
    }

    #[test]
    fn unavailable_collector_is_nonfatal() {
        assert_eq!(
            run(
                Command::new("moe4all-nonexistent-probe-6a2f"),
                b"",
                Duration::from_millis(50)
            )["collection"]["status"],
            "unavailable"
        );
    }

    fn test_shell() -> Command {
        #[cfg(windows)]
        {
            let path = Path::new(&std::env::var_os("SystemRoot").unwrap())
                .join("System32/WindowsPowerShell/v1.0/powershell.exe");
            let mut command = Command::new(path);
            command.args([
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$s=[Console]::In.ReadToEnd(); & ([scriptblock]::Create($s))",
            ]);
            command
        }
        #[cfg(not(windows))]
        {
            let mut command = Command::new("/bin/sh");
            command.arg("-s");
            command
        }
    }

    #[test]
    fn collector_invalid_output_and_timeout_do_not_escape() {
        #[cfg(windows)]
        let scripts = (
            b"Write-Output 'not-json'".as_slice(),
            b"Start-Sleep -Seconds 30".as_slice(),
        );
        #[cfg(not(windows))]
        let scripts = (b"printf not-json".as_slice(), b"sleep 30".as_slice());
        assert_eq!(
            run(test_shell(), scripts.0, Duration::from_secs(5))["collection"]["status"],
            "invalid_output"
        );
        let start = Instant::now();
        assert_eq!(
            run(test_shell(), scripts.1, Duration::from_millis(200))["collection"]["status"],
            "timeout"
        );
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn quantization_comes_from_tensor_types_not_the_filename() {
        struct Fixture {
            tensors: Vec<infr_core::TensorInfo>,
            metadata: infr_core::Metadata,
        }
        impl WeightSource for Fixture {
            fn tensors(&self) -> &[infr_core::TensorInfo] {
                &self.tensors
            }
            fn metadata(&self) -> &infr_core::Metadata {
                &self.metadata
            }
            fn tensor_bytes(&self, _: &str) -> infr_core::Result<&[u8]> {
                unreachable!()
            }
            fn chat_template(&self) -> Option<&str> {
                None
            }
        }
        let source = Fixture {
            metadata: Default::default(),
            tensors: vec![
                infr_core::TensorInfo {
                    name: "a".into(),
                    dtype: infr_core::DType::Q4K,
                    shape: vec![256],
                    offset: 0,
                    nbytes: 144,
                },
                infr_core::TensorInfo {
                    name: "b".into(),
                    dtype: infr_core::DType::Iq3S,
                    shape: vec![256],
                    offset: 144,
                    nbytes: 110,
                },
            ],
        };
        let result = model(&source, "misleading-Q8_0.gguf", 254);
        assert_eq!(result["quantization"], "IQ3_S + Q4_K");
        assert_eq!(result["weight_types"].as_array().unwrap().len(), 2);
        assert_eq!(
            result["bits_per_weight"].as_f64(),
            Some(254.0 * 8.0 / 512.0)
        );
    }

    #[test]
    #[ignore = "explicit local platform probe; no model or GPU work"]
    fn live_platform_probe_is_bounded_and_retains_hardware() {
        let result = collect(Path::new("missing-model-for-readonly-probe.gguf"), "", None);
        assert!(result["hardware"].is_object(), "{result}");
        assert!(result["hardware"]["cpu"]["name"].is_string(), "{result}");
        assert!(result["collection"].is_object(), "{result}");
    }
}
