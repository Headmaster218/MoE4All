//! Benchmark attribution, deliberately outside the scheduler and inference paths.

use super::ParallelSeam;
use crate::Config;
use infr_core::{tensor::DType, SizeSpec};
use serde_json::{json, Value};

impl ParallelSeam {
    /// Snapshot the initialized engine after a benchmark suite, with no GPU work.
    ///
    /// Call after requests have returned their slots, outside a private Rayon pool. The pool
    /// read is nonblocking; checked-out slots are unknown rather than guessed from configuration.
    /// Memory budgets are targets/ceilings, never measured usage or cache-pool capacities.
    pub fn benchmark_configuration(&self) -> Value {
        let ec = self.model.engine_cfg();
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        let (formats, slot_reason) = match self.pool.try_lock() {
            Ok(pool) => {
                let slots: Vec<_> = pool
                    .slots
                    .iter()
                    .map(|slot| {
                        if slot.busy {
                            None
                        } else {
                            slot.kv.as_ref().map(crate::seam::SeamKv::kv_formats)
                        }
                    })
                    .collect();
                (slots, None)
            }
            Err(std::sync::TryLockError::WouldBlock) => (Vec::new(), Some("slot_pool_busy")),
            Err(std::sync::TryLockError::Poisoned(_)) => (Vec::new(), Some("slot_pool_poisoned")),
        };
        let slots: Vec<_> = formats
            .into_iter()
            .map(|formats| formats.map(|(k, v)| kv_snapshot(self.model.config(), k, v)))
            .collect();
        let mut snapshot = aggregate_slots(&slots, slot_reason);
        let device = self.vk.benchmark_device_info();
        let (ram, ram_source, ram_reason) = ram_target(
            ec.device.ram_budget,
            ec.paging.dram_bypass && !device.unified_memory,
        );
        let object = snapshot.as_object_mut().expect("snapshot object");
        object.extend(
            json!({
                "backend": "vulkan",
                "actual_ubatch": crate::seam::ubatch_rows(ec),
                "actual_ubatch_scope": "placement_prefill_chunk",
                "actual_ubatch_parallel": crate::seam::ubatch_rows(ec)
                    .min(crate::seam::ubatch_rows_parallel(ec)),
                "actual_threads": rayon::current_num_threads(),
                "actual_threads_scope": "calling_rayon_pool",
                "actual_ram_budget_bytes": ram,
                "actual_ram_budget_scope": "process_resident_target",
                "actual_ram_budget_source": ram_source,
                "actual_ram_budget_unknown_reason": ram_reason,
                "actual_vram_budget_bytes": device.vram_budget_bytes,
                "actual_vram_budget_scope": "backend_allocation_ceiling",
                "actual_vram_budget_source": device.vram_budget_source,
                "actual_vram_budget_unknown_reason": device.vram_budget_unknown_reason,
                "submit_dispatches": self.vk.submit_cap_now(),
                "driver": {
                    "device_name": device.name,
                    "vendor_id": device.vendor_id,
                    "device_id": device.device_id,
                    "driver_id": device.driver_id,
                    "driver_id_name": device.driver_id_name,
                    "driver_name": device.driver_name,
                    "driver_info": device.driver_info,
                    "driver_version_raw": device.driver_version_raw,
                    "driver_version_encoding": "vulkan_vendor_specific",
                    "unknown_reason": device.driver_unknown_reason,
                    "pci_bdf": device.pci_bdf,
                    "pci_bdf_unknown_reason": device.pci_unknown_reason,
                },
            })
            .as_object()
            .expect("snapshot object")
            .clone(),
        );
        snapshot
    }
}

fn ram_target(
    spec: Option<SizeSpec>,
    bypass: bool,
) -> (Option<u64>, Option<&'static str>, Option<&'static str>) {
    if bypass {
        return (None, None, Some("host_ram_policy_bypassed"));
    }
    match spec {
        Some(SizeSpec::Bytes(bytes)) => (Some(bytes), Some("explicit_process_target"), None),
        Some(SizeSpec::FrozenAutomaticBytes(bytes)) => {
            (Some(bytes), Some("startup_frozen_process_target"), None)
        }
        Some(SizeSpec::Percent(_)) => {
            (None, None, Some("resolved_process_ram_target_not_retained"))
        }
        None => (None, None, Some("no_frozen_process_ram_target")),
    }
}

fn dtype_name(dtype: DType) -> String {
    match dtype {
        DType::Iq4Nl => "iq4_nl".to_owned(),
        _ => format!("{dtype:?}").to_ascii_lowercase(),
    }
}

fn kv_snapshot(cfg: &Config, k: DType, v: DType) -> Value {
    // Generic k_fmt/v_fmt are not the dtypes of V4's packed attention/indexer state.
    if cfg.deepseek4 {
        return json!({
            "kv_type_k": null, "kv_type_v": null,
            "kv_layout": "fp8-kv+mxfp4-index",
            "kv_unknown_reason": "packed_mixed_attention_and_indexer_state_has_no_uniform_k_v_dtype",
        });
    }
    let k = dtype_name(k);
    let v = dtype_name(v);
    let mut layout = if k == v {
        k.clone()
    } else {
        format!("k={k},v={v}")
    };
    if cfg.deepseek2 || cfg.bailingmoe3 {
        layout = format!("mla-latent={k},auxiliary={v}");
    }
    if (0..cfg.n_layer).any(|layer| cfg.is_recurrent_layer(layer)) {
        layout.push_str("+f32-recurrent");
    }
    if cfg.qwen4exp {
        layout.push_str("+f16-qsa-raw+f32-qsa-block");
    }
    if (0..cfg.n_layer).any(|layer| cfg.is_ple_layer(layer)) {
        layout.push_str("+f32-ple");
    }
    json!({ "kv_type_k": k, "kv_type_v": v, "kv_layout": layout, "kv_unknown_reason": null })
}

fn aggregate_slots(slots: &[Option<Value>], reason: Option<&str>) -> Value {
    let complete = !slots.is_empty() && slots.iter().all(Option::is_some);
    let uniform = complete && slots.iter().all(|slot| slot == &slots[0]);
    let mut snapshot = if uniform {
        slots[0].clone().expect("complete slots")
    } else {
        json!({
            "kv_type_k": null, "kv_type_v": null,
            "kv_layout": if complete { Some("mixed") } else { None },
            "kv_unknown_reason": reason.unwrap_or(if complete {
                "slot_layouts_differ"
            } else {
                "slots_uninitialized_or_checked_out"
            }),
        })
    };
    snapshot["kv_slots"] = json!(slots);
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_formats_override_unset_or_different_configuration() {
        let cfg = Config::default();
        let actual = kv_snapshot(&cfg, DType::Q8_0, DType::F16);
        assert_eq!(actual["kv_type_k"], "q8_0");
        assert_eq!(actual["kv_type_v"], "f16");
        assert_eq!(actual["kv_layout"], "k=q8_0,v=f16");
        assert_eq!(dtype_name(DType::Iq4Nl), "iq4_nl");
    }

    #[test]
    fn partial_and_mixed_slots_are_not_claimed_uniform() {
        let cfg = Config::default();
        let q8 = kv_snapshot(&cfg, DType::Q8_0, DType::Q8_0);
        let f16 = kv_snapshot(&cfg, DType::F16, DType::F16);
        let mixed = aggregate_slots(&[Some(q8.clone()), Some(f16)], None);
        assert_eq!(mixed["kv_layout"], "mixed");
        assert!(mixed["kv_type_k"].is_null());
        let partial = aggregate_slots(&[Some(q8), None], None);
        assert!(partial["kv_layout"].is_null());
        assert!(partial["kv_type_v"].is_null());
        assert_eq!(
            aggregate_slots(&[], Some("slot_pool_busy"))["kv_unknown_reason"],
            "slot_pool_busy"
        );
    }

    #[test]
    fn packed_deepseek_layout_does_not_claim_generic_f16() {
        let mut cfg = Config::default();
        cfg.deepseek4 = true;
        let actual = kv_snapshot(&cfg, DType::F16, DType::F16);
        assert_eq!(actual["kv_layout"], "fp8-kv+mxfp4-index");
        assert!(actual["kv_type_k"].is_null());
        assert!(actual["kv_type_v"].is_null());
    }

    #[test]
    fn ram_targets_never_derive_from_usage_or_cache_pool_size() {
        assert_eq!(
            ram_target(Some(SizeSpec::FrozenAutomaticBytes(123)), false).0,
            Some(123)
        );
        assert_eq!(ram_target(Some(SizeSpec::Bytes(0)), false).0, Some(0));
        assert!(ram_target(Some(SizeSpec::Percent(0.5)), false).0.is_none());
        assert!(ram_target(None, false).0.is_none());
        assert!(ram_target(None, false).2.is_some());
        assert!(ram_target(Some(SizeSpec::Bytes(123)), true).0.is_none());
    }

    #[test]
    fn hybrid_layout_includes_recurrent_qsa_and_ple_state() {
        let cfg = Config {
            qwen4exp: true,
            n_layer: 2,
            recurrent_layers: vec![true, false],
            ple_layers: vec![true, false],
            ..Config::default()
        };
        let actual = kv_snapshot(&cfg, DType::Q8_0, DType::Q8_0);
        assert_eq!(actual["kv_type_k"], "q8_0");
        assert_eq!(
            actual["kv_layout"],
            "q8_0+f32-recurrent+f16-qsa-raw+f32-qsa-block+f32-ple"
        );
        assert_eq!(
            aggregate_slots(&[Some(actual.clone()), Some(actual)], None)["kv_type_v"],
            "q8_0"
        );
    }
}
