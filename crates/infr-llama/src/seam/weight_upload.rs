//! Metadata-only description of the bytes the ordinary weight loader actually uploads.
//! The same rewrite choice drives allocation planning and payload materialization.
use super::WeightUploadSpec;
use crate::Config;
use anyhow::{anyhow, Result};
use infr_core::{loader::TensorInfo, DType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rewrite {
    Native,
    I2sF16,
    PleConvF32,
    AbsorbedF32,
}

pub(super) struct UploadLayout {
    pub spec: WeightUploadSpec,
    pub dtype: DType,
    pub numel: usize,
    pub rewrite: Rewrite,
}

pub(super) fn describe(c: &Config, tensors: &[TensorInfo], names: &[&str]) -> Result<UploadLayout> {
    let first = names
        .first()
        .ok_or_else(|| anyhow!("empty weight upload group"))?;
    let infos = names
        .iter()
        .map(|name| {
            tensors
                .iter()
                .find(|info| info.name == *name)
                .ok_or_else(|| anyhow!("tensor not found: {name}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let dtype = infos[0].dtype;
    if infos.iter().any(|info| info.dtype != dtype) {
        return Err(anyhow!("wload concat dtype mismatch: {names:?}"));
    }
    let numel = infos.iter().try_fold(0usize, |sum, info| {
        let elements = info
            .shape
            .iter()
            .try_fold(1usize, |product, &dim| product.checked_mul(dim))
            .ok_or_else(|| anyhow!("weight shape overflow: {}", info.name))?;
        sum.checked_add(elements)
            .ok_or_else(|| anyhow!("weight shape overflow: {first}"))
    })?;
    let rewrite = if dtype == DType::I2S {
        Rewrite::I2sF16
    } else if names.len() == 1 && c.qwen4exp && first.ends_with("ple_conv1d.weight") {
        Rewrite::PleConvF32
    } else if names.len() == 1
        && (((c.deepseek2 || c.bailingmoe3)
            && (first.ends_with("attn_k_b.weight") || first.ends_with("attn_v_b.weight")))
            || (c.deepseek4
                && (first.contains("_compressor_ape.weight")
                    || first.contains("_compressor_norm.weight"))))
    {
        Rewrite::AbsorbedF32
    } else {
        Rewrite::Native
    };
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| anyhow!("weight byte size overflow: {first}"))
    };
    let (dtype, numel, bytes) = match rewrite {
        Rewrite::I2sF16 => (DType::F16, numel, mul(numel, 2)?),
        Rewrite::AbsorbedF32 => (DType::F32, numel, mul(numel, 4)?),
        Rewrite::PleConvF32 => {
            let channels = mul(c.hc_mult, c.n_embd)?;
            let taps = c
                .ple_conv_kernel
                .checked_sub(1)
                .and_then(|v| v.checked_mul(c.ple_ngram_size))
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| anyhow!("invalid PLE convolution geometry: {first}"))?;
            if mul(channels, c.ple_conv_kernel)? != numel {
                return Err(anyhow!(
                    "{first}: {numel} values do not match {channels} channels x {} taps",
                    c.ple_conv_kernel
                ));
            }
            let elements = mul(channels, taps)?;
            (DType::F32, elements, mul(elements, 4)?)
        }
        Rewrite::Native => {
            let bytes = infos
                .iter()
                .try_fold(0usize, |sum, info| sum.checked_add(info.nbytes))
                .ok_or_else(|| anyhow!("weight byte size overflow: {first}"))?;
            (dtype, numel, bytes)
        }
    };
    let permuted = c.permute_qk_neox
        && names.iter().any(|name| {
            name.ends_with("attn_q.weight")
                || name.ends_with("attn_q.bias")
                || name.ends_with("attn_k.weight")
                || name.ends_with("attn_k.bias")
        });
    Ok(UploadLayout {
        spec: WeightUploadSpec {
            name: (*first).to_owned(),
            bytes,
            direct_mmap: names.len() == 1 && rewrite == Rewrite::Native && !permuted,
        },
        dtype,
        numel,
        rewrite,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, dtype: DType, shape: &[usize], nbytes: usize) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype,
            shape: shape.into(),
            nbytes,
            offset: 0,
        }
    }

    #[test]
    fn fixed_upload_native_concat_and_f32_are_exact() {
        let tensors = [
            tensor("q", DType::F32, &[3, 5], 60),
            tensor("k", DType::F32, &[2, 5], 40),
        ];
        let one = describe(&Config::default(), &tensors, &["q"]).unwrap();
        assert_eq!((one.spec.bytes, one.dtype, one.numel), (60, DType::F32, 15));
        assert!(one.spec.direct_mmap);
        let pair = describe(&Config::default(), &tensors, &["q", "k"]).unwrap();
        assert_eq!(
            (pair.spec.bytes, pair.dtype, pair.numel),
            (100, DType::F32, 25)
        );
        assert!(!pair.spec.direct_mmap);
    }

    #[test]
    fn fixed_upload_rewrites_and_permutation_match_resident_layout() {
        let mut c = Config {
            qwen4exp: true,
            hc_mult: 4,
            n_embd: 8,
            ple_conv_kernel: 3,
            ple_ngram_size: 4,
            ..Default::default()
        };
        let tensors = [tensor("blk.1.ple_conv1d.weight", DType::F16, &[3, 32], 192)];
        let ple = describe(&c, &tensors, &[&tensors[0].name]).unwrap();
        assert_eq!(
            (ple.spec.bytes, ple.dtype, ple.numel),
            (32 * 9 * 4, DType::F32, 32 * 9)
        );
        assert_eq!(ple.rewrite, Rewrite::PleConvF32);
        assert!(!ple.spec.direct_mmap);
        c.bailingmoe3 = true;
        let tensors = [tensor("blk.5.attn_k_b.weight", DType::Q4_0, &[32, 8], 144)];
        let mla = describe(&c, &tensors, &[&tensors[0].name]).unwrap();
        assert_eq!(
            (mla.spec.bytes, mla.dtype, mla.numel),
            (1024, DType::F32, 256)
        );
        let tensors = [tensor("w", DType::I2S, &[128], 36)];
        let i2s = describe(&c, &tensors, &["w"]).unwrap();
        assert_eq!(
            (i2s.spec.bytes, i2s.dtype, i2s.numel),
            (256, DType::F16, 128)
        );
        c.permute_qk_neox = true;
        let tensors = [tensor("blk.0.attn_q.weight", DType::F16, &[8, 8], 128)];
        assert!(
            !describe(&c, &tensors, &[&tensors[0].name])
                .unwrap()
                .spec
                .direct_mmap
        );
    }

    #[test]
    fn fixed_upload_bad_metadata_fails_before_payload_access() {
        let c = Config::default();
        assert!(describe(&c, &[], &[]).is_err());
        assert!(describe(&c, &[], &["missing"]).is_err());
        let tensors = [
            tensor("q", DType::F32, &[2], 8),
            tensor("k", DType::F16, &[2], 4),
        ];
        assert!(describe(&c, &tensors, &["q", "k"]).is_err());
        let tensors = [tensor("huge", DType::F32, &[usize::MAX, 2], 0)];
        assert!(describe(&c, &tensors, &["huge"]).is_err());
    }

    #[test]
    fn fixed_upload_paging_only_changes_true_mapped_banks() {
        let weights = [
            WeightUploadSpec {
                name: "blk.0.ffn_gate_exps.weight".into(),
                bytes: 1234,
                direct_mmap: true,
            },
            WeightUploadSpec {
                name: "blk.1.ffn_gate_exps.weight".into(),
                bytes: 1234,
                direct_mmap: true,
            },
            WeightUploadSpec {
                name: "blk.0.ffn_up_exps.weight".into(),
                bytes: 1234,
                direct_mmap: false,
            },
            WeightUploadSpec {
                name: "streamed".into(),
                bytes: 1234,
                direct_mmap: false,
            },
            WeightUploadSpec {
                name: "output_norm.weight".into(),
                bytes: 10,
                direct_mmap: true,
            },
        ];
        let streamed = ["streamed".to_owned()].into_iter().collect();
        assert_eq!(
            super::super::vulkan_fixed_allocation_sizes(&weights, &streamed, 1).unwrap(),
            [4, 1236, 1236, 4, 12]
        );
        assert_eq!(
            super::super::vulkan_fixed_allocation_sizes(&weights, &Default::default(), 0).unwrap(),
            [1236, 1236, 1236, 1236, 12]
        );
    }

    /// Use four original trunk layers (including DeltaNet and full attention), embedding and
    /// vocabulary head to exercise fully resident weights on a 24 GiB test GPU.
    #[test]
    #[ignore = "requires MOE4ALL_WEIGHT_TEST_MODEL and a Vulkan GPU"]
    fn gpu_real_qwen35_resident_weight_layout_parity() {
        real_qwen35_weight_layout_parity(false);
    }

    /// Full 40-layer real-model regression. Compare the very same kernels and paging policy;
    /// only the resident-weight allocation strategy changes.
    #[test]
    #[ignore = "requires MOE4ALL_WEIGHT_TEST_MODEL, a Vulkan GPU and enough host RAM"]
    fn gpu_real_qwen35_paged_weight_layout_parity() {
        real_qwen35_weight_layout_parity(true);
    }

    fn real_qwen35_weight_layout_parity(paged: bool) {
        use super::super::TokenEmbd;
        use infr_core::backend::Backend;
        let path =
            std::env::var_os("MOE4ALL_WEIGHT_TEST_MODEL").expect("set the real 35B GGUF path");
        let gguf = infr_gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
        let mut cfg = Config::from_gguf(&gguf).unwrap();
        assert!(cfg.qwen35 && cfg.moe.is_some());
        if !paged {
            cfg.n_layer = 4;
            cfg.n_layer_kv_from_start = 4;
        }
        let mut engine = crate::EngineConfig::default();
        engine.device.ubatch = Some(16);
        engine.device.ubatch_specified = true;
        engine.sampling.ignore_eos = true;
        if paged {
            engine.paging.cache = Some(infr_core::SizeSpec::Bytes(8 << 30));
            engine.device.ram_budget = Some(infr_core::SizeSpec::Bytes(32 << 30));
            engine.device.vram_budget = Some(infr_core::SizeSpec::Bytes(22 << 30));
            engine.device.vram_reserve = Some(infr_core::SizeSpec::Bytes(0));
        }
        let device = std::env::var("MOE4ALL_WEIGHT_TEST_DEVICE")
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let cell = std::sync::OnceLock::new();
        let mut outputs = Vec::new();
        let mut allocations = Vec::new();
        for planned in [false, true] {
            let be = infr_vulkan::VulkanBackend::new_on_with(
                device,
                std::sync::Arc::new(engine.clone()),
            )
            .unwrap();
            eprintln!("real_35b_test_device={}", be.capabilities().name);
            let (mut bind, finish) = if paged {
                super::super::vulkan_moe_binder(&be, &gguf, &cfg, &engine, true, 512).unwrap()
            } else {
                (super::super::vulkan_resident_upload_bind(&be), None)
            };
            if !planned {
                bind.prepare = None;
            }
            let mut state = None;
            let prompt = [151644, 872, 198, 40, 1085, 264, 3405, 13];
            let mut logits = Vec::new();
            let (out, _) = super::super::generate_dense_backend(
                &be,
                &bind,
                &gguf,
                &cfg,
                &engine,
                TokenEmbd::new(&cell, &gguf),
                None,
                &prompt,
                8,
                |_| {},
                &mut state,
                512,
                None,
                None,
                None,
                Some(&mut logits),
                None,
                None,
                None,
                None,
                finish.as_deref(),
                None,
            )
            .unwrap();
            assert!(!out.is_empty());
            assert!(!logits.is_empty());
            assert!(logits.iter().all(|v| v.is_finite()));
            let stats = be.resident_weight_allocation_stats();
            eprintln!("real_35b_layout paged={paged} layers={} planned={planned} requested={} capacity={} physical={} blocks={} tokens={out:?}",
                cfg.n_layer, stats.requested_bytes, stats.capacity_bytes, stats.physical_allocation_bytes, stats.block_count);
            // Warm reuse followed by a changed suffix must retain the exact uploaded views.
            let mut next = prompt.to_vec();
            next.extend_from_slice(&out);
            next.extend_from_slice(&[198, 40, 1097, 311, 1075, 13]);
            let (continuation, _) = super::super::generate_dense_backend(
                &be,
                &bind,
                &gguf,
                &cfg,
                &engine,
                TokenEmbd::new(&cell, &gguf),
                None,
                &next,
                8,
                |_| {},
                &mut state,
                512,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            assert_eq!(
                be.resident_weight_allocation_stats(),
                stats,
                "warm call reallocated weights"
            );
            allocations.push(stats);
            outputs.push((out, continuation, logits));
        }
        assert_eq!(outputs[0].0, outputs[1].0, "greedy output changed");
        assert_eq!(outputs[0].1, outputs[1].1, "warm suffix output changed");
        assert_eq!(outputs[0].2, outputs[1].2, "logits changed");
        assert_eq!(
            allocations[0].requested_bytes,
            allocations[1].requested_bytes
        );
        assert!(
            allocations[1].physical_allocation_bytes < allocations[0].physical_allocation_bytes
        );
        assert!(allocations[1].capacity_bytes < allocations[0].capacity_bytes);
    }
}
