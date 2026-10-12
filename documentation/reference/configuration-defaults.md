---
kind: reference
status: current
scope: configuration-defaults
last_verified: 2026-10-11
verified_commit: 7b4a9411e9c7524f8dc6ae2603c62d5f338190dc
baseline_tag: release-0.10.0
---

# Config 字段与构造默认值

本表逐项核对 `crates/infr-core/src/config/mod.rs` 的 `cfg_struct!` 声明，共261个叶字段。类型和默认表达式保持源码拼写；这是 **Config 构造默认值**，不是CLI完成模型推荐填充、Wizard参数注入或设备门控之后的最终值。加载顺序、语法、环境别名和模型默认见[配置参考](configuration.md)。

`None` 一般表示留给调用方/设备自动选择，不是数值0；是否支持环境变量以manifest/env实现为准。带 `_specified` 的字段用于保留输入来源/是否显式指定，不建议作为用户调参项。`device.ubatch_parallel=1024` 在没有显式输入时会由实际placement选中的ubatch继承覆盖；`spec.k=6` 在Qwen3.8传统路径被裁剪为2..4VERIFY rows。CPUmiss默认0，与向导主动启用的实验配方不同。

此表只列已存在字段，不列失败实验临时开关，不能凭原始benchmark命令重新启用已删除候选。

## `devic`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `device` | `DeviceCfg` | `> PartialDeviceCfg` |

## `samplin`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `sampling` | `SamplingCfg` | `> PartialSamplingCfg` |

## `k`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `kv` | `KvCfg` | `> PartialKvCfg` |

## `pagin`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `paging` | `PagingCfg` | `> PartialPagingCfg` |

## `kernel`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `kernels` | `KernelCfg` | `> PartialKernelCfg` |

## `spe`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `spec` | `SpecCfg` | `> PartialSpecCfg` |

## `mult`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `multi` | `MultiCfg` | `> PartialMultiCfg` |

## `pro`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `prof` | `ProfCfg` | `> PartialProfCfg` |

## `serv`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `serve` | `ServeCfg` | `> PartialServeCfg` |

## `hu`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `hub` | `HubCfg` | `> PartialHubCfg` |

## `debu`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `debug` | `DebugCfg` | `> PartialDebugCfg` |

## `device`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `device.dev` | `Option<String>` | `None` |
| `device.ctx` | `Option<SizeSpec>` | `None` |
| `device.auto_profile` | `AutoProfile` | `AutoProfile::Conservative` |
| `device.vram_budget` | `Option<SizeSpec>` | `None` |
| `device.ram_budget` | `Option<SizeSpec>` | `None` |
| `device.vram_reserve` | `Option<SizeSpec>` | `None` |
| `device.ubatch` | `Option<usize>` | `None` |
| `device.ubatch_specified` | `bool` | `false` |
| `device.ubatch_parallel` | `usize` | `1024` |
| `device.ubatch_parallel_specified` | `bool` | `false` |
| `device.submit_dispatches` | `Option<usize>` | `None` |
| `device.subgroup_pref` | `Option<u32>` | `None` |
| `device.threads` | `Option<usize>` | `None` |

## `sampling`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `sampling.temp` | `f32` | `0.0` |
| `sampling.top_k` | `usize` | `20` |
| `sampling.top_p` | `f32` | `0.95` |
| `sampling.seed` | `Option<u64>` | `None` |
| `sampling.max_new` | `usize` | `2048` |
| `sampling.ignore_eos` | `bool` | `false` |
| `sampling.no_think` | `bool` | `false` |
| `sampling.reasoning_effort` | `Option<ReasoningEffort>` | `None` |
| `sampling.preserve_thinking` | `Option<bool>` | `None` |

## `kv`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `kv.type_k` | `Option<DType>` | `None` |
| `kv.type_k_specified` | `bool` | `false` |
| `kv.type_v` | `Option<DType>` | `None` |
| `kv.type_v_specified` | `bool` | `false` |
| `kv.force_q8` | `bool` | `false` |
| `kv.slots` | `usize` | `4` |
| `kv.session_cache_dir` | `Option<PathBuf>` | `None` |
| `kv.session_idle_secs` | `u64` | `120` |
| `kv.session_cache_max` | `SizeSpec` | `SizeSpec::Bytes(5u64 << 30)` |
| `kv.session_cache_ttl_hours` | `u64` | `24` |
| `kv.ring` | `bool` | `true` |
| `kv.dynamic` | `bool` | `true` |
| `kv.inline_decode` | `bool` | `false` |
| `kv.coopmat_bda` | `bool` | `false` |
| `kv.overflow` | `bool` | `false` |
| `kv.overflow_vram_mb` | `Option<u64>` | `None` |
| `kv.overflow_reserve_mb` | `Option<u64>` | `None` |

## `paging`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `paging.cache` | `Option<SizeSpec>` | `None` |
| `paging.ring` | `Option<SizeSpec>` | `None` |
| `paging.ring_slots` | `usize` | `2` |
| `paging.moe_layer_stream` | `bool` | `true` |
| `paging.prefill_upload_async` | `bool` | `true` |
| `paging.prefill_cross_chunk` | `bool` | `true` |
| `paging.expert_prefetch` | `bool` | `false` |
| `paging.moe_size_cache_bias` | `Option<f32>` | `None` |
| `paging.stats` | `bool` | `false` |
| `paging.trace` | `Option<PathBuf>` | `None` |
| `paging.dram` | `Option<SizeSpec>` | `None` |
| `paging.dram_bypass` | `bool` | `false` |
| `paging.host_dma` | `bool` | `true` |
| `paging.layer_major` | `Option<bool>` | `None` |

## `kernels`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `kernels.vulkan` | `VulkanCfg` | `> PartialVulkanCfg` |
| `kernels.metal` | `MetalCfg` | `> PartialMetalCfg` |
| `kernels.cpu` | `CpuCfg` | `> PartialCpuCfg` |
| `kernels.qkv_fuse` | `bool` | `true` |
| `kernels.gated_rmsnorm` | `bool` | `true` |
| `kernels.delta_ab_fuse` | `bool` | `true` |
| `kernels.qwen_hc_down_inject` | `bool` | `true` |
| `kernels.ple_single_parallel` | `bool` | `true` |

## `kernels.vulkan`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `kernels.vulkan.gemv` | `GemvCfg` | `> PartialGemvCfg` |
| `kernels.vulkan.coopmat` | `bool` | `true` |
| `kernels.vulkan.coopmat_8x8` | `bool` | `false` |
| `kernels.vulkan.bf16_coopmat` | `bool` | `false` |
| `kernels.vulkan.f8_coopmat` | `bool` | `false` |
| `kernels.vulkan.f8_prepack` | `bool` | `false` |
| `kernels.vulkan.i8_coopmat` | `bool` | `false` |
| `kernels.vulkan.i8_row_scale` | `bool` | `false` |
| `kernels.vulkan.f16` | `bool` | `true` |
| `kernels.vulkan.i8_dot` | `bool` | `true` |
| `kernels.vulkan.gemm_warp` | `bool` | `true` |
| `kernels.vulkan.gemm_wide_tile` | `bool` | `false` |
| `kernels.vulkan.gemm_direct_a` | `bool` | `false` |
| `kernels.vulkan.small_bm` | `bool` | `true` |
| `kernels.vulkan.bm16` | `bool` | `true` |
| `kernels.vulkan.mmq` | `bool` | `true` |
| `kernels.vulkan.qwen_hc_down_prefetch` | `bool` | `true` |
| `kernels.vulkan.mmq_fallback` | `bool` | `true` |
| `kernels.vulkan.mmv` | `bool` | `true` |
| `kernels.vulkan.mmv_decode` | `bool` | `false` |
| `kernels.vulkan.mmv_m4` | `bool` | `true` |
| `kernels.vulkan.mmv_o4` | `bool` | `true` |
| `kernels.vulkan.mmv_mw` | `Option<bool>` | `None` |
| `kernels.vulkan.mmv_mw_warps` | `Option<usize>` | `None` |
| `kernels.vulkan.mmv_fuse_quant` | `bool` | `false` |
| `kernels.vulkan.mrow` | `bool` | `true` |
| `kernels.vulkan.mrow16` | `bool` | `true` |
| `kernels.vulkan.f32_mrow` | `bool` | `true` |
| `kernels.vulkan.f32_v4` | `bool` | `true` |
| `kernels.vulkan.moe_small_m` | `usize` | `8` |
| `kernels.vulkan.moe_topk_sg` | `bool` | `true` |
| `kernels.vulkan.qsa_score_decode8` | `bool` | `true` |
| `kernels.vulkan.qsa_prepare_fused` | `bool` | `false` |
| `kernels.vulkan.decode_down_overlap` | `Option<bool>` | `None` |
| `kernels.vulkan.decode_sparse_lut` | `Option<bool>` | `None` |
| `kernels.vulkan.decode_early_submit` | `Option<bool>` | `None` |
| `kernels.vulkan.cpu_miss_early_submit` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_input_tap` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_direct_result` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_host_result` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_threads` | `usize` | `0` |
| `kernels.vulkan.cpu_miss_max` | `usize` | `1` |
| `kernels.vulkan.cpu_miss_iq3` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_spin` | `u32` | `16384` |
| `kernels.vulkan.cpu_miss_poll` | `u32` | `0` |
| `kernels.vulkan.cpu_miss_idle_park` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_token_park` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_push` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_validate` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_split_acc` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_grouped_dot` | `bool` | `false` |
| `kernels.vulkan.cpu_miss_flush_denormals` | `bool` | `false` |
| `kernels.vulkan.qsa_topk_parallel` | `bool` | `true` |
| `kernels.vulkan.qsa_topk_scan` | `Option<bool>` | `None` |
| `kernels.vulkan.canvas_chunk_n` | `usize` | `3` |
| `kernels.vulkan.flash_splits` | `Option<usize>` | `None` |
| `kernels.vulkan.flash_bm32` | `bool` | `false` |
| `kernels.vulkan.flash_min_rows` | `usize` | `24` |
| `kernels.vulkan.flash_stage` | `bool` | `false` |
| `kernels.vulkan.flash_dequant` | `bool` | `false` |
| `kernels.vulkan.flash_warp` | `bool` | `true` |
| `kernels.vulkan.nc_fa` | `bool` | `true` |
| `kernels.vulkan.qk_warp` | `bool` | `true` |
| `kernels.vulkan.pv_warp` | `bool` | `true` |
| `kernels.vulkan.pv_splits` | `Option<usize>` | `None` |
| `kernels.vulkan.no_attn_hd_spec` | `bool` | `false` |
| `kernels.vulkan.attn_decode` | `bool` | `true` |
| `kernels.vulkan.q8_decode_d8` | `bool` | `true` |
| `kernels.vulkan.q8_decode_ls128` | `bool` | `true` |
| `kernels.vulkan.q8_decode_ls256` | `bool` | `true` |
| `kernels.vulkan.q8_decode_chunk1024` | `bool` | `true` |
| `kernels.vulkan.q8_qk_f16` | `bool` | `true` |
| `kernels.vulkan.q8_pv_f16` | `bool` | `true` |
| `kernels.vulkan.q8_decode_gqa2` | `bool` | `true` |
| `kernels.vulkan.q8_decode_gqa4` | `bool` | `true` |
| `kernels.vulkan.qsa_prefill_gqa` | `usize` | `12` |
| `kernels.vulkan.qsa_prefill_tile` | `usize` | `32` |
| `kernels.vulkan.qsa_prefill_coopmat` | `bool` | `true` |
| `kernels.vulkan.qsa_prefill_coopmat_pv` | `bool` | `true` |
| `kernels.vulkan.qsa_score_tile` | `usize` | `32` |
| `kernels.vulkan.q8_decode_combine_sg` | `bool` | `true` |
| `kernels.vulkan.mrows_attn` | `Option<bool>` | `None` |
| `kernels.vulkan.mla_sg` | `bool` | `true` |
| `kernels.vulkan.dn_chunk_scan` | `bool` | `true` |
| `kernels.vulkan.dn_chunk` | `bool` | `true` |
| `kernels.vulkan.dn_split` | `bool` | `true` |
| `kernels.vulkan.delta_strided` | `bool` | `true` |
| `kernels.vulkan.push_desc` | `bool` | `true` |
| `kernels.vulkan.pipeline_cache_disk` | `bool` | `true` |
| `kernels.vulkan.no_vram_guard` | `bool` | `false` |
| `kernels.vulkan.no_moe_sm_pool` | `bool` | `false` |
| `kernels.vulkan.no_replay` | `bool` | `false` |
| `kernels.vulkan.gpu_pos` | `bool` | `true` |
| `kernels.vulkan.fuse_add` | `bool` | `true` |
| `kernels.vulkan.bda_chunk_elems` | `Option<u64>` | `None` |
| `kernels.vulkan.bda_chunk_bytes` | `Option<u64>` | `None` |

## `kernels.vulkan.gemv`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `kernels.vulkan.gemv.no_rm` | `bool` | `false` |
| `kernels.vulkan.gemv.rm` | `u32` | `2` |
| `kernels.vulkan.gemv.rm_maxout` | `usize` | `usize::MAX` |
| `kernels.vulkan.gemv.rm_minout` | `usize` | `2048` |
| `kernels.vulkan.gemv.no_sg` | `bool` | `false` |
| `kernels.vulkan.gemv.no_id_sg` | `bool` | `false` |
| `kernels.vulkan.gemv.sg_minout` | `usize` | `2048` |
| `kernels.vulkan.gemv.sg_maxout` | `usize` | `8192` |
| `kernels.vulkan.gemv.sg_nr` | `u32` | `2` |
| `kernels.vulkan.gemv.id_grid_nr` | `u32` | `8` |
| `kernels.vulkan.gemv.id_grid_buffer_nr` | `u32` | `4` |
| `kernels.vulkan.gemv.id_grid_buffer` | `Option<bool>` | `None` |
| `kernels.vulkan.gemv.variant` | `Option<String>` | `Some("reg".to_string())` |

## `kernels.metal`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `kernels.metal.f16_native` | `bool` | `true` |
| `kernels.metal.f32_native` | `bool` | `true` |
| `kernels.metal.bf16_native` | `bool` | `true` |
| `kernels.metal.f16_cmm` | `bool` | `true` |
| `kernels.metal.bf16_cmm` | `bool` | `true` |
| `kernels.metal.f32_cmm` | `bool` | `true` |
| `kernels.metal.f16_rt` | `bool` | `true` |
| `kernels.metal.bf16_rt` | `bool` | `true` |
| `kernels.metal.f32_rt` | `bool` | `true` |
| `kernels.metal.kquant_native` | `bool` | `true` |
| `kernels.metal.q5k_rt` | `bool` | `true` |
| `kernels.metal.rmsnorm_vec4` | `bool` | `true` |
| `kernels.metal.conv1d_par` | `bool` | `true` |
| `kernels.metal.dn_gate_prep` | `bool` | `true` |
| `kernels.metal.dn_norm_prep` | `bool` | `true` |
| `kernels.metal.lmhead_mrv_uncapped` | `bool` | `false` |
| `kernels.metal.deltanet` | `bool` | `true` |
| `kernels.metal.moe` | `bool` | `true` |
| `kernels.metal.pipeline_cache` | `bool` | `true` |

## `kernels.cpu`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `kernels.cpu.spin` | `u32` | `32768` |
| `kernels.cpu.spinpool` | `bool` | `true` |
| `kernels.cpu.repack_mb` | `usize` | `4096` |
| `kernels.cpu.reference` | `bool` | `false` |

## `spec`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `spec.mtp` | `bool` | `false` |
| `spec.mtp_ckpt` | `bool` | `true` |
| `spec.mtp_reprime` | `bool` | `true` |
| `spec.mtp_draft_chain` | `bool` | `true` |
| `spec.mtp_ple_overlap` | `bool` | `true` |
| `spec.draft` | `Option<PathBuf>` | `None` |
| `spec.k` | `usize` | `6` |
| `spec.mtp_context` | `usize` | `32768` |
| `spec.mtp_max_context` | `usize` | `49152` |
| `spec.mtp_concurrent` | `bool` | `false` |
| `spec.debug` | `bool` | `false` |
| `spec.decode_chain` | `usize` | `8` |
| `spec.gpu_draft_prob` | `bool` | `true` |
| `spec.gpu_mtp_accept` | `bool` | `true` |
| `spec.gpu_argmax` | `bool` | `true` |
| `spec.gpu_sample` | `bool` | `true` |
| `spec.gpu_embed` | `bool` | `true` |

## `multi`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `multi.pipeline` | `Option<Vec<usize>>` | `None` |
| `multi.tensor_parallel` | `Option<Vec<usize>>` | `None` |
| `multi.expert_parallel` | `Option<Vec<usize>>` | `None` |
| `multi.pipeline_p2p` | `bool` | `true` |
| `multi.tp_p2p` | `bool` | `true` |
| `multi.ep_p2p` | `bool` | `true` |

## `prof`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `prof.ops` | `bool` | `false` |
| `prof.op_shapes` | `bool` | `false` |
| `prof.stages` | `bool` | `false` |
| `prof.vram` | `bool` | `false` |
| `prof.diffusion_trace` | `bool` | `false` |
| `prof.pager_profile` | `bool` | `false` |
| `prof.timeline_path` | `Option<PathBuf>` | `None` |
| `prof.cpu_miss_fixture_dir` | `Option<PathBuf>` | `None` |
| `prof.timeline_skip_steps` | `usize` | `64` |
| `prof.timeline_steps` | `usize` | `16` |
| `prof.timeline_stride` | `usize` | `128` |
| `prof.timeline_windows` | `usize` | `12` |
| `prof.timeline_lanes` | `usize` | `0` |
| `prof.out` | `Option<PathBuf>` | `None` |
| `prof.metal_device_time` | `MetalDeviceTime` | `MetalDeviceTime::Off` |
| `prof.metal_debug` | `bool` | `false` |

## `serve`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `serve.api_key` | `Option<String>` | `None` |
| `serve.max_tokens_cap` | `u32` | `131_072` |
| `serve.request_timeout_secs` | `u64` | `0` |
| `serve.stats_interval_secs` | `u64` | `5` |
| `serve.shutdown_file` | `Option<PathBuf>` | `None` |
| `serve.embedding_runner` | `Option<PathBuf>` | `None` |

## `hub`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `hub.endpoint` | `String` | `"https://huggingface.co".to_string()` |
| `hub.pull_jobs` | `usize` | `8` |

## `debug`

| 完整路径 | Rust 类型 | 构造默认值 |
| --- | --- | --- |
| `debug.bda_chunk` | `bool` | `false` |
| `debug.coopmat` | `bool` | `false` |
| `debug.wide_dispatch` | `bool` | `false` |
| `debug.chat` | `bool` | `false` |
| `debug.state_trace` | `bool` | `false` |
| `debug.moe_counts` | `bool` | `false` |
| `debug.moe_counts_dump` | `bool` | `false` |
| `debug.poison_uninit` | `bool` | `false` |
| `debug.no_barrier` | `bool` | `false` |
| `debug.full_barrier` | `bool` | `false` |
