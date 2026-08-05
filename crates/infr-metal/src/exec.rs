//! Graph execution: host-side staging (identical to the CPU interpreter) with each op's arithmetic
//! dispatched to a Metal compute kernel. See the crate docs for why this is host-orchestrated.

use crate::{metal_buf, MetalBackend};
use infr_core::backend::{Bindings, Plan};
use infr_core::config::MetalCfg;
use infr_core::error::Error;
use infr_core::exec::Provision;
use infr_core::graph::{Op, TensorKind};
use infr_core::tensor::{DType, TensorId};
use infr_core::Result;
use infr_gguf::dequant::dequant_block;
use metal::foreign_types::ForeignType;
use metal::{
    Buffer as MtlBuffer, CommandBuffer, ComputeCommandEncoder, ComputePipelineState,
    MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::sync::Arc;

/// A quantized weight in the compact FACTORED device form (`infr_gguf::dequant::Factored`):
/// bit-packed codes (4/6/8-bit, chosen by the format's max code), one `(sc, m)` i16 pair per
/// 16-element block, and one `(d, dmin)` f16 pair per `dblk` elements, so
/// `weight = (d*sc)*code + (dmin*m)` — bit-for-bit the `dequant` reference (same f32 multiplies).
/// Q4_K lands at ~6.1 bpw resident and Q6_K at ~8.1, vs 32 for a dequant-to-f32 weight; decode
/// GEMV is bound on exactly this stream.
pub(crate) struct QuiWeight {
    codes: MtlBuffer,
    scm: MtlBuffer,
    dd: MtlBuffer,
    /// Kernel matching the code packing: `linear_quik4`/`linear_quik6`/`linear_quik8`.
    kern: &'static str,
    /// log2(elements per `(d, dmin)` pair): 5 for legacy 32-element blocks, 8 for K-quants.
    dshift: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use infr_core::backend::{Backend, Buffer, BufferUsage};
    use infr_core::graph::{AttnMask, Graph};
    use infr_core::tensor::TensorDesc;

    #[test]
    fn replay_gpu_decode_op_eligibility() {
        let mut g = infr_core::graph::Graph::new();
        let logits = g.input(TensorDesc::new(vec![128], DType::F32));
        let uniform = g.input(TensorDesc::new(vec![64], DType::F32));
        let ids = g.input(TensorDesc::new(vec![2], DType::I32));
        let q4k = g.weight(TensorDesc::new(vec![8, 256], DType::Q4K));
        let q5k = g.weight(TensorDesc::new(vec![8, 256], DType::Q5K));
        let token = g.output(TensorDesc::new(vec![1], DType::F32));
        let gathered = g.output(TensorDesc::new(vec![2, 256], DType::F32));

        assert_eq!(
            replay_gpu_decode_op_supported(
                &Op::Argmax {
                    x: logits,
                    dst: token,
                    n: 128,
                    rows: 1,
                },
                &g,
            ),
            Some(true),
        );
        assert_eq!(
            replay_gpu_decode_op_supported(
                &Op::Argmax {
                    x: logits,
                    dst: token,
                    n: 128,
                    rows: 2,
                },
                &g,
            ),
            Some(false),
        );
        assert_eq!(
            replay_gpu_decode_op_supported(
                &Op::Sample {
                    x: logits,
                    u: uniform,
                    dst: token,
                    n: 128,
                    top_k: 40,
                    temp: 0.8,
                    top_p: 0.95,
                },
                &g,
            ),
            Some(true),
        );
        assert_eq!(
            replay_gpu_decode_op_supported(
                &Op::EmbedGather {
                    ids,
                    table: q4k,
                    dst: gathered,
                    rows: 1,
                    ne: 256,
                    scale: 1.0,
                },
                &g,
            ),
            Some(true),
        );
        assert_eq!(
            replay_gpu_decode_op_supported(
                &Op::EmbedGather {
                    ids,
                    table: q4k,
                    dst: gathered,
                    rows: 2,
                    ne: 256,
                    scale: 1.0,
                },
                &g,
            ),
            Some(false),
        );
        assert_eq!(
            replay_gpu_decode_op_supported(
                &Op::EmbedGather {
                    ids,
                    table: q5k,
                    dst: gathered,
                    rows: 1,
                    ne: 256,
                    scale: 1.0,
                },
                &g,
            ),
            Some(false),
        );
    }

    #[test]
    fn embed_gather_msl_reads_raw_i32_ids() {
        let src = include_str!("../shaders/embed_gather.metal");
        assert_eq!(
            src.matches("device const int*").count(),
            3,
            "quant, f16, and bf16 gather entry points must share the raw-i32 ID ABI",
        );
        assert!(
            !src.contains("device const float*  ids"),
            "numeric-f32 IDs require a host mirror and cannot be replayed",
        );
    }

    #[test]
    fn sample_msl_has_position_dynamic_entrypoint() {
        let src = include_str!("../shaders/elementwise_norms.metal");
        assert!(src.contains("kernel void sample_f32_dyn"));
        assert!(
            src.contains("positions[0]") && src.contains("& 63u"),
            "dynamic sampling must select the runner's position-indexed uniform slot",
        );
    }

    #[test]
    fn sample_msl_has_vocab_split_entrypoints() {
        let src = include_str!("../shaders/elementwise_norms.metal");
        assert!(src.contains("kernel void sample_f32_stage1"));
        assert!(src.contains("kernel void sample_f32_stage2"));
        assert!(src.contains("kernel void sample_f32_stage2_dyn"));
    }

    #[test]
    fn sample_stage1_tracks_selected_logits_per_lane() {
        let src = include_str!("../shaders/elementwise_norms.metal");
        assert!(src.contains("uint used_mask = 0u"));
        assert!(src.contains("used_mask |= 1u << slot"));
    }

    #[test]
    fn qknormrope_uses_combined_sincos() {
        let src = include_str!("../shaders/rope_ffn.metal");
        assert_eq!(src.matches("sincos(ang, c)").count(), 2);
    }

    #[test]
    fn rmsnorm_has_vec4_decode_entrypoint() {
        let src = include_str!("../shaders/elementwise_norms.metal");
        assert!(src.contains("kernel void rmsnorm_vec4_f32"));
        assert!(src.contains("device const float4* x"));
        assert!(src.contains("device const float4* w"));
        assert!(src.contains("device float4*       dst"));
    }

    #[test]
    fn rmsnorm_vec4_policy_only_targets_wide_decode_rows() {
        assert!(prefer_rmsnorm_vec4(1, 2048));
        assert!(prefer_rmsnorm_vec4(4, 5376));
        assert!(!prefer_rmsnorm_vec4(5, 5376));
        assert!(!prefer_rmsnorm_vec4(1, 1152));
        assert!(!prefer_rmsnorm_vec4(1, 2049));
    }

    #[test]
    fn gated_rmsnorm_msl_fuses_silu_on_store() {
        let src = include_str!("../shaders/elementwise_norms.metal");
        assert!(src.contains("kernel void gated_rmsnorm_f32"));
        assert!(src.contains("float silu = z / (1.0f + exp(-z))"));
    }

    /// Match a shader tripwire on TOKENS, not on exact source formatting — the same helper the
    /// `kernel_names` suite uses, and for the same reason: these `.metal` files are macro bodies
    /// with column-aligned `\` continuations, so adding one long line re-pads its neighbours and a
    /// formatter may respace an expression without changing a bit of generated code. A raw
    /// `contains` tripwire goes red on both, and a guard that cries wolf is one the next person
    /// relaxes.
    fn contains_tokens(src: &str, needle: &str) -> bool {
        let despace = |s: &str| -> String { s.chars().filter(|c| !c.is_whitespace()).collect() };
        despace(src).contains(&despace(needle))
    }

    #[test]
    fn deltanet_msl_has_hoisted_gate_scan() {
        let src = include_str!("../shaders/deltanet.metal");
        assert!(contains_tokens(src, "kernel void deltanet_gates_f32"));
        assert!(contains_tokens(src, "host_name(\"deltanet_gates_k4\")"));
        // The 8-row crossover itself: rows=4 measured NEUTRAL and rows=2 REGRESSED, so the
        // prepared path must not be taken below 8 (see `prefer_deltanet_gate_prep`).
        assert!(!prefer_deltanet_gate_prep(4));
        assert!(prefer_deltanet_gate_prep(8));
    }

    #[test]
    fn deltanet_msl_has_hoisted_qk_norm() {
        let src = include_str!("../shaders/deltanet.metal");
        assert!(contains_tokens(src, "kernel void deltanet_norm_f32_t"));
        assert!(contains_tokens(src, "host_name(\"deltanet_prepared_k4\")"));
        // Q/k prep first wins at 8 rows; 4 rows measured neutral because it stays inline.
        assert!(!prefer_deltanet_norm_prep(4));
        assert!(prefer_deltanet_norm_prep(8));
    }

    #[test]
    fn conv1d_msl_has_parallel_prefill_passes() {
        let src = include_str!("../shaders/deltanet.metal");
        assert!(contains_tokens(src, "kernel void conv1d_silu_par_f32"));
        assert!(contains_tokens(src, "kernel void conv1d_shift_f32"));
        assert!(!prefer_conv1d_parallel(2, 4));
        assert!(prefer_conv1d_parallel(3, 4));
    }

    #[test]
    fn argmax_split_policy_only_targets_vocab_scale_inputs() {
        assert_eq!(argmax_split_groups(151_936), Some(38));
        assert_eq!(argmax_split_groups(32_768), Some(8));
        assert_eq!(argmax_split_groups(8_192), None);
        assert_eq!(argmax_split_groups(4_099), None);
    }

    #[test]
    fn sample_split_policy_only_targets_vocab_scale_inputs() {
        assert_eq!(sample_split_groups(151_936), Some(38));
        assert_eq!(sample_split_groups(32_768), Some(8));
        assert_eq!(sample_split_groups(8_192), None);
        assert_eq!(sample_split_groups(4_099), None);
    }

    #[test]
    fn sample_split_sizes_merge_candidates_from_top_k() {
        assert_eq!(sample_split_shape(151_936, 20), Some((38, 760)));
        assert_eq!(sample_split_shape(151_936, 64), Some((38, 2_432)));
        assert_eq!(sample_split_shape(8_192, 20), None);
    }

    #[test]
    fn sample_top_k_clamps_to_group_cap() {
        assert_eq!(sample_top_k(20), 20);
        assert_eq!(sample_top_k(64), 64);
        // A top_k past the shader's per-group SAMPLE_KMAX must clamp so the packed stage-1 param
        // and the scratch sizing agree (no out-of-bounds device write).
        assert_eq!(sample_top_k(100), 64);
        // The candidate count in `sample_split_shape` is derived from the SAME clamp.
        assert_eq!(sample_split_shape(151_936, 100), Some((38, 38 * 64)));
    }

    #[test]
    fn softmax_tier_needs_full_256_wide_group() {
        assert_eq!(softmax_tier(256), SoftmaxTier::Wide);
        assert_eq!(softmax_tier(1024), SoftmaxTier::Wide);
        // A capacity-capped device can't run the 256-wide cross-lane reduction correctly.
        assert_eq!(softmax_tier(255), SoftmaxTier::HostScalar);
        assert_eq!(softmax_tier(0), SoftmaxTier::HostScalar);
    }

    #[test]
    fn qui_linear_kerns_are_format_matched() {
        // Every base `weight_qui` can produce. The single registry must map each base × suffix to
        // the format-matched kernel name for all four cooperative/row-tiled variants.
        let bases = [
            "linear_quik4",
            "linear_quik6",
            "linear_quik8",
            "linear_q4k",
            "linear_q5k",
            "linear_q2k",
            "linear_q3k",
            "linear_q6k",
            "linear_q8_0",
            "linear_q5_0",
            "linear_q4_0",
            "linear_q5_1",
            "linear_q4_1",
            "linear_mxfp4",
            "linear_nvfp4",
            "linear_q2_0",
            "linear_tq2_0",
            "linear_tq1_0",
            "linear_iq4xs",
            "linear_iq4nl",
            "linear_iq2xxs",
            "linear_iq3xxs",
            "linear_iq3s",
            "linear_iq2s",
            "linear_iq2xs",
            "linear_iq1s",
            "linear_iq1m",
        ];
        for base in bases {
            let k = qui_linear_kerns(base)
                .unwrap_or_else(|| panic!("no registry entry for base {base}"));
            assert_eq!(k.hmm, format!("{base}_hmm").as_str());
            assert_eq!(k.cmm, format!("{base}_cmm").as_str());
            assert_eq!(k.rt, format!("{base}_rt").as_str());
            let expected_cmm_ks = format!("{base}_cmm_ks");
            assert_eq!(k.cmm_ks, expected_cmm_ks.as_str());
        }
        // Q2_K/Q3_K are now NATIVE (their `linear_q2k`/`linear_q3k` decode kernels exist), so the
        // registry resolves them like every other native K-quant.
        assert!(qui_linear_kerns("linear_q2k").is_some());
        assert!(qui_linear_kerns("linear_q3k").is_some());
        // An unknown base is still a loud registry miss, never a silent quik8 default.
        assert!(qui_linear_kerns("linear_quik_bogus").is_none());
    }

    #[test]
    fn iq4nl_small_multirow_prefers_rt() {
        assert!(!prefer_iq4nl_rt("linear_iq4nl", 1));
        for m in 2..=4 {
            assert!(prefer_iq4nl_rt("linear_iq4nl", m));
        }
        assert!(!prefer_iq4nl_rt("linear_iq4nl", 5));
        assert!(!prefer_iq4nl_rt("linear_q4_0", 2));
    }

    /// The mrv m-ladder moved to the shared `infr_core::tier` policy (candidate F): the configured
    /// band must reproduce the OLD inline `(2..=8).contains(&m) && out_f < 65536` decision exactly.
    #[test]
    fn mrv_band_matches_inline_formula() {
        use infr_core::tier::{linear_tier, LinearTier};
        let old = |m: usize, out_f: usize| (2..=8).contains(&m) && out_f < 65536;
        for m in 0usize..=12 {
            for &out_f in &[1usize, 1152, 6912, 9216, 65_535, 65_536, 248_320] {
                let tier = linear_tier(m, out_f, &[MRV_BAND]);
                assert_eq!(
                    matches!(tier, LinearTier::MultiRow { .. }),
                    old(m, out_f),
                    "mrv band m={m} out_f={out_f}"
                );
                assert_eq!(m == 1, tier == LinearTier::Decode, "decode m={m}");
            }
        }
        // INFR_METAL_LMHEAD_MRV lifts ONLY the lm_head ceiling — the m window is untouched.
        let lifted = infr_core::tier::MultiRowBand {
            out_f_max: usize::MAX,
            ..MRV_BAND
        };
        assert!(lifted.contains(8, 248_320));
        assert!(!lifted.contains(9, 248_320));
        assert!(!lifted.contains(1, 6912));
    }

    #[test]
    fn attention_flash128_policy_is_exact() {
        assert_eq!(
            select_attention_flash2_kern(
                true,
                4,
                16,
                4096,
                128,
                || true,
                || Some("attnflash2_f16kv_hd128"),
            ),
            Some("attnflash2_c128_f16kv_hd128"),
        );
        assert_eq!(
            select_attention_flash2_kern(
                true,
                4,
                16,
                4096,
                128,
                || false,
                || Some("attnflash2_f16kv_hd128"),
            ),
            Some("attnflash2_f16kv_hd128"),
        );
        assert_eq!(
            select_attention_flash2_kern(true, 4, 16, 4096, 128, || false, || None),
            None,
        );

        let mut probed = false;
        assert_eq!(
            select_attention_flash2_kern(
                true,
                8,
                16,
                4096,
                128,
                || {
                    probed = true;
                    true
                },
                || Some("attnflash2_f16kv_hd128"),
            ),
            Some("attnflash2_f16kv_hd128"),
        );
        assert!(!probed, "non-four-row shapes must not build the C128 PSO");
    }

    #[test]
    fn q5k_row_tile_policy_is_limited_to_four_rows() {
        for m in [1usize, 2, 3, 5] {
            assert!(!prefer_q5k_rt("linear_q5k", m, true));
        }
        assert!(prefer_q5k_rt("linear_q5k", 4, true));
        assert!(!prefer_q5k_rt("linear_q5k", 4, false));
        assert!(!prefer_q5k_rt("linear_q4k", 4, true));
    }

    #[test]
    fn counter_profiles_use_selected_linear_kernel_for_every_row_count() {
        assert_eq!(
            counter_linear_label(true, "linear_q5k_cmm"),
            Some("linear_q5k_cmm"),
        );
        assert_eq!(counter_linear_label(false, "linear_q5k_cmm"), None);
    }

    #[test]
    fn attention_flash_policy_reuses_kv_at_four_rows() {
        assert!(prefer_attention_flash(true, 4, 16, 4096, 128, true));
        assert!(!prefer_attention_flash(true, 3, 16, 4096, 128, true));
        assert!(!prefer_attention_flash(true, 1, 64, 4096, 128, true));
        assert!(!prefer_attention_flash(true, 4, 16, 63, 128, true));
        assert!(!prefer_attention_flash(false, 4, 16, 4096, 128, true));
        assert!(!prefer_attention_flash(true, 4, 16, 4096, 128, false));
        assert!(!prefer_attention_flash(true, 4, 16, 4096, 256, false));
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn replay_gpu_decode_ops_observe_dynamic_inputs() {
        let be = MetalBackend::new().expect("Metal backend");
        let mut g = Graph::new();

        let rope_x = g.input(TensorDesc::new(vec![1, 1, 64], DType::F32));
        let positions = g.input(TensorDesc::new(vec![1], DType::I32));
        let rope_dst = g.internal(TensorDesc::new(vec![1, 1, 64], DType::F32));
        g.push(Op::Rope {
            x: rope_x,
            positions,
            dst: rope_dst,
            rows: 1,
            n_head: 1,
            head_dim: 64,
            rope_dim: 64,
            theta: 10_000.0,
            freq_factors: None,
            x_stride: 0,
        });

        let norm_weight = g.weight(TensorDesc::new(vec![64], DType::F32));
        let norm_dst = g.internal(TensorDesc::new(vec![1, 1, 64], DType::F32));
        g.push(Op::GatedRmsNorm {
            x: rope_x,
            weight: norm_weight,
            gate: rope_x,
            dst: norm_dst,
            rows: 1,
            n_head: 1,
            head_dim: 64,
            eps: 1e-6,
        });

        let q = g.input(TensorDesc::new(vec![1, 1, 64], DType::F32));
        let k_cache = g.input(TensorDesc::new(vec![8, 1, 64], DType::F16));
        let v_cache = g.input(TensorDesc::new(vec![8, 1, 64], DType::F16));
        let attn_dst = g.internal(TensorDesc::new(vec![1, 1, 64], DType::F32));
        g.push(Op::Attention {
            q,
            k_cache,
            v_cache,
            dst: attn_dst,
            rows: 1,
            kv_len: 1,
            n_head: 1,
            n_kv: 1,
            head_dim: 64,
            scale: 0.125,
            mask: AttnMask::Causal,
            pos: 0,
        });

        let ids = g.input(TensorDesc::new(vec![1], DType::I32));
        let table = g.weight(TensorDesc::new(vec![8, 64], DType::F16));
        let gathered = g.output(TensorDesc::new(vec![1, 64], DType::F32));
        g.push(Op::EmbedGather {
            ids,
            table,
            dst: gathered,
            rows: 1,
            ne: 64,
            scale: 1.0,
        });

        let logits = g.input(TensorDesc::new(vec![128], DType::F32));
        let uniform = g.input(TensorDesc::new(vec![64], DType::F32));
        let sampled = g.output(TensorDesc::new(vec![1], DType::F32));
        let greedy = g.output(TensorDesc::new(vec![1], DType::F32));
        g.push(Op::Sample {
            x: logits,
            u: uniform,
            dst: sampled,
            n: 128,
            top_k: 4,
            temp: 1.0,
            top_p: 1.0,
        });
        g.push(Op::Argmax {
            x: logits,
            dst: greedy,
            n: 128,
            rows: 1,
        });

        let alloc = |bytes: &[u8]| -> Box<dyn Buffer> {
            let buf = be
                .alloc(bytes.len().max(4), BufferUsage::Activations)
                .unwrap();
            be.upload(buf.as_ref(), bytes).unwrap();
            buf
        };
        let zeros_f32 = vec![0u8; 64 * 4];
        let zeros_f16_cache = vec![0u8; 8 * 64 * 2];
        let table_values: Vec<u16> = (0..8)
            .flat_map(|row| std::iter::repeat_n(half::f16::from_f32(row as f32).to_bits(), 64))
            .collect();
        let norm_weight_values = [1.0f32; 64];
        let logits_values: Vec<f32> = (0..128).map(|i| -(i as f32) * 0.1).collect();
        let mut uniform_values = [0.0f32; 64];
        uniform_values[0] = 0.01;

        let rope_x_buf = alloc(&zeros_f32);
        let norm_weight_buf = alloc(bytemuck::cast_slice(&norm_weight_values));
        let positions_buf = alloc(bytemuck::cast_slice(&[0i32]));
        let q_buf = alloc(&zeros_f32);
        let k_buf = alloc(&zeros_f16_cache);
        let v_buf = alloc(&zeros_f16_cache);
        let ids_buf = alloc(bytemuck::cast_slice(&[1i32]));
        let table_buf = alloc(bytemuck::cast_slice(&table_values));
        let gathered_buf = alloc(&vec![0u8; 64 * 4]);
        let logits_buf = alloc(bytemuck::cast_slice(&logits_values));
        let uniform_buf = alloc(bytemuck::cast_slice(&uniform_values));
        let sampled_buf = alloc(&[0u8; 4]);
        let greedy_buf = alloc(&[0u8; 4]);

        let mut bindings = Bindings::new();
        bindings.bind(rope_x, rope_x_buf.as_ref());
        bindings.bind(norm_weight, norm_weight_buf.as_ref());
        bindings.bind(positions, positions_buf.as_ref());
        bindings.bind(q, q_buf.as_ref());
        bindings.bind(k_cache, k_buf.as_ref());
        bindings.bind(v_cache, v_buf.as_ref());
        bindings.bind(ids, ids_buf.as_ref());
        bindings.bind(table, table_buf.as_ref());
        bindings.bind(gathered, gathered_buf.as_ref());
        bindings.bind(logits, logits_buf.as_ref());
        bindings.bind(uniform, uniform_buf.as_ref());
        bindings.bind(sampled, sampled_buf.as_ref());
        bindings.bind(greedy, greedy_buf.as_ref());

        let plan = be.compile(&g).unwrap();
        let replay_capable = be.replay_capable(&g, &bindings);
        be.execute(plan.as_ref(), &bindings).unwrap();
        if !replay_capable {
            assert!(
                be.replay.lock().unwrap().is_none(),
                "a capability-capped device must stay on static execution"
            );
            return;
        }
        let first_tape = {
            let replay = be.replay.lock().unwrap();
            let tape = replay.as_ref().expect("first execute must record a tape");
            (tape.fp, tape.entries.len())
        };

        let mut gathered_bytes = vec![0u8; 64 * 4];
        be.download(gathered_buf.as_ref(), &mut gathered_bytes)
            .unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&gathered_bytes)[0], 1.0);
        let mut token_bytes = [0u8; 4];
        be.download(sampled_buf.as_ref(), &mut token_bytes).unwrap();
        assert_eq!(u32::from_le_bytes(token_bytes), 0);

        be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&[6i32]))
            .unwrap();
        be.upload(positions_buf.as_ref(), bytemuck::cast_slice(&[7i32]))
            .unwrap();
        uniform_values[7] = 0.99;
        be.upload(uniform_buf.as_ref(), bytemuck::cast_slice(&uniform_values))
            .unwrap();
        be.execute(plan.as_ref(), &bindings).unwrap();

        let second_tape = {
            let replay = be.replay.lock().unwrap();
            let tape = replay
                .as_ref()
                .expect("second execute must retain the tape");
            (tape.fp, tape.entries.len())
        };
        assert_eq!(
            second_tape, first_tape,
            "second execute rebuilt the replay tape"
        );
        be.download(gathered_buf.as_ref(), &mut gathered_bytes)
            .unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&gathered_bytes)[0], 6.0);
        be.download(sampled_buf.as_ref(), &mut token_bytes).unwrap();
        assert_eq!(u32::from_le_bytes(token_bytes), 3);
        be.download(greedy_buf.as_ref(), &mut token_bytes).unwrap();
        assert_eq!(u32::from_le_bytes(token_bytes), 0);

        let mut chain_bindings = Bindings::new();
        chain_bindings.bind(rope_x, rope_x_buf.as_ref());
        chain_bindings.bind(norm_weight, norm_weight_buf.as_ref());
        chain_bindings.bind(positions, positions_buf.as_ref());
        chain_bindings.bind(q, q_buf.as_ref());
        chain_bindings.bind(k_cache, k_buf.as_ref());
        chain_bindings.bind(v_cache, v_buf.as_ref());
        chain_bindings.bind(ids, ids_buf.as_ref());
        chain_bindings.bind(table, table_buf.as_ref());
        chain_bindings.bind(gathered, gathered_buf.as_ref());
        chain_bindings.bind(logits, logits_buf.as_ref());
        chain_bindings.bind(uniform, uniform_buf.as_ref());
        chain_bindings.bind(sampled, ids_buf.as_ref());
        chain_bindings.bind(greedy, greedy_buf.as_ref());

        uniform_values[0] = 0.01;
        uniform_values[1] = 0.99;
        uniform_values[2] = 0.01;
        be.upload(uniform_buf.as_ref(), bytemuck::cast_slice(&uniform_values))
            .unwrap();
        be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&[6i32]))
            .unwrap();
        be.upload(positions_buf.as_ref(), bytemuck::cast_slice(&[0i32]))
            .unwrap();

        let chain_plan = be.compile(&g).unwrap();
        let ids = be
            .execute_chain(chain_plan.as_ref(), &chain_bindings, 3)
            .unwrap()
            .expect("Metal replay must support device-side token chains");
        assert_eq!(ids, vec![0, 3, 0]);

        be.download(gathered_buf.as_ref(), &mut gathered_bytes)
            .unwrap();
        assert_eq!(
            bytemuck::cast_slice::<u8, f32>(&gathered_bytes)[0],
            3.0,
            "the third replay must gather the second replay's sampled ID",
        );
        be.download(positions_buf.as_ref(), &mut token_bytes)
            .unwrap();
        assert_eq!(i32::from_le_bytes(token_bytes), 2);

        let mut unsupported = Graph::new();
        let bad_rope_x = unsupported.input(TensorDesc::new(vec![1, 1, 64], DType::F32));
        let bad_positions = unsupported.input(TensorDesc::new(vec![1], DType::I32));
        let bad_rope_dst = unsupported.internal(TensorDesc::new(vec![1, 1, 64], DType::F32));
        unsupported.push(Op::Rope {
            x: bad_rope_x,
            positions: bad_positions,
            dst: bad_rope_dst,
            rows: 1,
            n_head: 1,
            head_dim: 64,
            rope_dim: 64,
            theta: 10_000.0,
            freq_factors: None,
            x_stride: 0,
        });
        let bad_ids = unsupported.input(TensorDesc::new(vec![1], DType::I32));
        let bad_table = unsupported.weight(TensorDesc::new(vec![8, 64], DType::F16));
        let bad_gathered = unsupported.output(TensorDesc::new(vec![1, 64], DType::F32));
        unsupported.push(Op::EmbedGather {
            ids: bad_ids,
            table: bad_table,
            dst: bad_gathered,
            rows: 1,
            ne: 64,
            scale: 1.0,
        });
        let bad_logits = unsupported.input(TensorDesc::new(vec![128], DType::F32));
        let bad_uniform = unsupported.input(TensorDesc::new(vec![64], DType::F32));
        let bad_sampled = unsupported.output(TensorDesc::new(vec![1], DType::F32));
        unsupported.push(Op::Sample {
            x: bad_logits,
            u: bad_uniform,
            dst: bad_sampled,
            n: 128,
            top_k: 4,
            temp: 1.0,
            top_p: 1.0,
        });
        be.upload(ids_buf.as_ref(), bytemuck::cast_slice(&[6i32]))
            .unwrap();
        let mut bad_bindings = Bindings::new();
        bad_bindings.bind(bad_rope_x, rope_x_buf.as_ref());
        bad_bindings.bind(bad_positions, positions_buf.as_ref());
        bad_bindings.bind(bad_ids, ids_buf.as_ref());
        bad_bindings.bind(bad_table, table_buf.as_ref());
        bad_bindings.bind(bad_gathered, gathered_buf.as_ref());
        bad_bindings.bind(bad_logits, logits_buf.as_ref());
        bad_bindings.bind(bad_uniform, uniform_buf.as_ref());
        bad_bindings.bind(bad_sampled, ids_buf.as_ref());
        let bad_plan = be.compile(&unsupported).unwrap();
        assert!(
            be.execute_chain(bad_plan.as_ref(), &bad_bindings, 3)
                .unwrap()
                .is_none(),
            "a graph without a replayable decode shape cannot chain",
        );
        be.download(ids_buf.as_ref(), &mut token_bytes).unwrap();
        assert_eq!(
            i32::from_le_bytes(token_bytes),
            6,
            "declining a chain must not execute any graph side effects",
        );
    }
}

/// Where a tensor's current value lives. GPU ops keep their results on the device and only pay a
/// CPU↔GPU round-trip when a host-side op (or the final write-back) actually needs the bytes.
#[derive(Clone, Copy, PartialEq)]
enum Loc {
    Host,
    Device,
}

/// Per-forward execution state: the host mirror (`vals`), a device buffer per tensor (`dev`), where
/// each tensor currently lives (`loc`), and the open command buffer that batches consecutive GPU ops
/// so they share a single commit + wait instead of one per op. When `tape` is armed, every encoded
/// dispatch is also recorded for decode replay (see [`Tape`]), and `posbuf` carries the bound
/// positions buffer the dynamic-pos kernels read.
struct Resident {
    vals: Vec<Vec<f32>>,
    dev: Vec<Option<Arc<MtlBuffer>>>,
    loc: Vec<Loc>,
    cb: Option<CommandBuffer>,
    enc: Option<ComputeCommandEncoder>,
    tape: Option<Vec<TapeEntry>>,
    posbuf: Option<MtlBuffer>,
    /// Linear→Add residual fusion (see [`fusion_cfg`]): Linear op index → (residual tensor, final
    /// dst). The absorbed `Add`s are in the walk's skip set, which lives with the walk (`run_graph`'s
    /// local, handed to `infr_core::exec::run_ops`) rather than here — only `run_op` reads `fused`.
    fused: std::collections::HashMap<usize, (TensorId, TensorId)>,
    /// GPU-counter sampling state (`prof.metal_device_time=counters`): the timestamp sample buffer, the
    /// next free sample slot, the (op name, start-sample) log for this batch, and the op the
    /// walk is currently encoding (sub-dispatches attribute to their parent op).
    csb: Option<metal::CounterSampleBuffer>,
    csb_idx: u64,
    op_samples: Vec<(&'static str, u64)>,
    cur_op: &'static str,
    /// Host-read override of the graph's baked position (see `run_graph`): the seam's replay
    /// loop re-executes one compiled decode graph (baked pos=0) and only rewrites the bound
    /// positions buffer, so on any decode-shaped graph the position-consuming ops (Attention,
    /// WriteKv) must take the CURRENT position — the tape path reads it on the GPU; this covers
    /// every graph the tape can't (MoE, gemma shapes, hd without a dyn instantiation).
    dynpos: Option<u32>,
}

/// Weight-dtype predicate for the shared `Linear→Add` residual fusion (see
/// [`infr_core::fusion::plan_fusions`]): this backend's fused-residual GEMV
/// (`linear_q4k_add`/`linear_q6k_add`/…) coverage — the k-quants Q4_K/Q6_K plus the legacy-round
/// formats (Q8_0/Q5_0/Q4_0/Q5_1/Q4_1).
fn linear_add_weight_ok(dt: DType) -> bool {
    dt.is_legacy_round() || matches!(dt, DType::Q4K | DType::Q6K)
}
static LINEAR_ADD_WEIGHT_OK: fn(DType) -> bool = linear_add_weight_ok;

/// This backend's [`FusionCfg`](infr_core::fusion::FusionCfg): only the `Linear→Add` residual
/// fusion (one dispatch + one dependency stage instead of two, no round-trip of the sublayer
/// output — the decode o_proj/down_proj shape). `plan_fusions` returns the same `(fused, skip)`
/// the former private `linear_add_peephole` did.
fn fusion_cfg() -> infr_core::fusion::FusionCfg<'static> {
    infr_core::fusion::FusionCfg {
        linear_add: Some(infr_core::fusion::LinearAddCfg {
            weight_ok: &LINEAR_ADD_WEIGHT_OK,
            // No escape hatch on this backend (Vulkan has one); always planned.
            enabled: true,
        }),
        rmsnorm_linear: None,
        // F1c's `MoeFfn → Add` fold needs a residual epilogue on the expert accumulate, which this
        // backend's MoE arm does not have; un-planned, so the standalone `Add` still dispatches.
        moe_add: None,
        kv_write: false,
    }
}

/// This backend's hook into the shared op walk ([`infr_core::exec::run_ops`]): the ambient state
/// [`MetalBackend::run_op`] needs, plus the per-op profiling and the seal-the-encoder-before-
/// unwinding cleanup that used to sit in `run_graph`'s two loops. One method over the whole `&Op` —
/// the `match Op::` and every Metal decision inside it stay in `run_op`.
///
/// `PROF` is a CONST parameter, not a field: `run_graph` picks the instantiation once from
/// `profiling_enabled()` and `run_ops` monomorphizes, so the profiling half is compiled out of the
/// hot walk exactly as the two separate loops it replaces kept it out — no per-op branch added.
struct MetalDispatch<'a, 'b, const PROF: bool> {
    be: &'a MetalBackend,
    g: &'a infr_core::graph::Graph,
    bindings: &'a Bindings<'b>,
    r: &'a mut Resident,
}

impl<const PROF: bool> infr_core::exec::OpDispatch for MetalDispatch<'_, '_, PROF> {
    fn dispatch(&mut self, idx: usize, op: &Op) -> Result<()> {
        let t0 = PROF.then(std::time::Instant::now);
        if PROF {
            self.r.cur_op = op_name(op);
        }
        if let Err(e) = self.be.run_op(op, idx, self.g, self.bindings, self.r) {
            // Seal the open encoder before unwinding — dropping it un-ended is a Metal
            // assertion failure that masks the real error.
            self.be.flush(self.r);
            return Err(e);
        }
        if PROF {
            // Counter mode: seal this op's encoder (the command buffer stays open) so the
            // next op's dispatches land in their own sampled encoder.
            if self.be.active_counter_set().is_some() {
                if let Some(e) = self.r.enc.take() {
                    e.end_encoding();
                }
            }
            let enc = t0.expect("PROF ⇒ t0 taken").elapsed();
            // Per-op mode: flush now so this op's GPU wall is isolable (breaks batching).
            let gpu = if self.be.flush_per_op {
                let tg = std::time::Instant::now();
                self.be.flush(self.r);
                Some(tg.elapsed())
            } else {
                None
            };
            let mut pr = self.be.prof.lock().unwrap();
            let name = if self.be.active_counter_set().is_some() {
                self.r.cur_op
            } else {
                op_name(op)
            };
            pr.add_op(name, enc);
            if let Some(gpu) = gpu {
                pr.add_op_gpu(op_name(op), gpu);
            }
        }
        Ok(())
    }
}

/// One recorded dispatch: everything `encode_tg` needs to re-encode it verbatim. The buffer clones
/// are retains, so the tape keeps every transient (intermediate activations, cached weights) alive
/// across replays — replaying reuses the exact buffers the recording ran on.
pub(crate) struct TapeEntry {
    pso: ComputePipelineState,
    bufs: Vec<MtlBuffer>,
    /// Per-buffer byte offset for `set_buffer` (parallel to `bufs`; all zero except the
    /// fused-QKV weight slices — see the Linear arm's `w_off`).
    offs: Vec<u64>,
    /// Bitmask over `bufs`: which buffers this dispatch WRITES. Replay runs a CONCURRENT
    /// encoder and derives the exact barrier placement from these (see `replay_tape`); sites
    /// that don't annotate record `u32::MAX` — treated as writing everything, i.e. serialized.
    wmask: u32,
    params: Vec<u8>,
    threads: usize,
    tg: usize,
}

/// A recorded decode forward for the seam's record-once replay (`Capabilities::decode_replay`):
/// the engine compiles the decode graph once (baked pos=0), binds everything once, and re-executes
/// the same plan per token after uploading the new embedding + position. Metal command buffers are
/// single-use, so "replay" here is re-encoding this flat dispatch list — no graph walk, no host
/// mirror, no transient allocation, no routing. Ops whose behavior depends on the position use
/// DYNAMIC-POS kernels that read the bound positions buffer (`attnvec_dyn_*`, `writekv_dyn_f16`;
/// RoPE already reads it), so the recorded params stay valid for every token.
pub(crate) struct Tape {
    /// Fingerprint of (op discriminants, tensor count, bound buffer addresses): a tape is replayed
    /// only for a graph with the identical op sequence over the identical bindings. Baked pos /
    /// kv_len MAY differ across tokens — the dynamic-pos kernels ignore them by construction.
    fp: u64,
    entries: Vec<TapeEntry>,
}

/// Fingerprint the (graph shape, bindings) pair for tape matching. Op kind + tensor count pins the
/// graph structure; the bound buffer addresses pin the weights/caches/IO. Two graphs that agree on
/// all of that and pass [`replay_shape`] differ at most in baked positions, which replay reads
/// dynamically.
fn replay_fp(g: &infr_core::graph::Graph, bindings: &Bindings) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    };
    mix(g.tensors.len() as u64);
    for op in &g.ops {
        mix(op_name(op).as_ptr() as u64);
    }
    for i in 0..g.tensors.len() {
        if let Some(b) = bindings.get(TensorId(i as u32)) {
            mix(metal_buf(b) as *const _ as u64);
        }
    }
    h
}

fn metal_embed_gather_kern(dt: DType) -> Option<&'static str> {
    match dt {
        DType::F16 => Some("embed_gather_f16"),
        DType::Bf16 => Some("embed_gather_bf16"),
        DType::Q8_0 => Some("embed_gather_q8_0"),
        DType::Q4_0 => Some("embed_gather_q4_0"),
        DType::Q5_0 => Some("embed_gather_q5_0"),
        DType::Q4_1 => Some("embed_gather_q4_1"),
        DType::Q5_1 => Some("embed_gather_q5_1"),
        DType::Mxfp4 => Some("embed_gather_mxfp4"),
        DType::Nvfp4 => Some("embed_gather_nvfp4"),
        DType::Q2_0 => Some("embed_gather_q2_0"),
        DType::Tq2_0 => Some("embed_gather_tq2_0"),
        DType::Tq1_0 => Some("embed_gather_tq1_0"),
        DType::Q4K => Some("embed_gather_q4k"),
        DType::Q6K => Some("embed_gather_q6k"),
        DType::Iq4Nl => Some("embed_gather_iq4nl"),
        DType::Iq4Xs => Some("embed_gather_iq4xs"),
        _ => None,
    }
}

const ARGMAX_SPLIT_CHUNK: usize = 4096;
const SAMPLE_SPLIT_CHUNK: usize = 4096;

fn argmax_split_groups(n: usize) -> Option<usize> {
    (n > 8192).then(|| n.div_ceil(ARGMAX_SPLIT_CHUNK))
}

fn sample_split_groups(n: usize) -> Option<usize> {
    (n > 8192).then(|| n.div_ceil(SAMPLE_SPLIT_CHUNK))
}

/// Per-group candidate cap the split sampler's stage-1 writes (`SAMPLE_KMAX` in the shader). The
/// scratch sizing AND the packed stage-1 `top_k` param MUST both use this clamped value: stage-1
/// writes `min(top_k, 64)` candidates per group into a `min(top_k, 64)`-per-group buffer, so a
/// `top_k > 64` caller that packed the RAW k while the buffer was sized clamped would drive an
/// out-of-bounds device write. Clamp once, here, and use it for both.
fn sample_top_k(top_k: usize) -> usize {
    top_k.min(64)
}

fn sample_split_shape(n: usize, top_k: usize) -> Option<(usize, usize)> {
    sample_split_groups(n).map(|groups| (groups, groups * sample_top_k(top_k)))
}

/// Which softmax path a device can run. `softmax_wide_f32` needs its full 256-lane threadgroup for
/// the cross-lane max/sum reductions; `encode_tg_w` silently clamps the launch to the pipeline cap,
/// so on a capacity-capped device (CI paravirtual) the wide kernel would reduce over `red[8]` slots
/// its <256 launched lanes never wrote. When the 256-wide group doesn't fit, compute the softmax on
/// the host instead of silently returning garbage.
#[derive(Debug, PartialEq, Eq)]
enum SoftmaxTier {
    Wide,
    HostScalar,
}

fn softmax_tier(cap: usize) -> SoftmaxTier {
    if cap >= 256 {
        SoftmaxTier::Wide
    } else {
        SoftmaxTier::HostScalar
    }
}

fn prefer_iq4nl_rt(kern: &str, m: usize) -> bool {
    kern == "linear_iq4nl" && (2..=4).contains(&m)
}

fn prefer_q5k_rt(kern: &str, m: usize, enabled: bool) -> bool {
    enabled && kern == "linear_q5k" && m == 4
}

/// The cooperative / row-tiled kernel variants a factored-or-native quant `Linear` can dispatch,
/// all derived from ONE canonical base per `QuiWeight::kern`. Every base has a real `_hmm`/`_cmm`/
/// `_rt` shader kernel, so those derive as `base + suffix`; a base absent from this registry is a
/// REGISTRY error (a new dtype wired into `weight_qui` without its Linear kernels), never the old
/// silent `_ => "linear_quik8_*"` default that would read foreign-packed codes as quik8.
///
struct QuiLinearKerns {
    hmm: &'static str,
    cmm: &'static str,
    cmm_ks: &'static str,
    rt: &'static str,
}

fn qui_linear_kerns(base: &str) -> Option<QuiLinearKerns> {
    macro_rules! kset {
        ($b:literal) => {
            QuiLinearKerns {
                hmm: concat!($b, "_hmm"),
                cmm: concat!($b, "_cmm"),
                cmm_ks: concat!($b, "_cmm_ks"),
                rt: concat!($b, "_rt"),
            }
        };
    }
    Some(match base {
        "linear_quik4" => kset!("linear_quik4"),
        "linear_quik6" => kset!("linear_quik6"),
        "linear_quik8" => kset!("linear_quik8"),
        "linear_q4k" => kset!("linear_q4k"),
        "linear_q5k" => kset!("linear_q5k"),
        "linear_q2k" => kset!("linear_q2k"),
        "linear_q3k" => kset!("linear_q3k"),
        "linear_q6k" => kset!("linear_q6k"),
        "linear_q8_0" => kset!("linear_q8_0"),
        "linear_q5_0" => kset!("linear_q5_0"),
        "linear_q4_0" => kset!("linear_q4_0"),
        "linear_q5_1" => kset!("linear_q5_1"),
        "linear_q4_1" => kset!("linear_q4_1"),
        "linear_mxfp4" => kset!("linear_mxfp4"),
        "linear_nvfp4" => kset!("linear_nvfp4"),
        "linear_q2_0" => kset!("linear_q2_0"),
        "linear_tq2_0" => kset!("linear_tq2_0"),
        "linear_tq1_0" => kset!("linear_tq1_0"),
        "linear_iq4xs" => kset!("linear_iq4xs"),
        "linear_iq4nl" => kset!("linear_iq4nl"),
        "linear_iq2xxs" => kset!("linear_iq2xxs"),
        "linear_iq3xxs" => kset!("linear_iq3xxs"),
        "linear_iq3s" => kset!("linear_iq3s"),
        "linear_iq2s" => kset!("linear_iq2s"),
        "linear_iq2xs" => kset!("linear_iq2xs"),
        "linear_iq1s" => kset!("linear_iq1s"),
        "linear_iq1m" => kset!("linear_iq1m"),
        _ => return None,
    })
}

/// The multi-row GEMV (`*_mrv`) band of the dense `Linear` m-ladder — the SHAPE half of the
/// decision, in the shared [`infr_core::tier`] policy. `m = 2..=8` (weight bytes hoisted to
/// registers and reused across the rows: one DRAM stream per 8 tokens, where the cooperative tile
/// is latency-bound on its serial k-loop) at any per-layer width, up to the lm_head ceiling: at
/// `out_f >= 65536` mrv spawns `out_f/2 * m` simdgroups and loses to the cmm GEMM tile. `m == 1`
/// keeps the exact-f32 GEMV (decode's precision contract). Which dtypes have an mrv kernel, and
/// the `INFR_METAL_LMHEAD_MRV` hatch, stay at the `Op::Linear` arm — see the rationale written up
/// there.
const MRV_BAND: infr_core::tier::MultiRowBand = infr_core::tier::MultiRowBand {
    min_m: 2,
    max_m: 8,
    narrow_max_m: 8,
    wide_out_f_max: usize::MAX,
    out_f_max: 65535,
};

fn prefer_rmsnorm_vec4(rows: usize, dim: usize) -> bool {
    rows <= 4 && dim >= 2048 && dim.is_multiple_of(4)
}

fn prefer_deltanet_gate_prep(rows: usize) -> bool {
    rows >= 8
}

fn prefer_deltanet_norm_prep(rows: usize) -> bool {
    rows >= 8
}

fn prefer_conv1d_parallel(rows: usize, kernel: usize) -> bool {
    rows >= kernel.saturating_sub(1).max(2)
}

fn prefer_attention_flash(
    f16: bool,
    rows: usize,
    n_head: usize,
    kv_len: usize,
    head_dim: usize,
    flash2_ok: bool,
) -> bool {
    let launch_wide = rows * n_head >= 128;
    let launch_four = rows == 4 && rows * n_head >= 64 && flash2_ok;
    f16 && (launch_wide || launch_four)
        && kv_len >= 64
        && head_dim.is_multiple_of(8)
        && (head_dim <= 128 || flash2_ok)
}

fn select_attention_flash2_kern(
    f16: bool,
    rows: usize,
    n_head: usize,
    kv_len: usize,
    head_dim: usize,
    c128_pipeline_ok: impl FnOnce() -> bool,
    standard_pipeline: impl FnOnce() -> Option<&'static str>,
) -> Option<&'static str> {
    if f16 && rows == 4 && n_head >= 16 && kv_len >= 64 && head_dim == 128 && c128_pipeline_ok() {
        Some("attnflash2_c128_f16kv_hd128")
    } else {
        standard_pipeline()
    }
}

fn counter_linear_label(enabled: bool, kern: &'static str) -> Option<&'static str> {
    enabled.then_some(kern)
}

/// Classify the GPU-resident decode tail ops that may appear on an otherwise replayable graph.
/// `None` leaves the op to the main replay-shape match; `Some(false)` is an explicit rejection.
fn replay_gpu_decode_op_supported(op: &Op, g: &infr_core::graph::Graph) -> Option<bool> {
    match op {
        Op::Argmax { rows, .. } => Some(*rows == 1),
        Op::Sample { .. } => Some(true),
        Op::EmbedGather { table, rows, .. } => {
            Some(*rows == 1 && metal_embed_gather_kern(g.desc(*table).dtype).is_some())
        }
        _ => None,
    }
}

/// Is this graph the decode shape the replay tape supports? Every op must be one the recorder
/// handles fully on-device, attention must be the rows=1 f16 shape with a dynamic-pos kernel
/// instantiation (hd 64/128), and a QkNormRope must exist to name the positions buffer.
///
/// `m` is the backend's [`MetalCfg`] (S6): the Conv1dSilu/DeltaNet gates mirror their arms'
/// `cfg.deltanet` (`INFR_METAL_NODELTA`) and the MoeFfn gate mirrors `cfg.moe`
/// (`INFR_METAL_NOMOE`) — see §6.12, where each key is read BOTH ways with one meaning.
fn replay_shape(g: &infr_core::graph::Graph, bindings: &Bindings, m: &MetalCfg) -> bool {
    use infr_core::graph::TensorKind;
    let mut has_rope = false;
    let mut has_attn = false;
    for op in &g.ops {
        if let Some(supported) = replay_gpu_decode_op_supported(op, g) {
            if !supported {
                return false;
            }
            continue;
        }
        match op {
            Op::RmsNorm { .. }
            | Op::RmsNormAdd { .. }
            | Op::Linear { .. }
            | Op::GatedAct { .. }
            | Op::GatedActFused { .. }
            | Op::Add { .. }
            // Qwen2 q/k/v bias: pos-independent elementwise over a bound weight — replay-safe.
            | Op::AddBias { .. } => {}
            Op::QkNormRope { .. } => has_rope = true,
            // Plain llama-family Rope must NOT tape. The `rope_f32` shader reads a `device const
            // float* pos` buffer (buffer(1)); the arm binds it via `ensure_device`, a FRESH f32
            // buffer widened from the host mirror at record time — so a recorded tape freezes
            // token-0's angle while the seam rewrites the LIVE i32 positions buffer between
            // replays. QkNormRope reads the bound i32 buffer directly and stays replay-valid, but
            // plain Rope has no i32 `rope_f32` variant to bind the live buffer to (adding one is a
            // new, GPU-unverifiable shader). So exclude Rope from replay and keep such graphs
            // (llama-family plain Rope + f16 KV) on the correct per-token walk.
            Op::Rope { .. } => return false,
            // qwen35 (Qwen3-Next) decode ops: all pos-independent, on-device, recurrent state
            // updated in the BOUND buffer — tape-safe when the arm's device gate holds (the
            // host fallbacks can't tape, so the gates mirror the arms').
            Op::QkNorm { .. }
            | Op::GatedRmsNorm { .. }
            | Op::Scale { .. }
            | Op::Copy { .. }
            | Op::CopyStrided { .. } => {}
            Op::Conv1dSilu { state, kernel, .. } => {
                if *kernel > 8
                    || !m.deltanet
                    || bindings.get(*state).is_none()
                {
                    return false;
                }
            }
            Op::DeltaNet {
                state,
                head_k,
                head_v,
                ..
            } => {
                if *head_k > 256
                    || !head_k.is_multiple_of(32)
                    || !(head_v.is_multiple_of(4) && *head_v <= 1024)
                    || !m.deltanet
                    || bindings.get(*state).is_none()
                {
                    return false;
                }
            }
            // MoE decode tapes only when the arm takes the fully-on-device path (the same gate
            // as the MoeFfn arm's `device_ok`): dtypes with expert kernels, in-bounds shapes,
            // escape hatch off. The host fallback computes on the CPU per token — a tape can't
            // represent it.
            Op::MoeFfn {
                gate_exps,
                up_exps,
                down_exps,
                ne,
                n_ff_exp,
                n_used,
                ..
            } => {
                let kq = |t: &TensorId| matches!(g.desc(*t).dtype, DType::Q4K | DType::Q6K);
                if !(kq(gate_exps) && kq(up_exps) && kq(down_exps))
                    || *n_used > 16
                    || *ne % 256 != 0
                    || *n_ff_exp % 256 != 0
                    || !m.moe
                {
                    return false;
                }
            }
            Op::WriteKv { cache, .. } => {
                if !matches!(
                    g.desc(*cache).dtype,
                    DType::F16 | DType::Q8_0 | DType::Q4_0 | DType::Iq4Nl
                ) {
                    return false;
                }
            }
            Op::Mla { .. } => return false,
            Op::Attention {
                rows,
                head_dim,
                k_cache,
                v_cache,
                ..
            } => {
                has_attn = true;
                let (kdt, vdt) = (g.desc(*k_cache).dtype, g.desc(*v_cache).dtype);
                // Only a COUPLED pair has a native-read dyn kernel, so require k == v for EVERY
                // format (q8/f32 are coupled by the runner's clamp anyway; f16/f16 is the
                // default). Checking K alone admitted the reachable MIXED K=f16 + V=q4_0/iq4_nl
                // pair (prepass formats compose per-side — the recommended high-K + quant-V
                // shape), and the tape's f16 dyn kernel would read the quantized V cache as
                // half — corrupt decode under replay. Mixed pairs keep the static prepass path.
                let native = kdt == vdt
                    && matches!(
                        kdt,
                        DType::F16 | DType::Q8_0 | DType::Q4_0 | DType::Iq4Nl
                    );
                if *rows != 1 || !matches!(*head_dim, 64 | 128 | 256) || !native {
                    return false;
                }
            }
            _ => return false,
        }
    }
    // Every non-in-place tensor the recording direct-binds must actually be bound.
    for (i, decl) in g.tensors.iter().enumerate() {
        let needs_binding = matches!(decl.kind, TensorKind::Input | TensorKind::Output);
        if needs_binding && bindings.get(TensorId(i as u32)).is_none() {
            return false;
        }
    }
    has_rope && has_attn
}

/// Dynamic-pos vector attention kernel for a KV cache dtype (`replay_shape` admitted it): the
/// name feeds both the recording route and the pipeline-cap check, so they can never disagree.
fn dyn_attnvec_kern(dt: DType, hd: usize) -> &'static str {
    match (dt, hd) {
        (DType::Q8_0, 64) => "attnvec_dyn_q8kv_hd64",
        (DType::Q8_0, 256) => "attnvec_dyn_q8kv_hd256",
        (DType::Q8_0, _) => "attnvec_dyn_q8kv_hd128",
        (DType::Q4_0, 64) => "attnvec_dyn_q4_0kv_hd64",
        (DType::Q4_0, 256) => "attnvec_dyn_q4_0kv_hd256",
        (DType::Q4_0, _) => "attnvec_dyn_q4_0kv_hd128",
        (DType::Iq4Nl, 64) => "attnvec_dyn_iq4nlkv_hd64",
        (DType::Iq4Nl, 256) => "attnvec_dyn_iq4nlkv_hd256",
        (DType::Iq4Nl, _) => "attnvec_dyn_iq4nlkv_hd128",
        (_, 64) => "attnvec_dyn_f16kv_hd64",
        (_, 256) => "attnvec_dyn_f16kv_hd256",
        _ => "attnvec_dyn_f16kv_hd128",
    }
}

/// Reinterpret raw buffer bytes as `f32` per `dtype` (dequantizing quant/f16/bf16, widening integer
/// tensors). The host-side "read a tensor's value", matching `infr-cpu`'s `bytes_to_f32`.
fn bytes_to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        DType::I32 => bytemuck::cast_slice::<u8, i32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        DType::U32 => bytemuck::cast_slice::<u8, u32>(bytes)
            .iter()
            .map(|&v| v as f32)
            .collect(),
        other => dequant_block(other, bytes).expect("metal backend: host dequant"),
    }
}

/// Stable op label for the profiler.
fn op_name(op: &Op) -> &'static str {
    match op {
        Op::RmsNorm { .. } => "RmsNorm",
        Op::RmsNormAdd { .. } => "RmsNormAdd",
        Op::Softmax { .. } => "Softmax",
        Op::Linear { .. } => "Linear",
        Op::QkNorm { .. } => "QkNorm",
        Op::GatedRmsNorm { .. } => "GatedRmsNorm",
        Op::Rope { .. } => "Rope",
        Op::QkNormRope { .. } => "QkNormRope",
        Op::WriteKv { .. } => "WriteKv",
        Op::Attention { .. } => "Attention",
        Op::Mla { .. } => "Mla",
        Op::GatedAct { .. } => "GatedAct",
        Op::GatedActFused { .. } => "GatedActFused",
        Op::Add { .. } => "Add",
        Op::AddBias { .. } => "AddBias",
        Op::Scale { .. } => "Scale",
        Op::MulVec { .. } => "MulVec",
        Op::Softcap { .. } => "Softcap",
        Op::Argmax { .. } => "Argmax",
        Op::ArgmaxProb { .. } => "ArgmaxProb",
        Op::Sample { .. } => "Sample",
        Op::EmbedGather { .. } => "EmbedGather",
        Op::Copy { .. } => "Copy",
        Op::CopyStrided { .. } => "CopyStrided",
        Op::MoeFfn { .. } => "MoeFfn",
        Op::Conv1dSilu { .. } => "Conv1dSilu",
        Op::DeltaNet { .. } => "DeltaNet",
        Op::MoeSharedExpertAdd { .. } => "MoeSharedExpertAdd",
    }
}

/// Gated-FFN activation applied to the gate value (matches `infr-cpu`'s `act_fn`).
fn act_fn(act: infr_core::graph::Activation, g: f32) -> f32 {
    use infr_core::graph::Activation;
    match act {
        Activation::Silu => g / (1.0 + (-g).exp()),
        Activation::Gelu => 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()),
        Activation::Sigmoid => 1.0 / (1.0 + (-g).exp()),
    }
}

impl MetalBackend {
    // ---- small device-buffer helpers (StorageModeShared = CPU-visible on Apple Silicon) ----

    /// A device buffer initialized from an f32 host slice.
    fn f32_buf(&self, data: &[f32]) -> MtlBuffer {
        let len = (data.len().max(1) * 4) as u64;
        let buf = self
            .device
            .new_buffer(len, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
        }
        buf
    }

    /// A zeroed device buffer of `n` f32s.
    fn zeros_buf(&self, n: usize) -> MtlBuffer {
        self.device
            .new_buffer((n.max(1) * 4) as u64, MTLResourceOptions::StorageModeShared)
    }

    /// A reusable op-scratch buffer (see `MetalBackend::scratch`): get-or-create by
    /// (f32 count, tag).
    fn scratch_buf(&self, n: usize, tag: u8) -> Arc<MtlBuffer> {
        self.scratch
            .lock()
            .unwrap()
            .entry((n, tag))
            .or_insert_with(|| Arc::new(self.zeros_buf(n)))
            .clone()
    }

    /// Read `n` f32s out of a device buffer.
    fn read_f32(buf: &MtlBuffer, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), n);
        }
        out
    }

    /// Expand a quantized / dense-alt KV prefix (`ne` elements) into an f16 scratch buffer `out`,
    /// so the standard f16 attention kernels can read it (the Vulkan dequant->f16 prepass, ported
    /// to Metal). One dispatch; the kernel is a bit-for-bit port of the CPU dequant so the scratch
    /// matches the CPU oracle. Block quants + bf16 run one thread per element; turbo one per
    /// 128-block (the inverse WHT couples all 128).
    fn dequant_kv_f16(
        &self,
        r: &mut Resident,
        dt: DType,
        cache: &MtlBuffer,
        out: &MtlBuffer,
        ne: usize,
    ) -> Result<()> {
        let (kern, threads) = match dt {
            DType::Q4_0 => ("dequant_q4_0_f16", ne),
            DType::Q4_1 => ("dequant_q4_1_f16", ne),
            DType::Q5_0 => ("dequant_q5_0_f16", ne),
            DType::Q5_1 => ("dequant_q5_1_f16", ne),
            DType::Iq4Nl => ("dequant_iq4_nl_f16", ne),
            DType::Bf16 => ("dequant_bf16_f16", ne),
            DType::Turbo2 => ("dequant_turbo2_f16", ne / 128),
            DType::Turbo3 => ("dequant_turbo3_f16", ne / 128),
            DType::Turbo4 => ("dequant_turbo4_f16", ne / 128),
            _ => unreachable!("dequant_kv_f16 for non-prepass dtype {dt:?}"),
        };
        let pso = self.pipelines.get(kern)?;
        self.encode(r, &pso, &[cache, out], &(ne as u32).to_ne_bytes(), threads);
        Ok(())
    }

    /// Encode one kernel dispatch and block until it completes (correctness-first: every op is its
    /// own command buffer, so ops run strictly in graph order with no manual barriers). `data_bufs`
    /// bind to buffer indices `0..k`; `params` (packed scalars) binds at index `k`.
    fn dispatch(
        &self,
        pso: &ComputePipelineState,
        data_bufs: &[&MtlBuffer],
        params: &[u8],
        threads: usize,
    ) {
        if threads == 0 {
            return;
        }
        objc::rc::autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pso);
            for (i, b) in data_bufs.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            if !params.is_empty() {
                enc.set_bytes(
                    data_bufs.len() as u64,
                    params.len() as u64,
                    params.as_ptr() as *const c_void,
                );
            }
            let tg = (pso.max_total_threads_per_threadgroup() as usize).min(threads.max(1)) as u64;
            enc.dispatch_threads(MTLSize::new(threads as u64, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            let t0 = self.profiling_enabled().then(std::time::Instant::now);
            cb.commit();
            cb.wait_until_completed();
            if let Some(t0) = t0 {
                // Wall time of one op's commit + GPU schedule + wait barrier.
                self.prof.lock().unwrap().add_dispatch(t0.elapsed());
            }
        });
    }

    /// Encode one kernel into the batch's shared command buffer (lazily opened) WITHOUT committing.
    /// Metal's automatic hazard tracking inserts the needed barriers between kernels that touch the
    /// same buffers, so the batch runs in graph order inside a single CPU↔GPU round-trip.
    fn encode(
        &self,
        r: &mut Resident,
        pso: &ComputePipelineState,
        bufs: &[&MtlBuffer],
        params: &[u8],
        threads: usize,
    ) {
        // Auto threadgroup: as wide as the pipeline allows (good for the big elementwise grids).
        self.encode_tg(r, pso, bufs, params, threads, 0);
    }

    /// `encode` with a write mask — see `encode_tg_w`.
    fn encode_w(
        &self,
        r: &mut Resident,
        pso: &ComputePipelineState,
        bufs: &[&MtlBuffer],
        wmask: u32,
        params: &[u8],
        threads: usize,
    ) {
        self.encode_tg_w(r, pso, bufs, wmask, params, threads, 0);
    }

    /// As `encode`, but with an explicit threadgroup width (`tg`; 0 = auto). The simdgroup-GEMV
    /// kernels pass 32 (one simdgroup per threadgroup) so a matvec launches `out_f` threadgroups
    /// instead of a handful of wide ones — far more threadgroups for the GPU's cores to interleave,
    /// which hides the memory latency the tiny per-output dot would otherwise stall on.
    fn encode_tg(
        &self,
        r: &mut Resident,
        pso: &ComputePipelineState,
        bufs: &[&MtlBuffer],
        params: &[u8],
        threads: usize,
        tg: usize,
    ) {
        // No write annotation: recorded as writes-everything, so replay serializes around it.
        self.encode_tg_w(r, pso, bufs, u32::MAX, params, threads, tg);
    }

    /// `encode_tg` with an explicit WRITE mask (bit i = `bufs[i]` is written by the dispatch).
    /// The mask only matters on the decode replay tape, where it drives exact barrier placement
    /// under the concurrent encoder — un-annotated dispatches replay serialized.
    #[allow(clippy::too_many_arguments)]
    fn encode_tg_w(
        &self,
        r: &mut Resident,
        pso: &ComputePipelineState,
        bufs: &[&MtlBuffer],
        wmask: u32,
        params: &[u8],
        threads: usize,
        tg: usize,
    ) {
        let with_offs: Vec<(&MtlBuffer, u64)> = bufs.iter().map(|b| (*b, 0)).collect();
        self.encode_tg_off(r, pso, &with_offs, wmask, params, threads, tg);
    }

    /// As `encode_tg_w`, but each buffer binds at a byte offset — the fused-QKV Linear slices
    /// bind the shared concatenated weight at each projection's row offset (`Op::Linear.w_off`).
    #[allow(clippy::too_many_arguments)]
    fn encode_tg_off(
        &self,
        r: &mut Resident,
        pso: &ComputePipelineState,
        bufs: &[(&MtlBuffer, u64)],
        wmask: u32,
        params: &[u8],
        threads: usize,
        tg: usize,
    ) {
        if threads == 0 {
            return;
        }
        if r.enc.is_none() {
            if r.cb.is_none() {
                r.cb = Some(self.queue.new_command_buffer().to_owned());
            }
            let cb = r.cb.as_ref().unwrap();
            // Counter profiling: every encoder carries a stage-boundary timestamp pair, so each
            // op's GPU time is measured IN CONTEXT (the batch still commits once). The walk ends
            // the encoder between ops; everything an op encodes lands in its encoder(s).
            r.enc = Some(if let Some(set) = self.active_counter_set() {
                const CSB_CAP: u64 = 4096;
                if r.csb.is_none() {
                    let d = metal::CounterSampleBufferDescriptor::new();
                    d.set_counter_set(set);
                    d.set_sample_count(CSB_CAP);
                    d.set_storage_mode(metal::MTLStorageMode::Shared);
                    r.csb = self
                        .device
                        .new_counter_sample_buffer_with_descriptor(&d)
                        .ok();
                }
                match r.csb.as_ref() {
                    Some(csb) if r.csb_idx + 2 <= CSB_CAP => {
                        let desc = metal::ComputePassDescriptor::new();
                        let att = desc.sample_buffer_attachments().object_at(0).unwrap();
                        att.set_sample_buffer(csb);
                        att.set_start_of_encoder_sample_index(r.csb_idx);
                        att.set_end_of_encoder_sample_index(r.csb_idx + 1);
                        r.op_samples.push((r.cur_op, r.csb_idx));
                        r.csb_idx += 2;
                        cb.compute_command_encoder_with_descriptor(desc).to_owned()
                    }
                    _ => cb.new_compute_command_encoder().to_owned(),
                }
            } else {
                cb.new_compute_command_encoder().to_owned()
            });
        }
        let enc = r.enc.as_ref().unwrap();
        enc.set_compute_pipeline_state(pso);
        for (i, (b, off)) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), *off);
        }
        if !params.is_empty() {
            enc.set_bytes(
                bufs.len() as u64,
                params.len() as u64,
                params.as_ptr() as *const c_void,
            );
        }
        let cap = pso.max_total_threads_per_threadgroup() as usize;
        let tgw = if tg == 0 { cap } else { tg.min(cap) }.min(threads.max(1)) as u64;
        enc.dispatch_threads(MTLSize::new(threads as u64, 1, 1), MTLSize::new(tgw, 1, 1));
        if let Some(tape) = r.tape.as_mut() {
            tape.push(TapeEntry {
                pso: pso.clone(),
                bufs: bufs.iter().map(|(b, _)| (*b).clone()).collect(),
                offs: bufs.iter().map(|(_, off)| *off).collect(),
                wmask,
                params: params.to_vec(),
                threads,
                tg,
            });
        }
    }

    fn encode_tape(&self, cb: &metal::CommandBufferRef, tape: &Tape) {
        // CONCURRENT dispatch: with the serial type every dispatch implicitly orders after the
        // previous one. Explicit barriers below preserve only the actual RAW/WAW/WAR hazards.
        let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent);
        let mut written: Vec<*const c_void> = Vec::with_capacity(8);
        let mut read: Vec<*const c_void> = Vec::with_capacity(24);
        let mut touched: Vec<&MtlBuffer> = Vec::with_capacity(32);
        for e in &tape.entries {
            let is_w = |i: usize| e.wmask & (1u32 << i.min(31)) != 0;
            let hazard = e.bufs.iter().enumerate().any(|(i, b)| {
                let p = b.as_ptr() as *const c_void;
                written.contains(&p) || (is_w(i) && read.contains(&p))
            });
            if hazard {
                let refs: Vec<&metal::ResourceRef> = touched
                    .iter()
                    .map(|b| b.as_ref() as &metal::ResourceRef)
                    .collect();
                enc.memory_barrier_with_resources(&refs);
                written.clear();
                read.clear();
                touched.clear();
            }
            for (i, b) in e.bufs.iter().enumerate() {
                let p = b.as_ptr() as *const c_void;
                if is_w(i) {
                    written.push(p);
                } else {
                    read.push(p);
                }
                if !touched.iter().any(|t| t.as_ptr() as *const c_void == p) {
                    touched.push(b);
                }
            }
            enc.set_compute_pipeline_state(&e.pso);
            for (i, b) in e.bufs.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), e.offs[i]);
            }
            if !e.params.is_empty() {
                enc.set_bytes(
                    e.bufs.len() as u64,
                    e.params.len() as u64,
                    e.params.as_ptr() as *const c_void,
                );
            }
            let cap = e.pso.max_total_threads_per_threadgroup() as usize;
            let tg = if e.tg == 0 { cap } else { e.tg.min(cap) }.min(e.threads.max(1)) as u64;
            enc.dispatch_threads(MTLSize::new(e.threads as u64, 1, 1), MTLSize::new(tg, 1, 1));
        }
        enc.end_encoding();
    }

    /// Re-encode a recorded tape: one command buffer, the flat dispatch list, commit + wait.
    /// This IS the per-token decode cost on the replay path — no graph walk, no host mirror,
    /// no allocation.
    fn replay_tape(&self, tape: &Tape) {
        objc::rc::autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            self.encode_tape(cb, tape);
            let t0 = self.profiling_enabled().then(std::time::Instant::now);
            cb.commit();
            cb.wait_until_completed();
            if let Some(t0) = t0 {
                let mut pr = self.prof.lock().unwrap();
                pr.add_dispatch(t0.elapsed());
                pr.add_forward();
            }
        });
    }

    /// Close the open batch: end encoding, commit, and wait. A no-op if nothing is buffered.
    fn flush(&self, r: &mut Resident) {
        if let Some(enc) = r.enc.take() {
            enc.end_encoding();
        }
        if let Some(cb) = r.cb.take() {
            let t0 = self.profiling_enabled().then(std::time::Instant::now);
            cb.commit();
            cb.wait_until_completed();
            if let Some(t0) = t0 {
                self.prof.lock().unwrap().add_dispatch(t0.elapsed());
            }
            // Counter mode: the batch is done — resolve this batch's stage-boundary timestamp
            // pairs and attribute per op. The samples are GPU-clock ticks, not wall time:
            // calibrate ticks→ns by bracketing the wait with two CPU/GPU timestamp
            // correlations (sampleTimestamps) and using their ratio.
            if r.csb_idx > 0 {
                if let Some(csb) = r.csb.as_ref() {
                    let (mut cpu1, mut gpu1) = (0u64, 0u64);
                    self.device.sample_timestamps(&mut cpu1, &mut gpu1);
                    let ns_per_tick = {
                        // one correlation before this resolve and the one cached from init
                        let (c0, g0) = self.ts_base;
                        if gpu1 > g0 && cpu1 > c0 {
                            (cpu1 - c0) as f64 / (gpu1 - g0) as f64
                        } else {
                            1.0
                        }
                    };
                    let ts = Self::resolve_counters(csb, r.csb_idx);
                    if self.cfg.prof.metal_debug {
                        eprintln!(
                            "ns_per_tick={ns_per_tick:.4} first samples: {:?}",
                            &ts[..8.min(ts.len())]
                        );
                    }
                    let mut pr = self.prof.lock().unwrap();
                    for &(name, i) in &r.op_samples {
                        if i as usize + 1 >= ts.len() {
                            break;
                        }
                        let (a, b) = (ts[i as usize], ts[i as usize + 1]);
                        if b > a && a != u64::MAX && b != u64::MAX {
                            let ns = ((b - a) as f64 * ns_per_tick) as u64;
                            pr.add_op_gpu(name, std::time::Duration::from_nanos(ns));
                        }
                    }
                }
                r.csb_idx = 0;
                r.op_samples.clear();
            }
        }
    }

    /// Ensure a tensor's value is a device buffer (uploading from the host mirror if needed), and
    /// return it. Tensors already produced on-device are returned as-is — no round-trip.
    fn ensure_device(&self, r: &mut Resident, id: TensorId) -> Arc<MtlBuffer> {
        let i = id.0 as usize;
        if r.dev[i].is_none() || r.loc[i] == Loc::Host {
            r.dev[i] = Some(Arc::new(self.f32_buf(&r.vals[i])));
            r.loc[i] = Loc::Device;
        }
        r.dev[i].clone().unwrap()
    }

    /// Get (or allocate) the persistent device output buffer for a tensor, sized for `n` f32s.
    fn dev_dst(&self, r: &mut Resident, id: TensorId, n: usize) -> Arc<MtlBuffer> {
        let i = id.0 as usize;
        let big_enough = matches!(&r.dev[i], Some(b) if b.length() as usize >= n * 4);
        if !big_enough {
            r.dev[i] = Some(Arc::new(self.zeros_buf(n)));
        }
        r.dev[i].clone().unwrap()
    }

    /// Ensure a tensor's value is in the host mirror `vals`, flushing the batch and downloading from
    /// the device if the latest value lives there. This is the only place a GPU→host stall happens.
    fn ensure_host(&self, r: &mut Resident, g: &infr_core::graph::Graph, id: TensorId) {
        let i = id.0 as usize;
        if r.loc[i] == Loc::Device {
            self.flush(r);
            let n = g.desc(id).numel();
            r.vals[i] = Self::read_f32(r.dev[i].as_ref().unwrap(), n);
            r.loc[i] = Loc::Host;
        }
    }

    fn replay_capable(&self, g: &infr_core::graph::Graph, bindings: &Bindings) -> bool {
        if self.active_counter_set().is_some() || !replay_shape(g, bindings, self.metal()) {
            return false;
        }
        let Some((kern, need)) = g.ops.iter().find_map(|op| match op {
            Op::Attention {
                head_dim: hd @ (64 | 128 | 256),
                k_cache,
                ..
            } => Some((
                dyn_attnvec_kern(g.desc(*k_cache).dtype, *hd as usize),
                if *hd == 256 { 512 } else { 1024 },
            )),
            _ => None,
        }) else {
            return false;
        };
        self.pipelines
            .get(kern)
            .map(|pl| pl.max_total_threads_per_threadgroup() >= need)
            .unwrap_or(false)
    }

    pub(crate) fn execute_graph(&self, plan: &dyn Plan, bindings: &Bindings) -> Result<()> {
        let g = &plan
            .as_any()
            .downcast_ref::<infr_core::backend::GraphPlan>()
            .expect("metal backend: plan is not a GraphPlan")
            .graph;

        // Decode replay: if this exact (graph shape, bindings) pair was recorded, re-encode the
        // tape and skip the graph walk entirely (see `Tape`). The engine's replay loop re-executes
        // one compiled plan with stable bindings, so after the first recorded token every
        // subsequent token takes this path. The dynamic-pos vector kernel REQUIRES its full
        // 1024-thread threadgroup (same silent-clamp hazard as the split kernels), so recording is
        // gated on its pipeline cap — a capped device (CI paravirtual) keeps the per-token path.
        // Counter profiling is per-op analysis: the tape would replay decode tokens without
        // walking ops (only the RECORDED token would ever be attributed — every per-token op
        // reading would silently be a sample of one), so it is disabled under PROFILE=3.
        if self.replay_capable(g, bindings) {
            let fp = replay_fp(g, bindings);
            if let Some(tape) = self.replay.lock().unwrap().as_ref() {
                if tape.fp == fp {
                    self.replay_tape(tape);
                    return Ok(());
                }
            }
            let entries = objc::rc::autoreleasepool(|| self.run_graph(g, bindings, true))?;
            if let Some(entries) = entries {
                *self.replay.lock().unwrap() = Some(Tape { fp, entries });
            }
            return Ok(());
        }

        // Wrap the whole forward in one autorelease pool: the batched command buffers/encoders are
        // retained owned handles, so we drain the pool once per forward instead of once per op.
        objc::rc::autoreleasepool(|| self.run_graph(g, bindings, false).map(|_| ()))
    }

    pub(crate) fn execute_graph_chain(
        &self,
        plan: &dyn Plan,
        bindings: &Bindings,
        n: usize,
    ) -> Result<Option<Vec<u32>>> {
        if n == 0 || n > 64 || self.active_counter_set().is_some() {
            return Ok(None);
        }
        let Some(g) = plan
            .as_any()
            .downcast_ref::<infr_core::backend::GraphPlan>()
            .map(|p| &p.graph)
        else {
            return Ok(None);
        };
        if !self.replay_capable(g, bindings) {
            return Ok(None);
        }
        let positions = g.ops.iter().find_map(|op| match op {
            Op::QkNormRope { positions, .. } | Op::Rope { positions, .. } => Some(*positions),
            _ => None,
        });
        let feed = g.ops.iter().find_map(|op| match op {
            Op::EmbedGather { ids, .. } => Some(*ids),
            _ => None,
        });
        let sampled = g
            .ops
            .iter()
            .find_map(|op| match op {
                Op::Sample { dst, .. } => Some(*dst),
                _ => None,
            })
            .or_else(|| {
                g.ops.iter().find_map(|op| match op {
                    Op::Argmax { dst, .. } => Some(*dst),
                    _ => None,
                })
            });
        let (Some(positions), Some(feed), Some(sampled)) = (positions, feed, sampled) else {
            return Ok(None);
        };
        let (Some(pos_buf), Some(feed_buf), Some(sampled_buf)) = (
            bindings.get(positions),
            bindings.get(feed),
            bindings.get(sampled),
        ) else {
            return Ok(None);
        };
        let pos_buf = metal_buf(pos_buf);
        let feed_buf = metal_buf(feed_buf);
        let sampled_buf = metal_buf(sampled_buf);
        if feed_buf.raw.as_ptr() != sampled_buf.raw.as_ptr() {
            return Ok(None);
        }

        let fp = replay_fp(g, bindings);
        let mut ids = Vec::with_capacity(n);
        let tape_ready = self
            .replay
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|t| t.fp == fp);
        if !tape_ready {
            self.execute_graph(plan, bindings)?;
            let recorded = self
                .replay
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|t| t.fp == fp);
            if !recorded {
                return Ok(None);
            }
            let id = unsafe { *(sampled_buf.raw.contents() as *const u32) };
            ids.push(id);
        }

        let remaining = n - ids.len();
        if remaining == 0 {
            return Ok(Some(ids));
        }
        let result = self.device.new_buffer(
            (remaining * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let advance = self.pipelines.get("advance_position_i32")?;
        let guard = self.replay.lock().unwrap();
        let tape = guard.as_ref().expect("chain checked recorded tape");
        objc::rc::autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            for i in 0..remaining {
                if !ids.is_empty() || i > 0 {
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&advance);
                    enc.set_buffer(0, Some(&pos_buf.raw), 0);
                    enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                    enc.end_encoding();
                }
                self.encode_tape(cb, tape);
                let blit = cb.new_blit_command_encoder();
                blit.copy_from_buffer(&sampled_buf.raw, 0, &result, (i * 4) as u64, 4);
                blit.end_encoding();
            }
            let t0 = self.profiling_enabled().then(std::time::Instant::now);
            cb.commit();
            cb.wait_until_completed();
            if let Some(t0) = t0 {
                let mut pr = self.prof.lock().unwrap();
                pr.add_dispatch(t0.elapsed());
                pr.add_forward();
            }
        });
        let chained = unsafe {
            std::slice::from_raw_parts(result.contents() as *const u32, remaining).to_vec()
        };
        ids.extend(chained);
        Ok(Some(ids))
    }

    fn run_graph(
        &self,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
        record: bool,
    ) -> Result<Option<Vec<TapeEntry>>> {
        // f32 host mirror for every Input/Internal/Output handle (mirrors the CPU interpreter); GPU
        // results stay on-device until a host op or the write-back needs them (see `Loc`/`Resident`).
        // KV caches are written/read in place from their bound buffers (see `direct`).
        let direct = g.in_place_inputs();
        let n = g.tensors.len();
        let mut r = Resident {
            vals: vec![Vec::new(); n],
            dev: vec![None; n],
            loc: vec![Loc::Host; n],
            cb: None,
            enc: None,
            tape: record.then(Vec::new),
            posbuf: None,
            fused: std::collections::HashMap::new(),
            csb: None,
            csb_idx: 0,
            op_samples: Vec::new(),
            cur_op: "",
            dynpos: None,
        };
        // Decode-shaped graph (a rows==1 Attention) with a bound positions buffer: read the
        // CURRENT position off the shared-memory buffer and override the baked pos/kv_len in
        // the position-consuming arms. Single-execute callers bind positions equal to the baked
        // value, so this is the identity for them; the seam's replay loop is where they differ.
        let decode_shaped = g
            .ops
            .iter()
            .any(|op| matches!(op, Op::Attention { rows: 1, .. }));
        if decode_shaped {
            let positions = g.ops.iter().find_map(|op| match op {
                Op::QkNormRope { positions, .. } | Op::Rope { positions, .. } => Some(*positions),
                _ => None,
            });
            if let Some(pid) = positions {
                if let Some(b) = bindings.get(pid) {
                    let buf = metal_buf(b);
                    if buf.len >= 4 {
                        let v = unsafe { *(buf.raw.contents() as *const i32) };
                        if v >= 0 {
                            r.dynpos = Some(v as u32);
                        }
                    }
                }
            }
        }
        let skip = {
            let plan = infr_core::fusion::plan_fusions(g, &fusion_cfg());
            r.fused = plan.linear_add;
            plan.skip
        };
        if record {
            // Recording (`replay_shape` held): the tape must read/write the BOUND buffers, not
            // per-execute host-mirror copies — the engine mutates the bound hidden/positions
            // buffers between replays and reads logits from its bound buffer. So f32 Inputs and
            // Outputs are direct-bound as the tensor's device buffer, and the positions buffer
            // (i32, named by the graph's QkNormRope) is stashed for the dynamic-pos kernels.
            for (i, decl) in g.tensors.iter().enumerate() {
                let id = TensorId(i as u32);
                if direct.contains(&id) {
                    continue;
                }
                let bound = match decl.kind {
                    TensorKind::Input if decl.desc.dtype == DType::F32 => true,
                    TensorKind::Output => true,
                    _ => false,
                };
                if bound {
                    let buf = metal_buf(bindings.get(id).expect("replay_shape checked bindings"));
                    r.dev[i] = Some(Arc::new(buf.raw.clone()));
                    if decl.kind == TensorKind::Input {
                        r.loc[i] = Loc::Device;
                    }
                }
            }
            let positions = g
                .ops
                .iter()
                .find_map(|op| match op {
                    Op::QkNormRope { positions, .. } | Op::Rope { positions, .. } => {
                        Some(*positions)
                    }
                    _ => None,
                })
                .expect("replay_shape checked QkNormRope/Rope");
            r.posbuf = Some(metal_buf(bindings.get(positions).unwrap()).raw.clone());
        }
        // Per-kind working-store setup, classified by the shared `infr_core::exec::Provision` (the
        // cpu interpreter makes the identical calls) — the one Metal-specific arm is the recording
        // case, whose f32 Inputs were direct-BOUND above and so need no host mirror.
        for (id, decl, p) in infr_core::exec::provisions(g) {
            let i = id.0 as usize;
            match p {
                Provision::Zero => r.vals[i] = vec![0f32; decl.desc.numel()],
                Provision::Load if record && decl.desc.dtype == DType::F32 => {}
                Provision::Load => {
                    let buf = metal_buf(bindings.get(id).expect("metal backend: unbound Input"));
                    let bytes = Self::read_bytes(buf);
                    r.vals[i] = bytes_to_f32(&bytes, decl.desc.dtype);
                }
                // A Weight (dequantized at its use site) or an in-place KV Input.
                Provision::Skip => {}
            }
        }

        // The walk (graph order, fused-away indices elided) is the shared `infr_core::exec`
        // skeleton; the per-op body — and the profiling/encoder-sealing that wraps it — stays here,
        // in `MetalDispatch`. `profiling_enabled()` picks the instantiation ONCE, outside the walk,
        // exactly as the two separate loops it replaces did.
        if self.profiling_enabled() {
            infr_core::exec::run_ops(
                &g.ops,
                &skip,
                &mut MetalDispatch::<true> {
                    be: self,
                    g,
                    bindings,
                    r: &mut r,
                },
            )?;
            self.prof.lock().unwrap().add_forward();
        } else {
            infr_core::exec::run_ops(
                &g.ops,
                &skip,
                &mut MetalDispatch::<false> {
                    be: self,
                    g,
                    bindings,
                    r: &mut r,
                },
            )?;
        }

        // Write back Outputs (and mutated f32 Inputs, e.g. recurrent state) to their bound buffers;
        // the in-place KV caches are already current. Shared predicate — `infr_core::exec::
        // writes_back`, the same set the cpu backend copies back.
        for id in infr_core::exec::write_back_targets(g) {
            let i = id.0 as usize;
            if bindings.get(id).is_some() {
                let b = metal_buf(bindings.get(id).unwrap());
                // Direct-bound (recording): the GPU already wrote the bound buffer — nothing to
                // copy, the trailing flush below makes it visible.
                if matches!(&r.dev[i], Some(d) if d.contents() == b.raw.contents()) {
                    continue;
                }
                // Pull the value back to the host mirror (flushes the batch if it's still on-device).
                self.ensure_host(&mut r, g, id);
                let src: &[u8] = bytemuck::cast_slice(&r.vals[i]);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        b.raw.contents() as *mut u8,
                        src.len().min(b.len),
                    );
                }
            }
        }
        // Close any trailing batch (an Internal-only tail never pulled to host).
        self.flush(&mut r);
        Ok(r.tape.take())
    }

    /// Fetch a dequantized weight as a device f32 buffer, cached by bound-buffer address.
    ///
    /// GUARDED against silent OOM corruption: this cache stores the weight at 4 bytes/element,
    /// ballooning a quant tensor 4-8x. Formats without native kernels (IQ2*/IQ3*/TQ*/fp4 — every
    /// Linear falls here) can push a model that FITS in its quantized form past the GPU working
    /// set, and Metal shared-storage allocation doesn't fail there — it silently corrupts under
    /// pressure (observed: a 4B IQ4_XS ballooned to 16 GB on a 12.9 GB working set and generated
    /// garbage with no error). Exceeding the budget is now a hard error naming the fix.
    fn weight_buf(
        &self,
        id: TensorId,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
    ) -> Result<Arc<MtlBuffer>> {
        let buf = metal_buf(bindings.get(id).expect("metal backend: unbound Weight"));
        // Key on the UNDERLYING MTLBuffer, not the wrapper: the runner rebuilds its bindings map
        // per prefill graph, so wrapper addresses change every forward and a wrapper-keyed cache
        // re-repacks each one — hundreds of ms per forward on a factored-format checkpoint. The
        // MTLBuffer lives as long as the uploaded weight, so it is stable across forwards.
        //
        // But its `contents()` POINTER is only the slot, never the identity — Metal re-hands a
        // released buffer's address to the next allocation, and this cache does not retain the
        // weight it dequantized from, so nothing keeps that address occupied. `buf.id()` carries
        // the allocation's uid alongside, and a hit is served only when it matches; see `idcache`.
        let key = buf.id();
        if let Some(w) = self.weight_cache.lock().unwrap().get(key) {
            return Ok(w.clone());
        }
        let dt = g.desc(id).dtype;
        let want = (g.desc(id).numel() * 4) as u64;
        let used = self.device.current_allocated_size();
        let budget = self.device.recommended_max_working_set_size();
        if budget > 0 && used + want > budget {
            return Err(Error::Backend(format!(
                "dequant-cached f32 weight would exceed the GPU working set: \
                 {:.2} GiB allocated + {:.2} GiB for this {dt:?} tensor > {:.2} GiB budget. \
                 {dt:?} has no native Metal kernel, so its weights are cached at f32 (4-8x the \
                 quantized size) and the total no longer fits — proceeding would corrupt \
                 silently. Use a natively-supported quantization (Q4_K_M / Q6_K / Q8_0 / Q5_0 / \
                 Q4_0 / Q5_1 / Q4_1 / MXFP4 / NVFP4 / Q2_0 / TQ1_0 / TQ2_0) or run this \
                 checkpoint on the CPU backend (INFR_DEV=cpu).",
                used as f64 / (1u64 << 30) as f64,
                want as f64 / (1u64 << 30) as f64,
                budget as f64 / (1u64 << 30) as f64,
            )));
        }
        let bytes = Self::read_bytes(buf);
        let f = bytes_to_f32(&bytes, dt);
        let w = Arc::new(self.f32_buf(&f));
        self.weight_cache.lock().unwrap().insert(key, w.clone());
        Ok(w)
    }

    /// A device buffer initialized from a raw byte slice.
    fn bytes_buf(&self, data: &[u8]) -> MtlBuffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            data.len().max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Build (or fetch from cache) a quantized weight in on-device form. Q4_K and Q6_K — the
    /// formats real checkpoints ship — have NATIVE kernels that decode the raw GGUF block bytes,
    /// so the bound weight buffer is used as-is: no host repack, no extra residency, and the
    /// weight stream stays at the format's true size (~4.5 / ~6.6 bpw vs the factored form's
    /// ~6.1 / ~8.1). Everything else goes through the factored form (`dequant_factored`), codes
    /// bit-packed to the narrowest width the format's max code fits. Decode GEMV is bound on the
    /// weight byte stream, so every bit shaved here is throughput.
    fn weight_qui(
        &self,
        id: TensorId,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
    ) -> Arc<QuiWeight> {
        let buf = metal_buf(bindings.get(id).expect("metal backend: unbound Weight"));
        // Identity of the underlying buffer — see `weight_buf` for why neither the wrapper address
        // nor the MTLBuffer's `contents()` pointer will do. (The native-kernel arm below happens to
        // retain the weight's MTLBuffer in `QuiWeight::codes`, which pins its address for as long
        // as the entry lives; the factored `linear_quik*` arm copies the bytes out and retains
        // nothing, so that arm has the same exposure as `weight_buf`.)
        let key = buf.id();
        if let Some(w) = self.qui_cache.lock().unwrap().get(key) {
            return w.clone();
        }
        // INFR_METAL_NO_KQUANT_NATIVE routes the non-Q4_K/Q6_K K-quants (Q5_K/Q2_K/Q3_K) back
        // through the factored quik path (A/B + escape hatch). Q5_K reuses Q4_K's cheap 6-bit scale
        // extraction; Q2_K is affine (per-16 4-bit scale+min) and Q3_K is signed (per-16 6-bit
        // aux-shuffled scale + hmask high bit) — all decode the raw GGUF block so the narrower
        // stream (2.5 / 3.44 bpw) is what the decode GEMV is bound on, no host repack.
        let kq_native = self.metal().kquant_native;
        let native_kern = match g.desc(id).dtype {
            DType::Q4K => Some("linear_q4k"),
            DType::Q5K if kq_native => Some("linear_q5k"),
            DType::Q2K if kq_native => Some("linear_q2k"),
            DType::Q3K if kq_native => Some("linear_q3k"),
            DType::Q6K => Some("linear_q6k"),
            DType::Q8_0 => Some("linear_q8_0"),
            DType::Q5_0 => Some("linear_q5_0"),
            DType::Q4_0 => Some("linear_q4_0"),
            DType::Q5_1 => Some("linear_q5_1"),
            DType::Q4_1 => Some("linear_q4_1"),
            DType::Mxfp4 => Some("linear_mxfp4"),
            DType::Nvfp4 => Some("linear_nvfp4"),
            DType::Q2_0 => Some("linear_q2_0"),
            DType::Tq2_0 => Some("linear_tq2_0"),
            DType::Tq1_0 => Some("linear_tq1_0"),
            DType::Iq4Xs => Some("linear_iq4xs"),
            DType::Iq4Nl => Some("linear_iq4nl"),
            DType::Iq2Xxs => Some("linear_iq2xxs"),
            DType::Iq3Xxs => Some("linear_iq3xxs"),
            DType::Iq3S => Some("linear_iq3s"),
            DType::Iq2S => Some("linear_iq2s"),
            DType::Iq2Xs => Some("linear_iq2xs"),
            DType::Iq1S => Some("linear_iq1s"),
            DType::Iq1M => Some("linear_iq1m"),
            _ => None,
        };
        if let Some(kern) = native_kern {
            // The kernel never reads scm/dd for native formats; bind tiny dummies.
            let w = Arc::new(QuiWeight {
                codes: buf.raw.clone(),
                scm: self.zeros_buf(1),
                dd: self.zeros_buf(1),
                kern,
                dshift: 0,
            });
            self.qui_cache.lock().unwrap().insert(key, w.clone());
            return w;
        }
        let bytes = Self::read_bytes(buf);
        let f = infr_gguf::dequant::dequant_factored(g.desc(id).dtype, &bytes)
            .expect("infr-metal: dequant_factored on an unsupported (non-affine) dtype");
        let maxcode = f.codes.iter().copied().max().unwrap_or(0);
        let (kern, codes_bytes): (&'static str, Vec<u8>) = if maxcode < 16 {
            // two codes per byte, low nibble first
            (
                "linear_quik4",
                f.codes
                    .chunks_exact(2)
                    .map(|p| p[0] | (p[1] << 4))
                    .collect(),
            )
        } else if maxcode < 64 {
            // four codes per three bytes, LSB-first 6-bit stream
            (
                "linear_quik6",
                f.codes
                    .chunks_exact(4)
                    .flat_map(|c| {
                        [
                            c[0] | (c[1] << 6),
                            (c[1] >> 2) | (c[2] << 4),
                            (c[2] >> 4) | (c[3] << 2),
                        ]
                    })
                    .collect(),
            )
        } else {
            ("linear_quik8", f.codes)
        };
        let w = Arc::new(QuiWeight {
            codes: self.bytes_buf(&codes_bytes),
            scm: self.bytes_buf(bytemuck::cast_slice(&f.scm)),
            dd: self.bytes_buf(bytemuck::cast_slice(
                &f.dd.iter().map(|h| h.to_bits()).collect::<Vec<u16>>(),
            )),
            kern,
            dshift: f.dblk.trailing_zeros(),
        });
        self.qui_cache.lock().unwrap().insert(key, w.clone());
        w
    }

    fn read_bytes(buf: &crate::MetalBuffer) -> Vec<u8> {
        let mut v = vec![0u8; buf.len];
        unsafe {
            std::ptr::copy_nonoverlapping(buf.raw.contents() as *const u8, v.as_mut_ptr(), v.len());
        }
        v
    }

    /// Resolve a counter-sample range to raw u64 timestamps. metal-rs 0.33's
    /// `resolve_counter_range` computes its copy size from an EMPTY vec (0 bytes) and then
    /// `set_len`s over uninitialized memory — this reads the returned NSData directly.
    #[allow(unexpected_cfgs)]
    fn resolve_counters(csb: &metal::CounterSampleBufferRef, len: u64) -> Vec<u64> {
        use objc::{msg_send, sel, sel_impl};
        unsafe {
            let range = metal::NSRange {
                location: 0,
                length: len,
            };
            let data: *mut objc::runtime::Object = msg_send![csb, resolveCounterRange: range];
            if data.is_null() {
                return Vec::new();
            }
            let bytes: *const u8 = msg_send![data, bytes];
            let nbytes: usize = msg_send![data, length];
            let n = (nbytes / 8).min(len as usize);
            let mut out = vec![0u64; n];
            std::ptr::copy_nonoverlapping(bytes, out.as_mut_ptr() as *mut u8, n * 8);
            out
        }
    }

    /// Read `len` bytes at `off` from a buffer's (unified-memory) contents.
    fn read_bytes_range(buf: &crate::MetalBuffer, off: usize, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                (buf.raw.contents() as *const u8).add(off),
                v.as_mut_ptr(),
                len,
            );
        }
        v
    }

    /// Dequantize a bound weight to host f32 (small weights: norm/conv/router-adjacent gains).
    fn weight_host(
        &self,
        id: TensorId,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
    ) -> Vec<f32> {
        let buf = metal_buf(bindings.get(id).expect("metal backend: unbound Weight"));
        bytes_to_f32(&Self::read_bytes(buf), g.desc(id).dtype)
    }

    /// Single-row matmul on the GPU: `out[out_f] = x[in_f] · Wᵀ`, W row-major [out_f, in_f] f32.
    fn gpu_matvec(&self, x: &[f32], w: &[f32], in_f: usize, out_f: usize) -> Result<Vec<f32>> {
        let bx = self.f32_buf(x);
        let bw = self.f32_buf(w);
        let bd = self.zeros_buf(out_f);
        let pso = self.pipelines.get("linear_f32")?;
        let mut p = 1u32.to_ne_bytes().to_vec(); // m = 1
        p.extend_from_slice(&(in_f as u32).to_ne_bytes());
        p.extend_from_slice(&(out_f as u32).to_ne_bytes());
        self.dispatch(&pso, &[&bx, &bw, &bd], &p, out_f * 32); // 32 lanes/output — see linear_f32
        Ok(Self::read_f32(&bd, out_f))
    }

    fn run_op(
        &self,
        op: &Op,
        idx: usize,
        g: &infr_core::graph::Graph,
        bindings: &Bindings,
        r: &mut Resident,
    ) -> Result<()> {
        match *op {
            Op::RmsNorm {
                x,
                weight,
                dst,
                rows,
                dim,
                eps,
            } => {
                let (rows, dim) = (rows as usize, dim as usize);
                let bx = self.ensure_device(r, x);
                let bw = self.weight_buf(weight, g, bindings)?;
                let bd = self.dev_dst(r, dst, rows * dim);
                // Decode (few rows): the WIDE kernel — 8 simdgroups per row; the 32-lane form
                // is latency-bound on dim/32 serial loads (~20 us/launch at dim 1152). Prefill
                // keeps one simdgroup per row (the rows themselves fill the GPU).
                // `var(..).is_err()` (not `var_os(..).is_none()`) to match every other
                // INFR_METAL_NO_* kill-switch in this file — one spelling for one meaning.
                // Fetch each wide candidate PSO once and reuse it for the dispatch (the cap read
                // and the launch shared two lookups before). `.ok().filter(cap)` reproduces the
                // old `get(..).map(cap>=256).unwrap_or(false)` — a get error or a sub-cap pipeline
                // both degrade to the next tier, never propagate.
                let vec4_pso = (self.metal().rmsnorm_vec4 && prefer_rmsnorm_vec4(rows, dim))
                    .then(|| self.pipelines.get("rmsnorm_vec4_f32").ok())
                    .flatten()
                    .filter(|pl| pl.max_total_threads_per_threadgroup() >= 256);
                let wide_pso = (rows <= 4)
                    .then(|| self.pipelines.get("rmsnorm_wide_f32").ok())
                    .flatten()
                    .filter(|pl| pl.max_total_threads_per_threadgroup() >= 256);
                let vec4 = vec4_pso.is_some();
                let wide = wide_pso.is_some();
                let pso = if let Some(pl) = vec4_pso {
                    pl
                } else if let Some(pl) = wide_pso {
                    pl
                } else {
                    self.pipelines.get("rmsnorm_f32")?
                };
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(dim as u32).to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                let tgw = if vec4 || wide { 256 } else { 32 };
                self.encode_tg_w(
                    r,
                    &pso,
                    &[bx.as_ref(), bw.as_ref(), bd.as_ref()],
                    1 << 2,
                    &p,
                    rows * tgw,
                    tgw,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::RmsNormAdd {
                x,
                weight,
                dst,
                rows,
                dim,
                eps,
            } => {
                // Metal fallback: read post_norm weight from host, compute on host values.
                self.ensure_host(r, g, x);
                self.ensure_host(r, g, dst);
                let w = self.weight_host(weight, g, bindings);
                let xs = &r.vals[x.0 as usize];
                let mut ds = r.vals[dst.0 as usize].clone();
                let (rows, dim) = (rows as usize, dim as usize);
                for ri in 0..rows {
                    let xrow = &xs[ri * dim..(ri + 1) * dim];
                    let ss: f32 = (0..dim).map(|i| xrow[i] * xrow[i]).sum::<f32>() / dim as f32;
                    let s = 1.0 / (ss + eps).sqrt();
                    for i in 0..dim {
                        ds[ri * dim + i] += xrow[i] * s * w[i];
                    }
                }
                r.vals[dst.0 as usize] = ds;
                r.loc[dst.0 as usize] = Loc::Host;
            }
            // Row-wise softmax (diffusion-gemma's in-graph self-conditioning — see
            // docs/diffusion-gemma.md's Phase-B). UNVERIFIED on real Metal hardware — added blind,
            // mirroring `softmax.comp` (see that shader's doc + `softmax_wide_f32`'s doc).
            Op::Softmax {
                x,
                dst,
                rows,
                dim,
                scale,
                scale_buf,
            } => {
                // Perf (DiffusionGemma denoise, Vulkan-only — see `Op::Softmax::scale_buf`'s
                // doc): Metal's denoise call site never sets this (left on the pre-existing host
                // premultiply path), so this can't fire today. Not implemented here; fail loudly
                // instead of silently ignoring a scale a future caller expected to take effect.
                assert!(
                    scale_buf.is_none(),
                    "Metal Op::Softmax: scale_buf is Vulkan-only, unimplemented here"
                );
                let (rows, dim) = (rows as usize, dim as usize);
                // Gate on the wide kernel's threadgroup cap (see `softmax_tier`); fall back to a
                // per-row host scalar softmax — bit-for-bit the `softmax_wide_f32` math (max over
                // x*scale, sum exp(x*scale - m0), divide) — when the 256-wide group doesn't fit,
                // mirroring the RmsNormAdd host fallback rather than reading uninitialized slots.
                let wide_pso = self.pipelines.get("softmax_wide_f32").ok();
                let cap = wide_pso
                    .as_ref()
                    .map(|pl| pl.max_total_threads_per_threadgroup() as usize)
                    .unwrap_or(0);
                if let (SoftmaxTier::Wide, Some(pso)) = (softmax_tier(cap), &wide_pso) {
                    let bx = self.ensure_device(r, x);
                    let bd = self.dev_dst(r, dst, rows * dim);
                    let mut p = (rows as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(dim as u32).to_ne_bytes());
                    p.extend_from_slice(&scale.to_ne_bytes());
                    self.encode_tg_w(
                        r,
                        pso,
                        &[bx.as_ref(), bd.as_ref()],
                        1 << 1,
                        &p,
                        rows * 256,
                        256,
                    );
                    r.loc[dst.0 as usize] = Loc::Device;
                } else {
                    self.ensure_host(r, g, x);
                    let xs = &r.vals[x.0 as usize];
                    let mut ds = vec![0f32; rows * dim];
                    for ri in 0..rows {
                        let xrow = &xs[ri * dim..(ri + 1) * dim];
                        let mut m0 = f32::NEG_INFINITY;
                        for &v in xrow {
                            m0 = m0.max(v * scale);
                        }
                        let mut sum = 0.0f32;
                        for i in 0..dim {
                            let e = (xrow[i] * scale - m0).exp();
                            ds[ri * dim + i] = e;
                            sum += e;
                        }
                        let inv = 1.0 / sum;
                        for v in &mut ds[ri * dim..(ri + 1) * dim] {
                            *v *= inv;
                        }
                    }
                    r.vals[dst.0 as usize] = ds;
                    r.loc[dst.0 as usize] = Loc::Host;
                }
            }
            Op::QkNorm {
                x,
                weight,
                dst,
                rows,
                n_head,
                head_dim,
                eps,
                x_stride,
            } => {
                // Strided QkNorm reads (qwen35 q+g interleave) are wired only into QkNormRope; the
                // runner always emits packed rows for the standalone QkNorm (llama4 L2-norm, gemma4
                // V-norm, DeltaNet ssm_norm split path).
                assert_eq!(
                    x_stride, 0,
                    "Metal QkNorm: strided x not supported (runner emits packed)"
                );
                let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                let bx = self.ensure_device(r, x);
                let bw = self.weight_buf(weight, g, bindings)?;
                let bd = self.dev_dst(r, dst, rows * nh * hd);
                let pso = self.pipelines.get("qknorm_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                // One simdgroup per (row, head) — see `qknorm_f32`.
                self.encode_tg_w(
                    r,
                    &pso,
                    &[bx.as_ref(), bw.as_ref(), bd.as_ref()],
                    1 << 2,
                    &p,
                    rows * nh * 32,
                    32,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::GatedRmsNorm {
                x,
                weight,
                gate,
                dst,
                rows,
                n_head,
                head_dim,
                eps,
            } => {
                let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                let bx = self.ensure_device(r, x);
                let bw = self.weight_buf(weight, g, bindings)?;
                let bg = self.ensure_device(r, gate);
                let bd = self.dev_dst(r, dst, rows * nh * hd);
                let pso = self.pipelines.get("gated_rmsnorm_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                self.encode_tg_w(
                    r,
                    &pso,
                    &[bx.as_ref(), bw.as_ref(), bg.as_ref(), bd.as_ref()],
                    1 << 3,
                    &p,
                    rows * nh * 32,
                    32,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Linear {
                x,
                weight,
                dst,
                m,
                in_f,
                out_f,
                w_off,
            } => {
                let (m, in_f, out_f) = (m as usize, in_f as usize, out_f as usize);
                let bx = self.ensure_device(r, x);
                // `dev_dst` is deferred into each dispatch branch below: the fused-residual
                // peephole writes the absorbed Add's dst via its OWN `dev_dst` and never touches
                // this one, so allocating it up front left a dead per-op buffer on that path.
                let mut p = (m as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(in_f as u32).to_ne_bytes());
                p.extend_from_slice(&(out_f as u32).to_ne_bytes());
                // 32 lanes (one simdgroup) per output element — see `linear_f32`/`linear_quik*`.
                let wdt = g.desc(weight).dtype;
                if infr_gguf::dequant::is_quant(wdt)
                    || matches!(
                        wdt,
                        DType::Iq4Xs
                            | DType::Iq4Nl
                            | DType::Iq2Xxs
                            | DType::Iq3Xxs
                            | DType::Iq2Xs
                            | DType::Iq2S
                            | DType::Iq3S
                            | DType::Iq1S
                            | DType::Iq1M
                            | DType::Mxfp4
                            | DType::Nvfp4
                            | DType::Q2_0
                            | DType::Tq2_0
                            | DType::Tq1_0
                    )
                {
                    // Native quant: decode the compact factored weight inline — no f32 blow-up.
                    // Kernel matches the weight's code packing (4/6/8-bit). Multi-row (prefill)
                    // takes the row-tiled variant: one simdgroup per (8-row tile, output), each
                    // weight block decoded once for all 8 rows, instead of the GEMV kernel
                    // re-streaming the whole weight matrix once per row.
                    let qw = self.weight_qui(weight, g, bindings);
                    // Fused-QKV slice: `w_off` is a whole-row element offset into the shared
                    // concatenated weight (the runner bakes it as `row0 * in_f`), so it lands on
                    // a block boundary in every stream — bind each stream at the corresponding
                    // byte offset and the kernels see the slice as a standalone [out_f, in_f]
                    // weight. Native K-quant streams are raw GGUF blocks (144 B / 210 B per 256
                    // elements); factored streams are bit-packed codes + 4 B scm per 16 elements
                    // + 4 B (d, dmin) per 2^dshift elements.
                    let e = w_off as u64;
                    let dd_off = (e >> qw.dshift) * 4;
                    let (codes_off, scm_off, dd_off) = match qw.kern {
                        _ if w_off == 0 => (0, 0, 0),
                        // Native kernels read raw GGUF blocks; scm/dd are dummy buffers.
                        "linear_q4k" => (e / 256 * 144, 0, 0),
                        "linear_q6k" => (e / 256 * 210, 0, 0),
                        "linear_q5k" => (e / 256 * 176, 0, 0),
                        "linear_q2k" => (e / 256 * 84, 0, 0),
                        "linear_q3k" => (e / 256 * 110, 0, 0),
                        "linear_q8_0" => (e / 32 * 34, 0, 0),
                        "linear_q5_0" => (e / 32 * 22, 0, 0),
                        "linear_q4_0" => (e / 32 * 18, 0, 0),
                        "linear_q5_1" => (e / 32 * 24, 0, 0),
                        "linear_q4_1" => (e / 32 * 20, 0, 0),
                        "linear_mxfp4" => (e / 32 * 17, 0, 0),
                        "linear_nvfp4" => (e / 64 * 36, 0, 0),
                        "linear_q2_0" => (e / 64 * 18, 0, 0),
                        "linear_tq2_0" => (e / 256 * 66, 0, 0),
                        "linear_tq1_0" => (e / 256 * 54, 0, 0),
                        "linear_iq4xs" => (e / 256 * 136, 0, 0),
                        "linear_iq2xxs" => (e / 256 * 66, 0, 0),
                        "linear_iq3xxs" => (e / 256 * 98, 0, 0),
                        "linear_iq3s" => (e / 256 * 110, 0, 0),
                        "linear_iq2s" => (e / 256 * 82, 0, 0),
                        "linear_iq2xs" => (e / 256 * 74, 0, 0),
                        "linear_iq1s" => (e / 256 * 50, 0, 0),
                        "linear_iq1m" => (e / 256 * 56, 0, 0),
                        "linear_iq4nl" => (e / 32 * 18, 0, 0),
                        "linear_quik4" => (e / 2, e / 4, dd_off),
                        "linear_quik6" => (e / 4 * 3, e / 4, dd_off),
                        _ => (e, e / 4, dd_off),
                    };
                    // `set_buffer` offsets must stay 4-byte aligned; every real fused shape does
                    // (in_f is a multiple of 512 or the format's stride is already 4-aligned) —
                    // reject the pathological remainder instead of tripping API validation.
                    if codes_off % 4 != 0 || scm_off % 4 != 0 || dd_off % 4 != 0 {
                        return Err(Error::Unsupported(
                            "metal: Linear w_off lands off 4-byte alignment".into(),
                        ));
                    }
                    // sgs = simdgroups to launch; tg = threadgroup width. GEMM tiles 8x16 per
                    // simdgroup (4 simdgroups per threadgroup for the staging tile). The GEMM
                    // kernel requires its full 128-thread threadgroup (per-simdgroup staging
                    // tiles indexed by simdgroup_index_in_threadgroup), so it is gated on the
                    // pipeline's own thread cap — see the Attention arm for why that cap can
                    // drop below the kernel's need — and degrades to the row-tiled kernel.
                    // One canonical registry for every `_hmm`/`_cmm`/`_cmm_ks`/`_rt` variant (see
                    // `qui_linear_kerns`) — a base with no entry is a loud registry error, not a
                    // silent quik8 default reading foreign-packed codes.
                    let kerns = qui_linear_kerns(qw.kern).ok_or_else(|| {
                        Error::Unsupported(format!(
                            "metal Linear: no kernel registry entry for base {}",
                            qw.kern
                        ))
                    })?;
                    let hmm_kern = kerns.hmm;
                    let cmm_kern = kerns.cmm;
                    // Prefer the cooperative 32x64 threadgroup tile; per-simdgroup HGEMM covers
                    // the out_f % 64 != 0 leftovers, and both need the full 128-thread group.
                    // m >= 2 (not 16): small multi-row batches — a chat turn's short suffix
                    // prefill, speculative VERIFY's k+1 candidate rows — are weight-stream
                    // bound, so the cooperative tile wins even mostly empty (it reads the
                    // stream ONCE at GEMM rate; the row-tiled shape re-streams at ~1/3 the
                    // bandwidth: 140 ms vs ~45 ms per 8B verify forward). m == 1 keeps the
                    // exact-f32 GEMV (decode's precision contract).
                    // IQ4_NL is the measured exception at m=2..4: its non-linear codebook keeps
                    // the mostly-empty CMM tile decode-bound, while RT reuses each decoded block
                    // across rows. Gemma 6912x1152 measured 39%/25%/8% faster; CMM wins at m=5.
                    // Q5_K is another exception at exactly m=4: RT cuts the affected Qwen3.5
                    // profile bucket by 22% and improves pp4 by 6.1%. Other row counts remain on
                    // their existing routes; INFR_METAL_NO_Q5K_RT restores CMM for A/B.
                    // Fetch each candidate PSO ONCE (the cap read and the later dispatch share it),
                    // preserving the original `?` propagation: `get` only runs — and can only
                    // error — when the shape pre-conditions hold, exactly as before.
                    let q5k_rt = self.metal().q5k_rt;
                    let cmm_pso = if m >= 2
                        && !prefer_iq4nl_rt(qw.kern, m)
                        && !prefer_q5k_rt(qw.kern, m, q5k_rt)
                        && out_f % 64 == 0
                    {
                        Some(self.pipelines.get(cmm_kern)?)
                    } else {
                        None
                    };
                    let cmm_ok = cmm_pso
                        .as_ref()
                        .is_some_and(|pl| pl.max_total_threads_per_threadgroup() >= 128);
                    let hmm_pso = if m >= 16 && out_f % 16 == 0 {
                        Some(self.pipelines.get(hmm_kern)?)
                    } else {
                        None
                    };
                    let hmm_ok = hmm_pso
                        .as_ref()
                        .is_some_and(|pl| pl.max_total_threads_per_threadgroup() >= 128);
                    p.extend_from_slice(&qw.dshift.to_ne_bytes());
                    // Multi-row mul_mv GEMV for m = 2..8 K-quant shapes (speculative verify,
                    // short suffix prefills): weight bytes hoisted to registers and reused
                    // across 8 token rows — one DRAM stream per 8 tokens at GEMV-class
                    // bandwidth, where the cooperative tile is latency-bound on its serial
                    // k-loop at these sizes.
                    let mr_kern = match qw.kern {
                        "linear_q4k" => Some("linear_q4k_mrv"),
                        "linear_q6k" => Some("linear_q6k_mrv"),
                        _ => None,
                    };
                    // mrv's edge (weight bytes hoisted to registers, reused across the 2..8 rows)
                    // was tuned on the per-layer projections (out_f <= 9216). The tied lm_head is
                    // out_f = 248320 — 27x wider — where mrv spawns out_f/2 * m (~870k) simdgroups
                    // and loses to the cmm GEMM tile, which streams the weight once at GEMM rate.
                    // Measured on qwen3.5-4B-MTP verify (m=7): routing the lm_head to cmm is +8% on
                    // the MTP cycle (1.30x -> 1.39x), token-identical to greedy. Gate at 65536 —
                    // comfortably above every per-layer out_f, below any real lm_head/vocab width.
                    // INFR_METAL_LMHEAD_MRV forces the old mrv path (A/B + escape hatch) by
                    // lifting the band's `out_f` ceiling. The m window + that ceiling are the
                    // shared `infr_core::tier` band ([`MRV_BAND`]); the per-dtype kernel choice
                    // above stays Metal's own.
                    let band = infr_core::tier::MultiRowBand {
                        out_f_max: if self.metal().lmhead_mrv_uncapped {
                            usize::MAX
                        } else {
                            MRV_BAND.out_f_max
                        },
                        ..MRV_BAND
                    };
                    let mrv_tier = infr_core::tier::linear_tier(m, out_f, &[band]);
                    if let (infr_core::tier::LinearTier::MultiRow { .. }, Some(kn), 0) =
                        (mrv_tier, mr_kern, w_off)
                    {
                        let pso = self.pipelines.get(kn)?;
                        let bd = self.dev_dst(r, dst, m * out_f);
                        let sgs = out_f.div_ceil(2) * m;
                        if let Some(label) = counter_linear_label(self.counter_set.is_some(), kn) {
                            r.cur_op = label;
                        }
                        self.encode_tg_off(
                            r,
                            &pso,
                            &[
                                (bx.as_ref(), 0),
                                (&qw.codes, codes_off),
                                (&qw.scm, scm_off),
                                (&qw.dd, dd_off),
                                (bd.as_ref(), 0),
                            ],
                            1 << 4,
                            &p,
                            sgs * 32,
                            32,
                        );
                        r.loc[dst.0 as usize] = Loc::Device;
                        return Ok(());
                    }
                    if cmm_ok {
                        // Split-K for small-m deep-k (verify rows, short suffix prefills): the
                        // plain tile's out_f/64 threadgroups underfill the GPU and serialize
                        // the whole k loop. ksplit tiles share the k-dim into an f32 partial
                        // plane, reduced in fixed order. Gated to shapes where each split
                        // still gets >= 4 K-steps.
                        let nto = out_f / 64;
                        let ntm = m.div_ceil(32);
                        let ks_split = if m < 16 {
                            (160 / (nto * ntm).max(1)).min(8).min(in_f / 32 / 4).max(1)
                        } else {
                            1
                        };
                        if ks_split > 1 {
                            let ks_kern = kerns.cmm_ks;
                            let kchunk = (in_f / 32 / ks_split).max(1) * 32;
                            // ceil to cover the tail: ksplit*kchunk must reach in_f
                            let kchunk = if kchunk * ks_split < in_f {
                                kchunk + 32
                            } else {
                                kchunk
                            };
                            let parts = self.scratch_buf(ks_split * m * out_f, 8);
                            let mut kp = p.clone();
                            kp.extend_from_slice(&(ks_split as u32).to_ne_bytes());
                            kp.extend_from_slice(&(kchunk as u32).to_ne_bytes());
                            let pso = self.pipelines.get(ks_kern)?;
                            if let Some(label) =
                                counter_linear_label(self.counter_set.is_some(), ks_kern)
                            {
                                r.cur_op = label;
                            }
                            self.encode_tg_off(
                                r,
                                &pso,
                                &[
                                    (bx.as_ref(), 0),
                                    (&qw.codes, codes_off),
                                    (&qw.scm, scm_off),
                                    (&qw.dd, dd_off),
                                    (parts.as_ref(), 0),
                                ],
                                1 << 4,
                                &kp,
                                ntm * nto * ks_split * 128,
                                128,
                            );
                            let rp_pso = self.pipelines.get("cmm_ks_reduce")?;
                            let bd = self.dev_dst(r, dst, m * out_f);
                            let mut rp = ((m * out_f) as u32).to_ne_bytes().to_vec();
                            rp.extend_from_slice(&(ks_split as u32).to_ne_bytes());
                            if let Some(label) =
                                counter_linear_label(self.counter_set.is_some(), "cmm_ks_reduce")
                            {
                                r.cur_op = label;
                            }
                            self.encode_w(
                                r,
                                &rp_pso,
                                &[parts.as_ref(), bd.as_ref()],
                                1 << 1,
                                &rp,
                                m * out_f,
                            );
                            r.loc[dst.0 as usize] = Loc::Device;
                            return Ok(());
                        }
                        // Reads f32 x directly (casts to f16 while staging) — no cast pass.
                        // `cmm_ok` implies `cmm_pso` is `Some` (both come from the same fetch).
                        let pso = cmm_pso.expect("cmm_ok implies a fetched cmm PSO");
                        let bd = self.dev_dst(r, dst, m * out_f);
                        if let Some(label) =
                            counter_linear_label(self.counter_set.is_some(), cmm_kern)
                        {
                            r.cur_op = label;
                        }
                        self.encode_tg_off(
                            r,
                            &pso,
                            &[
                                (bx.as_ref(), 0),
                                (&qw.codes, codes_off),
                                (&qw.scm, scm_off),
                                (&qw.dd, dd_off),
                                (bd.as_ref(), 0),
                            ],
                            1 << 4,
                            &p,
                            m.div_ceil(32) * (out_f / 64) * 128,
                            128,
                        );
                    } else if hmm_ok {
                        // Prefill GEMM runs on half fragments (see `HGEMM_KERNEL`): cast x to an
                        // f16 scratch first, then one GEMM dispatch reads it. Pooled by (size,tag)
                        // — the cast fully overwrites it, so zero-init is irrelevant.
                        let xh = self.scratch_buf((m * in_f * 2).max(4).div_ceil(4), 21);
                        let bd = self.dev_dst(r, dst, m * out_f);
                        let cast = self.pipelines.get("cast_f32_f16")?;
                        let n = (m * in_f) as u32;
                        if let Some(label) =
                            counter_linear_label(self.counter_set.is_some(), "cast_f32_f16")
                        {
                            r.cur_op = label;
                        }
                        self.encode_w(
                            r,
                            &cast,
                            &[bx.as_ref(), xh.as_ref()],
                            1 << 1,
                            &n.to_ne_bytes(),
                            m * in_f,
                        );
                        // `hmm_ok` implies `hmm_pso` is `Some` (same fetch).
                        let pso = hmm_pso.expect("hmm_ok implies a fetched hmm PSO");
                        if let Some(label) =
                            counter_linear_label(self.counter_set.is_some(), hmm_kern)
                        {
                            r.cur_op = label;
                        }
                        self.encode_tg_off(
                            r,
                            &pso,
                            &[
                                (xh.as_ref(), 0),
                                (&qw.codes, codes_off),
                                (&qw.scm, scm_off),
                                (&qw.dd, dd_off),
                                (bd.as_ref(), 0),
                            ],
                            1 << 4,
                            &p,
                            m.div_ceil(32) * (out_f / 16) * 32,
                            128,
                        );
                    } else {
                        let (kern, threads, tgw): (&'static str, usize, usize) = if m > 1 {
                            (kerns.rt, m.div_ceil(8) * out_f * 32, 32)
                        } else {
                            // The native mul_mv-shape GEMVs cover TWO output rows per simdgroup
                            // (FOUR for Q8_0/Q5_0). Small/mid row counts underfill the GPU with
                            // one simdgroup per row group AND serialize the whole k-dim, so they
                            // take the k-SPLIT variant: 4 cooperating simdgroups per row group
                            // (needs the full 128-thread threadgroup — cap-gated with a fall
                            // back to the single-simdgroup form). Big row counts (the LM head)
                            // already saturate and keep NSG=1.
                            let rpg = match qw.kern {
                                "linear_q4k" | "linear_q6k" => 2usize,
                                "linear_q8_0" | "linear_q5_0" | "linear_q4_0" | "linear_q5_1"
                                | "linear_q4_1" | "linear_iq4nl" => 4,
                                _ => 1,
                            };
                            let groups = out_f.div_ceil(rpg);
                            // The k-dim must be deep enough that each of the 4 simdgroups gets
                            // at least two of the body's blocks-in-flight passes — a shallow k
                            // (0.6B's in_f=1024 is FOUR Q4_K superblocks) leaves simdgroups
                            // idle and pays the reduce for nothing (measured -8% there).
                            let ks_kern = match qw.kern {
                                "linear_q4k" if in_f >= 8192 => Some("linear_q4k_ks"),
                                "linear_q6k" if in_f >= 4096 => Some("linear_q6k_ks"),
                                "linear_q8_0" if in_f >= 2048 => Some("linear_q8_0_ks"),
                                "linear_q5_0" if in_f >= 2048 => Some("linear_q5_0_ks"),
                                "linear_q4_0" if in_f >= 2048 => Some("linear_q4_0_ks"),
                                "linear_q5_1" if in_f >= 2048 => Some("linear_q5_1_ks"),
                                "linear_q4_1" if in_f >= 2048 => Some("linear_q4_1_ks"),
                                _ => None,
                            };
                            let ks = groups <= 4096
                                && ks_kern.is_some_and(|kn| {
                                    self.pipelines
                                        .get(kn)
                                        .map(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                                        .unwrap_or(false)
                                });
                            if ks {
                                (ks_kern.unwrap(), groups * 128, 128)
                            } else {
                                (qw.kern, groups * 32, 32)
                            }
                        };
                        // Residual peephole: this Linear absorbed the following Add — take the
                        // fused-residual variant and write the Add's dst directly.
                        if let Some(&(res, fdst)) = r.fused.get(&idx) {
                            let fk = match kern {
                                "linear_q4k" => "linear_q4k_add",
                                "linear_q8_0" => "linear_q8_0_add",
                                "linear_q5_0" => "linear_q5_0_add",
                                "linear_q4_0" => "linear_q4_0_add",
                                "linear_q5_1" => "linear_q5_1_add",
                                "linear_q4_1" => "linear_q4_1_add",
                                "linear_iq4nl" => "linear_iq4nl_add",
                                "linear_q4k_ks" => "linear_q4k_ks_add",
                                "linear_q6k_ks" => "linear_q6k_ks_add",
                                "linear_q8_0_ks" => "linear_q8_0_ks_add",
                                "linear_q5_0_ks" => "linear_q5_0_ks_add",
                                "linear_q4_0_ks" => "linear_q4_0_ks_add",
                                "linear_q5_1_ks" => "linear_q5_1_ks_add",
                                "linear_q4_1_ks" => "linear_q4_1_ks_add",
                                _ => "linear_q6k_add",
                            };
                            let bres = self.ensure_device(r, res);
                            let bfd = self.dev_dst(r, fdst, out_f);
                            let pso = self.pipelines.get(fk)?;
                            if let Some(label) =
                                counter_linear_label(self.counter_set.is_some(), fk)
                            {
                                r.cur_op = label;
                            }
                            self.encode_tg_off(
                                r,
                                &pso,
                                &[
                                    (bx.as_ref(), 0),
                                    (&qw.codes, codes_off),
                                    (&qw.scm, scm_off),
                                    (&qw.dd, dd_off),
                                    (bfd.as_ref(), 0),
                                    (bres.as_ref(), 0),
                                ],
                                1 << 4,
                                &p,
                                threads,
                                tgw,
                            );
                            r.loc[fdst.0 as usize] = Loc::Device;
                            return Ok(());
                        }
                        // Deferred past the fused check above (which returns early via `bfd`).
                        let bd = self.dev_dst(r, dst, m * out_f);
                        let pso = self.pipelines.get(kern)?;
                        if let Some(label) = counter_linear_label(self.counter_set.is_some(), kern)
                        {
                            r.cur_op = label;
                        }
                        self.encode_tg_off(
                            r,
                            &pso,
                            &[
                                (bx.as_ref(), 0),
                                (&qw.codes, codes_off),
                                (&qw.scm, scm_off),
                                (&qw.dd, dd_off),
                                (bd.as_ref(), 0),
                            ],
                            1 << 4,
                            &p,
                            threads,
                            tgw,
                        );
                    }
                } else {
                    // dev_dst deferred here (see the quant branch): every native-float sub-path
                    // below writes it, none returns before it.
                    let bd = self.dev_dst(r, dst, m * out_f);
                    let mcfg = self.metal();
                    let f16_native = wdt == DType::F16 && mcfg.f16_native;
                    let f32_native = wdt == DType::F32 && mcfg.f32_native;
                    let bf16_native = wdt == DType::Bf16 && mcfg.bf16_native;
                    // Fetch each candidate cmm PSO ONCE; the cap read and the dispatch share it
                    // (the three are mutually exclusive by `wdt`). `?` still only fires when the
                    // shape pre-conditions hold, exactly as the old `&& get(..)?` chain did.
                    let f16_cmm_pso = if f16_native && m >= 8 && out_f % 64 == 0 && mcfg.f16_cmm {
                        Some(self.pipelines.get("linear_f16_cmm")?)
                    } else {
                        None
                    };
                    let bf16_cmm_pso = if bf16_native && m >= 6 && out_f % 64 == 0 && mcfg.bf16_cmm
                    {
                        Some(self.pipelines.get("linear_bf16_cmm")?)
                    } else {
                        None
                    };
                    let f32_cmm_pso = if f32_native && m >= 8 && out_f % 64 == 0 && mcfg.f32_cmm {
                        Some(self.pipelines.get("linear_f32_cmm")?)
                    } else {
                        None
                    };
                    let cap_ok =
                        |p: &Option<ComputePipelineState>| -> Option<ComputePipelineState> {
                            p.clone()
                                .filter(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                        };
                    let native_cmm: Option<(&'static str, ComputePipelineState)> =
                        if let Some(pl) = cap_ok(&f16_cmm_pso) {
                            Some(("linear_f16_cmm", pl))
                        } else if let Some(pl) = cap_ok(&bf16_cmm_pso) {
                            Some(("linear_bf16_cmm", pl))
                        } else if let Some(pl) = cap_ok(&f32_cmm_pso) {
                            Some(("linear_f32_cmm", pl))
                        } else {
                            None
                        };
                    let f16_rt = f16_native && (2..16).contains(&m) && mcfg.f16_rt;
                    let bf16_rt = bf16_native && m >= 2 && mcfg.bf16_rt;
                    let f32_rt = f32_native && (2..16).contains(&m) && mcfg.f32_rt;
                    let native_rt = if f16_rt {
                        Some("linear_f16_rt")
                    } else if bf16_rt {
                        Some("linear_bf16_rt")
                    } else if f32_rt {
                        Some("linear_f32_rt")
                    } else {
                        None
                    };
                    let (kern, elem_bytes) = match wdt {
                        DType::F16 if f16_native => ("linear_f16", 2u64),
                        DType::F32 if f32_native => ("linear_f32", 4u64),
                        DType::Bf16 if bf16_native => ("linear_bf16", 2u64),
                        _ => ("linear_f32", 4u64),
                    };
                    if let Some((cmm_kern, cmm)) = native_cmm {
                        if let Some(label) =
                            counter_linear_label(self.counter_set.is_some(), cmm_kern)
                        {
                            r.cur_op = label;
                        }
                        let bw =
                            metal_buf(bindings.get(weight).expect("metal backend: unbound Weight"));
                        let dummy = self.scratch_buf(1, 14);
                        let mut cp = p.clone();
                        cp.extend_from_slice(&0u32.to_ne_bytes());
                        self.encode_tg_off(
                            r,
                            &cmm,
                            &[
                                (bx.as_ref(), 0),
                                (&bw.raw, w_off as u64 * elem_bytes),
                                (dummy.as_ref(), 0),
                                (dummy.as_ref(), 0),
                                (bd.as_ref(), 0),
                            ],
                            1 << 4,
                            &cp,
                            m.div_ceil(32) * (out_f / 64) * 128,
                            128,
                        );
                    } else if let Some(rt_kern) = native_rt {
                        let rt = self.pipelines.get(rt_kern)?;
                        if let Some(label) =
                            counter_linear_label(self.counter_set.is_some(), rt_kern)
                        {
                            r.cur_op = label;
                        }
                        let bw =
                            metal_buf(bindings.get(weight).expect("metal backend: unbound Weight"));
                        let dummy = self.scratch_buf(1, 14);
                        let mut rp = p.clone();
                        rp.extend_from_slice(&0u32.to_ne_bytes());
                        self.encode_tg_off(
                            r,
                            &rt,
                            &[
                                (bx.as_ref(), 0),
                                (&bw.raw, w_off as u64 * elem_bytes),
                                (dummy.as_ref(), 0),
                                (dummy.as_ref(), 0),
                                (bd.as_ref(), 0),
                            ],
                            1 << 4,
                            &rp,
                            m.div_ceil(8) * out_f * 32,
                            32,
                        );
                    } else if f16_native || f32_native || bf16_native {
                        // Read uploaded float weights directly. F16 also halves the stream; both
                        // avoid a host read and redundant f32 device-cache allocation.
                        let bw =
                            metal_buf(bindings.get(weight).expect("metal backend: unbound Weight"));
                        let pso = self.pipelines.get(kern)?;
                        if let Some(label) = counter_linear_label(self.counter_set.is_some(), kern)
                        {
                            r.cur_op = label;
                        }
                        self.encode_tg_off(
                            r,
                            &pso,
                            &[
                                (bx.as_ref(), 0),
                                (&bw.raw, w_off as u64 * elem_bytes),
                                (bd.as_ref(), 0),
                            ],
                            1 << 2,
                            &p,
                            m * out_f * 32,
                            32,
                        );
                    } else {
                        // bf16 or a forced-control float weight: dequant-to-f32 device cache.
                        let bw = self.weight_buf(weight, g, bindings)?;
                        let pso = self.pipelines.get(kern)?;
                        if let Some(label) = counter_linear_label(self.counter_set.is_some(), kern)
                        {
                            r.cur_op = label;
                        }
                        self.encode_tg_off(
                            r,
                            &pso,
                            &[
                                (bx.as_ref(), 0),
                                (bw.as_ref(), w_off as u64 * elem_bytes),
                                (bd.as_ref(), 0),
                            ],
                            1 << 2,
                            &p,
                            m * out_f * 32,
                            32,
                        );
                    }
                }
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Rope {
                x,
                positions,
                dst,
                rows,
                n_head,
                head_dim,
                rope_dim,
                theta,
                freq_factors,
                x_stride,
            } => {
                // The runner only strides the fused QkNormRope (qwen35 q+g interleave); the plain
                // llama-family Rope always reads packed rows.
                assert_eq!(
                    x_stride, 0,
                    "Metal Rope: strided x not supported (runner emits packed)"
                );
                let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                let bx = self.ensure_device(r, x);
                let bpos = self.ensure_device(r, positions);
                let bff = match freq_factors {
                    Some(f) => self.ensure_device(r, f),
                    None => Arc::new(self.zeros_buf(1)),
                };
                let bd = self.dev_dst(r, dst, rows * nh * hd);
                let pso = self.pipelines.get("rope_f32")?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&(rope_dim).to_ne_bytes());
                p.extend_from_slice(&theta.to_ne_bytes());
                p.extend_from_slice(&(freq_factors.is_some() as u32).to_ne_bytes());
                // One thread per (row, head, pair) + pass-through dims — see `rope_f32`.
                let per = (rope_dim as usize / 2) + (hd - rope_dim as usize);
                self.encode_w(
                    r,
                    &pso,
                    &[bx.as_ref(), bpos.as_ref(), bff.as_ref(), bd.as_ref()],
                    1 << 3,
                    &p,
                    rows * nh * per,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::QkNormRope {
                x,
                weight,
                positions,
                dst,
                rows,
                n_head,
                head_dim,
                rope_dim,
                theta,
                eps,
                freq_factors,
                x_stride,
            } => {
                let (rows, nh, hd) = (rows as usize, n_head as usize, head_dim as usize);
                let bx = self.ensure_device(r, x);
                let bw = self.weight_buf(weight, g, bindings)?;
                // The kernel reads the bound i32 positions buffer directly (exact widening in
                // the shader) — no host round-trip, and the decode-replay tape stays valid when
                // the engine rewrites the position between replays.
                let bpos = Arc::new(
                    metal_buf(
                        bindings
                            .get(positions)
                            .expect("metal backend: unbound positions"),
                    )
                    .raw
                    .clone(),
                );
                let bff = match freq_factors {
                    Some(f) => self.ensure_device(r, f),
                    None => Arc::new(self.zeros_buf(1)),
                };
                let bd = self.dev_dst(r, dst, rows * nh * hd);
                // Decode: the WIDE kernel — see `rmsnorm_wide_f32` (a decode row launches only
                // n_head simdgroups on the 32-lane form and serializes head_dim/32 loads).
                // Fetch the wide PSO once (cap read + dispatch shared two lookups before).
                let wide_pso = (rows <= 4)
                    .then(|| self.pipelines.get("qknormrope_wide_f32").ok())
                    .flatten()
                    .filter(|pl| pl.max_total_threads_per_threadgroup() >= 256);
                let wide = wide_pso.is_some();
                let pso = match wide_pso {
                    Some(pl) => pl,
                    None => self.pipelines.get("qknormrope_f32")?,
                };
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&(rope_dim).to_ne_bytes());
                p.extend_from_slice(&theta.to_ne_bytes());
                p.extend_from_slice(&eps.to_ne_bytes());
                p.extend_from_slice(&(freq_factors.is_some() as u32).to_ne_bytes());
                p.extend_from_slice(&x_stride.to_ne_bytes());
                // One simdgroup per (row, head) — 8 on the wide decode form.
                let tgw = if wide { 256 } else { 32 };
                self.encode_tg_w(
                    r,
                    &pso,
                    &[
                        bx.as_ref(),
                        bw.as_ref(),
                        bpos.as_ref(),
                        bff.as_ref(),
                        bd.as_ref(),
                    ],
                    1 << 4,
                    &p,
                    rows * nh * tgw,
                    tgw,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Add { a, b, dst, n } => {
                let n = n as usize;
                let ba = self.ensure_device(r, a);
                let bb = self.ensure_device(r, b);
                let bd = self.dev_dst(r, dst, n);
                let pso = self.pipelines.get("add_f32")?;
                self.encode_w(
                    r,
                    &pso,
                    &[ba.as_ref(), bb.as_ref(), bd.as_ref()],
                    1 << 2,
                    &(n as u32).to_ne_bytes(),
                    n,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            // Broadcast bias add (Qwen2 q/k/v `Wx + b`): `dst[i] = x[i] + bias[i % n]` over rows*n
            // elements. `bias` is a bound weight (f32); dst-write-mask 1<<2 for the replay barriers.
            Op::AddBias {
                x,
                bias,
                dst,
                rows,
                n,
            } => {
                let (rows, n) = (rows as usize, n as usize);
                let total = rows * n;
                let bx = self.ensure_device(r, x);
                let bb = self.weight_buf(bias, g, bindings)?;
                let bd = self.dev_dst(r, dst, total);
                let pso = self.pipelines.get("add_bias_f32")?;
                let mut p = (n as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(total as u32).to_ne_bytes());
                self.encode_w(
                    r,
                    &pso,
                    &[bx.as_ref(), bb.as_ref(), bd.as_ref()],
                    1 << 2,
                    &p,
                    total,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Scale { x, dst, s, n } => {
                let n = n as usize;
                let bx = self.ensure_device(r, x);
                let bd = self.dev_dst(r, dst, n);
                let pso = self.pipelines.get("scale_f32")?;
                let mut p = s.to_ne_bytes().to_vec();
                p.extend_from_slice(&(n as u32).to_ne_bytes());
                self.encode_w(r, &pso, &[bx.as_ref(), bd.as_ref()], 1 << 1, &p, n);
                r.loc[dst.0 as usize] = Loc::Device;
            }
            // Broadcast multiply: the length-`n` `vec` scales every one of `rows` rows
            // (diffusion-gemma's router input scale — the multiplicative twin of `AddBias`).
            Op::MulVec {
                x,
                vec,
                dst,
                rows,
                n,
            } => {
                let (rows, n) = (rows as usize, n as usize);
                let total = rows * n;
                let bx = self.ensure_device(r, x);
                let bv = self.weight_buf(vec, g, bindings)?;
                let bd = self.dev_dst(r, dst, total);
                let pso = self.pipelines.get("mul_vec_f32")?;
                let mut p = (n as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(total as u32).to_ne_bytes());
                self.encode_w(
                    r,
                    &pso,
                    &[bx.as_ref(), bv.as_ref(), bd.as_ref()],
                    1 << 2,
                    &p,
                    total,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Softcap { x, dst, cap, n } => {
                let n = n as usize;
                let bx = self.ensure_device(r, x);
                let bd = self.dev_dst(r, dst, n);
                let pso = self.pipelines.get("softcap_f32")?;
                let mut p = cap.to_ne_bytes().to_vec();
                p.extend_from_slice(&(n as u32).to_ne_bytes());
                self.encode(r, &pso, &[bx.as_ref(), bd.as_ref()], &p, n);
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Sample {
                x,
                u,
                dst,
                n,
                top_k,
                temp,
                top_p,
            } => {
                // Device-side stochastic sampling: only the 4-byte token id reads back, not the
                // `[vocab]` logits. See `sample_f32` in elementwise_norms.metal for the algorithm
                // (mirrors the host `sample_logits` order of operations exactly).
                let bx = self.ensure_device(r, x);
                let bu = self.ensure_device(r, u);
                let bd = self.dev_dst(r, dst, 1);
                if let Some((groups, candidates)) = sample_split_shape(n as usize, top_k as usize) {
                    let values = self.scratch_buf(candidates, 11);
                    let indices = self.scratch_buf(candidates, 12);
                    let mut p1 = n.to_ne_bytes().to_vec();
                    // Clamp to match the scratch sizing (`sample_split_shape`) — stage-1 writes
                    // `k` candidates per group into a `k`-per-group buffer; passing the raw
                    // `top_k > 64` here would over-write past `candidates`.
                    p1.extend_from_slice(&(sample_top_k(top_k as usize) as u32).to_ne_bytes());
                    p1.extend_from_slice(&(SAMPLE_SPLIT_CHUNK as u32).to_ne_bytes());
                    let stage1 = self.pipelines.get("sample_f32_stage1")?;
                    self.encode_tg_w(
                        r,
                        &stage1,
                        &[bx.as_ref(), values.as_ref(), indices.as_ref()],
                        (1 << 1) | (1 << 2),
                        &p1,
                        groups * 256,
                        256,
                    );

                    let mut p2 = (candidates as u32).to_ne_bytes().to_vec();
                    p2.extend_from_slice(&top_k.to_ne_bytes());
                    p2.extend_from_slice(&temp.to_ne_bytes());
                    p2.extend_from_slice(&top_p.to_ne_bytes());
                    if let Some(posbuf) = r.posbuf.clone() {
                        let stage2 = self.pipelines.get("sample_f32_stage2_dyn")?;
                        self.encode_tg_w(
                            r,
                            &stage2,
                            &[
                                values.as_ref(),
                                indices.as_ref(),
                                bu.as_ref(),
                                &posbuf,
                                bd.as_ref(),
                            ],
                            1 << 4,
                            &p2,
                            256,
                            256,
                        );
                    } else {
                        let stage2 = self.pipelines.get("sample_f32_stage2")?;
                        self.encode_tg_w(
                            r,
                            &stage2,
                            &[values.as_ref(), indices.as_ref(), bu.as_ref(), bd.as_ref()],
                            1 << 3,
                            &p2,
                            256,
                            256,
                        );
                    }
                } else {
                    let mut p = n.to_ne_bytes().to_vec();
                    p.extend_from_slice(&top_k.to_ne_bytes());
                    p.extend_from_slice(&temp.to_ne_bytes());
                    p.extend_from_slice(&top_p.to_ne_bytes());
                    if let Some(posbuf) = r.posbuf.clone() {
                        let pso = self.pipelines.get("sample_f32_dyn")?;
                        self.encode_tg_w(
                            r,
                            &pso,
                            &[bx.as_ref(), bu.as_ref(), &posbuf, bd.as_ref()],
                            1 << 3,
                            &p,
                            256,
                            256,
                        );
                    } else {
                        let pso = self.pipelines.get("sample_f32")?;
                        self.encode_tg_w(
                            r,
                            &pso,
                            &[bx.as_ref(), bu.as_ref(), bd.as_ref()],
                            1 << 2,
                            &p,
                            256,
                            256,
                        );
                    }
                }
                r.loc[dst.0 as usize] = Loc::Device;
            }
            // GPU embed gather: `dst[r, :] = dequant(table[ids[r], :]) * scale` — the table's
            // RAW GGUF bytes are bound as-is (the kernels reuse linear.metal's DEC16_* native
            // decode, no host repack, no dequant cache). One simdgroup per row.
            //
            // Covered formats: F16, BF16, Q8_0, Q4_0, Q5_0, Q4_K, Q6_K, IQ4_NL, IQ4_XS — every
            // dtype with a native 16-block decode in the MSL library. The runner gates gpu_embed
            // on the SHARED format list (infr_vulkan::linear::embed_gather_supported), which is
            // wider (Q4_1/Q5_1/Q2_K/Q3_K/Q5_K have factored-only Metal decode): those return
            // Unsupported here and fail loudly rather than silently gathering garbage.
            //
            // `ids` stays in its bound raw-I32 buffer. That buffer is stable across per-token
            // replay, and its bytes also match the uint token ID written by Argmax/Sample when
            // the runner aliases sampler output to the next gather input.
            Op::EmbedGather {
                ids,
                table,
                dst,
                rows,
                ne,
                scale,
            } => {
                let dt = g.desc(table).dtype;
                let kern = metal_embed_gather_kern(dt).ok_or_else(|| {
                    Error::Unsupported(format!(
                        "Metal Op::EmbedGather: no native gather kernel for {dt:?}"
                    ))
                })?;
                let (rows_u, ne_u) = (rows as usize, ne as usize);
                let bt = metal_buf(bindings.get(table).expect("metal backend: unbound Weight"));
                let bi = metal_buf(bindings.get(ids).expect("metal backend: unbound token IDs"));
                let bd = self.dev_dst(r, dst, rows_u * ne_u);
                let pso = self.pipelines.get(kern)?;
                let mut p = rows.to_ne_bytes().to_vec();
                p.extend_from_slice(&ne.to_ne_bytes());
                p.extend_from_slice(&scale.to_ne_bytes());
                self.encode_tg_w(
                    r,
                    &pso,
                    &[&bt.raw, &bi.raw, bd.as_ref()],
                    1 << 2,
                    &p,
                    rows_u * 32,
                    32,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Argmax { x, dst, n, rows } => {
                // Greedy device-side sampling: one 256-thread threadgroup, the token id (u32
                // bit-pattern in the f32 slot) is the only readback greedy decode needs.
                // Single-row only (argmax_f32 scans the whole buffer with no row offset) —
                // `Capabilities::argmax_rows` is false so the runner never builds a multi-row
                // Argmax for Metal (the MTP verify accept keeps the host-logits path there).
                if rows != 1 {
                    return Err(Error::Unsupported(
                        "metal backend: Op::Argmax rows > 1 (MTP verify accept) not implemented"
                            .into(),
                    ));
                }
                let bx = self.ensure_device(r, x);
                let bd = self.dev_dst(r, dst, 1);
                if let Some(groups) = argmax_split_groups(n as usize) {
                    let values = self.scratch_buf(groups, 9);
                    let indices = self.scratch_buf(groups, 10);
                    let mut p = n.to_ne_bytes().to_vec();
                    p.extend_from_slice(&(ARGMAX_SPLIT_CHUNK as u32).to_ne_bytes());
                    let stage1 = self.pipelines.get("argmax_f32_stage1")?;
                    self.encode_tg_w(
                        r,
                        &stage1,
                        &[bx.as_ref(), values.as_ref(), indices.as_ref()],
                        (1 << 1) | (1 << 2),
                        &p,
                        groups * 256,
                        256,
                    );
                    let stage2 = self.pipelines.get("argmax_f32_stage2")?;
                    self.encode_tg_w(
                        r,
                        &stage2,
                        &[values.as_ref(), indices.as_ref(), bd.as_ref()],
                        1 << 2,
                        &(groups as u32).to_ne_bytes(),
                        256,
                        256,
                    );
                } else {
                    let pso = self.pipelines.get("argmax_f32")?;
                    let p = n.to_ne_bytes().to_vec();
                    self.encode_tg_w(r, &pso, &[bx.as_ref(), bd.as_ref()], 1 << 1, &p, 256, 256);
                }
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::ArgmaxProb { .. } => {
                // No fused argmax+prob kernel on Metal yet (issue #33 follow-up, Vulkan-only so
                // far — `Capabilities::argmax_prob` is false, so the MTP driver never builds this
                // op for the Metal backend and keeps the host logits-download + `top1_softmax`
                // path there instead). Kept as an explicit Unsupported arm (not silently missing)
                // so a future caller that ignores the capability fails loudly.
                return Err(Error::Unsupported(
                    "metal backend: Op::ArgmaxProb not implemented (argmax_prob capability is \
                     false; MTP draft loop uses the host logits path on Metal)"
                        .into(),
                ));
            }
            Op::GatedAct {
                gate,
                up,
                dst,
                rows,
                nff,
                act,
                up_off,
                up_stride,
                gate_stride,
                gate_block_width,
            } => {
                let (rows, nff) = (rows as usize, nff as usize);
                let bg = self.ensure_device(r, gate);
                let bu = self.ensure_device(r, up);
                let bd = self.dev_dst(r, dst, rows * nff);
                let pso = self.pipelines.get("gatedact_f32")?;
                let act_code: u32 = match act {
                    infr_core::graph::Activation::Silu => 0,
                    infr_core::graph::Activation::Sigmoid => 2,
                    infr_core::graph::Activation::Gelu => 1,
                };
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nff as u32).to_ne_bytes());
                p.extend_from_slice(&act_code.to_ne_bytes());
                p.extend_from_slice(&up_off.to_ne_bytes());
                p.extend_from_slice(&up_stride.to_ne_bytes());
                p.extend_from_slice(&gate_stride.to_ne_bytes());
                p.extend_from_slice(&gate_block_width.to_ne_bytes());
                self.encode_w(
                    r,
                    &pso,
                    &[bg.as_ref(), bu.as_ref(), bd.as_ref()],
                    1 << 2,
                    &p,
                    rows * nff,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            // Only produced when `Capabilities::combined_gu` is set — Metal leaves it false (the
            // reference backend keeps the separate gate/up form), so this arm is unreachable.
            Op::GatedActFused {
                gu,
                dst,
                rows,
                nff,
                act,
            } => {
                let (rows, nff) = (rows as usize, nff as usize);
                let bg = self.ensure_device(r, gu);
                let bd = self.dev_dst(r, dst, rows * nff);
                let pso = self.pipelines.get("gatedactfused_f32")?;
                let act_code: u32 = match act {
                    infr_core::graph::Activation::Silu => 0,
                    infr_core::graph::Activation::Gelu => 1,
                    infr_core::graph::Activation::Sigmoid => 2,
                };
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(nff as u32).to_ne_bytes());
                p.extend_from_slice(&act_code.to_ne_bytes());
                p.extend_from_slice(&0u32.to_ne_bytes()); // pad to GatedParams
                self.encode_w(r, &pso, &[bg.as_ref(), bd.as_ref()], 1 << 1, &p, rows * nff);
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::WriteKv {
                src,
                cache,
                rows,
                row_stride,
                pos,
            } => {
                // Stateful cast-copy of `rows` rows into the persistent KV buffer, on the GPU so it
                // stays in the batch (a host write would force a per-layer flush). Metal's hazard
                // tracking orders this write before the Attention that reads the same cache buffer.
                let (rows, rs, mut pos) = (rows as usize, row_stride as usize, pos as usize);
                if let (Some(dp), 1) = (r.dynpos, rows) {
                    pos = dp as usize;
                }
                let bsrc = self.ensure_device(r, src);
                let cbuf = metal_buf(
                    bindings
                        .get(cache)
                        .expect("metal backend: unbound KV cache"),
                );
                let base = pos * rs;
                let n = rows * rs;
                if let Some(posbuf) = r.posbuf.clone() {
                    // Recording a replay tape: the row offset must come from the positions buffer
                    // (the baked `pos` is this token's only) — f16/q8/q4_0/iq4_nl cache
                    // guaranteed by the gate; the quantizing variants run one thread per block.
                    let dt = g.desc(cache).dtype;
                    let kern = match dt {
                        DType::Q8_0 => "writekv_dyn_q8",
                        DType::Q4_0 => "writekv_dyn_q4_0",
                        DType::Iq4Nl => "writekv_dyn_iq4_nl",
                        _ => "writekv_dyn_f16",
                    };
                    let pso = self.pipelines.get(kern)?;
                    let mut p = (n as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(rs as u32).to_ne_bytes());
                    let threads = if matches!(dt, DType::Q8_0 | DType::Q4_0 | DType::Iq4Nl) {
                        n / 32
                    } else {
                        n
                    };
                    self.encode_w(
                        r,
                        &pso,
                        &[bsrc.as_ref(), &cbuf.raw, &posbuf],
                        1 << 1,
                        &p,
                        threads,
                    );
                } else {
                    // Static per-token write. The decoupled quant formats (block quants, bf16,
                    // turbo) force static decode (the replay gate rejects them), so their WriteKv
                    // only ever lands here — one thread per block (32-elem blocks, 128 for turbo)
                    // or per element (dense). `base` is in elements; each quantize kernel converts
                    // to its block byte offset internally.
                    let dt = g.desc(cache).dtype;
                    let kern = match dt {
                        DType::Q8_0 => "writekv_q8",
                        DType::F16 => "writekv_f16",
                        DType::Q4_0 => "writekv_q4_0",
                        DType::Q4_1 => "writekv_q4_1",
                        DType::Q5_0 => "writekv_q5_0",
                        DType::Q5_1 => "writekv_q5_1",
                        DType::Iq4Nl => "writekv_iq4_nl",
                        DType::Bf16 => "writekv_bf16",
                        DType::Turbo2 => "writekv_turbo2",
                        DType::Turbo3 => "writekv_turbo3",
                        DType::Turbo4 => "writekv_turbo4",
                        _ => "writekv_f32",
                    };
                    let pso = self.pipelines.get(kern)?;
                    let mut p = (n as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(base as u32).to_ne_bytes());
                    // Per-format thread counts (block quants n/32, turbo n/128, dense n) — from
                    // the KV-quant work on main; `encode_w` with write-mask 1<<1 (WriteKv writes
                    // only the cache at binding index 1) — from #15's concurrent-replay barriers.
                    let threads = match dt {
                        DType::Q8_0
                        | DType::Q4_0
                        | DType::Q4_1
                        | DType::Q5_0
                        | DType::Q5_1
                        | DType::Iq4Nl => n / 32,
                        DType::Turbo2 | DType::Turbo3 | DType::Turbo4 => n / 128,
                        _ => n,
                    };
                    self.encode_w(r, &pso, &[bsrc.as_ref(), &cbuf.raw], 1 << 1, &p, threads);
                }
            }
            Op::Attention {
                q,
                k_cache,
                v_cache,
                dst,
                rows,
                kv_len,
                n_head,
                n_kv,
                head_dim,
                scale,
                mask,
                pos,
            } => {
                let (rows, kv_len, nh, nkv, hd) = (
                    rows as usize,
                    kv_len as usize,
                    n_head as usize,
                    n_kv as usize,
                    head_dim as usize,
                );
                // Replayed decode graph: the baked pos/kv_len are the compile-time token's;
                // take the current position (also steers the kv_len-based kernel routing).
                let (pos, kv_len) = match r.dynpos {
                    Some(dp) if rows == 1 => (dp, dp as usize + 1),
                    _ => (pos, kv_len),
                };
                if hd > 256 {
                    return Err(Error::Unsupported(format!(
                        "metal attention: head_dim {hd} exceeds MAX_HD 256 (shader acc[] cap)"
                    )));
                }
                // Read K/V straight from the bound cache buffers on-device (no host materialize):
                // f16 caches use the half-typed kernel, f32 caches the plain one. The WriteKv above
                // wrote this same buffer earlier in the batch; hazard tracking makes it visible here.
                let kbuf = metal_buf(bindings.get(k_cache).expect("metal: unbound k_cache"));
                let vbuf = metal_buf(bindings.get(v_cache).expect("metal: unbound v_cache"));
                let bq = self.ensure_device(r, q);
                let bd = self.dev_dst(r, dst, rows * nh * hd);
                // DiffusionGemma canvas denoise (docs/diffusion-gemma.md, `AttnMask::Canvas`):
                // EVERY row attends the SAME fixed bidirectional `[lo, kv_len)`, ignoring its own
                // causal position entirely. `window` stays meaningless for it (the dedicated
                // canvas kernel below never reads the field) — extract `lo` separately and route
                // around the ordinary flash/split/vec tiers further down (see `canvas_lo`'s use).
                let canvas_lo = match mask {
                    infr_core::graph::AttnMask::Canvas { lo } => Some(lo),
                    _ => None,
                };
                let window: u32 = match mask {
                    infr_core::graph::AttnMask::Causal => 0,
                    infr_core::graph::AttnMask::SlidingWindow(w) => w as u32,
                    infr_core::graph::AttnMask::Canvas { .. } => 0,
                };
                // Canvas is a denoise-prefill mask, never autoregressive decode, so it should
                // never reach the record-once replay tape (rows==1, dyn-pos) — no dyn-pos canvas
                // kernel exists. Fail loudly instead of silently running the wrong (causal) tape.
                if canvas_lo.is_some() && r.posbuf.is_some() {
                    return Err(Error::Unsupported(
                        "metal attention: AttnMask::Canvas on a decode-replay tape is unexpected \
                         (canvas is a denoise-prefill mask) — no dyn-pos canvas kernel exists"
                            .into(),
                    ));
                }
                // Q8_0's attention family (attention_q8kv/attnvec_q8kv/attnflash2_q8kv_*) has no
                // split-canvas sibling — only the f16 path below (native cache or the
                // dequant-prepass scratch, which covers every other KV dtype) supports canvas.
                if canvas_lo.is_some()
                    && (g.desc(k_cache).dtype == DType::Q8_0
                        || g.desc(v_cache).dtype == DType::Q8_0)
                {
                    return Err(Error::Unsupported(
                        "metal attention: AttnMask::Canvas over a Q8_0 KV cache has no kernel yet \
                         — f16 cache (native or block-quant dequant-prepassed) only so far"
                            .into(),
                    ));
                }
                if let Some(posbuf) = r.posbuf.clone() {
                    // Recording a replay tape (rows==1, f16/q8/coupled-q4_0/iq4_nl cache, hd
                    // 64/128/256 — checked by the gate): route straight to the dynamic-pos
                    // vector flash kernel, which reads pos from the positions buffer and covers
                    // every kv_len from 1 up. The usual shape routing below would bake this
                    // token's kv_len into the kernel CHOICE, which is exactly what a replayed
                    // tape can't have.
                    let kern = dyn_attnvec_kern(g.desc(k_cache).dtype, hd);
                    // hd=256 instantiates at NSG=16 (see the shader) — half the threadgroup.
                    let nsg = if hd == 256 { 16 } else { 32 };
                    let pso = self.pipelines.get(kern)?;
                    let mut p = (rows as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(kv_len as u32).to_ne_bytes());
                    p.extend_from_slice(&(nh as u32).to_ne_bytes());
                    p.extend_from_slice(&(nkv as u32).to_ne_bytes());
                    p.extend_from_slice(&(hd as u32).to_ne_bytes());
                    p.extend_from_slice(&scale.to_ne_bytes());
                    p.extend_from_slice(&window.to_ne_bytes());
                    p.extend_from_slice(&pos.to_ne_bytes());
                    self.encode_tg_w(
                        r,
                        &pso,
                        &[bq.as_ref(), &kbuf.raw, &vbuf.raw, bd.as_ref(), &posbuf],
                        1 << 3,
                        &p,
                        rows * nh * nsg * 32,
                        nsg * 32,
                    );
                    r.loc[dst.0 as usize] = Loc::Device;
                    return Ok(());
                }
                // Decoupled quant KV (block quants q4_0/…/iq4_nl, dense bf16, turbo2/3/4): expand
                // each quantized side into a transient f16 scratch, then fall through to the
                // standard f16 attention over the scratch (the Vulkan dequant->f16 prepass, ported).
                // A side already f16 keeps its native cache buffer (the high-precision-K +
                // quantized-V shape); q8 and f32 keep their own native-read paths below. These
                // formats force static decode, so no replay tape reaches here. The scratch is laid
                // out [kv_len, n_kv, head_dim] — identical stride to an f16 cache prefix, so the
                // f16 kernels index it unchanged.
                let prep = |dt| {
                    matches!(
                        dt,
                        DType::Q4_0
                            | DType::Q4_1
                            | DType::Q5_0
                            | DType::Q5_1
                            | DType::Iq4Nl
                            | DType::Bf16
                            | DType::Turbo2
                            | DType::Turbo3
                            | DType::Turbo4
                    )
                };
                let kdt = g.desc(k_cache).dtype;
                let vdt = g.desc(v_cache).dtype;
                // Coupled q4_0/iq4_nl at decode-class shapes (few rows, deep kv): read the
                // compact 18 B blocks natively with the vector flash — no scratch allocation,
                // no whole-prefix re-dequant per token (the prepass costs O(kv_len) dequant
                // work EVERY token, which measured 3x slower than f16 at d4096), and f32
                // accumulation over exactly-decoded values. Wide prefill keeps the prepass:
                // the vector kernel's one-threadgroup-per-(row, head) shape is serial over kv
                // and loses to dequant-once + flash there.
                let nat4_kern = || -> Option<&'static str> {
                    match (kdt, hd) {
                        (DType::Q4_0, 64) => Some("attnvec_q4_0kv_hd64"),
                        (DType::Q4_0, 128) => Some("attnvec_q4_0kv_hd128"),
                        (DType::Q4_0, 256) => Some("attnvec_q4_0kv_hd256"),
                        (DType::Iq4Nl, 64) => Some("attnvec_iq4nlkv_hd64"),
                        (DType::Iq4Nl, 128) => Some("attnvec_iq4nlkv_hd128"),
                        (DType::Iq4Nl, 256) => Some("attnvec_iq4nlkv_hd256"),
                        _ => None,
                    }
                };
                // No kv_len floor: the replay tape's dyn kernel reads natively from token 1, so
                // the static walk must too — a shallow-kv prepass here would make replay-on/off
                // decode diverge (f16 scratch rounding vs exact f32) on the earliest tokens.
                // Canvas never takes this fast path: `nat4_kern` reads `pos + ti`/`window` like
                // every other non-canvas kernel here, with no notion of a per-row-independent
                // `[lo, kv_len)` reach.
                if kdt == vdt && rows <= 8 && canvas_lo.is_none() {
                    if let Some(kern) = nat4_kern() {
                        let nsg = if hd == 256 { 16 } else { 32 };
                        let cap_ok = self
                            .pipelines
                            .get(kern)
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= nsg as u64 * 32)
                            .unwrap_or(false);
                        if cap_ok {
                            let pso = self.pipelines.get(kern)?;
                            let mut p = (rows as u32).to_ne_bytes().to_vec();
                            p.extend_from_slice(&(kv_len as u32).to_ne_bytes());
                            p.extend_from_slice(&(nh as u32).to_ne_bytes());
                            p.extend_from_slice(&(nkv as u32).to_ne_bytes());
                            p.extend_from_slice(&(hd as u32).to_ne_bytes());
                            p.extend_from_slice(&scale.to_ne_bytes());
                            p.extend_from_slice(&window.to_ne_bytes());
                            p.extend_from_slice(&pos.to_ne_bytes());
                            self.encode_tg(
                                r,
                                &pso,
                                &[bq.as_ref(), &kbuf.raw, &vbuf.raw, bd.as_ref()],
                                &p,
                                rows * nh * nsg * 32,
                                nsg * 32,
                            );
                            r.loc[dst.0 as usize] = Loc::Device;
                            return Ok(());
                        }
                    }
                }
                let ne = kv_len * nkv * hd;
                // f16 dequant scratch, pooled by (size, tag) rather than freshly allocated per
                // layer per token (the pool amortizes across every layer at a given depth; the
                // dequant fully overwrites `ne` elements, so zero-init is irrelevant). Distinct
                // K/V tags so both sides stay live simultaneously. This static-decode path never
                // tapes, and the serial walk's automatic hazard tracking serializes reuse.
                let ks: Option<Arc<MtlBuffer>> = if prep(kdt) {
                    let s = self.scratch_buf((ne * 2).max(4).div_ceil(4), 18);
                    self.dequant_kv_f16(r, kdt, &kbuf.raw, &s, ne)?;
                    Some(s)
                } else {
                    None
                };
                let vs: Option<Arc<MtlBuffer>> = if prep(vdt) {
                    let s = self.scratch_buf((ne * 2).max(4).div_ceil(4), 19);
                    self.dequant_kv_f16(r, vdt, &vbuf.raw, &s, ne)?;
                    Some(s)
                } else {
                    None
                };
                let prep_active = ks.is_some() || vs.is_some();
                // Effective K/V buffers the attention reads: the f16 scratch where prepassed, else
                // the native cache. (q8/f32 native paths below ignore these and read kbuf/vbuf.)
                let k_raw: &MtlBuffer = match &ks {
                    Some(s) => s,
                    None => &kbuf.raw,
                };
                let v_raw: &MtlBuffer = match &vs {
                    Some(s) => s,
                    None => &vbuf.raw,
                };
                // Q8_0 cache: rows==1 decode takes the q8 vector kernel (hd 64/128); every
                // other shape the scalar dequant-on-read fallback (prefill lives here until a
                // q8 flash port). Both dequantize exactly — reassociation-only vs the f16 math.
                if g.desc(k_cache).dtype == DType::Q8_0 {
                    // Prefill-wide launches take the q8 cooperative flash (dequant-staged KV
                    // tiles) — the same gate shape as the f16 flash.
                    let fq8 = rows * nh >= 128 && kv_len >= 64 && matches!(hd, 64 | 128) && {
                        let kn = if hd == 64 {
                            "attnflash2_q8kv_hd64"
                        } else {
                            "attnflash2_q8kv_hd128"
                        };
                        self.pipelines
                            .get(kn)
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                            .unwrap_or(false)
                    };
                    if fq8 {
                        let kern = if hd == 64 {
                            "attnflash2_q8kv_hd64"
                        } else {
                            "attnflash2_q8kv_hd128"
                        };
                        let pso = self.pipelines.get(kern)?;
                        let mut p = (rows as u32).to_ne_bytes().to_vec();
                        p.extend_from_slice(&(kv_len as u32).to_ne_bytes());
                        p.extend_from_slice(&(nh as u32).to_ne_bytes());
                        p.extend_from_slice(&(nkv as u32).to_ne_bytes());
                        p.extend_from_slice(&(hd as u32).to_ne_bytes());
                        p.extend_from_slice(&scale.to_ne_bytes());
                        p.extend_from_slice(&window.to_ne_bytes());
                        p.extend_from_slice(&pos.to_ne_bytes());
                        let n = rows * nh * hd;
                        // Pooled q-cast scratch (cast overwrites all `n`); tag 20 for the q8 path.
                        let qh = self.scratch_buf((n * 2).max(4).div_ceil(4), 20);
                        let cast = self.pipelines.get("cast_f32_f16")?;
                        self.encode(
                            r,
                            &cast,
                            &[bq.as_ref(), qh.as_ref()],
                            &(n as u32).to_ne_bytes(),
                            n,
                        );
                        self.encode_tg(
                            r,
                            &pso,
                            &[qh.as_ref(), &kbuf.raw, &vbuf.raw, bd.as_ref()],
                            &p,
                            rows.div_ceil(8) * nh * 128,
                            128,
                        );
                        r.loc[dst.0 as usize] = Loc::Device;
                        return Ok(());
                    }
                    // The vector kernel runs one threadgroup per (row, head), covering decode
                    // and any prefill shape the flash gate declined.
                    let vq8_kern = match hd {
                        64 => Some("attnvec_q8kv_hd64"),
                        128 => Some("attnvec_q8kv_hd128"),
                        256 => Some("attnvec_q8kv_hd256"),
                        _ => None,
                    };
                    // hd=256 instantiates at NSG=16 (threadgroup budget) — 512 threads.
                    let vnsg = if hd == 256 { 16 } else { 32 };
                    let vq8 = kv_len >= 128
                        && vq8_kern.is_some_and(|kn| {
                            self.pipelines
                                .get(kn)
                                .map(|pl| {
                                    pl.max_total_threads_per_threadgroup() >= vnsg as u64 * 32
                                })
                                .unwrap_or(false)
                        });
                    let kern = if vq8 {
                        vq8_kern.unwrap()
                    } else {
                        "attention_q8kv"
                    };
                    let pso = self.pipelines.get(kern)?;
                    let mut p = (rows as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(kv_len as u32).to_ne_bytes());
                    p.extend_from_slice(&(nh as u32).to_ne_bytes());
                    p.extend_from_slice(&(nkv as u32).to_ne_bytes());
                    p.extend_from_slice(&(hd as u32).to_ne_bytes());
                    p.extend_from_slice(&scale.to_ne_bytes());
                    p.extend_from_slice(&window.to_ne_bytes());
                    p.extend_from_slice(&pos.to_ne_bytes());
                    let nsg = if vq8 { vnsg } else { 1 };
                    self.encode_tg(
                        r,
                        &pso,
                        &[bq.as_ref(), &kbuf.raw, &vbuf.raw, bd.as_ref()],
                        &p,
                        rows * nh * nsg * 32,
                        nsg * 32,
                    );
                    r.loc[dst.0 as usize] = Loc::Device;
                    return Ok(());
                }
                // Route by kernel-latency shape. The one-simdgroup-per-(query, head) kernels are
                // latency-bound on their serial O(kv_len) online-softmax chain, so any long
                // context takes a split-KV kernel — NSG simdgroups per (query, head) merged in
                // threadgroup memory — 32-way when head_dim fits its threadgroup accumulator
                // (hd <= 128), 8-way otherwise. That holds for prefill too (measured, not just
                // decode): the extra parallelism is redundant there, but the 4x-8x shorter serial
                // chain still wins. Short contexts keep the lean unsplit kernel. (A
                // simdgroup_matrix flash-attention kernel was built and benched here; it never
                // beat the 32-way split on prefill — occupancy-starved by its threadgroup tiles —
                // so it was dropped.)
                // Prepassed quant KV reads an f16 scratch, so it takes the f16 attention path too.
                let f16 = g.desc(k_cache).dtype == DType::F16 || prep_active;
                // DiffusionGemma canvas denoise (docs/diffusion-gemma.md, `AttnMask::Canvas`):
                // route straight to the dedicated split-KV canvas kernel (`ATTNSPLIT_CANVAS_KERNEL`
                // in attention.metal) instead of the flash/vec/plain routing below — none of those
                // tiers can express a bound that's the SAME for every row regardless of its own
                // position. Mirrors the Vulkan adapter forcing the split-K tier whenever
                // `canvas_lo` is set (`adapter.rs`'s `split_ok`/`canvas_lo` handling). `k_raw`/
                // `v_raw` above already resolve to an f16 view for every KV dtype this reaches
                // (Q8_0 was rejected earlier — no split-canvas sibling for it yet).
                // UNVERIFIED: this whole branch is Phase D's blind Metal work — never run on
                // hardware; CPU + Vulkan remain the validated references for Canvas.
                if let Some(lo) = canvas_lo {
                    let split32_name = if f16 {
                        "attention_canvas32_f16kv"
                    } else {
                        "attention_canvas32_f32"
                    };
                    let split_name = if f16 {
                        "attention_canvas_f16kv"
                    } else {
                        "attention_canvas_f32"
                    };
                    // Same threadgroup-cap gating as the ordinary split32/split kernels below
                    // (maxTotalThreadsPerThreadgroup is per-pipeline; a paravirtual/CI device can
                    // cap it below the 32-wide kernel's 1024 threads, degrading to the 8-wide one).
                    let fits = |name: &'static str, threads: u64| -> Result<bool> {
                        Ok(self
                            .pipelines
                            .get(name)?
                            .max_total_threads_per_threadgroup()
                            >= threads)
                    };
                    let (kern, nsg) = if hd <= 128 && fits(split32_name, 1024)? {
                        (split32_name, 32usize)
                    } else if fits(split_name, 256)? {
                        (split_name, 8usize)
                    } else {
                        return Err(Error::Unsupported(format!(
                            "metal attention: canvas split kernel {split_name} doesn't fit this \
                             device's threadgroup cap"
                        )));
                    };
                    let pso = self.pipelines.get(kern)?;
                    let mut p = (rows as u32).to_ne_bytes().to_vec();
                    // Repurposed `kv_len` slot (dead in this kernel — see the macro's doc): the
                    // canvas `lo`, identical for every row.
                    p.extend_from_slice(&(lo as u32).to_ne_bytes());
                    p.extend_from_slice(&(nh as u32).to_ne_bytes());
                    p.extend_from_slice(&(nkv as u32).to_ne_bytes());
                    p.extend_from_slice(&(hd as u32).to_ne_bytes());
                    p.extend_from_slice(&scale.to_ne_bytes());
                    // window: unread by this kernel.
                    p.extend_from_slice(&0u32.to_ne_bytes());
                    // Repurposed `pos` slot: the fixed causal end `kv_len - 1`, identical for
                    // every row (not `pos + ti`).
                    p.extend_from_slice(&((kv_len - 1) as u32).to_ne_bytes());
                    self.encode_tg(
                        r,
                        &pso,
                        &[bq.as_ref(), k_raw, v_raw, bd.as_ref()],
                        &p,
                        rows * nh * nsg * 32,
                        nsg * 32,
                    );
                    r.loc[dst.0 as usize] = Loc::Device;
                    return Ok(());
                }
                // Wide launches on an f16 cache take the half-fragment flash kernel: K^T/V
                // fragments load straight from the cache, Q is cast once to f16 below. Small
                // kv_len stays scalar (also keeps the short-kv wide parity test on the exact
                // path). See `attnflash_f16kv` for why the f32 flash attempt lost and this wins.
                // The cooperative flash kernel (4 simdgroups per query tile, llama.cpp
                // flash_attn_ext structure) is instantiated per compile-time head size
                // (fully unrolled QK/PV loops) and needs a 128-thread threadgroup
                // (pipeline-cap gated like the split kernels); other head sizes keep the
                // single-simdgroup flash.
                let standard_flash2_kern = match hd {
                    64 => Some("attnflash2_f16kv_hd64"),
                    128 => Some("attnflash2_f16kv_hd128"),
                    256 => Some("attnflash2_f16kv_hd256"),
                    _ => None,
                };
                let flash2_kern = select_attention_flash2_kern(
                    f16,
                    rows,
                    nh,
                    kv_len,
                    hd,
                    || {
                        self.pipelines
                            .get("attnflash2_c128_f16kv_hd128")
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                            .unwrap_or(false)
                    },
                    || {
                        standard_flash2_kern.filter(|kn| {
                            self.pipelines
                                .get(kn)
                                .map(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                                .unwrap_or(false)
                        })
                    },
                );
                let flash2_ok = flash2_kern.is_some();
                // hd > 128 has no single-simdgroup flash fallback (its register accumulator
                // tops out at 128), so the wide gate only opens there when flash2 itself can run.
                let flash = prefer_attention_flash(f16, rows, nh, kv_len, hd, flash2_ok);
                let split = !flash && (rows * nh < 128 || kv_len >= 128);
                let flash2 = flash && flash2_ok;
                // The split kernels REQUIRE their full NSG*32-thread threadgroup: every simdgroup
                // owns a strided KV slice, so a smaller launch would silently skip positions and
                // merge uninitialized partials. maxTotalThreadsPerThreadgroup is per-PIPELINE
                // (register pressure or a paravirtual device can push it below 1024 — GitHub's
                // macOS CI runners do), so each width is gated on its own pipeline's cap and
                // degrades to the next-narrower kernel instead of letting `encode_tg` clamp.
                let fits = |name: &'static str, threads: u64| -> Result<bool> {
                    Ok(self
                        .pipelines
                        .get(name)?
                        .max_total_threads_per_threadgroup()
                        >= threads)
                };
                // The vector flash kernel (llama.cpp flash_attn_ext_vec structure) covers the
                // long-context decode shape split32 serves, at 32 KV positions per simdgroup
                // step instead of 1 — same head-size instantiations and cap gating as flash2.
                let vec_kern = match hd {
                    64 => Some("attnvec_f16kv_hd64"),
                    128 => Some("attnvec_f16kv_hd128"),
                    256 => Some("attnvec_f16kv_hd256"),
                    _ => None,
                };
                // hd=256 instantiates at NSG=16 (threadgroup budget) — 512 threads.
                let vec_nsg: usize = if hd == 256 { 16 } else { 32 };
                let vec = !flash
                    && f16
                    && split
                    && kv_len >= 128
                    && vec_kern.is_some_and(|kn| {
                        self.pipelines
                            .get(kn)
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= vec_nsg as u64 * 32)
                            .unwrap_or(false)
                    });
                let split32 = split
                    && !vec
                    && kv_len >= 128
                    && hd <= 128
                    && fits(
                        if f16 {
                            "attnsplit32_f16kv"
                        } else {
                            "attnsplit32_f32"
                        },
                        1024,
                    )?;
                let split = split
                    && (vec
                        || split32
                        || fits(
                            if f16 {
                                "attnsplit_f16kv"
                            } else {
                                "attnsplit_f32"
                            },
                            256,
                        )?);
                let kern = match (flash, f16, split, split32) {
                    (true, ..) if flash2 => flash2_kern.unwrap(),
                    (true, ..) => "attnflash_f16kv",
                    (_, true, true, _) if vec => vec_kern.unwrap(),
                    (_, true, false, _) => "attention_f16kv",
                    (_, true, _, true) => "attnsplit32_f16kv",
                    (_, true, true, false) => "attnsplit_f16kv",
                    (_, false, false, _) => "attention_f32",
                    (_, false, _, true) => "attnsplit32_f32",
                    (_, false, true, false) => "attnsplit_f32",
                };
                let pso = self.pipelines.get(kern)?;
                let mut p = (rows as u32).to_ne_bytes().to_vec();
                p.extend_from_slice(&(kv_len as u32).to_ne_bytes());
                p.extend_from_slice(&(nh as u32).to_ne_bytes());
                p.extend_from_slice(&(nkv as u32).to_ne_bytes());
                p.extend_from_slice(&(hd as u32).to_ne_bytes());
                p.extend_from_slice(&scale.to_ne_bytes());
                p.extend_from_slice(&window.to_ne_bytes());
                p.extend_from_slice(&pos.to_ne_bytes());
                if flash {
                    // Cast q to a transient f16 buffer (one pass), then one flash dispatch:
                    // one simdgroup per (8-query tile, head).
                    let n = rows * nh * hd;
                    // Pooled q-cast scratch (cast overwrites all `n`); tag 22 for the f16 flash.
                    let qh = self.scratch_buf((n * 2).max(4).div_ceil(4), 22);
                    let cast = self.pipelines.get("cast_f32_f16")?;
                    self.encode(
                        r,
                        &cast,
                        &[bq.as_ref(), qh.as_ref()],
                        &(n as u32).to_ne_bytes(),
                        n,
                    );
                    // flash2 runs 4 cooperating simdgroups per (query tile, head) threadgroup
                    let tgw = if flash2 { 128 } else { 32 };
                    self.encode_tg(
                        r,
                        &pso,
                        &[qh.as_ref(), k_raw, v_raw, bd.as_ref()],
                        &p,
                        rows.div_ceil(8) * nh * tgw,
                        tgw,
                    );
                } else {
                    // One simdgroup per (query, head); split/vec kernels use NSG simdgroups
                    // per pair, grid still exactly rows*n_head threadgroups.
                    let nsg = if vec {
                        vec_nsg
                    } else if split32 {
                        32
                    } else if split {
                        8
                    } else {
                        1
                    };
                    self.encode_tg(
                        r,
                        &pso,
                        &[bq.as_ref(), k_raw, v_raw, bd.as_ref()],
                        &p,
                        rows * nh * nsg * 32,
                        nsg * 32,
                    );
                }
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::MoeFfn {
                x,
                router_x,
                router,
                gate_exps,
                up_exps,
                down_exps,
                down_scale,
                fused_gate_up,
                dst,
                ne,
                n_expert,
                n_used,
                n_ff_exp,
                scale,
                act,
                gating,
                norm_w,
                weight_before,
                ep_band: _, // EP is a Vulkan-only path; Metal always runs full-expert
                ..
            } => {
                // The Metal MoE path implements only softmax gating + top-k renorm + output-
                // weighting; llama4's sigmoid/no-renorm/weight-before-FFN routing is CPU-only (see
                // the `llama4` arch note) and never reaches a GPU backend in-tree.
                assert!(
                    matches!(gating, infr_core::graph::MoeGating::Softmax)
                        && norm_w
                        && !weight_before,
                    "Metal MoeFfn: only softmax + renorm + output-weighting supported (llama4 is CPU-only)"
                );
                let (ne, n_expert, n_used, nffx) = (
                    ne as usize,
                    n_expert as usize,
                    n_used as usize,
                    n_ff_exp as usize,
                );
                // Rows inferred from the bound activation shape (the seam's batched prefill
                // passes [rows, ne]); each row routes independently.
                let rows = g.desc(x).numel() / ne;
                // Device path for K-quant experts (the shapes real MoE checkpoints ship): router
                // GEMV -> on-device top-k -> expert-BATCHED GEMV stages picking their weight
                // slices from the device expert table, weighted sum via a final reduce. Seven
                // dispatches, no expert bytes or activations on the host. Falls back to the host
                // path below for other dtypes (INFR_METAL_NOMOE forces it).
                let moe_kern = |dt: DType| -> Option<(&'static str, &'static str)> {
                    match dt {
                        DType::Q4K => Some(("linear_q4k_moe", "linear_q4k_moe_acc")),
                        DType::Q6K => Some(("linear_q6k_moe", "linear_q6k_moe_acc")),
                        _ => None,
                    }
                };
                let (gdt2, udt2, ddt2) = (
                    g.desc(gate_exps).dtype,
                    g.desc(up_exps).dtype,
                    g.desc(down_exps).dtype,
                );
                // The device path assumes SEPARATE full-width gate/up expert banks and no per-
                // expert down scale (qwen3moe's shape); diffusion-gemma's fused `gate_up_exps` +
                // `ffn_down_exps.scale` always falls to the host path below instead of teaching
                // the device kernels a layout they don't expect.
                let device_ok = n_used <= 16
                    && ne % 256 == 0
                    && nffx % 256 == 0
                    && !fused_gate_up
                    && down_scale.is_none()
                    && self.metal().moe
                    && moe_kern(gdt2).is_some()
                    && moe_kern(udt2).is_some()
                    && moe_kern(ddt2).is_some();
                if device_ok {
                    let bx = self.ensure_device(r, x);
                    let bd = self.dev_dst(r, dst, rows * ne);
                    // Router logits for ALL rows (one GEMV-per-(row, expert) dispatch over the
                    // dequant-cached router weight), then per-row top-k into the [rows, 32]
                    // expert table (idx in slots 0..16, f32 weights in 16..32).
                    let rw = self.weight_buf(router, g, bindings)?;
                    let logits = self.scratch_buf(rows * n_expert, 0);
                    let pso = self.pipelines.get("linear_f32")?;
                    let mut p = (rows as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(ne as u32).to_ne_bytes());
                    p.extend_from_slice(&(n_expert as u32).to_ne_bytes());
                    self.encode_tg_w(
                        r,
                        &pso,
                        &[bx.as_ref(), rw.as_ref(), logits.as_ref()],
                        1 << 2,
                        &p,
                        rows * n_expert * 32,
                        32,
                    );
                    let tbl = self.scratch_buf(rows * 32, 1);
                    let pso = self.pipelines.get("moe_topk")?;
                    let mut p = (n_expert as u32).to_ne_bytes().to_vec();
                    p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                    p.extend_from_slice(&scale.to_ne_bytes());
                    self.encode_tg_w(
                        r,
                        &pso,
                        &[logits.as_ref(), tbl.as_ref()],
                        1 << 1,
                        &p,
                        rows * 32,
                        32,
                    );
                    // Expert FFN, batched over (chunk rows x selected experts) — one dispatch per
                    // stage per chunk; chunking bounds the expert scratch (a full 8k-row prefill
                    // would need ~0.5 GB of it).
                    let gq = self.weight_qui(gate_exps, g, bindings);
                    let uq = self.weight_qui(up_exps, g, bindings);
                    let dq = self.weight_qui(down_exps, g, bindings);
                    const CHUNK: usize = 256;
                    let chunk = rows.min(CHUNK);
                    let gate_t = self.scratch_buf(chunk * n_used * nffx, 2);
                    let up_t = self.scratch_buf(chunk * n_used * nffx, 3);
                    let act_t = self.scratch_buf(chunk * n_used * nffx, 4);
                    let ydown = self.scratch_buf(chunk * n_used * ne, 5);
                    let act_code: u32 = match act {
                        infr_core::graph::Activation::Silu => 0,
                        infr_core::graph::Activation::Gelu => 1,
                        infr_core::graph::Activation::Sigmoid => 2,
                    };
                    let pack = |in_f: usize, out_f: usize, row0: usize| -> Vec<u8> {
                        let mut p = 1u32.to_ne_bytes().to_vec();
                        p.extend_from_slice(&(in_f as u32).to_ne_bytes());
                        p.extend_from_slice(&(out_f as u32).to_ne_bytes());
                        p.extend_from_slice(&0u32.to_ne_bytes()); // dshift (native kernels)
                        p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                        p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                        p
                    };
                    // Expert-grouped GEMM for prefill-width rows (the llama.cpp mul_mm_id
                    // shape): group the chunk's (token, slot) pairs by expert on-device, then
                    // run the cooperative-GEMM tile per expert over its token group — MMA
                    // compute instead of one GEMV per pair. Small row counts (decode) keep the
                    // GEMV stages below.
                    let grouped =
                        rows >= 16 && ne % 64 == 0 && nffx % 64 == 0 && n_expert <= 1024 && {
                            let kn = if gdt2 == DType::Q4K {
                                "linear_q4k_cmm_id"
                            } else {
                                "linear_q6k_cmm_id"
                            };
                            self.pipelines
                                .get(kn)
                                .map(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                                .unwrap_or(false)
                        };
                    let ids = self.scratch_buf(n_expert * rows.min(CHUNK), 6);
                    let tpe = self.scratch_buf(n_expert, 7);
                    let cmm_id = |dt: DType, scale: bool| -> &'static str {
                        match (dt, scale) {
                            (DType::Q4K, false) => "linear_q4k_cmm_id",
                            (DType::Q4K, true) => "linear_q4k_cmm_id_w",
                            (DType::Q6K, false) => "linear_q6k_cmm_id",
                            _ => "linear_q6k_cmm_id_w",
                        }
                    };
                    for row0 in (0..rows).step_by(CHUNK) {
                        let cr = (rows - row0).min(CHUNK);
                        if grouped {
                            let cap = rows.min(CHUNK);
                            // map: one thread per expert scans the chunk's routing table
                            let pso = self.pipelines.get("moe_map")?;
                            let mut p = (n_expert as u32).to_ne_bytes().to_vec();
                            p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                            p.extend_from_slice(&(cr as u32).to_ne_bytes());
                            p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                            p.extend_from_slice(&(cap as u32).to_ne_bytes());
                            self.encode_tg(
                                r,
                                &pso,
                                &[tbl.as_ref(), ids.as_ref(), tpe.as_ref()],
                                &p,
                                n_expert,
                                n_expert,
                            );
                            let ntt = cr.div_ceil(32);
                            let idpack = |in_f: usize, out_f: usize| -> Vec<u8> {
                                let mut p = (in_f as u32).to_ne_bytes().to_vec();
                                p.extend_from_slice(&(out_f as u32).to_ne_bytes());
                                p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                                p.extend_from_slice(&(cap as u32).to_ne_bytes());
                                p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                                p.extend_from_slice(&(ntt as u32).to_ne_bytes());
                                p
                            };
                            let gemm = |me: &Self,
                                        r: &mut Resident,
                                        kern: &'static str,
                                        xb: &MtlBuffer,
                                        q: &QuiWeight,
                                        db: &MtlBuffer,
                                        in_f: usize,
                                        out_f: usize|
                             -> Result<()> {
                                let pso = me.pipelines.get(kern)?;
                                me.encode_tg(
                                    r,
                                    &pso,
                                    &[
                                        xb,
                                        &q.codes,
                                        &q.scm,
                                        &q.dd,
                                        db,
                                        ids.as_ref(),
                                        tpe.as_ref(),
                                        tbl.as_ref(),
                                    ],
                                    &idpack(in_f, out_f),
                                    n_expert * ntt * (out_f / 64) * 128,
                                    128,
                                );
                                Ok(())
                            };
                            gemm(
                                self,
                                r,
                                cmm_id(gdt2, false),
                                bx.as_ref(),
                                &gq,
                                gate_t.as_ref(),
                                ne,
                                nffx,
                            )?;
                            gemm(
                                self,
                                r,
                                cmm_id(udt2, false),
                                bx.as_ref(),
                                &uq,
                                up_t.as_ref(),
                                ne,
                                nffx,
                            )?;
                            let pso = self.pipelines.get("gatedact_f32")?;
                            let mut p = ((cr * n_used) as u32).to_ne_bytes().to_vec();
                            p.extend_from_slice(&(nffx as u32).to_ne_bytes());
                            p.extend_from_slice(&act_code.to_ne_bytes());
                            // up_off + up_stride/gate_stride/gate_block_width — all 0 (packed):
                            // the shader reads the full 7-field GatedActParams, so every dispatch
                            // site must push all of it or the tail fields read garbage.
                            p.extend_from_slice(&[0u8; 16]);
                            self.encode(
                                r,
                                &pso,
                                &[gate_t.as_ref(), up_t.as_ref(), act_t.as_ref()],
                                &p,
                                cr * n_used * nffx,
                            );
                            gemm(
                                self,
                                r,
                                cmm_id(ddt2, true),
                                act_t.as_ref(),
                                &dq,
                                ydown.as_ref(),
                                nffx,
                                ne,
                            )?;
                            let pso = self.pipelines.get("moe_reduce")?;
                            let mut p = (ne as u32).to_ne_bytes().to_vec();
                            p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                            p.extend_from_slice(&(cr as u32).to_ne_bytes());
                            p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                            self.encode(r, &pso, &[ydown.as_ref(), bd.as_ref()], &p, cr * ne);
                            continue;
                        }
                        let pso = self.pipelines.get(moe_kern(gdt2).unwrap().0)?;
                        self.encode_tg_w(
                            r,
                            &pso,
                            &[
                                bx.as_ref(),
                                &gq.codes,
                                &gq.scm,
                                &gq.dd,
                                gate_t.as_ref(),
                                tbl.as_ref(),
                            ],
                            1 << 4,
                            &pack(ne, nffx, row0),
                            cr * n_used * (nffx / 2) * 32,
                            32,
                        );
                        let pso = self.pipelines.get(moe_kern(udt2).unwrap().0)?;
                        self.encode_tg_w(
                            r,
                            &pso,
                            &[
                                bx.as_ref(),
                                &uq.codes,
                                &uq.scm,
                                &uq.dd,
                                up_t.as_ref(),
                                tbl.as_ref(),
                            ],
                            1 << 4,
                            &pack(ne, nffx, row0),
                            cr * n_used * (nffx / 2) * 32,
                            32,
                        );
                        let pso = self.pipelines.get("gatedact_f32")?;
                        let mut p = ((cr * n_used) as u32).to_ne_bytes().to_vec();
                        p.extend_from_slice(&(nffx as u32).to_ne_bytes());
                        p.extend_from_slice(&act_code.to_ne_bytes());
                        // up_off + up_stride/gate_stride/gate_block_width — all 0 (packed): the
                        // shader reads the full 7-field GatedActParams, so every dispatch site
                        // must push all of it or the tail fields read garbage.
                        p.extend_from_slice(&[0u8; 16]);
                        self.encode_w(
                            r,
                            &pso,
                            &[gate_t.as_ref(), up_t.as_ref(), act_t.as_ref()],
                            1 << 2,
                            &p,
                            cr * n_used * nffx,
                        );
                        let pso = self.pipelines.get(moe_kern(ddt2).unwrap().1)?;
                        self.encode_tg_w(
                            r,
                            &pso,
                            &[
                                act_t.as_ref(),
                                &dq.codes,
                                &dq.scm,
                                &dq.dd,
                                ydown.as_ref(),
                                tbl.as_ref(),
                            ],
                            1 << 4,
                            &pack(nffx, ne, row0),
                            cr * n_used * (ne / 2) * 32,
                            32,
                        );
                        let pso = self.pipelines.get("moe_reduce")?;
                        let mut p = (ne as u32).to_ne_bytes().to_vec();
                        p.extend_from_slice(&(n_used as u32).to_ne_bytes());
                        p.extend_from_slice(&(cr as u32).to_ne_bytes());
                        p.extend_from_slice(&(row0 as u32).to_ne_bytes());
                        self.encode_w(r, &pso, &[ydown.as_ref(), bd.as_ref()], 1 << 1, &p, cr * ne);
                    }
                    r.loc[dst.0 as usize] = Loc::Device;
                    return Ok(());
                }
                self.ensure_host(r, g, x);
                self.ensure_host(r, g, router_x);
                let xs = r.vals[x.0 as usize].clone();
                // diffusion-gemma's router reads a DIFFERENTLY normalized/scaled row than the
                // experts (see the `Op::MoeFfn` doc); qwen3moe binds the same handle as `x`.
                let rxs = r.vals[router_x.0 as usize].clone();
                // `x` may hold several rows (the seam's batched prefill) — route + run per row,
                // mirroring the CPU reference's row loop.
                let rows = xs.len() / ne;
                // Router (host, mirroring the CPU reference). Structurally the same top-k selection,
                // but the logits use a naive f32 sum here vs the CPU's 8-accumulator dot, so the
                // summation order differs — top-k can pick differently on a near-tie logit.
                let rw = self.weight_host(router, g, bindings);
                // Per-expert stacked-weight byte-slice sizes (each expert = an equal slice). Fused:
                // `gate_exps` holds BOTH roles ([ne, 2*nffx, n_expert]); `up_exps` is the SAME
                // handle (unused below).
                let gbuf = metal_buf(bindings.get(gate_exps).expect("metal: unbound gate_exps"));
                let dbuf = metal_buf(bindings.get(down_exps).expect("metal: unbound down_exps"));
                let gdt = g.desc(gate_exps).dtype;
                let ddt = g.desc(down_exps).dtype;
                let gst = gbuf.len / n_expert;
                let dsz = dbuf.len / n_expert;
                let (ubuf, udt, ust) = if fused_gate_up {
                    (None, gdt, 0)
                } else {
                    let b = metal_buf(bindings.get(up_exps).expect("metal: unbound up_exps"));
                    let dt = g.desc(up_exps).dtype;
                    let s = b.len / n_expert;
                    (Some(b), dt, s)
                };
                let dscale = down_scale.map(|id| self.weight_host(id, g, bindings));
                let mut out = vec![0f32; rows * ne];
                for (row, orow) in out.chunks_mut(ne).enumerate() {
                    let xr = &xs[row * ne..row * ne + ne];
                    let xrr = &rxs[row * ne..row * ne + ne];
                    let logits: Vec<f32> = (0..n_expert)
                        .map(|e| (0..ne).map(|i| rw[e * ne + i] * xrr[i]).sum::<f32>())
                        .collect();
                    let maxl = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                    let psum: f32 = probs.iter().sum();
                    for p in probs.iter_mut() {
                        *p /= psum;
                    }
                    let mut idx: Vec<usize> = (0..n_expert).collect();
                    idx.sort_by(|&a, &b| {
                        probs[b]
                            .partial_cmp(&probs[a])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    idx.truncate(n_used);
                    let wsum: f32 = idx.iter().map(|&e| probs[e]).sum::<f32>().max(1e-20);
                    for &e in &idx {
                        // Dequant only this expert's slices, then matvec on the GPU.
                        let (gate, up) = if fused_gate_up {
                            let full =
                                bytes_to_f32(&Self::read_bytes_range(gbuf, e * gst, gst), gdt);
                            let full = self.gpu_matvec(xr, &full, ne, 2 * nffx)?;
                            (full[..nffx].to_vec(), full[nffx..].to_vec())
                        } else {
                            let ubuf = ubuf.as_ref().expect("split gate/up: up_exps missing");
                            let gw = bytes_to_f32(&Self::read_bytes_range(gbuf, e * gst, gst), gdt);
                            let uw = bytes_to_f32(&Self::read_bytes_range(ubuf, e * ust, ust), udt);
                            (
                                self.gpu_matvec(xr, &gw, ne, nffx)?,
                                self.gpu_matvec(xr, &uw, ne, nffx)?,
                            )
                        };
                        let actv: Vec<f32> =
                            (0..nffx).map(|i| act_fn(act, gate[i]) * up[i]).collect();
                        let dw = bytes_to_f32(&Self::read_bytes_range(dbuf, e * dsz, dsz), ddt);
                        let mut y = self.gpu_matvec(&actv, &dw, nffx, ne)?;
                        if let Some(ds) = &dscale {
                            let s = ds[e];
                            for v in y.iter_mut() {
                                *v *= s;
                            }
                        }
                        let w_e = probs[e] / wsum * scale;
                        for i in 0..ne {
                            orow[i] += w_e * y[i];
                        }
                    }
                }
                r.vals[dst.0 as usize] = out;
                r.loc[dst.0 as usize] = Loc::Host;
            }
            // Sequential per-token recurrences (control-flow heavy, tiny) — host-side, exactly
            // mirroring the CPU reference. The recurrent `state` is a bound f32 Input read/written
            // through `vals` (loaded in the preamble, written back after the op loop).
            Op::Conv1dSilu {
                x,
                weight,
                state,
                dst,
                rows,
                channels,
                kernel,
            } => {
                let (rr, cc, kk) = (rows as usize, channels as usize, kernel as usize);
                // Device path: each thread owns one channel and its state column — no cross-
                // thread deps, rows loop in-kernel, state updated in the BOUND buffer (the
                // write-back self-copy guard keeps it authoritative). kk > 8 exceeds the
                // kernel's register window; no real conv ships that (qwen3-next uses 4).
                if kk <= 8 && self.metal().deltanet {
                    if let Some(sb) = bindings.get(state) {
                        let bx = self.ensure_device(r, x);
                        let bw = self.weight_buf(weight, g, bindings)?;
                        let bd = self.dev_dst(r, dst, rr * cc);
                        let sbuf = metal_buf(sb);
                        let i = state.0 as usize;
                        r.dev[i] = Some(Arc::new(sbuf.raw.clone()));
                        r.loc[i] = Loc::Device;
                        let mut p = (rr as u32).to_ne_bytes().to_vec();
                        p.extend_from_slice(&(cc as u32).to_ne_bytes());
                        p.extend_from_slice(&(kk as u32).to_ne_bytes());
                        let parallel = prefer_conv1d_parallel(rr, kk) && self.metal().conv1d_par;
                        if parallel {
                            let pso = self.pipelines.get("conv1d_silu_par_f32")?;
                            self.encode_w(
                                r,
                                &pso,
                                &[bx.as_ref(), bw.as_ref(), &sbuf.raw, bd.as_ref()],
                                1 << 3,
                                &p,
                                rr * cc,
                            );
                            let shift = self.pipelines.get("conv1d_shift_f32")?;
                            self.encode_w(
                                r,
                                &shift,
                                &[bx.as_ref(), &sbuf.raw],
                                1 << 1,
                                &p,
                                (kk - 1) * cc,
                            );
                        } else {
                            let pso = self.pipelines.get("conv1d_silu_f32")?;
                            self.encode_w(
                                r,
                                &pso,
                                &[bx.as_ref(), bw.as_ref(), &sbuf.raw, bd.as_ref()],
                                (1 << 2) | (1 << 3),
                                &p,
                                cc,
                            );
                        }
                        r.loc[dst.0 as usize] = Loc::Device;
                        return Ok(());
                    }
                }
                self.ensure_host(r, g, x);
                self.ensure_host(r, g, state);
                let xs = r.vals[x.0 as usize].clone(); // [rows, channels]
                let ws = self.weight_host(weight, g, bindings); // [channels, kernel]
                let st = &mut r.vals[state.0 as usize]; // [(kernel-1), channels], oldest first
                let mut out = vec![0f32; rr * cc];
                for t in 0..rr {
                    let xt = &xs[t * cc..t * cc + cc];
                    for ch in 0..cc {
                        let mut acc = 0f32;
                        for j in 0..kk - 1 {
                            acc += st[j * cc + ch] * ws[ch * kk + j];
                        }
                        acc += xt[ch] * ws[ch * kk + (kk - 1)];
                        out[t * cc + ch] = acc / (1.0 + (-acc).exp()); // silu
                    }
                    for j in 0..kk.saturating_sub(2) {
                        for ch in 0..cc {
                            st[j * cc + ch] = st[(j + 1) * cc + ch];
                        }
                    }
                    if kk >= 2 {
                        for ch in 0..cc {
                            st[(kk - 2) * cc + ch] = xt[ch];
                        }
                    }
                }
                r.vals[dst.0 as usize] = out;
                r.loc[dst.0 as usize] = Loc::Host;
            }
            Op::DeltaNet {
                q,
                k,
                v,
                b,
                a,
                a_coef,
                dt_bias,
                state,
                dst,
                rows,
                n_vhead,
                n_khead,
                head_k,
                head_v,
                eps,
                src_stride,
            } => {
                // Strided q/k/v from a shared conv_out buffer is the INFR_DELTA_STRIDED experimental
                // decode path (Vulkan-only); the runner always emits src_stride == 0 for the Metal
                // graph (separate packed q/k/v buffers via CopyStrided).
                assert_eq!(
                    src_stride, 0,
                    "Metal DeltaNet: strided q/k/v not supported (runner emits packed buffers)"
                );
                let (rr, nv, nk, kd, vd) = (
                    rows as usize,
                    n_vhead as usize,
                    n_khead as usize,
                    head_k as usize,
                    head_v as usize,
                );
                // Device path: one SIMDGROUP per (value-dim, head) — 32 lanes split the k-dim
                // (kd/32 state entries per lane, register-resident for the whole chunk), so the
                // row scan needs only simd_sums; 4 simdgroups share a threadgroup for occupancy
                // (grid = nv * vd/4 threadgroups vs the old nv). State updates the BOUND buffer
                // in place. Gates match the kernel's register/split budgets (kd % 32 for the
                // per-lane split, kd <= 256 for the ls[] register cap, vd % 4 for the
                // simdgroup packing).
                if kd <= 256
                    && kd.is_multiple_of(32)
                    && vd.is_multiple_of(4)
                    && vd <= 1024
                    && self.metal().deltanet
                {
                    if let Some(sb) = bindings.get(state) {
                        // KPL = kd/32 is a compile-time template parameter (register
                        // promotion needs the fixed bound) — pick the instantiation.
                        let gate_prep = prefer_deltanet_gate_prep(rr) && self.metal().dn_gate_prep;
                        let norm_prep =
                            gate_prep && prefer_deltanet_norm_prep(rr) && self.metal().dn_norm_prep;
                        let dn_kern: &'static str = match (gate_prep, norm_prep, kd / 32) {
                            (false, _, 1) => "deltanet_f32_k1",
                            (false, _, 2) => "deltanet_f32_k2",
                            (false, _, 4) => "deltanet_f32_k4",
                            (false, _, _) => "deltanet_f32_k8",
                            (true, false, 1) => "deltanet_gates_k1",
                            (true, false, 2) => "deltanet_gates_k2",
                            (true, false, 4) => "deltanet_gates_k4",
                            (true, false, _) => "deltanet_gates_k8",
                            (true, true, 1) => "deltanet_prepared_k1",
                            (true, true, 2) => "deltanet_prepared_k2",
                            (true, true, 4) => "deltanet_prepared_k4",
                            (true, true, _) => "deltanet_prepared_k8",
                        };
                        let fits = self
                            .pipelines
                            .get(dn_kern)
                            .map(|pl| pl.max_total_threads_per_threadgroup() >= 128)
                            .unwrap_or(false);
                        if fits {
                            let bq = self.ensure_device(r, q);
                            let bk = self.ensure_device(r, k);
                            let bv = self.ensure_device(r, v);
                            let bb = self.ensure_device(r, b);
                            let ba = self.ensure_device(r, a);
                            let bac = self.weight_buf(a_coef, g, bindings)?;
                            let bdt = self.weight_buf(dt_bias, g, bindings)?;
                            let gates = if gate_prep {
                                self.scratch_buf(rr * nv * 2, 15)
                            } else {
                                bb.clone()
                            };
                            let (scan_q, scan_k) = if norm_prep {
                                (
                                    self.scratch_buf(rr * nk * kd, 16),
                                    self.scratch_buf(rr * nk * kd, 17),
                                )
                            } else {
                                (bq.clone(), bk.clone())
                            };
                            let bd = self.dev_dst(r, dst, rr * nv * vd);
                            let sbuf = metal_buf(sb);
                            let i = state.0 as usize;
                            r.dev[i] = Some(Arc::new(sbuf.raw.clone()));
                            r.loc[i] = Loc::Device;
                            let pso = self.pipelines.get(dn_kern)?;
                            let mut p = (rr as u32).to_ne_bytes().to_vec();
                            p.extend_from_slice(&(nv as u32).to_ne_bytes());
                            p.extend_from_slice(&(nk as u32).to_ne_bytes());
                            p.extend_from_slice(&(kd as u32).to_ne_bytes());
                            p.extend_from_slice(&(vd as u32).to_ne_bytes());
                            p.extend_from_slice(&eps.to_ne_bytes());
                            if gate_prep {
                                let gpso = self.pipelines.get("deltanet_gates_f32")?;
                                let mut gp = (rr as u32).to_ne_bytes().to_vec();
                                gp.extend_from_slice(&(nv as u32).to_ne_bytes());
                                self.encode_w(
                                    r,
                                    &gpso,
                                    &[
                                        bb.as_ref(),
                                        ba.as_ref(),
                                        bac.as_ref(),
                                        bdt.as_ref(),
                                        gates.as_ref(),
                                    ],
                                    1 << 4,
                                    &gp,
                                    rr * nv,
                                );
                            }
                            if norm_prep {
                                let norm_kern = match kd / 32 {
                                    1 => "deltanet_norm_k1",
                                    2 => "deltanet_norm_k2",
                                    4 => "deltanet_norm_k4",
                                    _ => "deltanet_norm_k8",
                                };
                                let npso = self.pipelines.get(norm_kern)?;
                                let mut np = (rr as u32).to_ne_bytes().to_vec();
                                np.extend_from_slice(&(nk as u32).to_ne_bytes());
                                np.extend_from_slice(&(kd as u32).to_ne_bytes());
                                np.extend_from_slice(&eps.to_ne_bytes());
                                self.encode_tg_w(
                                    r,
                                    &npso,
                                    &[bq.as_ref(), bk.as_ref(), scan_q.as_ref(), scan_k.as_ref()],
                                    (1 << 2) | (1 << 3),
                                    &np,
                                    (rr * nk).div_ceil(4) * 128,
                                    128,
                                );
                            }
                            self.encode_tg_w(
                                r,
                                &pso,
                                &[
                                    scan_q.as_ref(),
                                    scan_k.as_ref(),
                                    bv.as_ref(),
                                    bb.as_ref(),
                                    ba.as_ref(),
                                    bac.as_ref(),
                                    bdt.as_ref(),
                                    gates.as_ref(),
                                    &sbuf.raw,
                                    bd.as_ref(),
                                ],
                                (1 << 8) | (1 << 9),
                                &p,
                                nv * (vd / 4) * 128,
                                128,
                            );
                            r.loc[dst.0 as usize] = Loc::Device;
                            return Ok(());
                        }
                    }
                }
                for id in [q, k, v, b, a, state] {
                    self.ensure_host(r, g, id);
                }
                let qf = r.vals[q.0 as usize].clone(); // [rows, nk*kd]
                let kf = r.vals[k.0 as usize].clone();
                let vf = r.vals[v.0 as usize].clone(); // [rows, nv*vd]
                let bf = r.vals[b.0 as usize].clone(); // [rows, nv]
                let af = r.vals[a.0 as usize].clone();
                let acoef = self.weight_host(a_coef, g, bindings);
                let dtb = self.weight_host(dt_bias, g, bindings);
                let st = &mut r.vals[state.0 as usize]; // [nv, kd, vd]
                let mut out = vec![0f32; rr * nv * vd];
                let qscale = 1.0 / (kd as f32).sqrt();
                let l2 = |slice: &[f32]| -> f32 {
                    (slice.iter().map(|x| x * x).sum::<f32>() + eps).sqrt()
                };
                // Sequential scan over rows, carrying the per-head state S across tokens.
                for t in 0..rr {
                    let (qb, vb, bb) = (t * nk * kd, t * nv * vd, t * nv);
                    for h in 0..nv {
                        let kh_idx = h % nk; // GQA: q/k heads tiled to nv value heads
                        let mut qh = qf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                        let mut kh = kf[qb + kh_idx * kd..qb + kh_idx * kd + kd].to_vec();
                        let vh = &vf[vb + h * vd..vb + h * vd + vd];
                        let qn = l2(&qh);
                        let kn = l2(&kh);
                        for x in qh.iter_mut() {
                            *x = *x / qn * qscale;
                        }
                        for x in kh.iter_mut() {
                            *x /= kn;
                        }
                        let beta = 1.0 / (1.0 + (-bf[bb + h]).exp());
                        let sp = {
                            let z = af[bb + h] + dtb[h];
                            z.max(0.0) + (-z.abs()).exp().ln_1p()
                        };
                        let decay = (acoef[h] * sp).exp();
                        let sh = &mut st[h * kd * vd..(h + 1) * kd * vd]; // [kd, vd]
                        for x in sh.iter_mut() {
                            *x *= decay;
                        }
                        let mut kv = vec![0f32; vd];
                        for kk in 0..kd {
                            let kkv = kh[kk];
                            let row = &sh[kk * vd..kk * vd + vd];
                            for d in 0..vd {
                                kv[d] += kkv * row[d];
                            }
                        }
                        let delta: Vec<f32> = (0..vd).map(|d| (vh[d] - kv[d]) * beta).collect();
                        for kk in 0..kd {
                            let kkv = kh[kk];
                            let row = &mut sh[kk * vd..kk * vd + vd];
                            for d in 0..vd {
                                row[d] += kkv * delta[d];
                            }
                        }
                        let oh = &mut out[vb + h * vd..vb + h * vd + vd];
                        for kk in 0..kd {
                            let qv = qh[kk];
                            let row = &sh[kk * vd..kk * vd + vd];
                            for d in 0..vd {
                                oh[d] += qv * row[d];
                            }
                        }
                    }
                }
                r.vals[dst.0 as usize] = out;
                r.loc[dst.0 as usize] = Loc::Host;
            }
            // Pure data movement, ON-DEVICE (the rows=1 case of `copy_strided_f32`): the prefill
            // graph extracts the last row's hidden state for the LM head with a Copy, and a host
            // copy here forced a readback + a command-buffer break every prefill chunk.
            Op::Copy {
                src,
                src_off,
                dst,
                dst_off,
                n,
            } => {
                let bs = self.ensure_device(r, src);
                let bd = self.ensure_device(r, dst);
                let mut p = src_off.to_ne_bytes().to_vec();
                p.extend_from_slice(&0u32.to_ne_bytes()); // src_stride (unused at rows=1)
                p.extend_from_slice(&dst_off.to_ne_bytes());
                p.extend_from_slice(&0u32.to_ne_bytes()); // dst_stride
                p.extend_from_slice(&1u32.to_ne_bytes()); // rows
                p.extend_from_slice(&n.to_ne_bytes());
                let pso = self.pipelines.get("copy_strided_f32")?;
                self.encode_w(r, &pso, &[bs.as_ref(), bd.as_ref()], 1 << 1, &p, n as usize);
                r.loc[dst.0 as usize] = Loc::Device;
            }
            // On-device: the fused-QKV prefill emits one of these per projection right after the
            // wide GEMM — a host copy would round-trip the [m, qkv] activation mid-forward.
            Op::CopyStrided {
                src,
                src_off,
                src_stride,
                dst,
                dst_off,
                dst_stride,
                rows,
                n,
            } => {
                let bs = self.ensure_device(r, src);
                // The strided write may cover only part of dst — bring the rest along.
                let bd = self.ensure_device(r, dst);
                let mut p = src_off.to_ne_bytes().to_vec();
                p.extend_from_slice(&src_stride.to_ne_bytes());
                p.extend_from_slice(&dst_off.to_ne_bytes());
                p.extend_from_slice(&dst_stride.to_ne_bytes());
                p.extend_from_slice(&rows.to_ne_bytes());
                p.extend_from_slice(&n.to_ne_bytes());
                let pso = self.pipelines.get("copy_strided_f32")?;
                self.encode(
                    r,
                    &pso,
                    &[bs.as_ref(), bd.as_ref()],
                    &p,
                    rows as usize * n as usize,
                );
                r.loc[dst.0 as usize] = Loc::Device;
            }
            Op::Mla { .. } => {
                return Err(Error::Unsupported(
                    "Metal Op::Mla (DeepSeek V2 MLA attention) not yet implemented".into(),
                ));
            }
            Op::MoeSharedExpertAdd { .. } => {
                // qwen35moe (Qwen3.6 MoE) shared expert — landed on CPU + Vulkan only so far (see
                // `Op::MoeSharedExpertAdd`'s doc); no Metal kernel yet. Fails loudly instead of a
                // silent wrong-output run rather than pretending to support it blind.
                return Err(Error::Unsupported(
                    "Metal Op::MoeSharedExpertAdd (qwen35moe shared expert) not yet implemented"
                        .into(),
                ));
            }
        }
        Ok(())
    }
}
