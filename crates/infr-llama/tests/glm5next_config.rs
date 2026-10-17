//! M1 acceptance for the `glm5next` (GLM-5.3-Flash) port: parse the REAL GGUF metadata and
//! assert every hyper-parameter lands on the right `Config` field. The model is a 90 GiB
//! three-shard IQ2_XXS quant, so unlike the small-model goldens this test is opt-in: point
//! `INFR_TEST_GLM5NEXT` at shard 1 of the set and it runs; without the env var it skips.
//!
//! The values asserted here are the ones the port plan (`research/glm5next-gguf-keys.md`)
//! recorded from the Maya-S v2 export — they are the contract between the GGUF and the loader.

use std::path::PathBuf;

fn model() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("INFR_TEST_GLM5NEXT").ok()?);
    p.exists().then_some(p)
}

#[test]
fn glm5next_config_parses_real_gguf() {
    let Some(path) = model() else {
        eprintln!("skip: set INFR_TEST_GLM5NEXT to the glm5.3-flash GGUF shard 1");
        return;
    };
    let g = infr_gguf::Gguf::open(&path).expect("open gguf shard set");
    let c = infr_llama::Config::from_gguf(&g).expect("config parse");

    // Identity & topology.
    assert!(c.glm5next, "glm5next flag");
    assert!(!c.bailingmoe3 && !c.deepseek2 && !c.deepseek32 && !c.deepseek4);
    assert_eq!(c.n_layer, 45, "trunk = 46 blocks - 1 NextN");
    assert_eq!(c.n_layer_nextn, 1, "nextn_predict_layers");
    assert_eq!(c.n_embd, 4096);
    assert_eq!(c.vocab, 154_880);
    assert_eq!(c.n_ctx_train, 1_048_576);
    assert_eq!(c.n_head, 64);
    assert_eq!(c.n_kv, 1, "MLA arch pins a single shared KV head");

    // NoPE: rope.dimension_count = 0 must land as a zero rotary width, and the arch must not
    // claim any rope scaling.
    assert_eq!(c.qk_rope_dim, 0, "NoPE");

    // MLA geometry (deepseek2-shaped).
    assert_eq!(c.q_lora_rank, 1536);
    assert_eq!(c.kv_lora_rank, 512);
    assert_eq!(c.head_k_mla, 256);
    assert_eq!(c.v_head_dim, 256);

    // KDA geometry (bailingmoe3-shaped).
    assert_eq!(c.kda_head_dim, 128);
    assert!((c.kda_gate_lower_bound - -5.0).abs() < 1e-6);
    assert_eq!(c.ssm_d_conv, 4);
    assert_eq!(c.ssm_d_state, 128);
    assert_eq!(c.ssm_d_inner, 64 * 128);
    assert_eq!(c.ssm_n_group, 64);
    assert_eq!(c.ssm_dt_rank, 64);

    // Lightning indexer + GLM-only key pooling.
    assert_eq!(c.indexer_n_head, 32);
    assert_eq!(c.indexer_head_size, 128);
    assert_eq!(c.indexer_top_k, 2048);
    assert_eq!(c.indexer_kpool, 4, "attention.indexer.kpool");
    assert!((c.norm_eps - 1e-6).abs() < 1e-12, "indexer k_norm eps");

    // Hyper-connections (deepseek4-shaped triple, no low-rank / hash).
    assert_eq!(c.hc_mult, 4);
    assert_eq!(c.hc_low_rank, 0);
    assert_eq!(c.hc_sinkhorn_iters, 20);
    assert!(
        c.hc_eps > 0.0 && c.hc_eps < 1e-3,
        "hyper_connection.epsilon"
    );
    assert_eq!(c.o_group_count, 0);
    assert_eq!(c.hash_layer_count, 0);

    // MoE: 288 routed (sigmoid, renormalized, scale 2.5, bias) + 1 shared, 3 dense-lead.
    let moe = c.moe.clone().expect("moe config");
    assert_eq!(moe.n_expert, 288);
    assert_eq!(moe.n_used, 8);
    assert_eq!(moe.n_ff_exp, 2048);
    assert_eq!(c.n_expert_groups, 1);
    assert_eq!(c.n_expert_groups_used, 1);
    assert_eq!(moe.gating, infr_core::graph::MoeGating::Sigmoid);
    assert!(moe.norm_w, "expert_weights_norm");
    assert!((moe.scale - 2.5).abs() < 1e-6, "expert_weights_scale");
    assert_eq!(c.n_layer_dense_lead, 3);
    assert_eq!(c.shexp_ff, 2048);

    // Per-layer SwiGLU clamps: full arrays, all 10.0.
    assert_eq!(c.swiglu_clamp_exp.len(), 45);
    assert!(c.swiglu_clamp_exp.iter().all(|&v| (v - 10.0).abs() < 1e-6));
    assert_eq!(c.swiglu_clamp_shexp.len(), 45);
    assert!(c
        .swiglu_clamp_shexp
        .iter()
        .all(|&v| (v - 10.0).abs() < 1e-6));

    // Mixer selector: MLA at 3, 7, …, 43 (11 of 45), KDA everywhere else.
    assert_eq!(c.glm_mla_layers.len(), 45);
    for l in 0..45 {
        let want = l >= 3 && l % 4 == 3;
        assert_eq!(c.is_mla_layer(l), want, "is_mla_layer({l})");
        assert_eq!(c.is_recurrent_layer(l), !want, "is_recurrent_layer({l})");
    }
    assert_eq!(c.glm_mla_layers.iter().filter(|b| **b).count(), 11);

    // Dense-lead MoE threshold: layers 0..3 dense, 3..45 routed.
    for l in 0..45 {
        assert_eq!(c.is_moe_layer(l), l >= 3, "is_moe_layer({l})");
    }
}
