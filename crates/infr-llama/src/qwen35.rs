//! Qwen3.5 / Qwen3.6 (arch `qwen35`): hybrid gated-DeltaNet linear-attention + gated
//! full-attention. See `docs/qwen35.md`.
//!
//! NOT Qwen3-Next: llama.cpp's `qwen3next` is a SIBLING arch whose DeltaNet V heads broadcast in
//! a different pattern — Qwen3-Next blocks them (`[k0_v0, k0_v1, k1_v2, k1_v3]`) while Qwen3.5
//! interleaves (`[k0_v0, k1_v1, k0_v2, k1_v3]`, the `h % n_khead` tiling implemented here). A
//! `qwen3next` GGUF through this code would produce silently wrong output — the arch gate below
//! rejecting anything but `qwen35` is load-bearing, not cosmetic. (`qwen35moe` is likewise
//! unsupported until its expert FFN lands.)
//!
//! This module holds what's specific to qwen35 outside the shared skeleton: the raw GGUF
//! metadata parse ([`Cfg`], superseded in production by `Config::from_gguf`'s `qwen35` fields
//! but still used by tests that want the metadata without building a full `Config`), arch
//! detection ([`is_qwen35`]), and the chat-template renderer ([`render_chat`] /
//! [`render_chat_messages`]). The actual forward — batched/chunked prefill + per-token decode —
//! runs through the UNIFIED backend-agnostic seam (`crate::seam`, the shared model type at
//! `seam/model.rs`, `MixerW::DeltaNet` in `seam/weights.rs`), the SAME engine every other
//! architecture uses. There used to be a second, bespoke qwen35-only seam living in this file
//! (reachable via a temporary env-gated escape hatch) for cross-validation during the cutover;
//! it's gone now that the unified path is the only one (issue #30, phase 4).

use anyhow::{anyhow, bail, Result};
use infr_core::WeightSource;
use infr_gguf::Gguf;

/// Parsed `qwen35` hyper-parameters (subset needed for the 0.8B dense model).
#[derive(Debug, Clone)]
pub struct Cfg {
    pub n_layer: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub eps: f32,
    // attention layers
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize, // key_length == value_length (256)
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rope_sections: [u32; 4],
    pub full_attn_interval: usize,
    // linear (gated DeltaNet) layers
    pub d_conv: usize,  // ssm conv kernel (4)
    pub d_state: usize, // head_k_dim (128)
    pub d_inner: usize, // value_dim (2048)
    pub n_group: usize, // num_k_heads (16)
    pub dt_rank: usize, // num_v_heads (16)
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl Cfg {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.metadata().str("general.architecture").unwrap_or("");
        if arch != crate::arch::QWEN35 {
            bail!("not a qwen35 model (arch={arch:?})");
        }
        let u = |k: &str| g.metadata().u64(&format!("qwen35.{k}"));
        let req = |k: &str| u(k).ok_or_else(|| anyhow!("missing qwen35.{k}"));
        let f = |k: &str| -> Option<f32> {
            g.metadata()
                .get(&format!("qwen35.{k}"))
                .and_then(|v| match v {
                    infr_core::MetaValue::F64(x) => Some(*x as f32),
                    infr_core::MetaValue::U64(x) => Some(*x as f32),
                    infr_core::MetaValue::I64(x) => Some(*x as f32),
                    _ => None,
                })
        };
        // rope.dimension_sections is an array [11,11,10,0]
        let sections: [u32; 4] = {
            let mut s = [0u32; 4];
            if let Some(arr) = g
                .metadata()
                .get("qwen35.rope.dimension_sections")
                .and_then(|v| v.as_arr())
            {
                for (i, v) in arr.iter().take(4).enumerate() {
                    s[i] = v.as_u64().unwrap_or(0) as u32;
                }
            }
            s
        };
        Ok(Cfg {
            n_layer: req("block_count")? as usize,
            n_embd: req("embedding_length")? as usize,
            vocab: 0, // filled from token_embd shape
            eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            n_head: req("attention.head_count")? as usize,
            n_kv: req("attention.head_count_kv")? as usize,
            head_dim: req("attention.key_length")? as usize,
            rope_dim: u("rope.dimension_count").unwrap_or(64) as usize,
            rope_theta: f("rope.freq_base").unwrap_or(1e7),
            rope_sections: sections,
            full_attn_interval: u("full_attention_interval").unwrap_or(4) as usize,
            d_conv: req("ssm.conv_kernel")? as usize,
            d_state: req("ssm.state_size")? as usize,
            d_inner: req("ssm.inner_size")? as usize,
            n_group: req("ssm.group_count")? as usize,
            dt_rank: req("ssm.time_step_rank")? as usize,
        })
    }

    /// Attention (vs linear/SSM) layer test: every `full_attn_interval`-th layer is full attention.
    pub fn is_attn_layer(&self, i: usize) -> bool {
        (i + 1).is_multiple_of(self.full_attn_interval)
    }
    pub fn num_k_heads(&self) -> usize {
        self.n_group
    }
    pub fn num_v_heads(&self) -> usize {
        self.dt_rank
    }
    pub fn head_k_dim(&self) -> usize {
        self.d_state
    }
    pub fn head_v_dim(&self) -> usize {
        self.d_inner / self.dt_rank
    }
    pub fn conv_channels(&self) -> usize {
        self.d_inner + 2 * self.n_group * self.d_state
    }
}

/// Render a plain user message through the qwen35 GGUF's own jinja chat template (falls back to
/// ChatML — qwen35's native format — if there's no template). So `infr run` / tests pass plain text.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn render_chat(
    path: &std::path::Path,
    user: &str,
    ecfg: &crate::EngineConfig,
) -> Result<String> {
    render_chat_messages(path, &[("user", user)], ecfg)
}

/// Render a multi-turn conversation `(role, content)` through the qwen35 GGUF's own jinja chat
/// template — the [`crate::chat::ChatModel::render`] primitive for the qwen35 GPU + CPU paths, so
/// the shared [`crate::chat::Chat`] can drive a history-based REPL. Errors if the GGUF has no usable
/// `tokenizer.chat_template`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn render_chat_messages(
    path: &std::path::Path,
    messages: &[(&str, &str)],
    ecfg: &crate::EngineConfig,
) -> Result<String> {
    let g = Gguf::open(path).map_err(|e| anyhow!("open gguf: {e}"))?;
    let tok = crate::build_tokenizer(&g)?;
    let eos = g.metadata().u64("tokenizer.ggml.eos_token_id").unwrap_or(2) as u32;
    infr_chat::render_chat_jinja(&g, &tok, eos, messages, true, ecfg).ok_or_else(|| {
        anyhow!(
            "model GGUF has no usable chat template (no `tokenizer.chat_template`, or it failed to \
             render — set INFR_DEBUG_CHAT=1 for details)."
        )
    })
}

/// True if the GGUF at `path` is a `qwen35` (Qwen3.5/Qwen3.6) model.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn is_qwen35(path: &std::path::Path) -> bool {
    Gguf::open(path)
        .ok()
        .map(|g| g.metadata().str("general.architecture") == Some(crate::arch::QWEN35))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locate the Qwen3.5-0.8B GGUF in the HF Hub cache (or `INFR_TEST_MODEL`), or `None` if it isn't
    /// present (the test self-skips).
    fn model_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("INFR_TEST_MODEL") {
            return Some(std::path::PathBuf::from(p));
        }
        let hub = std::env::var("HOME").ok()? + "/.cache/huggingface/hub";
        let base = format!("{hub}/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots");
        std::fs::read_dir(&base).ok()?.find_map(|e| {
            let f = e.ok()?.path().join("Qwen3.5-0.8B-Q4_K_M.gguf");
            f.exists().then_some(f)
        })
    }

    /// Serialize the qwen35 tests: several toggle process-global env vars mid-generate, which
    /// would otherwise race with a concurrently-running test in the same process. Poison-tolerant.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn loads_and_dims() {
        let _s = serial();
        let Some(path) = model_path() else {
            eprintln!("skip: Qwen3.5-0.8B not present");
            return;
        };
        let g = Gguf::open(&path).unwrap();
        let c = Cfg::from_gguf(&g).unwrap();
        println!("cfg: {c:?}");
        println!(
            "k_heads={} head_k={} v_heads={} head_v={} conv_ch={}",
            c.num_k_heads(),
            c.head_k_dim(),
            c.num_v_heads(),
            c.head_v_dim(),
            c.conv_channels()
        );
        assert_eq!(c.n_layer, 24);
        assert_eq!(c.conv_channels(), 6144);
        assert_eq!(c.head_v_dim(), 128);
        let n_attn = (0..c.n_layer).filter(|&i| c.is_attn_layer(i)).count();
        assert_eq!(n_attn, 6, "expected 6 full-attention layers");
    }
}
