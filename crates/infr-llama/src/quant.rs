//! Host-side weight loading: GGUF tensor → dequantized f32 or f16 bytes.
//! Pure dequant kernels have moved to `infr_gguf::dequant`; re-exported here
//! so all existing `crate::dequant_block` / `crate::rdf16` call-sites continue to work.
use anyhow::{anyhow, Context, Result};
use infr_core::WeightSource;
use infr_gguf::Gguf;

// Re-export from infr_gguf::dequant so existing callers inside infr-llama resolve via `crate::`.
pub(crate) use infr_gguf::dequant::dequant_block;

// Test-only re-exports: transformer.rs `gpu_affine_tests` use these via `use crate::*;`.

/// Load a named tensor and dequantize it to host f32, returning (data, shape in GGUF ne order
/// `[in, out]`). The host/CPU-side dequant path — it does NOT load the bulk projection weights for
/// the GPU (those upload quantized/f16 in-VRAM). It feeds: the host embedding gather, the CPU norm
/// and SSM recurrence math (qwen35), the `Q35_CPU=1` oracle, and serves as the f32 source we
/// convert into f16/bf16/quant GPU weights. Survives even with full GPU format coverage.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub(crate) fn load_tensor_dequant(g: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let info = g
        .tensors()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("tensor not found: {name}"))?
        .clone();
    let bytes = g.tensor_bytes(name).map_err(|e| anyhow!("{e}"))?;
    let v = dequant_block(info.dtype, bytes).with_context(|| format!("tensor {name}"))?;
    Ok((v, info.shape))
}
