use anyhow::{anyhow, ensure, Context, Result};
use infr_core::backend::{Backend, Buffer, BufferUsage};
use infr_core::{DType, WeightSource};
use infr_gguf::Gguf;
use infr_vulkan::VulkanBackend;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

pub(super) struct DraftVocab {
    pub(super) weights: Box<dyn Buffer>,
    pub(super) id_map: Box<dyn Buffer>,
    pub(super) dtype: DType,
    pub(super) numel: usize,
    ids: Vec<i32>,
}

pub(super) fn validate_raw_ids(ids: &[u32], rows: usize) -> Result<()> {
    for (step, &id) in ids.iter().enumerate() {
        ensure!(
            (id as usize) < rows,
            "MTP draft argmax at step {step} returned invalid index {id} for {rows} rows"
        );
    }
    Ok(())
}

pub(super) fn load_identity_map(vk: &VulkanBackend, rows: usize) -> Result<Box<dyn Buffer>> {
    ensure!(
        rows > 0 && rows <= i32::MAX as usize,
        "invalid MTP vocabulary size"
    );
    let ids = (0..rows as i32).collect::<Vec<_>>();
    let map = vk
        .alloc(rows * 4, BufferUsage::Weights)
        .map_err(|e| anyhow!("{e}"))?;
    vk.upload(map.as_ref(), bytemuck::cast_slice(&ids))
        .map_err(|e| anyhow!("{e}"))?;
    tracing::info!(
        extra_fixed_identity_bytes = map.len_bytes(),
        "loaded MTP full-vocabulary safety identity map"
    );
    Ok(map)
}

fn parse_ids(bytes: &[u8], vocab: usize) -> Result<Vec<i32>> {
    ensure!(
        vocab > 0 && vocab <= i32::MAX as usize,
        "invalid MTP vocabulary size"
    );
    ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(4) && bytes.len() / 4 <= vocab,
        "MTP vocabulary IDs must be a nonempty little-endian I32 subset"
    );
    let mut ids = bytes
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(
        ids.iter().all(|&id| id >= 0 && (id as usize) < vocab),
        "MTP vocabulary contains an out-of-range token ID"
    );
    // Ascending original IDs preserve the full head's lowest-token-ID argmax tie rule.
    ids.sort_unstable();
    ensure!(
        ids.windows(2).all(|pair| pair[0] != pair[1]),
        "duplicate MTP vocabulary IDs"
    );
    Ok(ids)
}

fn pack_quant_rows(
    source: &[u8],
    dtype: DType,
    width: usize,
    vocab: usize,
    ids: &[i32],
) -> Result<(Vec<u8>, usize)> {
    ensure!(
        dtype.is_quant() && infr_vulkan::linear::native_dense_supported(dtype),
        "MTP compact vocabulary requires a supported native quantized output, got {dtype:?}"
    );
    let (block_elements, block_bytes) = infr_gguf::block_layout(dtype);
    ensure!(
        width > 0 && width.is_multiple_of(block_elements),
        "MTP output row is not quant-block aligned"
    );
    let row_bytes = (width / block_elements)
        .checked_mul(block_bytes)
        .ok_or_else(|| anyhow!("MTP quant row byte count overflow"))?;
    let source_bytes = row_bytes
        .checked_mul(vocab)
        .ok_or_else(|| anyhow!("MTP output tensor byte count overflow"))?;
    ensure!(
        source.len() == source_bytes,
        "MTP output tensor byte count does not match its shape"
    );
    ensure!(
        !ids.is_empty()
            && ids.iter().all(|&id| id >= 0 && (id as usize) < vocab)
            && ids.windows(2).all(|pair| pair[0] < pair[1]),
        "MTP compact rows require unique, ascending, in-range IDs"
    );
    let bytes = row_bytes
        .checked_mul(ids.len())
        .ok_or_else(|| anyhow!("MTP compact output byte count overflow"))?;
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(bytes)
        .context("allocate MTP compact output host rows")?;
    for &id in ids {
        let offset = id as usize * row_bytes;
        packed.extend_from_slice(&source[offset..offset + row_bytes]);
    }
    Ok((packed, row_bytes))
}

impl DraftVocab {
    #[cfg(test)]
    pub(super) fn mock(be: &dyn Backend, width: usize, ids: &[i32]) -> Result<Self> {
        let id_map = be
            .alloc(ids.len() * 4, BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(id_map.as_ref(), bytemuck::cast_slice(ids))
            .map_err(|e| anyhow!("{e}"))?;
        let numel = width * ids.len();
        Ok(Self {
            weights: be
                .alloc(numel * 4, BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?,
            id_map,
            dtype: DType::F32,
            numel,
            ids: ids.to_vec(),
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.ids.len()
    }

    pub(super) fn original_id(&self, validated_index: u32) -> u32 {
        self.ids[validated_index as usize] as u32
    }

    pub(super) fn load_vulkan(
        vk: &VulkanBackend,
        target: &Gguf,
        cfg: &crate::Config,
        path: &Path,
    ) -> Result<Self> {
        let max_bytes = cfg
            .vocab
            .checked_mul(4)
            .ok_or_else(|| anyhow!("MTP vocabulary size overflow"))?;
        let metadata = std::fs::metadata(path)
            .with_context(|| format!("stat MTP vocabulary {}", path.display()))?;
        ensure!(
            metadata.is_file() && metadata.len() <= max_bytes as u64,
            "MTP vocabulary file is too large or not a file"
        );
        let mut raw = Vec::new();
        std::fs::File::open(path)
            .with_context(|| format!("open MTP vocabulary {}", path.display()))?
            .take(max_bytes as u64 + 1)
            .read_to_end(&mut raw)
            .with_context(|| format!("read MTP vocabulary {}", path.display()))?;
        let ids = parse_ids(&raw, cfg.vocab)?;
        let name = if target.tensors().iter().any(|t| t.name == "output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let info = target
            .tensors()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("target GGUF has no shared output tensor"))?;
        ensure!(
            info.shape == [cfg.n_embd, cfg.vocab],
            "target shared output shape is incompatible with the MTP head"
        );
        let source = target.tensor_bytes_arc(name).map_err(|e| anyhow!("{e}"))?;
        let (packed, row_bytes) =
            pack_quant_rows(source.as_ref(), info.dtype, cfg.n_embd, cfg.vocab, &ids)?;
        let padded = infr_vulkan::linear::pad_to_u32_align(&packed);
        let weights = vk
            .alloc(padded.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        vk.upload(weights.as_ref(), &padded)
            .map_err(|e| anyhow!("{e}"))?;
        let map_bytes = ids
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>();
        let id_map = vk
            .alloc(map_bytes.len(), BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        vk.upload(id_map.as_ref(), &map_bytes)
            .map_err(|e| anyhow!("{e}"))?;
        let numel = cfg
            .n_embd
            .checked_mul(ids.len())
            .ok_or_else(|| anyhow!("MTP compact output element count overflow"))?;
        tracing::info!(
            path = %path.display(),
            asset_sha256 = %format!("{:x}", Sha256::digest(&raw)),
            source_tensor = name,
            dtype = ?info.dtype,
            full_vocab = cfg.vocab,
            draft_vocab = ids.len(),
            id_order = "ascending original token ID",
            row_bytes,
            extra_fixed_weight_bytes = padded.len(),
            extra_fixed_id_bytes = map_bytes.len(),
            extra_fixed_vram_bytes = weights.len_bytes() + id_map.len_bytes(),
            "loaded opt-in contiguous MTP draft vocabulary; target vocabulary unchanged"
        );
        Ok(Self {
            weights,
            id_map,
            dtype: info.dtype,
            numel,
            ids,
        })
    }
}

#[cfg(test)]
#[path = "qwen4_draft_vocab_tests.rs"]
mod vocab_tests;
