//! Pure geometry for lazily committed KV storage.
//!
//! This module deliberately owns no buffers and changes no execution path. It is the arithmetic
//! contract shared by the later allocator and Vulkan work: a conversation grows in 32K-token
//! steps, while recurrent layers remain fixed-size state and attention/QSA planes commit only the
//! segments their materialized token depth reaches.

use crate::Config;
use infr_core::backend::SegmentedKvSpec;
use infr_core::tensor::DType;

use super::{kv_row_elems, kv_side_bytes};

/// Public runtime policy: persistent per-token state grows in 32K-token increments.
pub(crate) const KV_GROW_ROWS: usize = 32 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlaneKind {
    K,
    V,
    QsaRaw,
    QsaBlock,
}

/// One independently addressed cache plane in one model layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneLayout {
    pub(crate) layer: usize,
    pub(crate) kind: PlaneKind,
    pub(crate) dtype: DType,
    pub(crate) row_elems: usize,
    /// Number of model tokens represented by one row. Ordinary K/V and QSA-raw rows use one;
    /// QSA block keys use the layer's compression ratio.
    pub(crate) tokens_per_row: usize,
}

impl PlaneLayout {
    pub(crate) fn rows_per_segment(self) -> usize {
        KV_GROW_ROWS / self.tokens_per_row
    }

    pub(crate) fn segment_bytes(self) -> usize {
        kv_side_bytes(self.dtype, self.rows_per_segment() * self.row_elems)
    }

    pub(crate) fn segment_elements(self) -> usize {
        self.rows_per_segment() * self.row_elems
    }

    pub(crate) fn logical_elements(self, max_ctx: usize) -> usize {
        let rows = (max_ctx / self.tokens_per_row).max(1);
        rows * self.row_elems
    }

    pub(crate) fn spec(self, max_ctx: usize) -> SegmentedKvSpec {
        let segment_elements = self.segment_elements();
        debug_assert!(segment_elements.is_power_of_two());
        SegmentedKvSpec {
            logical_bytes: kv_side_bytes(self.dtype, self.logical_elements(max_ctx)),
            segment_bytes: self.segment_bytes(),
            segment_elements,
            max_segments: max_ctx.div_ceil(KV_GROW_ROWS),
        }
    }
}

/// Model-level description of the per-token planes that may grow lazily. Fixed recurrent state is
/// intentionally absent: it remains allocated once per conversation and does not scale with depth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SegmentedKvLayout {
    pub(crate) max_ctx: usize,
    pub(crate) planes: Vec<PlaneLayout>,
}

impl SegmentedKvLayout {
    /// Build the layout used by the first implementation target: Qwen3.5/3.6 hybrid attention and
    /// Qwen3.8 full-attention/QSA layers. Other architectures keep their existing static cache.
    pub(crate) fn for_qwen(
        cfg: &Config,
        max_ctx: usize,
        k_fmt: DType,
        v_fmt: DType,
    ) -> Option<Self> {
        if !(cfg.qwen35 || cfg.qwen4exp) {
            return None;
        }
        let mut planes = Vec::new();
        for layer in 0..cfg.n_layer {
            if cfg.is_recurrent_layer(layer) {
                continue;
            }
            let (k_row, v_row) = kv_row_elems(cfg, layer);
            if k_row != 0 {
                planes.push(PlaneLayout {
                    layer,
                    kind: PlaneKind::K,
                    dtype: k_fmt,
                    row_elems: k_row,
                    tokens_per_row: 1,
                });
            }
            if v_row != 0 {
                planes.push(PlaneLayout {
                    layer,
                    kind: PlaneKind::V,
                    dtype: v_fmt,
                    row_elems: v_row,
                    tokens_per_row: 1,
                });
            }
            if cfg.qwen4exp && cfg.is_qwen_hybrid_attn_layer(layer) {
                planes.push(PlaneLayout {
                    layer,
                    kind: PlaneKind::QsaRaw,
                    dtype: DType::F16,
                    row_elems: cfg.indexer_head_size,
                    tokens_per_row: 1,
                });
                let ratio = cfg.layer_compress_ratio(layer).max(1);
                debug_assert!(KV_GROW_ROWS.is_multiple_of(ratio));
                planes.push(PlaneLayout {
                    layer,
                    kind: PlaneKind::QsaBlock,
                    dtype: DType::F32,
                    row_elems: cfg.indexer_head_size,
                    tokens_per_row: ratio,
                });
            }
        }
        Some(Self { max_ctx, planes })
    }

    #[cfg(test)]
    pub(crate) fn max_segments(&self) -> usize {
        self.max_ctx.div_ceil(KV_GROW_ROWS)
    }

    pub(crate) fn segments_for_tokens(&self, tokens: usize) -> usize {
        tokens.min(self.max_ctx).div_ceil(KV_GROW_ROWS)
    }

    pub(crate) fn committed_bytes(&self, tokens: usize) -> u64 {
        let segments = self.segments_for_tokens(tokens) as u64;
        self.planes
            .iter()
            .map(|plane| plane.segment_bytes() as u64 * segments)
            .sum()
    }

    pub(crate) fn plane(&self, layer: usize, kind: PlaneKind) -> Option<PlaneLayout> {
        self.planes
            .iter()
            .copied()
            .find(|plane| plane.layer == layer && plane.kind == kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen38() -> Config {
        Config {
            qwen4exp: true,
            n_layer: 48,
            recurrent_layers: (0usize..48)
                .map(|layer| !(layer + 1).is_multiple_of(4))
                .collect(),
            compress_ratios: (0usize..48)
                .map(|layer| if (layer + 1).is_multiple_of(4) { 4 } else { 0 })
                .collect(),
            n_kv: 2,
            head_dim: 256,
            indexer_head_size: 128,
            ..Default::default()
        }
    }

    #[test]
    fn depth_crosses_exactly_one_32k_boundary() {
        let layout =
            SegmentedKvLayout::for_qwen(&qwen38(), 262_144, DType::Q8_0, DType::Q8_0).unwrap();
        assert_eq!(layout.max_segments(), 8);
        assert_eq!(layout.segments_for_tokens(0), 0);
        assert_eq!(layout.segments_for_tokens(KV_GROW_ROWS - 1), 1);
        assert_eq!(layout.segments_for_tokens(KV_GROW_ROWS), 1);
        assert_eq!(layout.segments_for_tokens(KV_GROW_ROWS + 1), 2);
        assert_eq!(
            layout.committed_bytes(KV_GROW_ROWS + 1),
            2 * layout.committed_bytes(1)
        );
    }

    #[test]
    fn qwen38_layout_contains_only_twelve_attention_layers() {
        let layout =
            SegmentedKvLayout::for_qwen(&qwen38(), 262_144, DType::Q8_0, DType::Q8_0).unwrap();
        assert_eq!(layout.planes.len(), 12 * 4);
        assert_eq!(
            layout
                .planes
                .iter()
                .filter(|plane| plane.kind == PlaneKind::QsaBlock)
                .count(),
            12
        );
        assert!(layout
            .planes
            .iter()
            .filter(|plane| plane.kind == PlaneKind::QsaBlock)
            .all(|plane| plane.rows_per_segment() == KV_GROW_ROWS / 4));
    }

    #[test]
    fn q8_segment_bytes_use_a_local_quant_scale_plane() {
        let cfg = qwen38();
        let layout = SegmentedKvLayout::for_qwen(&cfg, 262_144, DType::Q8_0, DType::Q8_0).unwrap();
        let k = layout
            .planes
            .iter()
            .find(|plane| plane.kind == PlaneKind::K)
            .copied()
            .unwrap();
        assert_eq!(k.row_elems, cfg.n_kv * cfg.head_dim);
        assert_eq!(
            k.segment_bytes(),
            infr_core::budget::kv_fmt_bytes(DType::Q8_0, KV_GROW_ROWS * k.row_elems)
        );
    }

    #[test]
    fn fully_committed_qwen38_planes_equal_the_existing_static_geometry() {
        let cfg = qwen38();
        let ctx = 262_144;
        let layout = SegmentedKvLayout::for_qwen(&cfg, ctx, DType::Q8_0, DType::Q8_0).unwrap();
        let static_bytes: u64 = (0..cfg.n_layer)
            .filter(|&layer| !cfg.is_recurrent_layer(layer))
            .map(|layer| {
                let (k, v) = super::super::layer_state_bytes(
                    &cfg,
                    layer,
                    ctx,
                    false,
                    1024,
                    DType::Q8_0,
                    DType::Q8_0,
                );
                (k + v + super::super::qsa_cache_bytes(&cfg, layer, ctx)) as u64
            })
            .sum();
        assert_eq!(layout.committed_bytes(ctx), static_bytes);
    }
}
