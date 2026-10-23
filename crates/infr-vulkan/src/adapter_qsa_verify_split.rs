use super::{Graph, SequenceSpan};

pub(super) fn causal_row_views(
    graph: &Graph,
    enabled: bool,
    rows: u32,
    kv_len: u32,
    top_blocks: u32,
) -> Option<Vec<SequenceSpan>> {
    if !enabled || !graph.mtp_verify || graph.independent_rows || !(2..=8).contains(&rows) {
        return None;
    }
    // Recorder::vkb_byte_off requires 256-byte-aligned descriptor offsets. Do not round the
    // packed index stride: unaligned rows retain the original batched attention kernel.
    if top_blocks == 0 || !top_blocks.is_multiple_of(64) {
        return None;
    }
    let start = kv_len.checked_sub(rows)?;
    Some(
        (0..rows)
            .map(|row| SequenceSpan {
                row_start: row,
                rows: 1,
                start_pos: start + row,
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{qsa_lane_top_blocks, split_k_chunk_count_cap, ATTN_SPLIT};

    #[test]
    fn verify_gate_preserves_prefill_independent_rows_and_explicit_baseline() {
        let mut graph = Graph::new();
        assert!(causal_row_views(&graph, true, 4, 30_000, 512).is_none());
        graph.mtp_verify = true;
        assert!(causal_row_views(&graph, false, 4, 30_000, 512).is_none());
        for rows in [0, 1, 9, 16, 256] {
            assert!(causal_row_views(&graph, true, rows, 30_000, 512).is_none());
        }
        assert!(causal_row_views(&graph, true, 4, 3, 512).is_none());
        graph.independent_rows = true;
        assert!(causal_row_views(&graph, true, 4, 30_000, 512).is_none());
        graph.independent_rows = false;
        for rows in [2, 3, 4, 8] {
            let spans = causal_row_views(&graph, true, rows, u32::MAX, 512).unwrap();
            assert_eq!(spans.len(), rows as usize);
            for (row, span) in spans.iter().enumerate() {
                assert_eq!(span.row_start, row as u32);
                assert_eq!(span.rows, 1);
                assert_eq!(span.start_pos, u32::MAX - rows + row as u32);
            }
        }
        assert!(!graph.independent_rows);
        assert!(graph.sequence_spans.is_empty());
    }

    fn row_ids(visible: u32, ratio: u32, top: u32) -> Vec<u32> {
        let complete = visible / ratio;
        // Use different sparse subsets as the visible prefix grows, retaining chronological order.
        (complete.saturating_sub(top)..complete).collect()
    }

    #[test]
    fn index_descriptor_stride_requires_256_byte_alignment() {
        let mut graph = Graph::new();
        graph.mtp_verify = true;
        for rows in 2..=8 {
            for top in [0, 1, 63, 65, 511] {
                assert!(causal_row_views(&graph, true, rows, 32_768, top).is_none());
            }
            for top in [64, 128, 512] {
                assert!(causal_row_views(&graph, true, rows, 32_768, top).is_some());
            }
        }
    }

    fn scalar_positions(visible: u32, ratio: u32, top: u32) -> Vec<u32> {
        let mut positions = row_ids(visible, ratio, top)
            .into_iter()
            .flat_map(|block| block * ratio..(block + 1) * ratio)
            .collect::<Vec<_>>();
        positions.extend(visible / ratio * ratio..visible);
        positions
    }

    #[test]
    fn row_views_preserve_ratio_threshold_tail_and_32k_causality() {
        let mut graph = Graph::new();
        graph.mtp_verify = true;
        for rows in [2, 3, 4, 8] {
            for ratio in [1, 4, 8, 64] {
                let top = 512;
                for kv_len in [
                    rows,
                    top * ratio - 1,
                    top * ratio,
                    top * ratio + rows,
                    top * ratio + ratio - 1,
                    32_767,
                    32_768,
                    32_768 + rows / 2,
                    32_768 + rows,
                ] {
                    if kv_len < rows {
                        continue;
                    }
                    let spans = causal_row_views(&graph, true, rows, kv_len, top).unwrap();
                    for span in spans {
                        let visible = kv_len - rows + span.row_start + 1;
                        let complete = (span.start_pos + span.rows) / ratio;
                        let tail = (span.start_pos + span.rows) % ratio;
                        let selected = qsa_lane_top_blocks(&span, ratio, top);
                        let ids = row_ids(visible, ratio, top);
                        let keys = selected * ratio + tail;
                        let gathered = (0..keys)
                            .map(|logical| {
                                if logical < selected * ratio {
                                    ids[(logical / ratio) as usize] * ratio + logical % ratio
                                } else {
                                    complete * ratio + logical - selected * ratio
                                }
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(gathered, scalar_positions(visible, ratio, top));
                        assert!(gathered.iter().all(|&position| position < visible));
                        assert_eq!(gathered.last(), Some(&(visible - 1)));
                        assert_eq!(selected, top.min(complete));
                    }
                }
            }
        }
    }

    mod gpu {
        use super::*;
        use crate::adapter::*;
        use infr_core::backend::SegmentedKvSpec;
        use infr_core::{DType, TensorDesc};

        const HD: usize = 256;
        const NH: usize = 48;
        const NKV: usize = 4;
        const ROW_ELEMS: usize = NKV * HD;
        const SEGMENT_ROWS: usize = 32 * 1024;

        fn upload_cache(
            be: &VulkanBackend,
            kv_len: usize,
            q8: bool,
            segmented: bool,
            seed: usize,
        ) -> Box<dyn Buffer> {
            let elements = if segmented {
                SEGMENT_ROWS * ROW_ELEMS
            } else {
                kv_len * ROW_ELEMS
            };
            let bytes = if q8 { elements / 32 * 34 } else { elements * 2 };
            let segments = if segmented {
                kv_len.div_ceil(SEGMENT_ROWS)
            } else {
                1
            };
            let cache = if segmented {
                let cache = be
                    .alloc_segmented_kv(SegmentedKvSpec {
                        logical_bytes: segments * bytes,
                        segment_bytes: bytes,
                        segment_elements: elements,
                        max_segments: segments,
                    })
                    .unwrap()
                    .unwrap();
                be.ensure_segmented_kv(cache.as_ref(), segments).unwrap();
                cache
            } else {
                be.alloc(bytes, BufferUsage::KvCache).unwrap()
            };
            for segment in 0..segments {
                let mut data = vec![0u8; bytes];
                if q8 {
                    for (i, value) in data[..elements].iter_mut().enumerate() {
                        *value = (((i + segment * elements + seed) % 127) as i8 - 63) as u8;
                    }
                    for scale in data[elements..].chunks_exact_mut(2) {
                        scale.copy_from_slice(&half::f16::from_f32(1.0 / 128.0).to_le_bytes());
                    }
                } else {
                    for (i, value) in data.chunks_exact_mut(2).enumerate() {
                        let x = ((i + seed) % 127) as f32 / 128.0 - 0.5;
                        value.copy_from_slice(&half::f16::from_f32(x).to_le_bytes());
                    }
                }
                be.upload_range(cache.as_ref(), segment * bytes, &data)
                    .unwrap();
            }
            cache
        }

        fn run_case(
            be: &VulkanBackend,
            rows: usize,
            kv_len: usize,
            ratio: usize,
            configured_top: usize,
            q8: bool,
            segmented: bool,
        ) {
            let top = configured_top.min(kv_len / ratio);
            let cap = if segmented {
                kv_len.div_ceil(SEGMENT_ROWS) * SEGMENT_ROWS * ROW_ELEMS
            } else {
                kv_len * ROW_ELEMS
            };
            let kb = upload_cache(be, kv_len, q8, segmented, 11);
            let vb = upload_cache(be, kv_len, q8, segmented, 47);
            let q_bytes = (0..rows * NH * HD)
                .flat_map(|i| half::f16::from_f32((i % 113) as f32 / 128.0 - 0.4).to_le_bytes())
                .collect::<Vec<_>>();
            let mut ids = vec![u32::MAX; rows * top];
            for row in 0..rows {
                let visible = (kv_len - rows + row + 1) as u32;
                let selected = row_ids(visible, ratio as u32, top as u32);
                ids[row * top..row * top + selected.len()].copy_from_slice(&selected);
            }

            let mut graph = Graph::new();
            graph.mtp_verify = true;
            graph.no_decode_replay = true;
            let split = top.is_multiple_of(64);
            assert_eq!(
                causal_row_views(&graph, true, rows as u32, kv_len as u32, top as u32).is_some(),
                split,
            );
            let qi = graph.input(TensorDesc::new(vec![rows, NH, HD], DType::F16));
            let dtype = if q8 { DType::Q8_0 } else { DType::F16 };
            let ki = graph.input(TensorDesc::new(vec![cap], dtype));
            let vi = graph.input(TensorDesc::new(vec![cap], dtype));
            let ii = graph.input(TensorDesc::new(vec![rows, top], DType::I32));
            let oi = graph.output(TensorDesc::new(vec![rows, NH, HD], DType::F32));
            let scale = 1.0 / (HD as f32).sqrt();
            graph.push(Op::QsaBatchAttention {
                q: qi,
                k_cache: ki,
                v_cache: vi,
                indices: ii,
                dst: oi,
                rows: rows as u32,
                kv_len: kv_len as u32,
                n_head: NH as u32,
                n_kv: NKV as u32,
                head_dim: HD as u32,
                top_blocks: top as u32,
                ratio: ratio as u32,
                scale,
            });
            let qb = be.alloc(q_bytes.len(), BufferUsage::Activations).unwrap();
            let ib = be.alloc(ids.len() * 4, BufferUsage::Activations).unwrap();
            let ob = be.alloc(rows * NH * HD * 4, BufferUsage::Readback).unwrap();
            be.upload(qb.as_ref(), &q_bytes).unwrap();
            be.upload(ib.as_ref(), bytemuck::cast_slice(&ids)).unwrap();
            let plan = be.compile(&graph).unwrap();
            let mut bindings = Bindings::new();
            bindings.bind(qi, qb.as_ref());
            bindings.bind(ki, kb.as_ref());
            bindings.bind(vi, vb.as_ref());
            bindings.bind(ii, ib.as_ref());
            bindings.bind(oi, ob.as_ref());
            be.execute(plan.as_ref(), &bindings).unwrap();
            let mut got = vec![0u8; rows * NH * HD * 4];
            be.download(ob.as_ref(), &mut got).unwrap();

            let (k_binding, v_binding, shift) = if segmented {
                let k = crate::as_segmented_kv(kb.as_ref()).unwrap();
                let v = crate::as_segmented_kv(vb.as_ref()).unwrap();
                (
                    k.table_buffer() as &dyn Buffer,
                    v.table_buffer() as &dyn Buffer,
                    Some((SEGMENT_ROWS * ROW_ELEMS).trailing_zeros()),
                )
            } else {
                (kb.as_ref(), vb.as_ref(), None)
            };
            // Separate scalar submissions and independently sized scratch are the decode oracle.
            for row in 0..rows {
                let visible = kv_len - rows + row + 1;
                let complete = visible / ratio;
                let selected = top.min(complete);
                let tail = visible % ratio;
                let keys = selected * ratio + tail;
                let chunk = split_k_chunk_count_cap(
                    keys,
                    infr_core::tier::adaptive_chunk(keys, &ATTN_SPLIT),
                );
                let chunks = keys.div_ceil(chunk);
                let alloc = |bytes| be.alloc(bytes, BufferUsage::Activations).unwrap();
                let q = alloc(NH * HD * 2);
                let indices = alloc(top * 4);
                let kd = alloc(keys * ROW_ELEMS * 2);
                let vd = alloc(keys * ROW_ELEMS * 2);
                let out = alloc(NH * HD * 4);
                let pm = alloc(NH * chunks * 4);
                let pl = alloc(NH * chunks * 4);
                let pacc = alloc(NH * chunks * HD * 4);
                be.upload(
                    q.as_ref(),
                    &q_bytes[row * NH * HD * 2..(row + 1) * NH * HD * 2],
                )
                .unwrap();
                be.upload(
                    indices.as_ref(),
                    bytemuck::cast_slice(&ids[row * top..(row + 1) * top]),
                )
                .unwrap();
                let rec = be.recorder().unwrap();
                rec.qsa_gather(
                    k_binding,
                    v_binding,
                    indices.as_ref(),
                    kd.as_ref(),
                    vd.as_ref(),
                    selected as u32,
                    complete as u32,
                    tail as u32,
                    ratio as u32,
                    ROW_ELEMS as u32,
                    q8,
                    q8,
                    cap as u32,
                    cap as u32,
                    shift,
                );
                rec.attention_kv_split_off(
                    q.as_ref(),
                    0,
                    kd.as_ref(),
                    vd.as_ref(),
                    out.as_ref(),
                    0,
                    pm.as_ref(),
                    pl.as_ref(),
                    pacc.as_ref(),
                    1,
                    keys - 1,
                    keys,
                    NH,
                    NKV,
                    HD,
                    chunk,
                    chunks,
                    scale,
                    keys * ROW_ELEMS,
                );
                rec.finish().unwrap();
                let mut want = vec![0u8; NH * HD * 4];
                be.download(out.as_ref(), &mut want).unwrap();
                assert!(bytemuck::cast_slice::<u8, f32>(&want)
                    .iter()
                    .all(|v| v.is_finite()));
                let actual = &got[row * want.len()..(row + 1) * want.len()];
                if split {
                    assert_eq!(actual, &want,
                        "rows={rows} row={row} kv_len={kv_len} ratio={ratio} q8={q8} segmented={segmented}");
                } else {
                    for (&actual, &expected) in bytemuck::cast_slice::<u8, f32>(actual)
                        .iter()
                        .zip(bytemuck::cast_slice::<u8, f32>(&want))
                    {
                        assert!(
                            (actual - expected).abs() <= 3e-3 + 3e-3 * expected.abs(),
                            "unaligned fallback top={top} row={row}: got {actual}, want {expected}"
                        );
                    }
                }
            }
        }

        #[test]
        #[ignore = "requires an explicitly selected idle Vulkan GPU"]
        fn verify_split_matches_sequential_scalar_at_ratio_and_32k_boundaries() {
            let mut cfg = infr_core::config::Config::default();
            cfg.device.dev =
                Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select the test GPU explicitly"));
            cfg.kernels.vulkan.qsa_verify_split = true;
            let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
            let _pool = be.init_unified_vram(256 * 1024 * 1024).unwrap();
            for rows in [2, 3, 4, 8] {
                for (ratio, kv_len, top, q8, segmented) in [
                    (4, rows + 2, 512, false, false),
                    (4, rows + 2, 1, true, true),
                    (4, 260 + rows, 63, true, true),
                    (4, 2048 + rows / 2, 512, true, false),
                    (8, 4096 + rows / 2, 512, true, true),
                    (4, SEGMENT_ROWS + rows / 2, 512, true, true),
                ] {
                    run_case(&be, rows, kv_len, ratio, top, q8, segmented);
                }
            }
        }
    }
}
