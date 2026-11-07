#[cfg(test)]
mod tests {
    use super::super::super::{
        build_draft_graph_impl, build_draft_graph_vocab_impl, Qwen4MtpSession, SharedWeights,
    };
    use super::super::{pack_quant_rows, parse_ids, validate_raw_ids, DraftVocab};
    use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
    use infr_core::graph::{Graph, MoeGating, Op, TensorKind};
    use infr_core::{DType, TensorDesc, WeightSource};
    use infr_gguf::Gguf;
    use infr_vulkan::VulkanBackend;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    fn encoded(ids: &[i32]) -> Vec<u8> {
        ids.iter().flat_map(|id| id.to_le_bytes()).collect()
    }

    #[test]
    fn external_ids_are_checked_and_sorted_without_silent_deduplication() {
        assert_eq!(parse_ids(&encoded(&[9, 0, 3]), 10).unwrap(), [0, 3, 9]);
        for bytes in [
            vec![],
            vec![0],
            encoded(&[-1]),
            encoded(&[10]),
            encoded(&[3, 3]),
        ] {
            assert!(parse_ids(&bytes, 10).is_err());
        }
        assert!(parse_ids(&encoded(&[0, 1]), 1).is_err());
        assert!(parse_ids(&encoded(&[0]), 0).is_err());
        assert!(parse_ids(&encoded(&[0]), i32::MAX as usize + 1).is_err());
    }

    #[test]
    fn raw_argmax_validation_rejects_every_bad_row_without_substitution() {
        for rows in [1, 7, 128, 40_525, 106_299, 248_320] {
            for width in 2..=4 {
                let valid = vec![rows as u32 - 1; width - 1];
                validate_raw_ids(&valid, rows).unwrap();
                for step in 0..width - 1 {
                    for bad in [rows as u32, rows as u32 + 1, u32::MAX] {
                        let mut raw = valid.clone();
                        raw[step] = bad;
                        let before = raw.clone();
                        assert!(validate_raw_ids(&raw, rows).is_err());
                        assert_eq!(raw, before);
                    }
                }
            }
        }
        assert!(validate_raw_ids(&[0], 0).is_err());
    }

    #[test]
    fn compact_rows_preserve_original_quant_bytes_and_reject_invalid_geometry() {
        for dtype in [DType::Q8_0, DType::Q4K, DType::Q6K] {
            let width = 512;
            let row_bytes = infr_gguf::nbytes(dtype, width);
            let source = (0..12 * row_bytes)
                .map(|i| (i * 17 + i / row_bytes) as u8)
                .collect::<Vec<_>>();
            let (packed, stride) = pack_quant_rows(&source, dtype, width, 12, &[0, 4, 11]).unwrap();
            assert_eq!(stride, row_bytes);
            assert_eq!(packed.len(), 3 * row_bytes);
            for (row, id) in [0, 4, 11].into_iter().enumerate() {
                assert_eq!(
                    &packed[row * row_bytes..(row + 1) * row_bytes],
                    &source[id * row_bytes..(id + 1) * row_bytes]
                );
            }
            assert!(pack_quant_rows(&source[..source.len() - 1], dtype, width, 12, &[0]).is_err());
            assert!(pack_quant_rows(&source, dtype, width - 1, 12, &[0]).is_err());
            for ids in [vec![], vec![-1], vec![12], vec![1, 1], vec![4, 0]] {
                assert!(pack_quant_rows(&source, dtype, width, 12, &ids).is_err());
            }
        }
        assert!(pack_quant_rows(&[0; 64], DType::F16, 32, 1, &[0]).is_err());
        assert!(pack_quant_rows(&[], DType::Q8_0, 0, 1, &[0]).is_err());
        assert!(pack_quant_rows(&[], DType::Q8_0, 32, usize::MAX, &[0]).is_err());
    }

    fn shape_config() -> (crate::Config, Vec<(DType, usize)>) {
        let cfg = crate::Config {
            n_embd: 64,
            n_head: 4,
            n_kv: 2,
            head_dim: 16,
            rope_dim: 16,
            rope_theta: 1_000_000.0,
            rope_sections: [2, 2, 4, 0],
            rms_eps: 1e-6,
            vocab: 128,
            hc_mult: 4,
            hc_low_rank: 32,
            shexp_ff: 64,
            qwen4exp: true,
            moe: Some(crate::MoeConfig {
                n_expert: 4,
                n_used: 2,
                n_ff_exp: 64,
                scale: 1.0,
                gating: MoeGating::Softmax,
                norm_w: true,
                weight_before: false,
                n_expert_groups: 0,
                n_expert_groups_used: 0,
            }),
            ..Default::default()
        };
        let (ne, hc, lr) = (64, 4, 32);
        let hcw = ne * hc;
        let qrow = 64;
        let kvrow = 32;
        let exps = ne * 64 * 4;
        let shexp = ne * 64;
        let specs = [
            hcw,
            hcw * lr,
            lr * hcw,
            hcw * hc,
            hcw,
            hcw * lr,
            lr * hcw,
            hcw * hc,
            ne * 2 * qrow,
            ne * kvrow,
            ne * kvrow,
            16,
            16,
            qrow * ne,
            ne * 4,
            exps,
            exps,
            exps,
            ne,
            shexp,
            shexp,
            shexp,
            2 * ne * ne,
            ne,
            hcw,
            hcw,
            hcw * lr,
            lr * hcw,
        ]
        .into_iter()
        .map(|n| (DType::F32, n))
        .collect();
        (cfg, specs)
    }

    #[test]
    fn graph_remaps_each_compact_argmax_before_the_next_bridge() {
        let (cfg, specs) = shape_config();
        let full = (DType::F32, cfg.n_embd * cfg.vocab);
        let compact = (DType::Q8_0, cfg.n_embd * 7);
        for steps in [2, 3, 4] {
            for start in [30_000, 32_767, 32_769] {
                for tail in [false, true] {
                    let build = if tail {
                        build_draft_graph_vocab_impl::<true>
                    } else {
                        build_draft_graph_vocab_impl::<false>
                    };
                    let (g, h) = build(
                        &cfg,
                        &specs,
                        [full, compact],
                        32_768,
                        32_768,
                        start,
                        steps,
                        false,
                        Some(7),
                    );
                    assert!(g.no_decode_replay);
                    let map = h.vocab_map.unwrap();
                    assert_eq!(g.kind(map), TensorKind::Weight);
                    assert_eq!(g.desc(map), &TensorDesc::new(vec![7], DType::I32));
                    assert_eq!(g.desc(h.embd).numel(), cfg.n_embd * cfg.vocab);
                    assert_eq!(g.desc(h.lm_head).dtype, DType::Q8_0);
                    assert_eq!(g.desc(h.lm_head).numel(), cfg.n_embd * 7);
                    let bridges = g
                        .ops
                        .iter()
                        .filter_map(|op| match op {
                            Op::EmbedGather { ids, table, .. } if *table == h.embd => Some(*ids),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    let expected = std::iter::once(h.id)
                        .chain(h.ids.iter().copied())
                        .collect::<Vec<_>>();
                    assert_eq!(bridges, expected);
                    let mut remaps = 0;
                    for (index, op) in g.ops.iter().enumerate() {
                        if let Op::GatherI32 {
                            ids,
                            table,
                            dst,
                            rows,
                            ne,
                        } = op
                        {
                            assert_eq!((*table, *rows, *ne), (map, 1, 1));
                            assert_eq!(*dst, h.ids[remaps]);
                            assert!(
                                matches!(&g.ops[index - 1], Op::Argmax { dst, n: 7, rows: 1, .. } if dst == ids)
                            );
                            assert_eq!(g.kind(*ids), TensorKind::Output);
                            assert_eq!(*ids, h.raw_ids[remaps]);
                            assert_eq!(g.kind(*dst), TensorKind::Internal);
                            assert_eq!(
                                g.ops.iter().filter(|op| op.io().0.contains(ids)).count(),
                                1
                            );
                            remaps += 1;
                        }
                        if let Op::Linear {
                            weight, out_f, dst, ..
                        } = op
                        {
                            if *weight == h.lm_head {
                                assert_eq!(*out_f, 7);
                                assert_eq!(g.desc(*dst).numel(), 7);
                            }
                        }
                    }
                    assert_eq!(remaps, steps - 1);
                    let baseline = if tail {
                        build_draft_graph_impl::<true>
                    } else {
                        build_draft_graph_impl::<false>
                    };
                    let (g, h) =
                        baseline(&cfg, &specs, [full; 2], 32_768, 32_768, start, steps, false);
                    assert!(h.vocab_map.is_none());
                    assert!(!g.ops.iter().any(|op| matches!(op, Op::GatherI32 { .. })));
                    assert_eq!(g.desc(h.lm_head).numel(), cfg.n_embd * cfg.vocab);
                }
            }
        }
    }

    type Uploaded = (Box<dyn Buffer>, DType, usize);

    #[test]
    fn full_vocab_safety_graph_keeps_full_projection_and_only_chains_mapped_ids() {
        let (cfg, specs) = shape_config();
        let full = (DType::Q8_0, cfg.n_embd * cfg.vocab);
        for width in 2..=4 {
            for tail in [false, true] {
                let build = if tail {
                    build_draft_graph_vocab_impl::<true>
                } else {
                    build_draft_graph_vocab_impl::<false>
                };
                let (g, h) = build(
                    &cfg,
                    &specs,
                    [full; 2],
                    32_768,
                    32_768,
                    32_767,
                    width,
                    false,
                    Some(cfg.vocab),
                );
                let map = h.vocab_map.unwrap();
                assert_eq!(g.desc(map).numel(), cfg.vocab);
                assert_eq!(g.desc(h.lm_head).numel(), full.1);
                assert_eq!(g.outputs, h.raw_ids);
                for (raw, mapped) in h.raw_ids.iter().zip(&h.ids) {
                    assert_eq!(g.kind(*raw), TensorKind::Output);
                    assert_eq!(g.kind(*mapped), TensorKind::Internal);
                    assert!(g.ops.iter().any(|op| matches!(op, Op::GatherI32 { ids, table, dst, .. } if ids == raw && *table == map && dst == mapped)));
                    assert_eq!(g.ops.iter().filter(|op| op.io().0.contains(raw)).count(), 1);
                    assert!(g
                        .ops
                        .iter()
                        .any(|op| matches!(op, Op::EmbedGather { ids, .. } if ids == mapped)));
                }
                assert!(g
                    .ops
                    .iter()
                    .filter_map(|op| match op {
                        Op::Argmax { n, .. } => Some(*n),
                        _ => None,
                    })
                    .all(|n| n as usize == cfg.vocab));
            }
        }
    }

    fn bounded_gather_padding_checks(be: &VulkanBackend) {
        for table_rows in [1, 7, 63] {
            for ne in [1, 3] {
                let ids = [
                    0,
                    table_rows - 1,
                    table_rows,
                    table_rows + 1,
                    255,
                    u32::MAX,
                    0x8000_0000,
                ];
                let mut table = vec![777i32; 256];
                for (i, value) in table[..table_rows as usize * ne].iter_mut().enumerate() {
                    *value = i as i32 + 5;
                }
                let mut g = Graph::new();
                g.no_decode_replay = true;
                let input = g.input(TensorDesc::new(vec![ids.len()], DType::I32));
                // The bound deliberately differs from the much larger allocated descriptor.
                let weight = g.weight(TensorDesc::new(vec![table_rows as usize * ne], DType::I32));
                let output = g.output(TensorDesc::new(vec![ids.len() * ne], DType::I32));
                g.push(Op::GatherI32 {
                    ids: input,
                    table: weight,
                    dst: output,
                    rows: ids.len() as u32,
                    ne: ne as u32,
                });
                let ib = be.alloc(ids.len() * 4, BufferUsage::Staging).unwrap();
                let tb = be.alloc(table.len() * 4, BufferUsage::Weights).unwrap();
                let ob = be.alloc(ids.len() * ne * 4, BufferUsage::Readback).unwrap();
                be.upload(ib.as_ref(), bytemuck::cast_slice(&ids)).unwrap();
                be.upload(tb.as_ref(), bytemuck::cast_slice(&table))
                    .unwrap();
                let mut bindings = Bindings::new();
                bindings.bind(input, ib.as_ref());
                bindings.bind(weight, tb.as_ref());
                bindings.bind(output, ob.as_ref());
                be.execute(be.compile(&g).unwrap().as_ref(), &bindings)
                    .unwrap();
                let mut actual = vec![0i32; ids.len() * ne];
                be.download(ob.as_ref(), bytemuck::cast_slice_mut(&mut actual))
                    .unwrap();
                for (row, &id) in ids.iter().enumerate() {
                    for j in 0..ne {
                        assert_eq!(
                            actual[row * ne + j],
                            if id < table_rows {
                                table[id as usize * ne + j]
                            } else {
                                0
                            }
                        );
                    }
                }
            }
        }
    }

    fn nonfinite_projection_checks(
        be: &VulkanBackend,
        cfg: &crate::Config,
        embedding: &Uploaded,
        vocab: Option<&DraftVocab>,
        identity_map: &dyn Buffer,
    ) {
        let rows = vocab.map_or(cfg.vocab, DraftVocab::rows);
        for width in 2..=4 {
            let predictions = width - 1;
            for bad_step in 0..predictions {
                for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut logits = vec![0.0f32; rows * predictions];
                    logits[bad_step * rows + rows - 1] = bad;
                    let mut g = Graph::new();
                    g.no_decode_replay = true;
                    let x = g.input(TensorDesc::new(vec![logits.len()], DType::F32));
                    let raw = g.output(TensorDesc::new(vec![predictions], DType::I32));
                    let mapped = g.output(TensorDesc::new(vec![predictions], DType::I32));
                    let table = g.weight(TensorDesc::new(vec![rows], DType::I32));
                    let emb = g.weight(TensorDesc::new(vec![embedding.2], embedding.1));
                    let dst = g.output(TensorDesc::new(vec![predictions * cfg.n_embd], DType::F32));
                    g.push(Op::Argmax {
                        x,
                        dst: raw,
                        n: rows as u32,
                        rows: predictions as u32,
                    });
                    g.push(Op::GatherI32 {
                        ids: raw,
                        table,
                        dst: mapped,
                        rows: predictions as u32,
                        ne: 1,
                    });
                    g.push(Op::EmbedGather {
                        ids: mapped,
                        table: emb,
                        dst,
                        rows: predictions as u32,
                        ne: cfg.n_embd as u32,
                        scale: 1.0,
                    });
                    let xb = be.alloc(logits.len() * 4, BufferUsage::Staging).unwrap();
                    let rb = be.alloc(predictions * 4, BufferUsage::Readback).unwrap();
                    let mb = be.alloc(predictions * 4, BufferUsage::Readback).unwrap();
                    let eb = be
                        .alloc(predictions * cfg.n_embd * 4, BufferUsage::Readback)
                        .unwrap();
                    be.upload(xb.as_ref(), bytemuck::cast_slice(&logits))
                        .unwrap();
                    let mut bindings = Bindings::new();
                    bindings.bind(x, xb.as_ref());
                    bindings.bind(raw, rb.as_ref());
                    bindings.bind(mapped, mb.as_ref());
                    bindings.bind(table, vocab.map_or(identity_map, |v| v.id_map.as_ref()));
                    bindings.bind(emb, embedding.0.as_ref());
                    bindings.bind(dst, eb.as_ref());
                    be.execute(be.compile(&g).unwrap().as_ref(), &bindings)
                        .unwrap();
                    let mut raw_ids = vec![0u32; predictions];
                    let mut mapped_ids = vec![0u32; predictions];
                    let mut embedded = vec![0.0f32; predictions * cfg.n_embd];
                    be.download(rb.as_ref(), bytemuck::cast_slice_mut(&mut raw_ids))
                        .unwrap();
                    be.download(mb.as_ref(), bytemuck::cast_slice_mut(&mut mapped_ids))
                        .unwrap();
                    be.download(eb.as_ref(), bytemuck::cast_slice_mut(&mut embedded))
                        .unwrap();
                    assert_eq!(raw_ids[bad_step], u32::MAX);
                    assert_eq!(mapped_ids[bad_step], 0);
                    assert!(embedded.iter().all(|v| v.is_finite()));
                    assert!(validate_raw_ids(&raw_ids, rows).is_err());
                    for step in 0..predictions {
                        if step != bad_step {
                            assert_eq!(raw_ids[step], 0);
                            assert_eq!(mapped_ids[step], vocab.map_or(0, |v| v.ids[0] as u32));
                        }
                    }
                }
            }
        }
    }

    fn upload_shared(be: &VulkanBackend, target: &Gguf, name: &str) -> Uploaded {
        let info = target.tensors().iter().find(|t| t.name == name).unwrap();
        assert!(info.dtype.is_quant());
        let bytes = target.tensor_bytes_arc(name).unwrap();
        let padded = infr_vulkan::linear::pad_to_u32_align(bytes.as_ref());
        let buffer = be.alloc(padded.len(), BufferUsage::Weights).unwrap();
        be.upload(buffer.as_ref(), &padded).unwrap();
        (buffer, info.dtype, info.shape.iter().product())
    }

    fn projection(
        be: &VulkanBackend,
        cfg: &crate::Config,
        full: &Uploaded,
        vocab: Option<&DraftVocab>,
        input: &[f32],
    ) -> (Vec<f32>, u32) {
        let rows = vocab.map_or(cfg.vocab, DraftVocab::rows);
        let dtype = vocab.map_or(full.1, |v| v.dtype);
        let numel = vocab.map_or(full.2, |v| v.numel);
        let mut g = Graph::new();
        g.no_decode_replay = true;
        let x = g.input(TensorDesc::new(vec![cfg.n_embd], DType::F32));
        let w = g.weight(TensorDesc::new(vec![numel], dtype));
        let logits = g.output(TensorDesc::new(vec![rows], DType::F32));
        let id = g.output(TensorDesc::new(vec![1], DType::F32));
        let raw_id = if vocab.is_some() {
            g.internal(TensorDesc::new(vec![1], DType::F32))
        } else {
            id
        };
        g.push(Op::Linear {
            x,
            weight: w,
            dst: logits,
            m: 1,
            in_f: cfg.n_embd as u32,
            out_f: rows as u32,
            w_off: 0,
        });
        g.push(Op::Argmax {
            x: logits,
            dst: raw_id,
            n: rows as u32,
            rows: 1,
        });
        let map = vocab.map(|_| g.weight(TensorDesc::new(vec![rows], DType::I32)));
        if let Some(table) = map {
            g.push(Op::GatherI32 {
                ids: raw_id,
                table,
                dst: id,
                rows: 1,
                ne: 1,
            });
        }
        let xb = be.alloc(input.len() * 4, BufferUsage::Staging).unwrap();
        let yb = be.alloc(rows * 4, BufferUsage::Readback).unwrap();
        let ib = be.alloc(4, BufferUsage::Readback).unwrap();
        be.upload(xb.as_ref(), bytemuck::cast_slice(input)).unwrap();
        let plan = be.compile(&g).unwrap();
        let mut bindings = Bindings::new();
        bindings.bind(x, xb.as_ref());
        bindings.bind(w, vocab.map_or(full.0.as_ref(), |v| v.weights.as_ref()));
        bindings.bind(logits, yb.as_ref());
        bindings.bind(id, ib.as_ref());
        if let (Some(map), Some(v)) = (map, vocab) {
            bindings.bind(map, v.id_map.as_ref());
        }
        be.execute(plan.as_ref(), &bindings).unwrap();
        let mut values = vec![0.0f32; rows];
        let mut token = 0u32;
        be.download(yb.as_ref(), bytemuck::cast_slice_mut(&mut values))
            .unwrap();
        be.download(ib.as_ref(), bytemuck::bytes_of_mut(&mut token))
            .unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        (values, token)
    }

    fn reset_history(be: &VulkanBackend, heads: &[Qwen4MtpSession; 2], zero: &dyn Buffer) {
        for head in heads {
            head.settle_draft(be, 0).unwrap();
        }
        let a = heads[0].kv_for_depth(be, 160_000).unwrap();
        let b = heads[1].kv_for_depth(be, 160_000).unwrap();
        assert!(a.segmented && b.segmented);
        let mut copies = Vec::new();
        for dst in [a.k.as_ref(), a.v.as_ref(), b.k.as_ref(), b.v.as_ref()] {
            for offset in (0..heads[0].kv_spec.logical_bytes).step_by(zero.len_bytes()) {
                copies.push((
                    zero,
                    0,
                    dst,
                    offset,
                    zero.len_bytes()
                        .min(heads[0].kv_spec.logical_bytes - offset),
                ));
            }
        }
        be.copy_buffer_ranges(&copies).unwrap();
    }

    fn compare_common_prefix(
        be: &VulkanBackend,
        heads: &[Qwen4MtpSession; 2],
        vocab: &DraftVocab,
        ids: &[Vec<u32>; 2],
        start: usize,
    ) {
        assert!(ids[1]
            .iter()
            .all(|&id| vocab.ids.binary_search(&(id as i32)).is_ok()));
        let mut common = 0;
        for (&full, &compact) in ids[0].iter().zip(&ids[1]) {
            if vocab.ids.binary_search(&(full as i32)).is_err() {
                break;
            }
            assert_eq!(
                full, compact,
                "full argmax in subset must match before divergence"
            );
            common += 1;
        }
        let a = heads[0].kv.lock().unwrap();
        let b = heads[1].kv.lock().unwrap();
        let row_bytes = heads[0].kv_spec.logical_bytes / 32_768;
        let mut left = vec![0u8; row_bytes];
        let mut right = vec![0u8; row_bytes];
        for row in 0..=common {
            let offset = (start + row) % 32_768 * row_bytes;
            for (a, b) in [(a.k.as_ref(), b.k.as_ref()), (a.v.as_ref(), b.v.as_ref())] {
                be.download_range(a, offset, &mut left).unwrap();
                be.download_range(b, offset, &mut right).unwrap();
                assert!(
                    left == right,
                    "common-prefix KV mismatch start={start} row={row}"
                );
            }
        }
        eprintln!(
            "compact_head_parity start={start} common_candidates={common} full={:?} compact={:?}",
            ids[0], ids[1]
        );
    }

    fn saved_live_rows(
        be: &VulkanBackend,
        head: &Qwen4MtpSession,
        start: usize,
        width: usize,
    ) -> Vec<(usize, [Vec<u8>; 2])> {
        let kv = head.kv_for_depth(be, start + width).unwrap();
        let capacity = head.kv_capacity;
        let row_bytes = head.kv_spec.logical_bytes / capacity;
        let first_saved = width.min(capacity.saturating_sub(start));
        (first_saved..width)
            .map(|step| {
                let offset = (start + step) % capacity * row_bytes;
                let mut bytes = [vec![0; row_bytes], vec![0; row_bytes]];
                be.download_range(kv.k.as_ref(), offset, &mut bytes[0])
                    .unwrap();
                be.download_range(kv.v.as_ref(), offset, &mut bytes[1])
                    .unwrap();
                (offset, bytes)
            })
            .collect()
    }

    fn assert_restored_rows(
        be: &VulkanBackend,
        head: &Qwen4MtpSession,
        saved: &[(usize, [Vec<u8>; 2])],
    ) {
        let kv = head.kv.lock().unwrap();
        assert!(kv.draft_journal.pending.is_none());
        for (offset, expected) in saved {
            for (cache, expected) in [(kv.k.as_ref(), &expected[0]), (kv.v.as_ref(), &expected[1])]
            {
                let mut bytes = vec![0; expected.len()];
                be.download_range(cache, *offset, &mut bytes).unwrap();
                assert_eq!(
                    &bytes, expected,
                    "aborted draft changed live KV at {offset}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires MOE4ALL_TEST_DEVICE, MOE4ALL_SPEC_TEST_MODEL, MOE4ALL_MTP_TEST_DRAFT, MOE4ALL_MTP_TEST_VOCAB"]
    fn actual_draft_nonfinite_safety_and_wrapped_journal() {
        let required =
            |key| PathBuf::from(std::env::var_os(key).unwrap_or_else(|| panic!("set {key}")));
        let target = Gguf::open(&required("MOE4ALL_SPEC_TEST_MODEL")).unwrap();
        let cfg = crate::Config::from_gguf(&target).unwrap();
        let mut engine = infr_core::config::Config::default();
        engine.device.dev =
            Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select the test GPU"));
        engine.spec.mtp = true;
        let be = VulkanBackend::new_with(Arc::new(engine)).unwrap();
        bounded_gather_padding_checks(&be);
        let embedding = upload_shared(&be, &target, "token_embd.weight");
        let output_name = if target.tensors().iter().any(|t| t.name == "output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output = upload_shared(&be, &target, output_name);
        let fixed = Qwen4MtpSession::load_fixed_vulkan_with_vocab(
            &be,
            &required("MOE4ALL_MTP_TEST_DRAFT"),
            &cfg,
            &target,
            Some(&required("MOE4ALL_MTP_TEST_VOCAB")),
        )
        .unwrap();
        let vocab = fixed.draft_vocab.as_ref().unwrap();
        let identity = fixed.draft_identity_map.as_deref().unwrap();
        for projection in [None, Some(vocab)] {
            nonfinite_projection_checks(&be, &cfg, &embedding, projection, identity);
        }
        // The pinned FlashNext fixture has Q8_0 shared output. Poison one native scale,
        // not hidden state: activation quantization must not mask the injected NaN/Inf.
        assert_eq!(output.1, DType::Q8_0);
        assert_eq!(vocab.dtype, DType::Q8_0);
        let spec = super::super::super::qwen4_mtp_kv_spec(&cfg, 32_768).unwrap();
        be.init_unified_vram(4 * spec.segment_bytes + 256 * 1024 * 1024)
            .unwrap();
        let catch = Qwen4MtpSession::shared_catch_workspace(&be, &fixed, 160_000, 4).unwrap();
        let heads = [
            Qwen4MtpSession::with_fixed_and_catch(
                &be,
                Arc::clone(&fixed),
                160_000,
                Arc::clone(&catch),
                32_768,
            )
            .unwrap(),
            Qwen4MtpSession::with_fixed_and_catch(&be, Arc::clone(&fixed), 160_000, catch, 32_768)
                .unwrap(),
        ];
        let shared = (
            (embedding.0.as_ref(), embedding.1, embedding.2),
            (output.0.as_ref(), output.1, output.2),
        );
        let hidden = (0..heads[0].h_width())
            .map(|i| ((i * 17 + 23) % 101) as f32 / 256.0 - 0.2)
            .collect::<Vec<_>>();
        let zero = be.alloc(4 * 1024 * 1024, BufferUsage::Staging).unwrap();
        be.upload(zero.as_ref(), &vec![0; zero.len_bytes()])
            .unwrap();
        reset_history(&be, &heads, zero.as_ref());
        let mut cases = 0;
        for slot in 0..2 {
            let head = &heads[slot];
            let projection = if slot == 0 { None } else { Some(vocab) };
            let weight = projection.map_or(output.0.as_ref(), |v| v.weights.as_ref());
            // Four-byte aligned transfer includes, but does not change, the next scale.
            let mut original = [0u8; 36];
            be.download_range(weight, 0, &mut original).unwrap();
            for scale in [0x7e00u16, 0x7c00, 0xfc00] {
                let mut poison = original;
                poison[2..34].fill(127);
                poison[..2].copy_from_slice(&scale.to_le_bytes());
                be.upload_range(weight, 0, &poison).unwrap();
                for width in 2..=4 {
                    for start in [32_767, 32_769] {
                        for tail in [false, true] {
                            let saved = saved_live_rows(&be, head, start, width);
                            let committed_before = head.kv.lock().unwrap().committed_tokens;
                            let error = head
                                .draft_with_projection(
                                    &be,
                                    1234,
                                    &hidden,
                                    start,
                                    start as i32,
                                    width,
                                    shared,
                                    tail,
                                    projection,
                                )
                                .unwrap_err();
                            assert!(error.to_string().contains("invalid index"), "{error:#}");
                            let mut raw = 0u32;
                            be.download(
                                head.draft_ids[0].as_ref(),
                                bytemuck::bytes_of_mut(&mut raw),
                            )
                            .unwrap();
                            assert_eq!(raw, u32::MAX);
                            assert_restored_rows(&be, head, &saved);
                            assert_eq!(head.kv.lock().unwrap().committed_tokens, committed_before);
                            head.settle_draft(&be, 0).unwrap();
                            cases += 1;
                        }
                    }
                }
            }
            be.upload_range(weight, 0, &original).unwrap();
            let saved = saved_live_rows(&be, head, 32_769, 4);
            let ids = head
                .draft_with_projection(
                    &be, 1234, &hidden, 32_769, 32_769, 4, shared, true, projection,
                )
                .unwrap();
            assert!(ids.iter().all(|&id| (id as usize) < cfg.vocab));
            head.settle_draft(&be, 0).unwrap();
            assert_restored_rows(&be, head, &saved);
        }
        assert_eq!(cases, 72);
        eprintln!("draft_nonfinite_safety cases={cases} widths=2,3,4 full_and_subset=true both_tail_modes=true logical_padding_guard=true raw_sentinel_preserved=true wrapped_live_kv_byte_exact=true retry_passed=true");
    }

    #[test]
    #[ignore = "requires MOE4ALL_TEST_DEVICE, MOE4ALL_SPEC_TEST_MODEL, MOE4ALL_MTP_TEST_DRAFT, MOE4ALL_MTP_TEST_VOCAB"]
    fn actual_compact_vocab_gpu_parity_and_module_timing() {
        let required =
            |key| PathBuf::from(std::env::var_os(key).unwrap_or_else(|| panic!("set {key}")));
        let target = Gguf::open(&required("MOE4ALL_SPEC_TEST_MODEL")).unwrap();
        let cfg = crate::Config::from_gguf(&target).unwrap();
        let sidecar = required("MOE4ALL_MTP_TEST_DRAFT");
        let ids_path = required("MOE4ALL_MTP_TEST_VOCAB");
        let mut engine = infr_core::config::Config::default();
        engine.device.dev =
            Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select the test GPU"));
        engine.spec.mtp = true;
        let be = VulkanBackend::new_with(Arc::new(engine)).unwrap();
        let embedding = upload_shared(&be, &target, "token_embd.weight");
        let name = if target.tensors().iter().any(|t| t.name == "output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output = upload_shared(&be, &target, name);
        let fixed = Qwen4MtpSession::load_fixed_vulkan_with_vocab(
            &be,
            &sidecar,
            &cfg,
            &target,
            Some(&ids_path),
        )
        .unwrap();
        let vocab = fixed.draft_vocab.as_ref().unwrap();
        let input = (0..cfg.n_embd)
            .map(|i| ((i * 17 + 23) % 101) as f32 / 256.0 - 0.2)
            .collect::<Vec<_>>();
        // The zero input also exercises the lowest-original-ID tie rule. The pinned EN/CJK
        // assets include zero, so that case guarantees a full argmax inside their subset.
        for (case, input) in [input, vec![0.0; cfg.n_embd]].iter().enumerate() {
            let (full_logits, full_id) = projection(&be, &cfg, &output, None, input);
            let (compact_logits, compact_id) = projection(&be, &cfg, &output, Some(vocab), input);
            for (index, &id) in vocab.ids.iter().enumerate() {
                assert_eq!(
                    compact_logits[index].to_bits(),
                    full_logits[id as usize].to_bits(),
                    "native quant row {id}"
                );
            }
            let mut best = 0;
            for index in 1..vocab.rows() {
                if compact_logits[index] > compact_logits[best] {
                    best = index;
                }
            }
            assert_eq!(compact_id, vocab.ids[best] as u32);
            let full_in_subset = vocab.ids.binary_search(&(full_id as i32)).is_ok();
            if full_in_subset {
                assert_eq!(full_id, compact_id);
            }
            if case == 1 {
                assert_eq!(full_id, 0);
                assert_eq!(compact_id, vocab.ids[0] as u32);
            }
            eprintln!("compact_projection case={case} full_argmax={full_id} compact_argmax={compact_id} full_argmax_in_subset={full_in_subset} quant_rows_bit_exact=true");
        }

        let spec = super::super::super::qwen4_mtp_kv_spec(&cfg, 32_768).unwrap();
        be.init_unified_vram(4 * spec.segment_bytes + 256 * 1024 * 1024)
            .unwrap();
        let catch = Qwen4MtpSession::shared_catch_workspace(&be, &fixed, 160_000, 4).unwrap();
        let heads = [
            Qwen4MtpSession::with_fixed_and_catch(
                &be,
                Arc::clone(&fixed),
                160_000,
                Arc::clone(&catch),
                32_768,
            )
            .unwrap(),
            Qwen4MtpSession::with_fixed_and_catch(&be, Arc::clone(&fixed), 160_000, catch, 32_768)
                .unwrap(),
        ];
        let shared: SharedWeights<'_> = (
            (embedding.0.as_ref(), embedding.1, embedding.2),
            (output.0.as_ref(), output.1, output.2),
        );
        let hidden = (0..heads[0].h_width())
            .map(|i| ((i * 17 + 23) % 101) as f32 / 256.0 - 0.2)
            .collect::<Vec<_>>();
        let zero = be.alloc(4 * 1024 * 1024, BufferUsage::Staging).unwrap();
        be.upload(zero.as_ref(), &vec![0u8; zero.len_bytes()])
            .unwrap();
        let mut samples = Vec::new();
        // Six shape/wrap checks, two warmup pairs, then twelve paired AB/BA measurements.
        let cases = [2, 3, 4]
            .into_iter()
            .flat_map(|w| [(w, 30_000, false), (w, 32_769, false)])
            .chain((0..14).map(|i| (4, 30_000, i >= 2)));
        for (trial, (width, start, measured)) in cases.enumerate() {
            reset_history(&be, &heads, zero.as_ref());
            let mut ids = [Vec::new(), Vec::new()];
            let mut draft_us = [0.0; 2];
            let mut settle_us = [0.0; 2];
            let order = if trial % 2 == 0 { [0, 1] } else { [1, 0] };
            for slot in order {
                be.sync().unwrap();
                let begin = Instant::now();
                ids[slot] = if slot == 0 {
                    heads[slot].draft_with_projection(
                        &be,
                        1234,
                        &hidden,
                        start,
                        start as i32,
                        width,
                        shared,
                        true,
                        None,
                    )
                } else {
                    heads[slot].draft_with_tail_mode(
                        &be,
                        1234,
                        &hidden,
                        start,
                        start as i32,
                        width,
                        shared,
                        true,
                    )
                }
                .unwrap();
                draft_us[slot] = begin.elapsed().as_secs_f64() * 1e6;
                assert_eq!(ids[slot].len(), width - 1);
            }
            compare_common_prefix(&be, &heads, vocab, &ids, start);
            for slot in 0..2 {
                let begin = Instant::now();
                heads[slot].settle_draft(&be, 0).unwrap();
                settle_us[slot] = begin.elapsed().as_secs_f64() * 1e6;
            }
            eprintln!("compact_head_pair trial={trial} measured={measured} width={width} start={start} order={order:?} draft_us={draft_us:?} settle_us={settle_us:?}");
            if measured {
                samples.push(draft_us);
            }
        }
        assert_eq!(samples.len(), 12);
        let median = |mut values: Vec<f64>| {
            values.sort_by(f64::total_cmp);
            (values[5] + values[6]) / 2.0
        };
        eprintln!("compact_head_summary pairs=12 full_median_us={:.3} compact_median_us={:.3} paired_delta_percent={:.3} tail_kv_only=true draft_includes_build_compile_execution_readback=true target_acceptance_measured=false e2e_tps_measured=false",
            median(samples.iter().map(|p| p[0]).collect()), median(samples.iter().map(|p| p[1]).collect()),
            median(samples.iter().map(|p| (p[1] / p[0] - 1.0) * 100.0).collect()));
    }
}
