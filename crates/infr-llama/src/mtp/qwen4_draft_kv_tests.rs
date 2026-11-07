use super::*;
use infr_core::graph::{MoeGating, TensorKind};

const HEAD_WINDOW: usize = 32_768;

fn config() -> crate::Config {
    crate::Config {
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
    }
}

fn weight_specs(cfg: &crate::Config, down_dtype: DType) -> Vec<(DType, usize)> {
    let ne = cfg.n_embd;
    let hc = cfg.hc_mult;
    let hcw = hc * ne;
    let lr = cfg.hc_low_rank;
    let qrow = cfg.n_head * cfg.head_dim;
    let kvrow = cfg.n_kv * cfg.head_dim;
    let moe = cfg.moe.unwrap();
    let exps = ne * moe.n_ff_exp * moe.n_expert;
    let shexp = ne * cfg.shexp_ff;
    let mut specs = [
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
        cfg.head_dim,
        cfg.head_dim,
        qrow * ne,
        ne * moe.n_expert,
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
    .map(|numel| (DType::F32, numel))
    .to_vec();
    for index in [1, 5, 26] {
        specs[index].0 = down_dtype;
    }
    specs
}

fn shared_specs(cfg: &crate::Config) -> [(DType, usize); 2] {
    [(DType::F32, cfg.n_embd * cfg.vocab); 2]
}

fn ops_equal(actual: &[Op], expected: &[Op]) {
    // Op has no PartialEq; Debug includes every operand and scalar parameter.
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

fn without_query(ops: &[Op], q_weight: TensorId) -> Vec<Op> {
    ops.iter()
        .filter(|op| !matches!(op, Op::Linear { weight, .. } if *weight == q_weight))
        .cloned()
        .collect()
}

fn assert_kv_writes(
    g: &Graph,
    cfg: &crate::Config,
    k_cache: TensorId,
    v_cache: TensorId,
    start_pos: usize,
    rows: usize,
    steps: usize,
) {
    let kvrow = cfg.n_kv * cfg.head_dim;
    for cache in [k_cache, v_cache] {
        assert_eq!(g.kind(cache), TensorKind::Input);
        assert_eq!(
            g.desc(cache),
            &TensorDesc::new(vec![HEAD_WINDOW * kvrow], DType::F16)
        );
    }
    let writes: Vec<_> = g
        .ops
        .iter()
        .filter_map(|op| match op {
            Op::WriteKv {
                src,
                cache,
                rows,
                row_stride,
                pos,
            } => Some((*src, *cache, *rows, *row_stride, *pos)),
            _ => None,
        })
        .collect();
    assert_eq!(writes.len(), 2 * steps);
    for (step, pair) in writes.chunks_exact(2).enumerate() {
        for (write, cache, source_dtype) in [
            (pair[0], k_cache, DType::F16),
            (pair[1], v_cache, DType::F32),
        ] {
            let (src, actual_cache, actual_rows, stride, pos) = write;
            assert_eq!(
                (actual_cache, actual_rows, stride, pos),
                (cache, rows as u32, kvrow as u32, (start_pos + step) as u32)
            );
            assert_eq!(
                g.desc(src),
                &TensorDesc::new(vec![rows * kvrow], source_dtype)
            );
        }
    }
    for op in &g.ops {
        for dst in op.io().1 {
            if g.kind(dst) == TensorKind::Input {
                assert!(
                    matches!(op, Op::WriteKv { cache, .. } if *cache == k_cache || *cache == v_cache)
                );
            } else {
                assert!(matches!(
                    g.kind(dst),
                    TensorKind::Internal | TensorKind::Output
                ));
            }
        }
    }
}

#[test]
fn draft_last_step_preserves_kv_producers_and_all_candidate_consumers() {
    let cfg = config();
    for steps in 2..=DRAFT_TOKENS {
        for start_pos in [
            0,
            HEAD_WINDOW - 4,
            HEAD_WINDOW - 1,
            HEAD_WINDOW,
            HEAD_WINDOW + 1,
            2 * HEAD_WINDOW - 2,
            150_000,
        ] {
            for down_dtype in [DType::F32, DType::Q8_0] {
                for fuse in [false, true] {
                    let specs = weight_specs(&cfg, down_dtype);
                    let (g, h) = build_draft_graph(
                        &cfg,
                        &specs,
                        shared_specs(&cfg),
                        HEAD_WINDOW,
                        HEAD_WINDOW,
                        start_pos,
                        steps,
                        fuse,
                    );
                    let (baseline, old) = build_draft_graph_impl::<false>(
                        &cfg,
                        &specs,
                        shared_specs(&cfg),
                        HEAD_WINDOW,
                        HEAD_WINDOW,
                        start_pos,
                        steps,
                        fuse,
                    );
                    assert!(g.no_decode_replay, "KV-only draft width {steps}");
                    assert!(baseline.no_decode_replay, "full-tail draft width {steps}");
                    assert_eq!(g.inputs, baseline.inputs);
                    assert_eq!(g.weights, baseline.weights);
                    assert_eq!(g.outputs, baseline.outputs);
                    assert_eq!(h.ids, old.ids);
                    assert_eq!(h.positions, old.positions);
                    assert_eq!(h.ids.len(), steps - 1);
                    assert_eq!(h.positions.len(), steps);
                    assert_eq!(g.tensors.len() + 1, baseline.tensors.len());
                    for (a, b) in g.tensors.iter().zip(&baseline.tensors) {
                        assert_eq!(a.desc, b.desc);
                        assert_eq!(a.kind, b.kind);
                    }
                    let final_start = baseline
                        .ops
                        .iter()
                        .rposition(|op| matches!(op, Op::EmbedGather { .. }))
                        .unwrap();
                    // Every earlier full step, prediction and final bridge is unchanged.
                    ops_equal(&g.ops[..final_start], &baseline.ops[..final_start]);
                    let last_kv = baseline
                        .ops
                        .iter()
                        .rposition(|op| matches!(op, Op::WriteKv { .. }))
                        .unwrap();
                    let q_weight = h.weights[8];
                    let kv_prefix = &baseline.ops[final_start..=last_kv];
                    let queries: Vec<_> = kv_prefix
                        .iter()
                        .filter_map(|op| match op {
                            Op::Linear { weight, dst, .. } if *weight == q_weight => Some(*dst),
                            _ => None,
                        })
                        .collect();
                    assert_eq!(queries.len(), 1);
                    ops_equal(&g.ops[final_start..], &without_query(kv_prefix, q_weight));
                    assert!(g.ops[final_start..]
                        .iter()
                        .all(|op| !op.io().0.contains(&queries[0])));
                    assert_eq!(
                        g.ops[final_start..]
                            .iter()
                            .filter(|op| matches!(op, Op::QwenHcDownInject { .. }))
                            .count(),
                        usize::from(fuse && down_dtype == DType::Q8_0)
                    );
                    assert!(
                        matches!(g.ops.last(), Some(Op::WriteKv { cache, .. }) if *cache == h.v_cache)
                    );
                    assert_kv_writes(&g, &cfg, h.k_cache, h.v_cache, start_pos, 1, steps);
                    assert_kv_writes(
                        &baseline,
                        &cfg,
                        old.k_cache,
                        old.v_cache,
                        start_pos,
                        1,
                        steps,
                    );

                    // The discarded suffix has no persistent writes or observable hidden output.
                    for op in &baseline.ops[last_kv + 1..] {
                        assert!(op
                            .io()
                            .1
                            .iter()
                            .all(|&dst| baseline.kind(dst) == TensorKind::Internal));
                    }
                    let h_next = match baseline.ops.last().unwrap() {
                        Op::QwenHcInject { dst, .. } => *dst,
                        op => panic!("unexpected baseline last op: {op:?}"),
                    };
                    assert_eq!(baseline.kind(h_next), TensorKind::Internal);
                    assert!(baseline.ops.iter().all(|op| !op.io().0.contains(&h_next)));
                    for (graph, count) in [(&g, steps - 1), (&baseline, steps)] {
                        assert_eq!(
                            graph
                                .ops
                                .iter()
                                .filter(|op| matches!(op, Op::Attention { .. }))
                                .count(),
                            count
                        );
                        assert_eq!(
                            graph
                                .ops
                                .iter()
                                .filter(|op| matches!(op, Op::MoeFfn { .. }))
                                .count(),
                            count
                        );
                        assert_eq!(
                            graph
                                .ops
                                .iter()
                                .filter(|op| matches!(op, Op::Argmax { .. }))
                                .count(),
                            steps - 1
                        );
                        for op in &graph.ops {
                            if let Op::Attention { mask, .. } = op {
                                assert!(matches!(mask, AttnMask::SlidingWindow(HEAD_WINDOW)));
                            }
                        }
                    }
                    for &id in &h.ids {
                        assert_eq!(g.kind(id), TensorKind::Output);
                        assert_eq!(g.desc(id), &TensorDesc::new(vec![1], DType::F32));
                        assert!(g
                            .ops
                            .iter()
                            .any(|op| matches!(op, Op::EmbedGather { ids, .. } if *ids == id)));
                    }
                }
            }
        }
    }
}

// Byte-only backend: a segmented cache has independent allocations rather than a contiguous
// backing Vec. The production journal still goes through the real Backend range-copy API.
struct JournalBuffer {
    segments: Mutex<Vec<Vec<u8>>>,
    segment_bytes: usize,
    len: usize,
}

impl JournalBuffer {
    fn new(bytes: &[u8], segment_bytes: usize) -> Self {
        Self {
            segments: Mutex::new(bytes.chunks(segment_bytes).map(<[u8]>::to_vec).collect()),
            segment_bytes,
            len: bytes.len(),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        self.segments.lock().unwrap().concat()
    }
}

impl Buffer for JournalBuffer {
    fn len_bytes(&self) -> usize {
        self.len
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Default)]
struct JournalBackend {
    fail_after: Mutex<Option<usize>>,
    copy_batches: Mutex<Vec<Vec<(usize, usize, usize)>>>,
    compiled: Mutex<Option<Graph>>,
    injected_ids: Mutex<Option<Vec<u32>>>,
    chain_ids: Mutex<Vec<u32>>,
}

impl Backend for JournalBackend {
    fn name(&self) -> &str {
        "draft-journal-bytes"
    }
    fn capabilities(&self) -> infr_core::backend::Capabilities {
        Default::default()
    }
    fn alloc(&self, bytes: usize, _: BufferUsage) -> infr_core::error::Result<Box<dyn Buffer>> {
        Ok(Box::new(JournalBuffer::new(&vec![0; bytes], bytes)))
    }
    fn upload(&self, dst: &dyn Buffer, src: &[u8]) -> infr_core::error::Result<()> {
        self.upload_range(dst, 0, src)
    }
    fn download(&self, src: &dyn Buffer, dst: &mut [u8]) -> infr_core::error::Result<()> {
        self.download_range(src, 0, dst)
    }
    fn download_range(
        &self,
        src: &dyn Buffer,
        offset: usize,
        dst: &mut [u8],
    ) -> infr_core::error::Result<()> {
        let src = src.as_any().downcast_ref::<JournalBuffer>().unwrap();
        assert!(offset + dst.len() <= src.len);
        let segments = src.segments.lock().unwrap();
        let mut copied = 0;
        while copied < dst.len() {
            let pos = offset + copied;
            let within = pos % src.segment_bytes;
            let segment = &segments[pos / src.segment_bytes];
            let n = (dst.len() - copied).min(segment.len() - within);
            dst[copied..copied + n].copy_from_slice(&segment[within..within + n]);
            copied += n;
        }
        Ok(())
    }
    fn upload_range(
        &self,
        dst: &dyn Buffer,
        offset: usize,
        src: &[u8],
    ) -> infr_core::error::Result<()> {
        let dst = dst.as_any().downcast_ref::<JournalBuffer>().unwrap();
        assert!(offset + src.len() <= dst.len);
        let mut segments = dst.segments.lock().unwrap();
        let mut copied = 0;
        while copied < src.len() {
            let pos = offset + copied;
            let within = pos % dst.segment_bytes;
            let segment = &mut segments[pos / dst.segment_bytes];
            let n = (src.len() - copied).min(segment.len() - within);
            segment[within..within + n].copy_from_slice(&src[copied..copied + n]);
            copied += n;
        }
        Ok(())
    }
    fn copy_buffer_ranges(
        &self,
        copies: &[(&dyn Buffer, usize, &dyn Buffer, usize, usize)],
    ) -> infr_core::error::Result<()> {
        self.copy_batches.lock().unwrap().push(
            copies
                .iter()
                .map(|&(_, src, _, dst, n)| (src, dst, n))
                .collect(),
        );
        let fail_after = self.fail_after.lock().unwrap().take();
        for (i, &(src, src_off, dst, dst_off, n)) in copies.iter().enumerate() {
            if fail_after == Some(i) {
                return Err(infr_core::error::Error::backend(
                    "injected journal copy failure",
                ));
            }
            let mut bytes = vec![0; n];
            self.download_range(src, src_off, &mut bytes)?;
            self.upload_range(dst, dst_off, &bytes)?;
        }
        Ok(())
    }
    fn compile(&self, graph: &Graph) -> infr_core::error::Result<Box<dyn Plan>> {
        *self.compiled.lock().unwrap() = Some(graph.clone());
        if self.injected_ids.lock().unwrap().is_some() {
            return Ok(infr_core::backend::GraphPlan::boxed(graph));
        }
        Err(infr_core::error::Error::backend(
            "byte test does not compile graphs",
        ))
    }
    fn execute(&self, plan: &dyn Plan, bindings: &Bindings) -> infr_core::error::Result<()> {
        let plan = plan
            .as_any()
            .downcast_ref::<infr_core::backend::GraphPlan>()
            .unwrap();
        let graph = &plan.graph;
        let injected = self.injected_ids.lock().unwrap().clone().unwrap();
        let mut values = vec![None; graph.tensors.len()];
        let mut step = 0;
        self.chain_ids.lock().unwrap().clear();
        for op in &graph.ops {
            match *op {
                Op::Argmax { dst, .. } => {
                    let id = injected[step];
                    step += 1;
                    values[dst.0 as usize] = Some(id);
                    self.upload(bindings.get(dst).unwrap(), &id.to_le_bytes())?;
                }
                Op::GatherI32 {
                    ids,
                    table,
                    dst,
                    ne,
                    ..
                } => {
                    assert_eq!(ne, 1);
                    let id = values[ids.0 as usize].unwrap() as usize;
                    let rows = graph.desc(table).numel();
                    // Match the shader's logical bound; never read invalid or padding rows.
                    let mapped = if id < rows {
                        let mut word = [0; 4];
                        self.download_range(bindings.get(table).unwrap(), id * 4, &mut word)?;
                        u32::from_le_bytes(word)
                    } else {
                        0
                    };
                    values[dst.0 as usize] = Some(mapped);
                }
                Op::EmbedGather { ids, table, ne, .. } => {
                    let id = match values[ids.0 as usize] {
                        Some(id) => id,
                        None => {
                            let mut word = [0; 4];
                            self.download(bindings.get(ids).unwrap(), &mut word)?;
                            u32::from_le_bytes(word)
                        }
                    };
                    assert!((id as usize) < graph.desc(table).numel() / ne as usize);
                    self.chain_ids.lock().unwrap().push(id);
                }
                Op::WriteKv {
                    cache,
                    row_stride,
                    pos,
                    rows,
                    ..
                } => {
                    let buffer = bindings.get(cache).unwrap();
                    let row_bytes = row_stride as usize * 2;
                    let capacity = buffer.len_bytes() / row_bytes;
                    for row in 0..rows as usize {
                        let offset = (pos as usize + row) % capacity * row_bytes;
                        self.upload_range(buffer, offset, &vec![0xa7; row_bytes])?;
                    }
                }
                _ => {}
            }
        }
        assert_eq!(step, injected.len());
        Ok(())
    }
    fn sync(&self) -> infr_core::error::Result<()> {
        Ok(())
    }
}

fn row_pattern(row: usize, plane: u8, generation: u8) -> [u8; 8] {
    let mut bytes = [plane, generation, 0x80, 0xff, 0, 0, 0, 0];
    bytes[4..].copy_from_slice(&(row as u32).to_le_bytes());
    bytes
}

fn ring_pattern(capacity: usize, plane: u8) -> Vec<u8> {
    (0..capacity)
        .flat_map(|row| row_pattern(row, plane, 0))
        .collect()
}

fn write_draft_rows(
    be: &JournalBackend,
    k: &JournalBuffer,
    v: &JournalBuffer,
    capacity: usize,
    start: usize,
    rows: usize,
) {
    for step in 0..rows {
        let off = ((start + step) % capacity) * 8;
        for (cache, plane) in [(k, 0), (v, 1)] {
            be.upload_range(
                cache,
                off,
                &row_pattern(start + step, plane, step as u8 + 1),
            )
            .unwrap();
        }
    }
}

fn expected_ring(capacity: usize, plane: u8, start: usize, retained: usize) -> Vec<u8> {
    let mut expected = ring_pattern(capacity, plane);
    for step in 0..retained {
        let off = ((start + step) % capacity) * 8;
        expected[off..off + 8].copy_from_slice(&row_pattern(start + step, plane, step as u8 + 1));
    }
    expected
}

fn assert_visible_ring(actual: &[u8], expected: &[u8], capacity: usize, depth: usize) {
    assert_eq!(actual.len(), capacity * 8);
    assert_eq!(expected.len(), actual.len());
    // Unpopulated suffix slots may keep rejected draft bytes; only live rows are observable.
    for row in depth.saturating_sub(capacity)..depth {
        let off = (row % capacity) * 8;
        assert_eq!(
            &actual[off..off + 8],
            &expected[off..off + 8],
            "row {row} at depth {depth}"
        );
    }
}

#[test]
fn standalone_retained_rows_apply_delivery_cap_and_first_eos_inclusively() {
    for width in 2..=DRAFT_TOKENS {
        for accepted in 1..=width {
            let tokens = &([10, 11, 12, 13])[..accepted];
            for remaining in [0, 1, 2, 3, 4, 8] {
                for eos in [None, Some(10), Some(11), Some(12), Some(13)] {
                    for ignore_eos in [false, true] {
                        let retained = retained_draft_rows(tokens, remaining, |token| {
                            !ignore_eos && eos == Some(token)
                        });
                        let mut expected = 0;
                        for &token in tokens {
                            if expected == remaining {
                                break;
                            }
                            expected += 1;
                            if !ignore_eos && eos == Some(token) {
                                break;
                            }
                        }
                        assert_eq!(retained, expected);
                    }
                }
            }
        }
    }
    assert_eq!(retained_draft_rows(&[10, 11, 12, 13], 1, |_| false), 1);
    assert_eq!(retained_draft_rows(&[10, 0, 11, 0], 4, |t| t == 0), 2);
    // A pending EOS predicted by VERIFY is outside this cycle's accepted input rows.
    assert_eq!(retained_draft_rows(&[10, 11], 4, |t| t == 77), 2);
    assert_eq!(retained_draft_rows(&[], 4, |_| false), 0);
}

#[test]
fn standalone_cap_and_eos_retention_restore_the_live_head_ring_suffix() {
    let capacity = 8;
    let tokens = [10, 11, 12, 13];
    for start in [7, 8] {
        for width in 2..=DRAFT_TOKENS {
            for (remaining, eos, ignore_eos, expected) in [
                (1, None, false, 1),
                (width, None, false, width),
                (width, Some(10), false, 1),
                (width, Some(11), false, 2),
                (width, Some(11), true, width),
                (1, Some(11), false, 1),
            ] {
                let be = JournalBackend::default();
                let k = JournalBuffer::new(&ring_pattern(capacity, 0), 3 * 8);
                let v = JournalBuffer::new(&ring_pattern(capacity, 1), 3 * 8);
                let mut journal = DraftKvJournal::new(&be, capacity, 8).unwrap();
                journal.begin(&be, &k, &v, start, width).unwrap();
                write_draft_rows(&be, &k, &v, capacity, start, width);
                let retained = retained_draft_rows(&tokens[..width], remaining, |token| {
                    !ignore_eos && eos == Some(token)
                });
                assert_eq!(retained, expected);
                journal.settle(&be, &k, &v, retained).unwrap();
                assert!(journal.pending.is_none());
                assert_visible_ring(
                    &k.bytes(),
                    &expected_ring(capacity, 0, start, expected),
                    capacity,
                    start + expected,
                );
                assert_visible_ring(
                    &v.bytes(),
                    &expected_ring(capacity, 1, start, expected),
                    capacity,
                    start + expected,
                );
            }
        }
    }
}

#[test]
fn journal_restores_exact_wrapped_and_segmented_bytes_for_every_retained_prefix() {
    for (capacity, segment_rows) in [
        (8, 8),
        (8, 3),
        (HEAD_WINDOW, HEAD_WINDOW),
        (2 * HEAD_WINDOW, HEAD_WINDOW),
    ] {
        for start in [
            0,
            capacity - 4,
            capacity - 1,
            capacity,
            segment_rows - 1,
            30_000,
            150_000,
        ] {
            for width in 2..=DRAFT_TOKENS {
                for retained in 0..=width {
                    let be = JournalBackend::default();
                    let k = JournalBuffer::new(&ring_pattern(capacity, 0), segment_rows * 8);
                    let v = JournalBuffer::new(&ring_pattern(capacity, 1), segment_rows * 8);
                    let mut journal = DraftKvJournal::new(&be, capacity, 8).unwrap();
                    assert_eq!(journal.k.len_bytes(), DRAFT_TOKENS * 8);
                    assert_eq!(journal.v.len_bytes(), DRAFT_TOKENS * 8);
                    journal.begin(&be, &k, &v, start, width).unwrap();
                    assert!(journal.pending.is_some());
                    assert!(journal.begin(&be, &k, &v, start, width).is_err());
                    write_draft_rows(&be, &k, &v, capacity, start, width);
                    assert!(journal.settle(&be, &k, &v, width + 1).is_err());
                    journal.settle(&be, &k, &v, retained).unwrap();
                    assert!(journal.pending.is_none());
                    assert_visible_ring(
                        &k.bytes(),
                        &expected_ring(capacity, 0, start, retained),
                        capacity,
                        start + retained,
                    );
                    assert_visible_ring(
                        &v.bytes(),
                        &expected_ring(capacity, 1, start, retained),
                        capacity,
                        start + retained,
                    );
                    let batches = be.copy_batches.lock().unwrap();
                    let first_saved = width.min(capacity.saturating_sub(start));
                    let first_restored = retained.max(first_saved);
                    let mut expected_batches = Vec::new();
                    if first_saved < width {
                        expected_batches.push(
                            (first_saved..width)
                                .flat_map(|step| {
                                    let range = (((start + step) % capacity) * 8, step * 8, 8);
                                    [range; 2]
                                })
                                .collect::<Vec<_>>(),
                        );
                    }
                    if first_restored < width {
                        expected_batches.push(
                            (first_restored..width)
                                .flat_map(|step| {
                                    let range = (step * 8, ((start + step) % capacity) * 8, 8);
                                    [range; 2]
                                })
                                .collect::<Vec<_>>(),
                        );
                    }
                    assert_eq!(*batches, expected_batches);
                    drop(batches);
                    journal.settle(&be, &k, &v, 0).unwrap();
                    assert!(journal.settle(&be, &k, &v, 1).is_err());
                }
            }
        }
    }
}

#[test]
fn journal_copy_failures_preserve_retry_or_full_abort_and_never_widen_retention() {
    for retained_after_retry in [0, 1] {
        let be = JournalBackend::default();
        let (capacity, start, width) = (8, 7, 4);
        let k = JournalBuffer::new(&ring_pattern(capacity, 0), 3 * 8);
        let v = JournalBuffer::new(&ring_pattern(capacity, 1), 3 * 8);
        let mut journal = DraftKvJournal::new(&be, capacity, 8).unwrap();
        *be.fail_after.lock().unwrap() = Some(1);
        assert!(journal.begin(&be, &k, &v, start, width).is_err());
        assert!(journal.pending.is_none());
        assert_eq!(k.bytes(), ring_pattern(capacity, 0));
        assert_eq!(v.bytes(), ring_pattern(capacity, 1));
        journal.begin(&be, &k, &v, start, width).unwrap();
        write_draft_rows(&be, &k, &v, capacity, start, width);
        *be.fail_after.lock().unwrap() = Some(1);
        assert!(journal.settle(&be, &k, &v, 1).is_err());
        assert!(journal.pending.is_some());
        assert!(journal.settle(&be, &k, &v, 2).is_err());
        journal.settle(&be, &k, &v, retained_after_retry).unwrap();
        assert_visible_ring(
            &k.bytes(),
            &expected_ring(capacity, 0, start, retained_after_retry),
            capacity,
            start + retained_after_retry,
        );
        assert_visible_ring(
            &v.bytes(),
            &expected_ring(capacity, 1, start, retained_after_retry),
            capacity,
            start + retained_after_retry,
        );
    }
}

#[test]
fn journal_rejects_aliasing_rows_and_position_overflow_before_copying() {
    let be = JournalBackend::default();
    let mut journal = DraftKvJournal::new(&be, 2, 8).unwrap();
    let k = JournalBuffer::new(&ring_pattern(2, 0), 8);
    let v = JournalBuffer::new(&ring_pattern(2, 1), 8);
    assert!(journal.begin(&be, &k, &v, 0, 3).is_err());
    assert!(journal.begin(&be, &k, &v, usize::MAX, 2).is_err());
    assert!(be.copy_batches.lock().unwrap().is_empty());
}

#[test]
fn journal_reuse_preserves_previous_cycles_across_repeated_wraps_and_aborts() {
    let be = JournalBackend::default();
    let capacity = 8;
    let k = JournalBuffer::new(&ring_pattern(capacity, 0), 3 * 8);
    let v = JournalBuffer::new(&ring_pattern(capacity, 1), 3 * 8);
    let mut expected = [k.bytes(), v.bytes()];
    let mut journal = DraftKvJournal::new(&be, capacity, 8).unwrap();
    let mut start = 7;
    for cycle in 0..24 {
        let width = 2 + cycle % 3;
        let retained = cycle % (width + 1);
        journal.begin(&be, &k, &v, start, width).unwrap();
        write_draft_rows(&be, &k, &v, capacity, start, width);
        journal.settle(&be, &k, &v, retained).unwrap();
        for (plane, bytes) in expected.iter_mut().enumerate() {
            for step in 0..retained {
                let off = ((start + step) % capacity) * 8;
                bytes[off..off + 8].copy_from_slice(&row_pattern(
                    start + step,
                    plane as u8,
                    step as u8 + 1,
                ));
            }
        }
        assert_visible_ring(&k.bytes(), &expected[0], capacity, start + retained);
        assert_visible_ring(&v.bytes(), &expected[1], capacity, start + retained);
        start += retained;
    }
}

fn byte_test_session(be: &JournalBackend) -> Qwen4MtpSession {
    let cfg = config();
    let fixed = Arc::new(Qwen4MtpFixed {
        cfg: cfg.clone(),
        weights: Vec::new(),
        specs: weight_specs(&cfg, DType::F32),
        fuse_down_inject: false,
        draft_vocab: None,
        draft_identity_map: Some(Box::new(JournalBuffer::new(
            bytemuck::cast_slice(&(0..cfg.vocab as i32).collect::<Vec<_>>()),
            cfg.vocab * 4,
        ))),
        device_catch_plans: Mutex::new(VecDeque::new()),
    });
    Qwen4MtpSession::with_fixed(be, fixed, 128, 4, 8).unwrap()
}

#[test]
fn invalid_raw_argmax_aborts_the_session_and_restores_wrapped_live_rows() {
    for compact in [false, true] {
        for width in 2..=DRAFT_TOKENS {
            for tail in [false, true] {
                for start in [7, 8, 9] {
                    for bad_step in 0..width - 1 {
                        let be = JournalBackend::default();
                        let mut head = byte_test_session(&be);
                        if compact {
                            let vocab =
                                draft_vocab::DraftVocab::mock(&be, head.cfg.n_embd, &[3, 9, 17])
                                    .unwrap();
                            Arc::get_mut(&mut head.fixed).unwrap().draft_vocab = Some(vocab);
                        }
                        let rows = if compact { 3 } else { head.cfg.vocab };
                        let numel = head.cfg.n_embd * head.cfg.vocab;
                        let table = be.alloc(numel * 4, BufferUsage::Weights).unwrap();
                        let shared = (
                            (table.as_ref(), DType::F32, numel),
                            (table.as_ref(), DType::F32, numel),
                        );
                        let hidden = vec![0.0; head.h_width()];
                        let before = {
                            let kv = head.kv_for_depth(&be, start + width).unwrap();
                            for (cache, byte) in [(kv.k.as_ref(), 0x31), (kv.v.as_ref(), 0x62)] {
                                be.upload(cache, &vec![byte; cache.len_bytes()]).unwrap();
                            }
                            [
                                kv.k.as_any()
                                    .downcast_ref::<JournalBuffer>()
                                    .unwrap()
                                    .bytes(),
                                kv.v.as_any()
                                    .downcast_ref::<JournalBuffer>()
                                    .unwrap()
                                    .bytes(),
                            ]
                        };
                        for bad in [rows as u32, u32::MAX] {
                            let mut injected = vec![1; width - 1];
                            injected[bad_step] = bad;
                            *be.injected_ids.lock().unwrap() = Some(injected);
                            let error = head
                                .draft_with_tail_mode(
                                    &be, 1, &hidden, start, 700, width, shared, tail,
                                )
                                .unwrap_err();
                            assert!(error.to_string().contains("invalid index"));
                            let chain = be.chain_ids.lock().unwrap();
                            assert_eq!(chain[bad_step + 1], 0);
                            assert_eq!(chain.len(), width);
                            drop(chain);
                            let kv = head.kv.lock().unwrap();
                            assert!(kv.draft_journal.pending.is_none());
                            let row_bytes = kv.k.len_bytes() / 8;
                            for (cache, expected) in
                                [(kv.k.as_ref(), &before[0]), (kv.v.as_ref(), &before[1])]
                            {
                                let actual = cache
                                    .as_any()
                                    .downcast_ref::<JournalBuffer>()
                                    .unwrap()
                                    .bytes();
                                for row in start.saturating_sub(8)..start {
                                    let offset = row % 8 * row_bytes;
                                    assert_eq!(
                                        &actual[offset..offset + row_bytes],
                                        &expected[offset..offset + row_bytes]
                                    );
                                }
                            }
                        }
                        *be.injected_ids.lock().unwrap() = Some(vec![1; width - 1]);
                        let ids = head
                            .draft_with_tail_mode(&be, 1, &hidden, start, 700, width, shared, tail)
                            .unwrap();
                        assert_eq!(ids, vec![if compact { 9 } else { 1 }; width - 1]);
                        head.settle_draft(&be, 0).unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn session_settlement_blocks_unsettled_kv_access_and_restores_bytes() {
    let be = JournalBackend::default();
    let head = byte_test_session(&be);
    let cfg = config();
    let row_bytes = cfg.n_kv * cfg.head_dim * 2;
    {
        let mut kv = head.kv_for_depth(&be, 11).unwrap();
        let Qwen4MtpKvCache {
            k,
            v,
            draft_journal,
            ..
        } = &mut *kv;
        draft_journal
            .begin(&be, k.as_ref(), v.as_ref(), 7, 4)
            .unwrap();
        for step in 0..4 {
            let off = ((7 + step) % 8) * row_bytes;
            be.upload_range(k.as_ref(), off, &vec![0x71; row_bytes])
                .unwrap();
            be.upload_range(v.as_ref(), off, &vec![0xe3; row_bytes])
                .unwrap();
        }
    }
    assert!(head.kv_prefix(&be, 8).is_err());
    head.settle_draft(&be, 1).unwrap();
    {
        let view = head.kv_prefix(&be, 8).unwrap();
        for (cache, byte) in [(view.k(), 0x71), (view.v(), 0xe3)] {
            let mut bytes = vec![0; 8 * row_bytes];
            be.download(cache, &mut bytes).unwrap();
            let mut expected = vec![0; 8 * row_bytes];
            expected[7 * row_bytes..].fill(byte);
            assert_eq!(bytes, expected);
        }
    }
    head.settle_draft(&be, 0).unwrap();
    assert!(head.settle_draft(&be, 1).is_err());
}

#[test]
fn explicit_tail_mode_selects_baseline_or_kv_only_in_the_same_session() {
    let be = JournalBackend::default();
    let head = byte_test_session(&be);
    let cfg = config();
    let hidden = vec![0.0; cfg.hc_mult * cfg.n_embd];
    let numel = cfg.n_embd * cfg.vocab;
    let table = be.alloc(numel * 4, BufferUsage::Weights).unwrap();
    let shared = (
        (table.as_ref(), DType::F32, numel),
        (table.as_ref(), DType::F32, numel),
    );
    for tail_kv_only in [false, true] {
        // The backend captures the graph and returns before execution or journaling.
        assert!(head
            .draft_with_tail_mode(&be, 1, &hidden, 7, 700, 4, shared, tail_kv_only)
            .is_err());
        let compiled = be.compiled.lock().unwrap().take().unwrap();
        assert!(compiled.no_decode_replay);
        let (expected, _) = if tail_kv_only {
            build_draft_graph_vocab_impl::<true>(
                &cfg,
                &head.fixed.specs,
                shared_specs(&cfg),
                8,
                8,
                7,
                4,
                false,
                Some(cfg.vocab),
            )
        } else {
            build_draft_graph_vocab_impl::<false>(
                &cfg,
                &head.fixed.specs,
                shared_specs(&cfg),
                8,
                8,
                7,
                4,
                false,
                Some(cfg.vocab),
            )
        };
        ops_equal(&compiled.ops, &expected.ops);
    }
    assert!(head.draft(&be, 1, &hidden, 7, 700, 4, shared).is_err());
    let compiled = be.compiled.lock().unwrap().take().unwrap();
    assert!(compiled.no_decode_replay);
    let (expected, _) = build_draft_graph_vocab_impl::<true>(
        &cfg,
        &head.fixed.specs,
        shared_specs(&cfg),
        8,
        8,
        7,
        4,
        false,
        Some(cfg.vocab),
    );
    ops_equal(&compiled.ops, &expected.ops);
    assert!(head.kv.lock().unwrap().draft_journal.pending.is_none());
    assert!(be.copy_batches.lock().unwrap().is_empty());
}

#[test]
fn kv_only_matches_full_kv_emission_with_multimodal_bridge_and_mrope() {
    let cfg = config();
    let specs = weight_specs(&cfg, DType::Q8_0);
    let kvrow = cfg.n_kv * cfg.head_dim;
    // Draft continuation uses scalar positions, including after multimodal prefill. Exercise
    // the shared KV emitter's existing MRoPE/override path separately; do not change that policy.
    for rows in [1, 4] {
        let ranges = if rows == 1 {
            vec![(0, 1)]
        } else {
            vec![(0, 1), (2, 2)]
        };
        for start_pos in [0, HEAD_WINDOW - 1, HEAD_WINDOW, 150_000] {
            for mrope in [false, true] {
                for fuse in [false, true] {
                    let make_graph = |full| {
                        let mut g = Graph::new();
                        let ids = g.input(TensorDesc::new(vec![rows], DType::I32));
                        let h = g.input(TensorDesc::new(
                            vec![rows * cfg.hc_mult * cfg.n_embd],
                            DType::F32,
                        ));
                        let positions = g.input(TensorDesc::new(vec![rows], DType::I32));
                        let positions4 =
                            mrope.then(|| g.input(TensorDesc::new(vec![rows, 4], DType::I32)));
                        let override_rows: usize = ranges.iter().map(|&(_, n)| n).sum();
                        let overrides = g.input(TensorDesc::new(
                            vec![override_rows * cfg.n_embd],
                            DType::F32,
                        ));
                        let k = g.input(TensorDesc::new(vec![HEAD_WINDOW * kvrow], DType::F16));
                        let v = g.input(TensorDesc::new(vec![HEAD_WINDOW * kvrow], DType::F16));
                        let (w, _) = declare_weights(
                            &mut g,
                            &specs,
                            shared_specs(&cfg)[0],
                            shared_specs(&cfg)[1],
                        );
                        let scratch = step_scratch(&mut g, &cfg, rows);
                        emit_bridge(
                            &mut g,
                            &cfg,
                            rows,
                            ids,
                            h,
                            &w,
                            &scratch,
                            Some((overrides, &ranges)),
                        );
                        emit_attention_kv(
                            &mut g, &cfg, rows, start_pos, positions, positions4, k, v, &w,
                            &scratch, fuse, full,
                        );
                        (g, w, k, v, positions4, overrides)
                    };
                    let (g, w, k, v, positions4, overrides) = make_graph(false);
                    let (baseline, _, _, _, _, _) = make_graph(true);
                    ops_equal(&g.ops, &without_query(&baseline.ops, w.q));
                    assert_kv_writes(&g, &cfg, k, v, start_pos, rows, 1);
                    assert_kv_writes(&baseline, &cfg, k, v, start_pos, rows, 1);
                    assert_eq!(
                        g.ops
                            .iter()
                            .filter(
                                |op| matches!(op, Op::CopyStrided { src, .. } if *src == overrides)
                            )
                            .count(),
                        ranges.len()
                    );
                    let norm = &g.ops[g.ops.len() - 3];
                    match (norm, positions4) {
                        (
                            Op::QkNormMrope {
                                positions4,
                                sections,
                                rows: actual,
                                ..
                            },
                            Some(expected),
                        ) => {
                            assert_eq!(*positions4, expected);
                            assert_eq!(*sections, cfg.rope_sections);
                            assert_eq!(*actual, rows as u32);
                        }
                        (Op::QkNormRope { rows: actual, .. }, None) => {
                            assert_eq!(*actual, rows as u32)
                        }
                        _ => panic!("unexpected K normalization: {norm:?}"),
                    }
                }
            }
        }
    }
}
