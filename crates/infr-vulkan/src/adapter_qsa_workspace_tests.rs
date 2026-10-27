use super::*;
use infr_core::{DType, TensorDesc};

fn graph(positions: &[u32], ratio: u32, top: u32, dtype: DType) -> Graph {
    let mut graph = Graph::new();
    graph.independent_rows = true;
    graph.sequence_spans = positions
        .iter()
        .enumerate()
        .map(|(row, &start_pos)| SequenceSpan {
            row_start: row as u32,
            rows: 1,
            start_pos,
        })
        .collect();
    let rows = positions.len();
    let cap = positions.iter().copied().max().unwrap_or(0) as usize + 1;
    let q = graph.input(TensorDesc::new(vec![rows, 16, 256], DType::F16));
    let k = graph.input(TensorDesc::new(vec![cap, 2, 256], dtype));
    let v = graph.input(TensorDesc::new(vec![cap, 2, 256], dtype));
    let indices = graph.input(TensorDesc::new(vec![rows, top as usize], DType::I32));
    let dst = graph.output(TensorDesc::new(vec![rows, 16, 256], DType::F32));
    graph.push(Op::QsaBatchAttention {
        q,
        k_cache: k,
        v_cache: v,
        indices,
        dst,
        rows: rows as u32,
        kv_len: cap as u32,
        n_head: 16,
        n_kv: 2,
        head_dim: 256,
        top_blocks: top,
        ratio,
        scale: 0.0625,
    });
    graph
}

#[test]
fn qsa_workspace_preflight_covers_every_scalar_row_lowering_buffer() {
    for positions in [
        vec![0, 3, 127],
        vec![127],
        vec![29_999, 30_003],
        vec![29_999, 149_999, 70_001],
    ] {
        for ratio in [4, 128] {
            for dtype in [DType::F16, DType::Q8_0] {
                let graph = graph(&positions, ratio, 16, dtype);
                let spans = sequence_spans(&graph, positions.len()).unwrap();
                let runtime = geometry(spans, ratio, 16, 16, 2, 256).unwrap();
                let admitted: HashMap<_, _> =
                    requests(&graph, false).unwrap().into_iter().collect();
                assert_eq!(admitted.len(), 5);
                for (tag, bytes) in runtime.requests {
                    assert_eq!(admitted[tag], bytes, "exact lowering/admission tag={tag}");
                }
                // Gather scratch is reused across lanes, not multiplied by cohort size.
                assert_eq!(
                    admitted["qsa_independent_gather_k"],
                    runtime.max_keys * 512 * 2
                );
                for (row, span) in runtime.rows.iter().zip(spans) {
                    let visible = span.start_pos as usize + 1;
                    assert_eq!(row.complete, visible / ratio as usize);
                    assert_eq!(row.tail, visible % ratio as usize);
                    assert_eq!(row.selected, (visible / ratio as usize).min(16));
                    assert_eq!(row.keys, row.selected * ratio as usize + row.tail);
                    assert!(runtime.max_keys >= row.keys);
                    assert!(admitted["qsa_independent_split_pm"] >= 16 * row.chunks * 4);
                    assert!(admitted["qsa_independent_split_pacc"] >= 16 * row.chunks * 256 * 4);
                }
            }
        }
    }
}

#[test]
fn qsa_workspace_tail_growth_reuses_one_block_capacity() {
    for visible in 2048..2176 {
        let graph = graph(&[visible - 1], 128, 16, DType::Q8_0);
        let workspace = geometry(&graph.sequence_spans, 128, 16, 16, 2, 256).unwrap();
        assert_eq!(workspace.max_keys, 17 * 128, "visible={visible}");
    }
    for position in 30_080..30_207 {
        let graph = graph(&[position, position + 1], 128, 16, DType::Q8_0);
        let workspace = geometry(&graph.sequence_spans, 128, 16, 16, 2, 256).unwrap();
        assert_eq!(workspace.max_keys, 17 * 128);
        assert_eq!(workspace.requests[0].1, 17 * 128 * 512 * 2);
    }
}

#[test]
fn qsa_workspace_merges_layer_maxima_and_skips_batched_prefill() {
    let mut graph = graph(&[511, 149_999], 128, 16, DType::Q8_0);
    let mut larger = graph.ops[0].clone();
    if let Op::QsaBatchAttention {
        n_head, head_dim, ..
    } = &mut larger
    {
        *n_head = 32;
        *head_dim = 128;
    }
    graph.push(larger);
    let admitted: HashMap<_, _> = requests(&graph, false).unwrap().into_iter().collect();
    let first = geometry(&graph.sequence_spans, 128, 16, 16, 2, 256).unwrap();
    let second = geometry(&graph.sequence_spans, 128, 16, 32, 2, 128).unwrap();
    for (tag, bytes) in first.requests.into_iter().chain(second.requests) {
        assert!(admitted[tag] >= bytes);
    }
    graph.independent_rows = false;
    graph.sequence_spans.clear();
    assert!(requests(&graph, false).unwrap().is_empty());
    graph.independent_rows = true;
    graph.sequence_spans = vec![SequenceSpan {
        row_start: 0,
        rows: 2,
        start_pos: 510,
    }];
    assert!(requests(&graph, false).unwrap().is_empty());
}

#[test]
fn qsa_workspace_rejects_invalid_spans_before_any_resource_admission() {
    let mut graph = graph(&[100, 200], 4, 16, DType::F16);
    graph.sequence_spans[1].row_start = 0;
    assert!(requests(&graph, false).is_err());
    assert!(geometry(&[], 4, 16, 16, 2, 256).is_err());
    assert!(geometry(&graph.sequence_spans[..1], 0, 16, 16, 2, 256).is_err());
}
