//! Shared sizing for scalar-row QSA lowering and pre-execution workspace admission.
use super::*;

pub(super) struct Row {
    pub complete: usize,
    pub tail: usize,
    pub selected: usize,
    pub keys: usize,
    pub chunk: usize,
    pub chunks: usize,
}

pub(super) struct Workspace {
    pub rows: Vec<Row>,
    pub max_keys: usize,
    pub requests: [(&'static str, usize); 5],
}

pub(super) fn geometry(
    spans: &[SequenceSpan],
    ratio: u32,
    top_blocks: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
) -> Result<Workspace> {
    if ratio == 0 || spans.is_empty() || spans.iter().any(|span| span.rows != 1) {
        return Err(be(
            "scalar-row QSA workspace requires nonempty single-row spans and ratio>0",
        ));
    }
    let ratio_size = ratio as usize;
    let rows = spans
        .iter()
        .map(|span| {
            let visible = (span.start_pos as usize)
                .checked_add(1)
                .ok_or_else(|| be("QSA visible length overflow"))?;
            let complete = visible / ratio_size;
            let tail = visible % ratio_size;
            let selected = qsa_lane_top_blocks(span, ratio, top_blocks) as usize;
            let keys = selected
                .checked_mul(ratio_size)
                .and_then(|n| n.checked_add(tail))
                .ok_or_else(|| be("QSA selected key count overflow"))?;
            if keys == 0 {
                return Err(be("scalar-row QSA requires at least one visible key"));
            }
            let chunk =
                split_k_chunk_count_cap(keys, infr_core::tier::adaptive_chunk(keys, &ATTN_SPLIT));
            Ok(Row {
                complete,
                tail,
                selected,
                keys,
                chunk,
                chunks: keys.div_ceil(chunk),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // The incomplete causal tail grows every token. Admit its complete next block now,
    // rather than changing the arena inside a later layer or on every tail increment.
    let max_keys = rows.iter().map(|row| row.keys).max().unwrap();
    let max_keys = (max_keys / ratio_size)
        .checked_add(1)
        .ok_or_else(|| be("QSA gathered capacity overflow"))?
        .checked_mul(ratio_size)
        .ok_or_else(|| be("QSA gathered capacity overflow"))?;
    let reserved_chunk = split_k_chunk_count_cap(max_keys, ATTN_SPLIT.min_chunk);
    let chunks = rows
        .iter()
        .map(|row| row.chunks)
        .max()
        .unwrap()
        .max(max_keys.div_ceil(reserved_chunk));
    let bytes = |a: usize, b: usize, c: usize| {
        a.checked_mul(b)
            .and_then(|n| n.checked_mul(c))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| be("QSA workspace size overflow"))
    };
    let row_elements = bytes(kv_heads as usize, head_dim as usize, 1)?;
    let gathered = bytes(max_keys, row_elements, 2)?;
    let scalars = bytes(heads as usize, chunks, 4)?;
    let accumulators = bytes(scalars, head_dim as usize, 1)?;
    Ok(Workspace {
        rows,
        max_keys,
        requests: [
            ("qsa_independent_gather_k", gathered),
            ("qsa_independent_gather_v", gathered),
            ("qsa_independent_split_pm", scalars),
            ("qsa_independent_split_pl", scalars),
            ("qsa_independent_split_pacc", accumulators),
        ],
    })
}

pub(super) fn requests(graph: &Graph, verify_split: bool) -> Result<Vec<(&'static str, usize)>> {
    let mut maxima = HashMap::new();
    for op in &graph.ops {
        let Op::QsaBatchAttention {
            rows,
            kv_len,
            top_blocks,
            ratio,
            n_head,
            n_kv,
            head_dim,
            ..
        } = op
        else {
            continue;
        };
        let independent = graph
            .independent_rows
            .then(|| sequence_spans(graph, *rows as usize))
            .transpose()?;
        let verify =
            qsa_verify_split::causal_row_views(graph, verify_split, *rows, *kv_len, *top_blocks);
        let Some(spans) = independent.or(verify.as_deref()) else {
            continue;
        };
        if spans.iter().any(|span| span.rows != 1) {
            continue;
        }
        for (tag, size) in geometry(spans, *ratio, *top_blocks, *n_head, *n_kv, *head_dim)?.requests
        {
            record_scratch_max(&mut maxima, tag, size);
        }
    }
    let mut requests = maxima.into_iter().collect::<Vec<_>>();
    requests.sort_unstable_by_key(|&(tag, _)| tag);
    Ok(requests)
}

#[cfg(test)]
#[path = "adapter_qsa_workspace_tests.rs"]
mod tests;
