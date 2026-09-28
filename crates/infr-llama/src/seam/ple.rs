//! Qwen3.8 PLE host tier: hash a token's short n-gram context and gather the selected rows from
//! the mmap-backed GGUF table. The table is intentionally never uploaded or fully materialized.

use crate::Config;
use anyhow::{anyhow, bail, Context, Result};
use infr_core::{pager_profile, tensor::DType, WeightSource};
use infr_gguf::{Gguf, TensorBytes};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

const TABLE_NAME: &str = "per_layer_token_embd.weight";
const SOURCE_PAGE_BYTES: usize = 4096;
const MAX_GATHER_THREADS: usize = 4;
const PARALLEL_MIN_UNIQUE_ROWS: usize = 256;
const OUTPUT_POOL_DEPTH: usize = 2;

struct JobSpan {
    tokens: Vec<u32>,
    start: usize,
    rows: usize,
}

struct Job {
    spans: Vec<JobSpan>,
    reply: SyncSender<Result<PleRows>>,
}

/// One persistent model-level worker. The bounded channel prevents an unbounded random-I/O queue;
/// each job owns its reply channel, so several conversation slots can share the immutable table.
pub(super) struct PleWorker {
    tx: SyncSender<Job>,
}

pub(super) struct PleTicket(Receiver<Result<PleRows>>);

impl PleTicket {
    pub(super) fn wait(self) -> Result<PleRows> {
        let t0 = pager_profile::start();
        let rows = self
            .0
            .recv()
            .map_err(|_| anyhow!("qwen4exp PLE worker stopped before returning a row batch"))?;
        if let Some(elapsed) = pager_profile::elapsed(t0) {
            pager_profile::record_ple_wait(elapsed);
        }
        rows
    }
}

/// A gathered batch backed by one of two reusable host buffers. More buffers may be allocated
/// temporarily when several API slots reach PLE at once, so buffer reuse can never deadlock the
/// shared model worker.
pub(super) struct PleRows {
    values: Option<Vec<f32>>,
    pool: Arc<PleOutputPool>,
}

impl PleRows {
    fn new(pool: Arc<PleOutputPool>, len: usize) -> Self {
        let mut values = pool.take();
        values.resize(len, 0.0);
        Self {
            values: Some(values),
            pool,
        }
    }

    pub(super) fn as_slice(&self) -> &[f32] {
        self.values
            .as_deref()
            .expect("PLE rows retain their buffer until drop")
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        self.values
            .as_deref_mut()
            .expect("PLE rows retain their buffer until drop")
    }

    pub(super) fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl Drop for PleRows {
    fn drop(&mut self) {
        let Some(values) = self.values.take() else {
            return;
        };
        self.pool.put(values);
    }
}

struct PleOutputPool {
    buffers: Mutex<Vec<Vec<f32>>>,
}

impl PleOutputPool {
    fn new() -> Self {
        Self {
            buffers: Mutex::new((0..OUTPUT_POOL_DEPTH).map(|_| Vec::new()).collect()),
        }
    }

    fn take(&self) -> Vec<f32> {
        self.buffers.lock().unwrap().pop().unwrap_or_default()
    }

    fn put(&self, values: Vec<f32>) {
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.len() < OUTPUT_POOL_DEPTH {
            buffers.push(values);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RowRequest {
    row: usize,
    src_offset: usize,
    dst_row: usize,
}

#[derive(Clone, Copy, Debug)]
struct RowGroup {
    request_start: usize,
    request_end: usize,
    row: usize,
    src_offset: usize,
}

/// The planner assigns every request a unique output row before source sorting. Parallel groups
/// therefore write disjoint ranges even though they are no longer in output order.
#[derive(Clone, Copy)]
struct SharedOutput {
    ptr: *mut f32,
    len: usize,
}

// SAFETY: `process_row_group` is the only user. The planner gives each RowRequest a unique
// `dst_row`; exact source-row duplicates are folded into one group and copied to distinct rows.
unsafe impl Send for SharedOutput {}
unsafe impl Sync for SharedOutput {}

struct WorkerState {
    table: TensorBytes,
    dtype: DType,
    row_bytes: usize,
    row_dim: usize,
    rows: usize,
    ngram: usize,
    heads_per_ngram: usize,
    eos: u32,
    multipliers: Vec<u64>,
    offsets: Vec<u64>,
    vocab_sizes: Vec<u64>,
    gather_threads: usize,
    single_row_parallel: bool,
    gather_pool: ThreadPool,
    output_pool: Arc<PleOutputPool>,
    context_scratch: Vec<u64>,
    indices_scratch: Vec<u64>,
    requests: Vec<RowRequest>,
    groups: Vec<RowGroup>,
}

impl PleWorker {
    pub(super) fn new(g: &Gguf, cfg: &Config, single_row_parallel: bool) -> Result<Option<Self>> {
        if !cfg.qwen4exp || !cfg.ple_layers.iter().any(|&v| v) {
            return Ok(None);
        }
        let info = g
            .tensors()
            .iter()
            .find(|t| t.name == TABLE_NAME)
            .with_context(|| format!("qwen4exp PLE tensor `{TABLE_NAME}` missing"))?;
        if info.shape.len() != 2 || info.shape[0] != cfg.ple_head_dim {
            bail!(
                "qwen4exp `{TABLE_NAME}` shape {:?} does not match row dim {}",
                info.shape,
                cfg.ple_head_dim
            );
        }
        let rows = info.shape[1];
        if rows == 0 || !info.nbytes.is_multiple_of(rows) {
            bail!(
                "qwen4exp `{TABLE_NAME}` has {} bytes for {rows} rows",
                info.nbytes
            );
        }
        let row_bytes = info.nbytes / rows;
        let max_row = cfg
            .ple_head_offsets
            .iter()
            .zip(&cfg.ple_head_vocab_sizes)
            .map(|(&o, &n)| o.checked_add(n))
            .collect::<Option<Vec<_>>>()
            .context("qwen4exp PLE row range overflow")?
            .into_iter()
            .max()
            .unwrap_or(0);
        if max_row > rows as u64 {
            bail!(
                "qwen4exp PLE metadata addresses row {max_row}, but `{TABLE_NAME}` has only \
                 {rows} rows"
            );
        }
        let gather_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_GATHER_THREADS);
        let gather_pool = ThreadPoolBuilder::new()
            .num_threads(gather_threads)
            .thread_name(|i| format!("infr-qwen4-ple-{i}"))
            .build()
            .context("build qwen4exp PLE gather pool")?;
        let state = WorkerState {
            table: g.tensor_bytes_arc(TABLE_NAME).map_err(|e| anyhow!("{e}"))?,
            dtype: info.dtype,
            row_bytes,
            row_dim: cfg.ple_head_dim,
            rows,
            ngram: cfg.ple_ngram_size,
            heads_per_ngram: cfg.ple_heads_per_ngram,
            eos: cfg.ple_eos,
            multipliers: cfg.ple_layer_multipliers.clone(),
            offsets: cfg.ple_head_offsets.clone(),
            vocab_sizes: cfg.ple_head_vocab_sizes.clone(),
            gather_threads,
            single_row_parallel,
            gather_pool,
            output_pool: Arc::new(PleOutputPool::new()),
            context_scratch: Vec::with_capacity(cfg.ple_ngram_size),
            indices_scratch: Vec::with_capacity((cfg.ple_ngram_size - 1) * cfg.ple_heads_per_ngram),
            requests: Vec::new(),
            groups: Vec::new(),
        };
        let (tx, rx) = mpsc::sync_channel::<Job>(1);
        std::thread::Builder::new()
            .name("infr-qwen4-ple".into())
            .spawn(move || {
                let mut state = state;
                while let Ok(job) = rx.recv() {
                    let _ = job.reply.send(state.gather_spans(&job.spans));
                }
            })
            .context("spawn qwen4exp PLE worker")?;
        Ok(Some(Self { tx }))
    }

    /// Start the SSD/mmap gather before layer 0 is submitted. Only the current token and at most
    /// `ngram-1` predecessors cross the channel.
    pub(super) fn submit(&self, tokens: &[u32], pos: usize, ngram: usize) -> Result<PleTicket> {
        self.submit_range(tokens, pos, 1, ngram)
    }

    /// Gather a consecutive known-prompt range in token order. Only the range and its at most
    /// `ngram-1` predecessors cross the worker channel.
    pub(super) fn submit_range(
        &self,
        tokens: &[u32],
        start: usize,
        rows: usize,
        ngram: usize,
    ) -> Result<PleTicket> {
        self.submit_spans(vec![ple_job_span(tokens, start, rows, ngram)?])
    }

    /// Gather independent conversation rows in one source-sorted job. This lets concurrent decode
    /// share deduplication and issue mmap faults in parallel instead of serializing one worker job
    /// per conversation.
    pub(super) fn submit_batch<'a>(
        &self,
        requests: impl IntoIterator<Item = (&'a [u32], usize)>,
        ngram: usize,
    ) -> Result<PleTicket> {
        let spans = requests
            .into_iter()
            .map(|(tokens, pos)| ple_job_span(tokens, pos, 1, ngram))
            .collect::<Result<Vec<_>>>()?;
        if spans.is_empty() {
            bail!("qwen4exp PLE batch cannot be empty");
        }
        self.submit_spans(spans)
    }

    fn submit_spans(&self, spans: Vec<JobSpan>) -> Result<PleTicket> {
        let (reply, rx) = mpsc::sync_channel(1);
        self.tx
            .send(Job { spans, reply })
            .map_err(|_| anyhow!("qwen4exp PLE worker is not running"))?;
        Ok(PleTicket(rx))
    }
}

fn ple_job_span(tokens: &[u32], start: usize, rows: usize, ngram: usize) -> Result<JobSpan> {
    let (tokens, start) = ple_job_tokens(tokens, start, rows, ngram)?;
    Ok(JobSpan {
        tokens,
        start,
        rows,
    })
}

fn ple_job_tokens(
    tokens: &[u32],
    start: usize,
    rows: usize,
    ngram: usize,
) -> Result<(Vec<u32>, usize)> {
    let end = start
        .checked_add(rows)
        .context("PLE token range overflow")?;
    let begin = start.saturating_sub(ngram.saturating_sub(1));
    let context = tokens
        .get(begin..end)
        .with_context(|| {
            format!(
                "PLE token range {start}..{end} outside stream of {}",
                tokens.len()
            )
        })?
        .to_vec();
    Ok((context, start - begin))
}

impl WorkerState {
    fn gather_spans(&mut self, spans: &[JobSpan]) -> Result<PleRows> {
        let profile = pager_profile::active();
        let plan_t0 = profile.then(std::time::Instant::now);
        let heads = (self.ngram - 1) * self.heads_per_ngram;
        let rows = spans.iter().try_fold(0usize, |total, span| {
            total
                .checked_add(span.rows)
                .context("PLE batch row count overflow")
        })?;
        let request_count = rows
            .checked_mul(heads)
            .context("PLE row request count overflow")?;
        let output_len = request_count
            .checked_mul(self.row_dim)
            .context("PLE output size overflow")?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .context("PLE output byte size overflow")?;
        self.requests.clear();
        self.requests.reserve(request_count);
        for span in spans {
            let token_end = span
                .start
                .checked_add(span.rows)
                .context("PLE local token range overflow")?;
            for pos in span.start..token_end {
                let recent = span
                    .tokens
                    .get(..=pos)
                    .context("PLE local token range is inconsistent")?;
                ple_context_into(recent, self.ngram, self.eos, &mut self.context_scratch)?;
                ple_row_indices_into(
                    &self.context_scratch,
                    self.ngram,
                    self.heads_per_ngram,
                    &self.multipliers,
                    &self.offsets,
                    &self.vocab_sizes,
                    &mut self.indices_scratch,
                );
                for &row in &self.indices_scratch {
                    let row = usize::try_from(row).context("PLE row index overflow")?;
                    if row >= self.rows {
                        bail!(
                            "qwen4exp PLE row {row} is outside table with {} rows",
                            self.rows
                        );
                    }
                    let off = row
                        .checked_mul(self.row_bytes)
                        .context("PLE byte offset overflow")?;
                    let end = off
                        .checked_add(self.row_bytes)
                        .context("PLE byte range overflow")?;
                    if end > self.table.len() {
                        bail!(
                            "qwen4exp PLE row {row} byte range {off}..{end} exceeds table size {}",
                            self.table.len()
                        );
                    }
                    self.requests.push(RowRequest {
                        row,
                        src_offset: off,
                        dst_row: self.requests.len(),
                    });
                }
            }
        }
        debug_assert_eq!(self.requests.len(), request_count);
        self.requests
            .sort_unstable_by_key(|request| request.src_offset);
        let logical_pages = rebuild_row_groups(&self.requests, self.row_bytes, &mut self.groups);
        let plan_elapsed = pager_profile::elapsed(plan_t0).unwrap_or_default();

        let mut out = PleRows::new(Arc::clone(&self.output_pool), output_len);
        let shared_out = SharedOutput {
            ptr: out.as_mut_slice().as_mut_ptr(),
            len: output_len,
        };
        let independent_batch = spans.len() > 1;
        let multirow_span = spans.iter().any(|span| span.rows > 1);
        let parallel = self.gather_threads > 1
            && (self.groups.len() >= PARALLEL_MIN_UNIQUE_ROWS
                || ((independent_batch || multirow_span)
                    && self.groups.len() >= self.gather_threads * 2)
                || (self.single_row_parallel
                    && !independent_batch
                    && !multirow_span
                    && self.groups.len() >= self.gather_threads * 2));
        let work_t0 = profile.then(std::time::Instant::now);
        let table: &[u8] = &self.table;
        let requests = &self.requests;
        let groups = &self.groups;
        let dtype = self.dtype;
        let row_bytes = self.row_bytes;
        let row_dim = self.row_dim;
        if parallel {
            // Row groups own disjoint output ranges, so an independent multi-slot batch is not
            // limited by its number of request spans. Capping this at spans.len() left a two-slot
            // decode using only two workers from the four-thread pool, even with dozens of row
            // groups ready to gather.
            let task_count = self.gather_threads.min(groups.len());
            let groups_per_task = groups.len().div_ceil(task_count);
            self.gather_pool.install(|| {
                groups
                    .par_chunks(groups_per_task)
                    .try_for_each(|group_chunk| -> Result<()> {
                        for group in group_chunk {
                            process_row_group(
                                table, dtype, row_bytes, row_dim, requests, *group, shared_out,
                            )?;
                        }
                        Ok(())
                    })
            })?;
        } else {
            for &group in groups {
                process_row_group(
                    table, dtype, row_bytes, row_dim, requests, group, shared_out,
                )?;
            }
        }
        let work_elapsed = pager_profile::elapsed(work_t0).unwrap_or_default();
        if profile {
            pager_profile::record_ple_gather(
                rows,
                request_count,
                groups.len(),
                logical_pages,
                output_bytes,
                parallel,
                plan_elapsed,
                work_elapsed,
            );
        }
        Ok(out)
    }
}

fn process_row_group(
    table: &[u8],
    dtype: DType,
    row_bytes: usize,
    row_dim: usize,
    requests: &[RowRequest],
    group: RowGroup,
    output: SharedOutput,
) -> Result<()> {
    let end = group
        .src_offset
        .checked_add(row_bytes)
        .context("PLE byte range overflow")?;
    let source = table
        .get(group.src_offset..end)
        .with_context(|| format!("qwen4exp PLE row {} is outside its table", group.row))?;
    let first = requests[group.request_start];
    let first_offset = first
        .dst_row
        .checked_mul(row_dim)
        .context("PLE destination offset overflow")?;
    let first_end = first_offset
        .checked_add(row_dim)
        .context("PLE destination range overflow")?;
    if first_end > output.len {
        bail!("qwen4exp PLE destination row is outside its output buffer");
    }

    // SAFETY: every planned dst_row is unique, and groups partition the request array by source
    // row. Parallel groups therefore write disjoint output slices. `out` remains alive until all
    // pool work has joined.
    let first_ptr = unsafe { output.ptr.add(first_offset) };
    let first_out = unsafe { std::slice::from_raw_parts_mut(first_ptr, row_dim) };
    dequant_ple_row_into(dtype, source, first_out)
        .with_context(|| format!("dequant qwen4exp PLE row {}", group.row))?;

    // Exact duplicate source rows share one dequantization, but retain their original output
    // positions. Overlap is impossible because each request owns one complete output row.
    for request in &requests[group.request_start + 1..group.request_end] {
        let dst_offset = request
            .dst_row
            .checked_mul(row_dim)
            .context("PLE duplicate destination offset overflow")?;
        let dst_end = dst_offset
            .checked_add(row_dim)
            .context("PLE duplicate destination range overflow")?;
        if dst_end > output.len {
            bail!("qwen4exp PLE duplicate destination is outside its output buffer");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(first_ptr, output.ptr.add(dst_offset), row_dim);
        }
    }
    Ok(())
}

fn rebuild_row_groups(
    requests: &[RowRequest],
    row_bytes: usize,
    groups: &mut Vec<RowGroup>,
) -> usize {
    groups.clear();
    for (i, request) in requests.iter().enumerate() {
        match groups.last_mut() {
            Some(group) if group.src_offset == request.src_offset => group.request_end = i + 1,
            _ => groups.push(RowGroup {
                request_start: i,
                request_end: i + 1,
                row: request.row,
                src_offset: request.src_offset,
            }),
        }
    }

    let mut pages = 0usize;
    let mut last_page: Option<usize> = None;
    for group in groups.iter() {
        let first_page = group.src_offset / SOURCE_PAGE_BYTES;
        let last = group.src_offset.saturating_add(row_bytes.saturating_sub(1)) / SOURCE_PAGE_BYTES;
        let first_unseen = last_page.map_or(first_page, |page| first_page.max(page + 1));
        if first_unseen <= last {
            pages = pages.saturating_add(last - first_unseen + 1);
        }
        last_page = Some(last_page.map_or(last, |page| page.max(last)));
    }
    pages
}

fn dequant_ple_row_into(dtype: DType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        DType::Q5_1 => dequant_q5_1_into(bytes, out),
        _ => {
            let values = infr_gguf::dequant::dequant_block(dtype, bytes)?;
            if values.len() != out.len() {
                bail!(
                    "PLE row dequantized to {} values, expected {}",
                    values.len(),
                    out.len()
                );
            }
            out.copy_from_slice(&values);
            Ok(())
        }
    }
}

fn dequant_q5_1_into(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_BYTES: usize = 24;
    const BLOCK_ELEMS: usize = 32;
    if !out.len().is_multiple_of(BLOCK_ELEMS)
        || bytes.len() != out.len() / BLOCK_ELEMS * BLOCK_BYTES
    {
        bail!(
            "invalid Q5_1 PLE row geometry: {} bytes for {} values",
            bytes.len(),
            out.len()
        );
    }
    for (block, values) in bytes
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<BLOCK_ELEMS>().0.iter_mut())
    {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes(block[4..8].try_into().unwrap());
        let qs = &block[8..24];
        for j in 0..16 {
            let high0 = ((qh >> j) << 4) & 0x10;
            let high1 = (qh >> (j + 12)) & 0x10;
            let q0 = (qs[j] as u32 & 0x0f) | high0;
            let q1 = (qs[j] as u32 >> 4) | high1;
            values[j] = d * q0 as f32 + m;
            values[j + 16] = d * q1 as f32 + m;
        }
    }
    Ok(())
}

fn ple_context_into(recent: &[u32], ngram: usize, eos: u32, ctx: &mut Vec<u64>) -> Result<()> {
    let current = *recent.last().context("empty PLE token context")?;
    ctx.clear();
    ctx.resize(ngram, eos as u64);
    ctx[0] = current as u64;
    let mut cut = false;
    for s in 1..ngram {
        let tok = if cut || s >= recent.len() {
            eos
        } else {
            recent[recent.len() - 1 - s]
        };
        ctx[s] = tok as u64;
        if tok == eos {
            cut = true;
        }
    }
    Ok(())
}

fn ple_row_indices_into(
    ctx: &[u64],
    ngram: usize,
    heads_per_ngram: usize,
    multipliers: &[u64],
    offsets: &[u64],
    vocab_sizes: &[u64],
    rows: &mut Vec<u64>,
) {
    rows.clear();
    rows.reserve((ngram - 1) * heads_per_ngram);
    for n in 2..=ngram {
        let mut mixed = ctx[0].wrapping_mul(multipliers[0]);
        for j in 1..n {
            mixed ^= ctx[j].wrapping_mul(multipliers[j]);
        }
        let base = (n - 2) * heads_per_ngram;
        for h in base..base + heads_per_ngram {
            rows.push(mixed % vocab_sizes[h] + offsets[h]);
        }
    }
}

#[cfg(test)]
fn ple_context(recent: &[u32], ngram: usize, eos: u32) -> Result<Vec<u64>> {
    let mut ctx = Vec::new();
    ple_context_into(recent, ngram, eos, &mut ctx)?;
    Ok(ctx)
}

#[cfg(test)]
fn ple_row_indices(
    ctx: &[u64],
    ngram: usize,
    heads_per_ngram: usize,
    multipliers: &[u64],
    offsets: &[u64],
    vocab_sizes: &[u64],
) -> Vec<u64> {
    let mut rows = Vec::new();
    ple_row_indices_into(
        ctx,
        ngram,
        heads_per_ngram,
        multipliers,
        offsets,
        vocab_sizes,
        &mut rows,
    );
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_wrapping_u64_and_head_partitioned() {
        let ctx = [u32::MAX as u64, 7, 3];
        let mul = [u64::MAX - 2, 11, 13];
        let offsets = [0, 100, 200, 300];
        let sizes = [97, 89, 83, 79];
        let got = ple_row_indices(&ctx, 3, 2, &mul, &offsets, &sizes);
        let h2 = ctx[0].wrapping_mul(mul[0]) ^ ctx[1].wrapping_mul(mul[1]);
        let h3 = h2 ^ ctx[2].wrapping_mul(mul[2]);
        assert_eq!(
            got,
            vec![h2 % 97, h2 % 89 + 100, h3 % 83 + 200, h3 % 79 + 300]
        );
    }

    #[test]
    fn eos_strictly_before_current_cuts_older_tokens() {
        let ctx = ple_context(&[99, 2, 7], 3, 2).unwrap();
        assert_eq!(ctx, vec![7, 2, 2]);
        // The current token's own EOS does not hide its predecessors.
        assert_eq!(ple_context(&[9, 8, 2], 3, 2).unwrap(), vec![2, 8, 9]);
    }

    #[test]
    fn batched_range_preserves_each_scalar_ngram_context() {
        let tokens = [5, 7, 11, 13, 17, 19, 23, 29];
        let (local, start) = ple_job_tokens(&tokens, 3, 4, 4).unwrap();
        for row in 0..4 {
            let batched = ple_context(&local[..=start + row], 4, 2).unwrap();
            let scalar = ple_context(&tokens[..=3 + row], 4, 2).unwrap();
            assert_eq!(batched, scalar);
        }
    }

    #[test]
    fn independent_spans_preserve_lane_order_and_context() {
        let lane0 = [3, 5, 7, 11, 13];
        let lane1 = [17, 19, 23, 29, 31, 37];
        let spans = [(&lane0[..], 4usize), (&lane1[..], 5usize)]
            .into_iter()
            .map(|(tokens, pos)| ple_job_span(tokens, pos, 1, 4).unwrap())
            .collect::<Vec<_>>();

        let contexts = spans
            .iter()
            .map(|span| ple_context(&span.tokens[..=span.start], 4, 2).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(contexts[0], ple_context(&lane0, 4, 2).unwrap());
        assert_eq!(contexts[1], ple_context(&lane1, 4, 2).unwrap());
    }

    #[test]
    fn q5_1_into_matches_the_general_dequantizer() {
        let mut bytes = vec![0u8; 5 * 24];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = i.wrapping_mul(37).wrapping_add(11) as u8;
        }
        // Keep both f16 scale fields finite for every block.
        for block in bytes.as_chunks_mut::<24>().0 {
            block[0..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
            block[2..4].copy_from_slice(&half::f16::from_f32(-0.75).to_le_bytes());
        }
        let expected = infr_gguf::dequant::dequant_block(DType::Q5_1, &bytes).unwrap();
        let mut actual = vec![0.0; expected.len()];
        dequant_q5_1_into(&bytes, &mut actual).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn source_sorting_groups_duplicate_rows_and_counts_pages() {
        let mut requests = vec![
            RowRequest {
                row: 40,
                src_offset: 4800,
                dst_row: 0,
            },
            RowRequest {
                row: 1,
                src_offset: 120,
                dst_row: 1,
            },
            RowRequest {
                row: 40,
                src_offset: 4800,
                dst_row: 2,
            },
        ];
        requests.sort_unstable_by_key(|request| request.src_offset);
        let mut groups = Vec::new();
        let pages = rebuild_row_groups(&requests, 120, &mut groups);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[1].request_end - groups[1].request_start, 2);
        assert_eq!(pages, 2);
        assert_eq!(
            requests
                .iter()
                .map(|request| request.dst_row)
                .sum::<usize>(),
            3
        );
    }

    #[test]
    fn source_sorted_parallel_gather_restores_output_order() {
        const ROW_DIM: usize = 160;
        const ROW_BYTES: usize = 120;
        let mut table = vec![0u8; 4 * ROW_BYTES];
        for (row, row_bytes) in table.as_chunks_mut::<ROW_BYTES>().0.iter_mut().enumerate() {
            for (block_index, block) in row_bytes.as_chunks_mut::<24>().0.iter_mut().enumerate() {
                block[0..2].copy_from_slice(&half::f16::from_f32(0.125 + row as f32).to_le_bytes());
                block[2..4].copy_from_slice(&half::f16::from_f32(block_index as f32).to_le_bytes());
                for (i, byte) in block[4..].iter_mut().enumerate() {
                    *byte = (row * 53 + block_index * 17 + i) as u8;
                }
            }
        }

        let source_rows = [3usize, 0, 3, 1, 2, 0, 1, 3];
        let mut requests = source_rows
            .iter()
            .enumerate()
            .map(|(dst_row, &row)| RowRequest {
                row,
                src_offset: row * ROW_BYTES,
                dst_row,
            })
            .collect::<Vec<_>>();
        requests.sort_unstable_by_key(|request| request.src_offset);
        let mut groups = Vec::new();
        rebuild_row_groups(&requests, ROW_BYTES, &mut groups);

        let mut actual = vec![0.0; source_rows.len() * ROW_DIM];
        let output = SharedOutput {
            ptr: actual.as_mut_ptr(),
            len: actual.len(),
        };
        groups
            .par_iter()
            .try_for_each(|&group| {
                process_row_group(
                    &table,
                    DType::Q5_1,
                    ROW_BYTES,
                    ROW_DIM,
                    &requests,
                    group,
                    output,
                )
            })
            .unwrap();

        for (dst_row, &source_row) in source_rows.iter().enumerate() {
            let expected = infr_gguf::dequant::dequant_block(
                DType::Q5_1,
                &table[source_row * ROW_BYTES..(source_row + 1) * ROW_BYTES],
            )
            .unwrap();
            assert_eq!(
                &actual[dst_row * ROW_DIM..(dst_row + 1) * ROW_DIM],
                expected.as_slice()
            );
        }
    }
}
