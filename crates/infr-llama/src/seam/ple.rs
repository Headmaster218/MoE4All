//! Qwen3.8 PLE host tier: hash a token's short n-gram context and gather the selected rows from
//! the mmap-backed GGUF table. The table is intentionally never uploaded or fully materialized.

use crate::Config;
use anyhow::{anyhow, bail, Context, Result};
use infr_core::tensor::DType;
use infr_core::WeightSource;
use infr_gguf::{Gguf, TensorBytes};
use std::sync::mpsc::{self, Receiver, SyncSender};

const TABLE_NAME: &str = "per_layer_token_embd.weight";

struct Job {
    tokens: Vec<u32>,
    start: usize,
    rows: usize,
    reply: SyncSender<Result<Vec<f32>>>,
}

/// One persistent model-level worker. `SyncSender` keeps submission lock-free at the model layer
/// and bounds queued random I/O; each job owns its reply channel, so several conversation slots
/// can share the immutable table safely.
pub(super) struct PleWorker {
    tx: SyncSender<Job>,
}

pub(super) struct PleTicket(Receiver<Result<Vec<f32>>>);

impl PleTicket {
    pub(super) fn wait(self) -> Result<Vec<f32>> {
        self.0
            .recv()
            .map_err(|_| anyhow!("qwen4exp PLE worker stopped before returning a row batch"))?
    }
}

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
}

impl PleWorker {
    pub(super) fn new(g: &Gguf, cfg: &Config) -> Result<Option<Self>> {
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
        };
        let (tx, rx) = mpsc::sync_channel::<Job>(1);
        std::thread::Builder::new()
            .name("infr-qwen4-ple".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let _ = job
                        .reply
                        .send(state.gather_range(&job.tokens, job.start, job.rows));
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
        let (tokens, local_start) = ple_job_tokens(tokens, start, rows, ngram)?;
        let (reply, rx) = mpsc::sync_channel(1);
        self.tx
            .send(Job {
                tokens,
                start: local_start,
                rows,
                reply,
            })
            .map_err(|_| anyhow!("qwen4exp PLE worker is not running"))?;
        Ok(PleTicket(rx))
    }
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
    fn gather_range(&self, tokens: &[u32], start: usize, rows: usize) -> Result<Vec<f32>> {
        let heads = (self.ngram - 1) * self.heads_per_ngram;
        let mut out = Vec::with_capacity(rows * heads * self.row_dim);
        for pos in start..start + rows {
            let recent = tokens
                .get(..=pos)
                .context("PLE local token range is inconsistent")?;
            let ctx = ple_context(recent, self.ngram, self.eos)?;
            let indices = ple_row_indices(
                &ctx,
                self.ngram,
                self.heads_per_ngram,
                &self.multipliers,
                &self.offsets,
                &self.vocab_sizes,
            );
            for row in indices {
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
                let values = infr_gguf::dequant::dequant_block(self.dtype, &self.table[off..end])
                    .with_context(|| format!("dequant qwen4exp PLE row {row}"))?;
                if values.len() != self.row_dim {
                    bail!(
                        "qwen4exp PLE row {row} dequantized to {} values, expected {}",
                        values.len(),
                        self.row_dim
                    );
                }
                out.extend_from_slice(&values);
            }
        }
        Ok(out)
    }
}

fn ple_context(recent: &[u32], ngram: usize, eos: u32) -> Result<Vec<u64>> {
    let current = *recent.last().context("empty PLE token context")?;
    let mut ctx = vec![eos as u64; ngram];
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
    Ok(ctx)
}

fn ple_row_indices(
    ctx: &[u64],
    ngram: usize,
    heads_per_ngram: usize,
    multipliers: &[u64],
    offsets: &[u64],
    vocab_sizes: &[u64],
) -> Vec<u64> {
    let mut rows = Vec::with_capacity((ngram - 1) * heads_per_ngram);
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
}
