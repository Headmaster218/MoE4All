use anyhow::{anyhow, bail, Context, Result};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage, Plan, SegmentedKvSpec};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::{TensorInfo, WeightSource};
use infr_gguf::Gguf;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use super::{BindWeightFn, MtpTensor};

#[path = "qwen4_draft_vocab.rs"]
mod draft_vocab;

pub const DRAFT_TOKENS: usize = 4;

fn refill_verify_feed(
    feed: &mut Vec<u32>,
    committed: &[u32],
    pending_token: u32,
    candidates: &[u32],
) {
    feed.clear();
    feed.reserve(committed.len() + 1 + candidates.len());
    feed.extend_from_slice(committed);
    feed.push(pending_token);
    feed.extend_from_slice(candidates);
}

fn generation_budget(prompt_rows: usize, requested: usize, max_ctx: usize) -> Result<usize> {
    let reserved = prompt_rows
        .checked_add(DRAFT_TOKENS)
        .ok_or_else(|| anyhow!("Qwen3.8 MTP context row count overflow"))?;
    if reserved > max_ctx {
        bail!(
            "Qwen3.8 MTP needs at least {reserved} context rows for a {prompt_rows}-token prompt, but its fixed runtime has {max_ctx}"
        );
    }
    Ok(requested.min(max_ctx - reserved))
}

fn retained_draft_rows(
    accepted_tokens: &[u32],
    remaining: usize,
    mut is_eos: impl FnMut(u32) -> bool,
) -> usize {
    let limit = accepted_tokens.len().min(remaining);
    accepted_tokens[..limit]
        .iter()
        .position(|&token| is_eos(token))
        .map_or(limit, |row| row + 1)
}

pub(crate) type SharedWeight<'a> = (&'a dyn Buffer, DType, usize);
pub(crate) type SharedWeights<'a> = (SharedWeight<'a>, SharedWeight<'a>);

struct PhaseProfile {
    name: &'static str,
    start: std::time::Instant,
    before: Option<infr_core::pager_profile::Snapshot>,
}

impl PhaseProfile {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
            before: infr_core::pager_profile::active().then(infr_core::pager_profile::snapshot),
        }
    }
}

impl Drop for PhaseProfile {
    fn drop(&mut self) {
        let Some(before) = self.before.as_ref() else {
            return;
        };
        let after = infr_core::pager_profile::snapshot();
        let delta = |a: u64, b: u64| a.saturating_sub(b);
        let ms = |a: u64, b: u64| delta(a, b) as f64 / 1e6;
        // Backend sub-timers overlap; these are scoped counters, not additive wall-time shares.
        tracing::info!(
            "[qwen4 mtp phase] name={} wall={:.2}ms backend={:.2}ms setup={:.2}ms scratch={:.2}ms record={:.2}ms sync={:.2}ms idle={:.2}ms paging_sync={:.2}ms submits={} hits={} misses={} push={:.2}MiB/{:.2}ms dma={:.2}MiB dma_gpu={:.2}ms ple_wait={:.2}ms",
            self.name,
            self.start.elapsed().as_secs_f64() * 1e3,
            ms(after.backend_execute_ns, before.backend_execute_ns),
            ms(after.backend_setup_ns, before.backend_setup_ns),
            ms(after.backend_setup_phase_scratch_ns, before.backend_setup_phase_scratch_ns),
            ms(after.command_record_ns, before.command_record_ns),
            ms(after.sync_wait_ns, before.sync_wait_ns),
            ms(after.queue_idle_wait_ns, before.queue_idle_wait_ns),
            ms(after.paging_sync_wait_ns, before.paging_sync_wait_ns),
            delta(after.queue_submits, before.queue_submits),
            delta(after.gpu_hits, before.gpu_hits),
            delta(after.gpu_misses, before.gpu_misses),
            delta(after.memcpy_bytes, before.memcpy_bytes) as f64 / 1048576.0,
            ms(after.memcpy_ns, before.memcpy_ns),
            delta(after.dedicated_transfer_bytes, before.dedicated_transfer_bytes) as f64 / 1048576.0,
            ms(after.dedicated_transfer_gpu_ns, before.dedicated_transfer_gpu_ns),
            ms(after.ple_wait_ns, before.ple_wait_ns),
        );
    }
}

struct Qwen4MtpWeights {
    // Shipped sidecars also contain the four QSA indexer tensors. Head QSA is deferred,
    // not weight-blocked: it needs bounded-ring index state, journaling and frozen selection.
    hc_attn_norm: MtpTensor,
    hc_attn_down: MtpTensor,
    hc_attn_up: MtpTensor,
    hc_attn_inject: MtpTensor,
    hc_ffn_norm: MtpTensor,
    hc_ffn_down: MtpTensor,
    hc_ffn_up: MtpTensor,
    hc_ffn_inject: MtpTensor,
    attn_q: MtpTensor,
    attn_k: MtpTensor,
    attn_v: MtpTensor,
    attn_q_norm: MtpTensor,
    attn_k_norm: MtpTensor,
    attn_output: MtpTensor,
    ffn_gate_inp: MtpTensor,
    ffn_gate_exps: MtpTensor,
    ffn_up_exps: MtpTensor,
    ffn_down_exps: MtpTensor,
    ffn_gate_inp_shexp: MtpTensor,
    ffn_gate_shexp: MtpTensor,
    ffn_up_shexp: MtpTensor,
    ffn_down_shexp: MtpTensor,
    eh_proj: MtpTensor,
    enorm: MtpTensor,
    hnorm: MtpTensor,
    hc_head_norm: MtpTensor,
    hc_head_down: MtpTensor,
    hc_head_up: MtpTensor,
}

fn tensor(g: &Gguf, name: &str, shape: &[usize]) -> Result<TensorInfo> {
    let info = g
        .tensors()
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| anyhow!("Qwen3.8 MTP sidecar is missing `{name}`"))?;
    if info.shape != shape {
        bail!(
            "Qwen3.8 MTP sidecar `{name}` has shape {:?}, expected {shape:?}",
            info.shape
        );
    }
    Ok(info.clone())
}

impl Qwen4MtpWeights {
    fn load(g: &Gguf, cfg: &crate::Config) -> Result<Self> {
        if !cfg.qwen4exp {
            bail!("Qwen3.8 MTP sidecar requires an arch=qwen4exp target");
        }
        let arch = g.metadata().str("general.architecture").unwrap_or("");
        if arch != "qwen4exp" {
            bail!("MTP sidecar architecture is `{arch}`, expected `qwen4exp`");
        }
        let nextn = g
            .metadata()
            .u64("qwen4exp.nextn_predict_layers")
            .unwrap_or(0);
        if nextn != 1 {
            bail!("Qwen3.8 MTP sidecar declares {nextn} prediction layers, expected 1");
        }
        let il = cfg.n_layer;
        let p = |suffix: &str| format!("blk.{il}.{suffix}");
        let ne = cfg.n_embd;
        let hcw = cfg.hc_mult * ne;
        let lr = cfg.hc_low_rank;
        let qrow = cfg.n_head * cfg.head_dim;
        let kvrow = cfg.n_kv * cfg.head_dim;
        let moe = cfg.moe.context("Qwen3.8 target has no MoE configuration")?;
        let nff = moe.n_ff_exp;
        let nex = moe.n_expert;
        let sff = cfg.shexp_ff;
        Ok(Self {
            hc_attn_norm: tensor(g, &p("hc_attn_norm.weight"), &[hcw])?,
            hc_attn_down: tensor(g, &p("hc_attn_down.weight"), &[hcw, lr])?,
            hc_attn_up: tensor(g, &p("hc_attn_up.weight"), &[lr, hcw])?,
            hc_attn_inject: tensor(g, &p("hc_attn_inject.weight"), &[hcw, cfg.hc_mult])?,
            hc_ffn_norm: tensor(g, &p("hc_ffn_norm.weight"), &[hcw])?,
            hc_ffn_down: tensor(g, &p("hc_ffn_down.weight"), &[hcw, lr])?,
            hc_ffn_up: tensor(g, &p("hc_ffn_up.weight"), &[lr, hcw])?,
            hc_ffn_inject: tensor(g, &p("hc_ffn_inject.weight"), &[hcw, cfg.hc_mult])?,
            attn_q: tensor(g, &p("attn_q.weight"), &[ne, 2 * qrow])?,
            attn_k: tensor(g, &p("attn_k.weight"), &[ne, kvrow])?,
            attn_v: tensor(g, &p("attn_v.weight"), &[ne, kvrow])?,
            attn_q_norm: tensor(g, &p("attn_q_norm.weight"), &[cfg.head_dim])?,
            attn_k_norm: tensor(g, &p("attn_k_norm.weight"), &[cfg.head_dim])?,
            attn_output: tensor(g, &p("attn_output.weight"), &[qrow, ne])?,
            ffn_gate_inp: tensor(g, &p("ffn_gate_inp.weight"), &[ne, nex])?,
            ffn_gate_exps: tensor(g, &p("ffn_gate_exps.weight"), &[ne, nff, nex])?,
            ffn_up_exps: tensor(g, &p("ffn_up_exps.weight"), &[ne, nff, nex])?,
            ffn_down_exps: tensor(g, &p("ffn_down_exps.weight"), &[nff, ne, nex])?,
            ffn_gate_inp_shexp: tensor(g, &p("ffn_gate_inp_shexp.weight"), &[ne])?,
            ffn_gate_shexp: tensor(g, &p("ffn_gate_shexp.weight"), &[ne, sff])?,
            ffn_up_shexp: tensor(g, &p("ffn_up_shexp.weight"), &[ne, sff])?,
            ffn_down_shexp: tensor(g, &p("ffn_down_shexp.weight"), &[sff, ne])?,
            eh_proj: tensor(g, &p("nextn.eh_proj.weight"), &[2 * ne, ne])?,
            enorm: tensor(g, &p("nextn.enorm.weight"), &[ne])?,
            hnorm: tensor(g, &p("nextn.hnorm.weight"), &[hcw])?,
            hc_head_norm: tensor(g, &p("nextn.hc_head_norm.weight"), &[hcw])?,
            hc_head_down: tensor(g, &p("nextn.hc_head_down.weight"), &[hcw, lr])?,
            hc_head_up: tensor(g, &p("nextn.hc_head_up.weight"), &[lr, hcw])?,
        })
    }

    fn ordered(&self) -> [&MtpTensor; 28] {
        [
            &self.hc_attn_norm,
            &self.hc_attn_down,
            &self.hc_attn_up,
            &self.hc_attn_inject,
            &self.hc_ffn_norm,
            &self.hc_ffn_down,
            &self.hc_ffn_up,
            &self.hc_ffn_inject,
            &self.attn_q,
            &self.attn_k,
            &self.attn_v,
            &self.attn_q_norm,
            &self.attn_k_norm,
            &self.attn_output,
            &self.ffn_gate_inp,
            &self.ffn_gate_exps,
            &self.ffn_up_exps,
            &self.ffn_down_exps,
            &self.ffn_gate_inp_shexp,
            &self.ffn_gate_shexp,
            &self.ffn_up_shexp,
            &self.ffn_down_shexp,
            &self.eh_proj,
            &self.enorm,
            &self.hnorm,
            &self.hc_head_norm,
            &self.hc_head_down,
            &self.hc_head_up,
        ]
    }
}

#[derive(Clone, Copy)]
struct HcW {
    norm: TensorId,
    down: TensorId,
    up: TensorId,
    inject: Option<TensorId>,
}

#[derive(Clone, Copy)]
struct GraphW {
    attn_hc: HcW,
    ffn_hc: HcW,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    q_norm: TensorId,
    k_norm: TensorId,
    output: TensorId,
    router: TensorId,
    gate_exps: TensorId,
    up_exps: TensorId,
    down_exps: TensorId,
    shexp_gate_inp: TensorId,
    shexp_gate: TensorId,
    shexp_up: TensorId,
    shexp_down: TensorId,
    eh_proj: TensorId,
    enorm: TensorId,
    hnorm: TensorId,
    head_hc: HcW,
    embd: TensorId,
    lm_head: TensorId,
}

fn declare_weights(
    g: &mut Graph,
    specs: &[(DType, usize)],
    embd: (DType, usize),
    lm_head: (DType, usize),
) -> (GraphW, Vec<TensorId>) {
    let mut handles = Vec::with_capacity(specs.len());
    for &(dtype, numel) in specs {
        handles.push(g.weight(TensorDesc::new(vec![numel], dtype)));
    }
    let mut i = 0usize;
    let mut next = || {
        let id = handles[i];
        i += 1;
        id
    };
    let attn_hc = HcW {
        norm: next(),
        down: next(),
        up: next(),
        inject: Some(next()),
    };
    let ffn_hc = HcW {
        norm: next(),
        down: next(),
        up: next(),
        inject: Some(next()),
    };
    let q = next();
    let k = next();
    let v = next();
    let q_norm = next();
    let k_norm = next();
    let output = next();
    let router = next();
    let gate_exps = next();
    let up_exps = next();
    let down_exps = next();
    let shexp_gate_inp = next();
    let shexp_gate = next();
    let shexp_up = next();
    let shexp_down = next();
    let eh_proj = next();
    let enorm = next();
    let hnorm = next();
    let head_hc = HcW {
        norm: next(),
        down: next(),
        up: next(),
        inject: None,
    };
    debug_assert_eq!(i, specs.len());
    let embd_id = g.weight(TensorDesc::new(vec![embd.1], embd.0));
    let lm_id = g.weight(TensorDesc::new(vec![lm_head.1], lm_head.0));
    (
        GraphW {
            attn_hc,
            ffn_hc,
            q,
            k,
            v,
            q_norm,
            k_norm,
            output,
            router,
            gate_exps,
            up_exps,
            down_exps,
            shexp_gate_inp,
            shexp_gate,
            shexp_up,
            shexp_down,
            eh_proj,
            enorm,
            hnorm,
            head_hc,
            embd: embd_id,
            lm_head: lm_id,
        },
        handles,
    )
}

struct HcScratch {
    normed: TensorId,
    low: TensorId,
    gate: TensorId,
    inject: TensorId,
}

#[allow(clippy::too_many_arguments)]
fn emit_hc_mix(
    g: &mut Graph,
    cfg: &crate::Config,
    rows: usize,
    weights: HcW,
    residual: TensorId,
    dst: TensorId,
    scratch: HcScratch,
    fuse_down_inject: bool,
    injection: Option<(TensorId, TensorId, TensorId)>,
) {
    let ne = cfg.n_embd;
    let hcw = cfg.hc_mult * ne;
    if let Some((source_residual, block, gate)) = injection {
        g.push(Op::QwenHcInjectNorm {
            residual: source_residual,
            block,
            gate,
            norm: weights.norm,
            residual_dst: residual,
            normed_dst: scratch.normed,
            rows: rows as u32,
            hc: cfg.hc_mult as u32,
            n_embd: ne as u32,
            eps: cfg.rms_eps,
        });
    } else {
        g.push(Op::QwenHcNorm {
            x: residual,
            norm: weights.norm,
            dst: scratch.normed,
            rows: rows as u32,
            hc: cfg.hc_mult as u32,
            n_embd: ne as u32,
            eps: cfg.rms_eps,
        });
    }
    let inject = weights.inject;
    let fused_down_inject = fuse_down_inject
        && rows == 1
        && inject.is_some_and(|inject| {
            g.desc(weights.down).dtype == DType::Q8_0
                && g.desc(inject).dtype == DType::F32
                && hcw.is_multiple_of(32)
        });
    if let Some(inject) = inject.filter(|_| fused_down_inject) {
        g.push(Op::QwenHcDownInject {
            x: scratch.normed,
            down_weight: weights.down,
            inject_weight: inject,
            low_dst: scratch.low,
            inject_dst: scratch.inject,
            in_f: hcw as u32,
            low_rank: cfg.hc_low_rank as u32,
            hc: cfg.hc_mult as u32,
            silu_scale: 1.0 / cfg.hc_mult as f32,
        });
    } else {
        g.push(Op::Linear {
            x: scratch.normed,
            weight: weights.down,
            dst: scratch.low,
            m: rows as u32,
            in_f: hcw as u32,
            out_f: cfg.hc_low_rank as u32,
            w_off: 0,
        });
        g.push(Op::Silu {
            x: scratch.low,
            dst: scratch.low,
            n: (rows * cfg.hc_low_rank) as u32,
            scale: 1.0 / cfg.hc_mult as f32,
        });
    }
    g.push(Op::Linear {
        x: scratch.low,
        weight: weights.up,
        dst: scratch.gate,
        m: rows as u32,
        in_f: cfg.hc_low_rank as u32,
        out_f: hcw as u32,
        w_off: 0,
    });
    g.push(Op::QwenHcMix {
        x: scratch.normed,
        gate: scratch.gate,
        dst,
        rows: rows as u32,
        hc: cfg.hc_mult as u32,
        n_embd: ne as u32,
    });
    if let Some(inject) = inject.filter(|_| !fused_down_inject) {
        g.push(Op::Linear {
            x: scratch.normed,
            weight: inject,
            dst: scratch.inject,
            m: rows as u32,
            in_f: hcw as u32,
            out_f: cfg.hc_mult as u32,
            w_off: 0,
        });
    }
}

struct StepScratch {
    e: TensorId,
    e_norm: TensorId,
    h_norm: TensorId,
    concat: TensorId,
    residual: TensorId,
    alt: TensorId,
    hidden: TensorId,
    hc: HcScratch,
    qg: TensorId,
    k: TensorId,
    v: TensorId,
    q16: TensorId,
    k16: TensorId,
    attn: TensorId,
    block: TensorId,
    moe: TensorId,
    shexp_gate: TensorId,
    shexp_g: TensorId,
    shexp_u: TensorId,
    shexp_a: TensorId,
    shexp_out: TensorId,
}

fn step_scratch(g: &mut Graph, cfg: &crate::Config, rows: usize) -> StepScratch {
    let f32d = |n| TensorDesc::new(vec![n], DType::F32);
    let f16d = |n| TensorDesc::new(vec![n], DType::F16);
    let ne = cfg.n_embd;
    let hcw = cfg.hc_mult * ne;
    let qrow = cfg.n_head * cfg.head_dim;
    let kvrow = cfg.n_kv * cfg.head_dim;
    let sff = cfg.shexp_ff;
    StepScratch {
        e: g.internal(f32d(rows * ne)),
        e_norm: g.internal(f32d(rows * ne)),
        h_norm: g.internal(f32d(rows * hcw)),
        concat: g.internal(f32d(rows * cfg.hc_mult * 2 * ne)),
        residual: g.internal(f32d(rows * hcw)),
        alt: g.internal(f32d(rows * hcw)),
        hidden: g.internal(f32d(rows * ne)),
        hc: HcScratch {
            normed: g.internal(f32d(rows * hcw)),
            low: g.internal(f32d(rows * cfg.hc_low_rank)),
            gate: g.internal(f32d(rows * hcw)),
            inject: g.internal(f32d(rows * cfg.hc_mult)),
        },
        qg: g.internal(f32d(rows * 2 * qrow)),
        k: g.internal(f32d(rows * kvrow)),
        v: g.internal(f32d(rows * kvrow)),
        q16: g.internal(f16d(rows * qrow)),
        k16: g.internal(f16d(rows * kvrow)),
        attn: g.internal(f32d(rows * qrow)),
        block: g.internal(f32d(rows * ne)),
        moe: g.internal(f32d(rows * ne)),
        shexp_gate: g.internal(f32d(rows)),
        shexp_g: g.internal(f32d(rows * sff)),
        shexp_u: g.internal(f32d(rows * sff)),
        shexp_a: g.internal(f32d(rows * sff)),
        shexp_out: g.internal(f32d(rows * ne)),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_bridge(
    g: &mut Graph,
    cfg: &crate::Config,
    rows: usize,
    ids: TensorId,
    h_in: TensorId,
    weights: &GraphW,
    scratch: &StepScratch,
    embedding_overrides: Option<(TensorId, &[(usize, usize)])>,
) {
    let ne = cfg.n_embd;
    g.push(Op::EmbedGather {
        ids,
        table: weights.embd,
        dst: scratch.e,
        rows: rows as u32,
        ne: ne as u32,
        scale: 1.0,
    });
    if let Some((source, ranges)) = embedding_overrides {
        let mut source_row = 0usize;
        for &(row_start, range_rows) in ranges {
            g.push(Op::CopyStrided {
                src: source,
                src_off: (source_row * ne) as u32,
                src_stride: ne as u32,
                dst: scratch.e,
                dst_off: (row_start * ne) as u32,
                dst_stride: ne as u32,
                rows: range_rows as u32,
                n: ne as u32,
            });
            source_row += range_rows;
        }
    }
    g.push(Op::RmsNorm {
        x: scratch.e,
        weight: weights.enorm,
        dst: scratch.e_norm,
        rows: rows as u32,
        dim: ne as u32,
        eps: cfg.rms_eps,
    });
    g.push(Op::QwenHcNorm {
        x: h_in,
        norm: weights.hnorm,
        dst: scratch.h_norm,
        rows: rows as u32,
        hc: cfg.hc_mult as u32,
        n_embd: ne as u32,
        eps: cfg.rms_eps,
    });
    for row in 0..rows {
        g.push(Op::CopyStrided {
            src: scratch.e_norm,
            src_off: (row * ne) as u32,
            src_stride: 0,
            dst: scratch.concat,
            dst_off: (row * cfg.hc_mult * 2 * ne) as u32,
            dst_stride: (2 * ne) as u32,
            rows: cfg.hc_mult as u32,
            n: ne as u32,
        });
    }
    g.push(Op::CopyStrided {
        src: scratch.h_norm,
        src_off: 0,
        src_stride: ne as u32,
        dst: scratch.concat,
        dst_off: ne as u32,
        dst_stride: (2 * ne) as u32,
        rows: (rows * cfg.hc_mult) as u32,
        n: ne as u32,
    });
    g.push(Op::Linear {
        x: scratch.concat,
        weight: weights.eh_proj,
        dst: scratch.residual,
        m: (rows * cfg.hc_mult) as u32,
        in_f: (2 * ne) as u32,
        out_f: ne as u32,
        w_off: 0,
    });
}

#[allow(clippy::too_many_arguments)]
fn emit_attention_kv(
    g: &mut Graph,
    cfg: &crate::Config,
    rows: usize,
    start_pos: usize,
    positions: TensorId,
    positions4: Option<TensorId>,
    k_cache: TensorId,
    v_cache: TensorId,
    weights: &GraphW,
    scratch: &StepScratch,
    fuse_down_inject: bool,
    full: bool,
) {
    let ne = cfg.n_embd;
    let qrow = cfg.n_head * cfg.head_dim;
    let kvrow = cfg.n_kv * cfg.head_dim;
    emit_hc_mix(
        g,
        cfg,
        rows,
        weights.attn_hc,
        scratch.residual,
        scratch.hidden,
        HcScratch {
            normed: scratch.hc.normed,
            low: scratch.hc.low,
            gate: scratch.hc.gate,
            inject: scratch.hc.inject,
        },
        fuse_down_inject,
        None,
    );
    if full {
        g.push(Op::Linear {
            x: scratch.hidden,
            weight: weights.q,
            dst: scratch.qg,
            m: rows as u32,
            in_f: ne as u32,
            out_f: (2 * qrow) as u32,
            w_off: 0,
        });
    }
    g.push(Op::Linear {
        x: scratch.hidden,
        weight: weights.k,
        dst: scratch.k,
        m: rows as u32,
        in_f: ne as u32,
        out_f: kvrow as u32,
        w_off: 0,
    });
    g.push(Op::Linear {
        x: scratch.hidden,
        weight: weights.v,
        dst: scratch.v,
        m: rows as u32,
        in_f: ne as u32,
        out_f: kvrow as u32,
        w_off: 0,
    });
    if let Some(positions4) = positions4 {
        g.push(Op::QkNormMrope {
            x: scratch.k,
            weight: weights.k_norm,
            positions4,
            dst: scratch.k16,
            rows: rows as u32,
            n_head: cfg.n_kv as u32,
            head_dim: cfg.head_dim as u32,
            rope_dim: cfg.rope_dim as u32,
            theta: cfg.rope_theta,
            eps: cfg.rms_eps,
            sections: cfg.rope_sections,
            x_stride: 0,
        });
    } else {
        g.push(Op::QkNormRope {
            x: scratch.k,
            weight: weights.k_norm,
            positions,
            dst: scratch.k16,
            rows: rows as u32,
            n_head: cfg.n_kv as u32,
            head_dim: cfg.head_dim as u32,
            rope_dim: cfg.rope_dim as u32,
            theta: cfg.rope_theta,
            eps: cfg.rms_eps,
            freq_factors: None,
            x_stride: 0,
        });
    }
    g.push(Op::WriteKv {
        src: scratch.k16,
        cache: k_cache,
        rows: rows as u32,
        row_stride: kvrow as u32,
        pos: start_pos as u32,
    });
    g.push(Op::WriteKv {
        src: scratch.v,
        cache: v_cache,
        rows: rows as u32,
        row_stride: kvrow as u32,
        pos: start_pos as u32,
    });
}

#[allow(clippy::too_many_arguments)]
fn emit_full_step(
    g: &mut Graph,
    cfg: &crate::Config,
    start_pos: usize,
    attention_window: usize,
    positions: TensorId,
    k_cache: TensorId,
    v_cache: TensorId,
    weights: &GraphW,
    scratch: &StepScratch,
    h_next: TensorId,
    fuse_down_inject: bool,
) {
    let ne = cfg.n_embd;
    let hcw = cfg.hc_mult * ne;
    let qrow = cfg.n_head * cfg.head_dim;
    let moe = cfg.moe.expect("validated Qwen3.8 MoE config");
    emit_attention_kv(
        g,
        cfg,
        1,
        start_pos,
        positions,
        None,
        k_cache,
        v_cache,
        weights,
        scratch,
        fuse_down_inject,
        true,
    );
    g.push(Op::QkNormRope {
        x: scratch.qg,
        weight: weights.q_norm,
        positions,
        dst: scratch.q16,
        rows: 1,
        n_head: cfg.n_head as u32,
        head_dim: cfg.head_dim as u32,
        rope_dim: cfg.rope_dim as u32,
        theta: cfg.rope_theta,
        eps: cfg.rms_eps,
        freq_factors: None,
        x_stride: (2 * qrow) as u32,
    });
    g.push(Op::Attention {
        q: scratch.q16,
        k_cache,
        v_cache,
        dst: scratch.attn,
        rows: 1,
        kv_len: (start_pos + 1) as u32,
        n_head: cfg.n_head as u32,
        n_kv: cfg.n_kv as u32,
        head_dim: cfg.head_dim as u32,
        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        mask: AttnMask::SlidingWindow(attention_window),
        pos: start_pos as u32,
        sinks: None,
    });
    g.push(Op::GatedAct {
        gate: scratch.qg,
        up: scratch.attn,
        dst: scratch.attn,
        rows: 1,
        nff: qrow as u32,
        act: Activation::Sigmoid,
        up_off: 0,
        up_stride: 0,
        gate_stride: (2 * qrow) as u32,
        gate_block_width: (2 * cfg.head_dim) as u32,
        swiglu_clamp: None,
    });
    g.push(Op::Linear {
        x: scratch.attn,
        weight: weights.output,
        dst: scratch.block,
        m: 1,
        in_f: qrow as u32,
        out_f: ne as u32,
        w_off: 0,
    });
    emit_hc_mix(
        g,
        cfg,
        1,
        weights.ffn_hc,
        scratch.alt,
        scratch.hidden,
        HcScratch {
            normed: scratch.hc.normed,
            low: scratch.hc.low,
            gate: scratch.hc.gate,
            inject: scratch.hc.inject,
        },
        fuse_down_inject,
        Some((scratch.residual, scratch.block, scratch.hc.inject)),
    );
    g.push(Op::MoeFfn {
        x: scratch.hidden,
        router_x: scratch.hidden,
        router: weights.router,
        gate_exps: weights.gate_exps,
        up_exps: weights.up_exps,
        down_exps: weights.down_exps,
        down_scale: None,
        fused_gate_up: false,
        dst: scratch.moe,
        ne: ne as u32,
        n_expert: moe.n_expert as u32,
        n_used: moe.n_used as u32,
        n_ff_exp: moe.n_ff_exp as u32,
        scale: moe.scale,
        act: Activation::Silu,
        gating: moe.gating,
        norm_w: moe.norm_w,
        weight_before: moe.weight_before,
        ep_band: None,
        exp_probs_b: None,
        n_expert_groups: moe.n_expert_groups,
        n_expert_groups_used: moe.n_expert_groups_used,
        swiglu_clamp: None,
        expert_ids: None,
    });
    g.push(Op::Linear {
        x: scratch.hidden,
        weight: weights.shexp_gate_inp,
        dst: scratch.shexp_gate,
        m: 1,
        in_f: ne as u32,
        out_f: 1,
        w_off: 0,
    });
    g.push(Op::Linear {
        x: scratch.hidden,
        weight: weights.shexp_gate,
        dst: scratch.shexp_g,
        m: 1,
        in_f: ne as u32,
        out_f: cfg.shexp_ff as u32,
        w_off: 0,
    });
    g.push(Op::Linear {
        x: scratch.hidden,
        weight: weights.shexp_up,
        dst: scratch.shexp_u,
        m: 1,
        in_f: ne as u32,
        out_f: cfg.shexp_ff as u32,
        w_off: 0,
    });
    g.push(Op::GatedAct {
        gate: scratch.shexp_g,
        up: scratch.shexp_u,
        dst: scratch.shexp_a,
        rows: 1,
        nff: cfg.shexp_ff as u32,
        act: Activation::Silu,
        up_off: 0,
        up_stride: 0,
        gate_stride: 0,
        gate_block_width: 0,
        swiglu_clamp: None,
    });
    g.push(Op::Linear {
        x: scratch.shexp_a,
        weight: weights.shexp_down,
        dst: scratch.shexp_out,
        m: 1,
        in_f: cfg.shexp_ff as u32,
        out_f: ne as u32,
        w_off: 0,
    });
    g.push(Op::MoeSharedExpertAdd {
        moe: scratch.moe,
        shexp: scratch.shexp_out,
        gate: scratch.shexp_gate,
        dst: scratch.block,
        rows: 1,
        n: ne as u32,
    });
    g.push(Op::QwenHcInject {
        residual: scratch.alt,
        block: scratch.block,
        gate: scratch.hc.inject,
        dst: h_next,
        rows: 1,
        hc: cfg.hc_mult as u32,
        n_embd: ne as u32,
    });
    debug_assert_eq!(hcw, cfg.hc_mult * ne);
}

struct CatchHandles {
    ids: TensorId,
    h: TensorId,
    positions: TensorId,
    positions4: Option<TensorId>,
    embedding_overrides: Option<TensorId>,
    k_cache: TensorId,
    v_cache: TensorId,
    weights: Vec<TensorId>,
    embd: TensorId,
    lm_head: TensorId,
}

#[derive(Clone)]
struct DeviceCatchHandles {
    ids: TensorId,
    target_h: TensorId,
    previous_h: TensorId,
    last_h: TensorId,
    positions: TensorId,
    positions4: Option<TensorId>,
    embedding_overrides: Option<TensorId>,
    k_cache: TensorId,
    v_cache: TensorId,
    weights: Vec<TensorId>,
    embd: TensorId,
    lm_head: TensorId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DeviceCatchPlanKey {
    rows: usize,
    start_pos: usize,
    kv_capacity: usize,
    mrope: bool,
    embedding_overrides: Vec<(usize, usize)>,
    shared: [(DType, usize); 2],
}

struct DeviceCatchPlan {
    key: DeviceCatchPlanKey,
    plan: Arc<dyn Plan>,
    handles: DeviceCatchHandles,
}

const DEVICE_CATCH_PLAN_CACHE: usize = 64;

fn push_bounded<T>(cache: &mut VecDeque<T>, value: T, capacity: usize) {
    if capacity == 0 {
        return;
    }
    while cache.len() >= capacity {
        cache.pop_front();
    }
    cache.push_back(value);
}

fn hidden_shift_copy_lengths(rows: usize, h_width: usize) -> (usize, usize) {
    (h_width, rows.saturating_sub(1) * h_width)
}

#[allow(clippy::too_many_arguments)]
fn build_device_catch_graph(
    cfg: &crate::Config,
    specs: &[(DType, usize)],
    shared: [(DType, usize); 2],
    kv_capacity: usize,
    rows: usize,
    start_pos: usize,
    embedding_overrides: &[(usize, usize)],
    mrope: bool,
    fuse_down_inject: bool,
) -> (Graph, DeviceCatchHandles) {
    let mut g = Graph::new();
    let hcw = cfg.hc_mult * cfg.n_embd;
    let kvrow = cfg.n_kv * cfg.head_dim;
    let ids = g.input(TensorDesc::new(vec![rows], DType::I32));
    let target_h = g.input(TensorDesc::new(vec![rows * hcw], DType::F32));
    let previous_h = g.input(TensorDesc::new(vec![hcw], DType::F32));
    let last_h = g.output(TensorDesc::new(vec![hcw], DType::F32));
    let shifted_h = g.internal(TensorDesc::new(vec![rows * hcw], DType::F32));
    let positions = g.input(TensorDesc::new(vec![rows], DType::I32));
    let positions4 = mrope.then(|| g.input(TensorDesc::new(vec![rows, 4], DType::I32)));
    let override_rows = embedding_overrides
        .iter()
        .map(|&(_, range_rows)| range_rows)
        .sum::<usize>();
    let embedding_override_input = (override_rows > 0).then(|| {
        g.input(TensorDesc::new(
            vec![override_rows * cfg.n_embd],
            DType::F32,
        ))
    });
    let k_cache = g.input(TensorDesc::new(vec![kv_capacity * kvrow], DType::F16));
    let v_cache = g.input(TensorDesc::new(vec![kv_capacity * kvrow], DType::F16));
    let (weights, handles) = declare_weights(&mut g, specs, shared[0], shared[1]);
    let scratch = step_scratch(&mut g, cfg, rows);

    let (previous_len, shifted_tail_len) = hidden_shift_copy_lengths(rows, hcw);
    g.push(Op::Copy {
        src: previous_h,
        src_off: 0,
        dst: shifted_h,
        dst_off: 0,
        n: previous_len as u32,
    });
    if shifted_tail_len != 0 {
        g.push(Op::Copy {
            src: target_h,
            src_off: 0,
            dst: shifted_h,
            dst_off: hcw as u32,
            n: shifted_tail_len as u32,
        });
    }
    emit_bridge(
        &mut g,
        cfg,
        rows,
        ids,
        shifted_h,
        &weights,
        &scratch,
        embedding_override_input.map(|input| (input, embedding_overrides)),
    );
    emit_attention_kv(
        &mut g,
        cfg,
        rows,
        start_pos,
        positions,
        positions4,
        k_cache,
        v_cache,
        &weights,
        &scratch,
        fuse_down_inject,
        false,
    );
    // This runs after the first-row copy above. `previous_h` and `last_h` may therefore bind the
    // same persistent buffer without overwriting the value consumed by this chunk.
    g.push(Op::Copy {
        src: target_h,
        src_off: ((rows - 1) * hcw) as u32,
        dst: last_h,
        dst_off: 0,
        n: hcw as u32,
    });
    (
        g,
        DeviceCatchHandles {
            ids,
            target_h,
            previous_h,
            last_h,
            positions,
            positions4,
            embedding_overrides: embedding_override_input,
            k_cache,
            v_cache,
            weights: handles,
            embd: weights.embd,
            lm_head: weights.lm_head,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn build_catch_graph(
    cfg: &crate::Config,
    specs: &[(DType, usize)],
    shared: [(DType, usize); 2],
    kv_capacity: usize,
    rows: usize,
    start_pos: usize,
    embedding_overrides: &[(usize, usize)],
    mrope: bool,
    fuse_down_inject: bool,
) -> (Graph, CatchHandles) {
    let mut g = Graph::new();
    let hcw = cfg.hc_mult * cfg.n_embd;
    let kvrow = cfg.n_kv * cfg.head_dim;
    let ids = g.input(TensorDesc::new(vec![rows], DType::I32));
    let h = g.input(TensorDesc::new(vec![rows * hcw], DType::F32));
    let positions = g.input(TensorDesc::new(vec![rows], DType::I32));
    let positions4 = mrope.then(|| g.input(TensorDesc::new(vec![rows, 4], DType::I32)));
    let override_rows = embedding_overrides
        .iter()
        .map(|&(_, range_rows)| range_rows)
        .sum::<usize>();
    let embedding_override_input = (override_rows > 0).then(|| {
        g.input(TensorDesc::new(
            vec![override_rows * cfg.n_embd],
            DType::F32,
        ))
    });
    let k_cache = g.input(TensorDesc::new(vec![kv_capacity * kvrow], DType::F16));
    let v_cache = g.input(TensorDesc::new(vec![kv_capacity * kvrow], DType::F16));
    let (weights, handles) = declare_weights(&mut g, specs, shared[0], shared[1]);
    let scratch = step_scratch(&mut g, cfg, rows);
    emit_bridge(
        &mut g,
        cfg,
        rows,
        ids,
        h,
        &weights,
        &scratch,
        embedding_override_input.map(|input| (input, embedding_overrides)),
    );
    emit_attention_kv(
        &mut g,
        cfg,
        rows,
        start_pos,
        positions,
        positions4,
        k_cache,
        v_cache,
        &weights,
        &scratch,
        fuse_down_inject,
        false,
    );
    (
        g,
        CatchHandles {
            ids,
            h,
            positions,
            positions4,
            embedding_overrides: embedding_override_input,
            k_cache,
            v_cache,
            weights: handles,
            embd: weights.embd,
            lm_head: weights.lm_head,
        },
    )
}

struct DraftHandles {
    id: TensorId,
    h: TensorId,
    positions: Vec<TensorId>,
    k_cache: TensorId,
    v_cache: TensorId,
    weights: Vec<TensorId>,
    embd: TensorId,
    lm_head: TensorId,
    vocab_map: Option<TensorId>,
    raw_ids: Vec<TensorId>,
    #[cfg(test)]
    ids: Vec<TensorId>,
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn build_draft_graph(
    cfg: &crate::Config,
    specs: &[(DType, usize)],
    shared: [(DType, usize); 2],
    kv_capacity: usize,
    attention_window: usize,
    start_pos: usize,
    steps: usize,
    fuse_down_inject: bool,
) -> (Graph, DraftHandles) {
    build_draft_graph_impl::<true>(
        cfg,
        specs,
        shared,
        kv_capacity,
        attention_window,
        start_pos,
        steps,
        fuse_down_inject,
    )
}

// The production full-vocabulary path also uses the bounded identity-map gather.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn build_draft_graph_impl<const LAST_STEP_KV_ONLY: bool>(
    cfg: &crate::Config,
    specs: &[(DType, usize)],
    shared: [(DType, usize); 2],
    kv_capacity: usize,
    attention_window: usize,
    start_pos: usize,
    steps: usize,
    fuse_down_inject: bool,
) -> (Graph, DraftHandles) {
    build_draft_graph_vocab_impl::<LAST_STEP_KV_ONLY>(
        cfg,
        specs,
        shared,
        kv_capacity,
        attention_window,
        start_pos,
        steps,
        fuse_down_inject,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_draft_graph_vocab_impl<const LAST_STEP_KV_ONLY: bool>(
    cfg: &crate::Config,
    specs: &[(DType, usize)],
    shared: [(DType, usize); 2],
    kv_capacity: usize,
    attention_window: usize,
    start_pos: usize,
    steps: usize,
    fuse_down_inject: bool,
    vocab_rows: Option<usize>,
) -> (Graph, DraftHandles) {
    let mut g = Graph::new();
    // Unrolled steps have distinct KV/RoPE positions, even with only one Attention at width two.
    g.no_decode_replay = true;
    let f32d = |n| TensorDesc::new(vec![n], DType::F32);
    let hcw = cfg.hc_mult * cfg.n_embd;
    let kvrow = cfg.n_kv * cfg.head_dim;
    let id = g.input(TensorDesc::new(vec![1], DType::I32));
    let h = g.input(f32d(hcw));
    let positions = (0..steps)
        .map(|_| g.input(TensorDesc::new(vec![1], DType::I32)))
        .collect::<Vec<_>>();
    let k_cache = g.input(TensorDesc::new(vec![kv_capacity * kvrow], DType::F16));
    let v_cache = g.input(TensorDesc::new(vec![kv_capacity * kvrow], DType::F16));
    let (weights, handles) = declare_weights(&mut g, specs, shared[0], shared[1]);
    let vocab_map = vocab_rows.map(|rows| g.weight(TensorDesc::new(vec![rows], DType::I32)));
    let output_rows = vocab_rows.unwrap_or(cfg.vocab);
    let scratch = step_scratch(&mut g, cfg, 1);
    let mut prev_id = id;
    let mut prev_h = h;
    #[cfg(test)]
    let mut out_ids = Vec::with_capacity(steps.saturating_sub(1));
    let mut raw_ids = Vec::with_capacity(steps.saturating_sub(1));
    for (step, &position) in positions.iter().enumerate() {
        emit_bridge(&mut g, cfg, 1, prev_id, prev_h, &weights, &scratch, None);
        if LAST_STEP_KV_ONLY && step + 1 == steps {
            // The last row has no hidden/prediction consumer. Its only persistent writes are
            // K/V; the remaining Q, Attention and FFN work would only update internal scratch.
            emit_attention_kv(
                &mut g,
                cfg,
                1,
                start_pos + step,
                position,
                None,
                k_cache,
                v_cache,
                &weights,
                &scratch,
                fuse_down_inject,
                false,
            );
            break;
        }
        let h_next = g.internal(f32d(hcw));
        emit_full_step(
            &mut g,
            cfg,
            start_pos + step,
            attention_window,
            position,
            k_cache,
            v_cache,
            &weights,
            &scratch,
            h_next,
            fuse_down_inject,
        );
        // The baseline also omits the last prediction, which is outside this target batch.
        if step + 1 == steps {
            break;
        }
        let head_hidden = g.internal(f32d(cfg.n_embd));
        emit_hc_mix(
            &mut g,
            cfg,
            1,
            weights.head_hc,
            h_next,
            head_hidden,
            HcScratch {
                normed: scratch.hc.normed,
                low: scratch.hc.low,
                gate: scratch.hc.gate,
                inject: scratch.hc.inject,
            },
            fuse_down_inject,
            None,
        );
        let logits = g.internal(f32d(output_rows));
        g.push(Op::Linear {
            x: head_hidden,
            weight: weights.lm_head,
            dst: logits,
            m: 1,
            in_f: cfg.n_embd as u32,
            out_f: output_rows as u32,
            w_off: 0,
        });
        let argmax_id = g.output(f32d(1));
        let next_id = if vocab_map.is_some() {
            g.internal(f32d(1))
        } else {
            argmax_id
        };
        g.push(Op::Argmax {
            x: logits,
            dst: argmax_id,
            n: output_rows as u32,
            rows: 1,
        });
        if let Some(table) = vocab_map {
            // Vulkan Argmax/GatherI32 carry raw integer words, including in the F32 id outputs.
            // Remap before the next bridge; a host-only readback remap would chain the wrong token.
            g.push(Op::GatherI32 {
                ids: argmax_id,
                table,
                dst: next_id,
                rows: 1,
                ne: 1,
            });
        }
        raw_ids.push(argmax_id);
        #[cfg(test)]
        out_ids.push(next_id);
        prev_id = next_id;
        prev_h = h_next;
    }
    (
        g,
        DraftHandles {
            id,
            h,
            positions,
            k_cache,
            v_cache,
            weights: handles,
            embd: weights.embd,
            lm_head: weights.lm_head,
            vocab_map,
            raw_ids,
            #[cfg(test)]
            ids: out_ids,
        },
    )
}

pub(crate) struct Qwen4MtpFixed {
    cfg: crate::Config,
    weights: Vec<Box<dyn Buffer>>,
    specs: Vec<(DType, usize)>,
    fuse_down_inject: bool,
    draft_vocab: Option<draft_vocab::DraftVocab>,
    draft_identity_map: Option<Box<dyn Buffer>>,
    device_catch_plans: Mutex<VecDeque<Arc<DeviceCatchPlan>>>,
}

struct Qwen4MtpCatchBuffers {
    max_batch: usize,
    ids: Box<dyn Buffer>,
    h: Box<dyn Buffer>,
    positions: Box<dyn Buffer>,
    positions4: Box<dyn Buffer>,
    embedding_overrides: Box<dyn Buffer>,
}

pub(crate) struct Qwen4MtpCatchWorkspace {
    h_width: usize,
    n_embd: usize,
    buffers: Mutex<Qwen4MtpCatchBuffers>,
}

struct Qwen4MtpKvCache {
    k: Box<dyn Buffer>,
    v: Box<dyn Buffer>,
    segmented: bool,
    committed_tokens: usize,
    draft_journal: DraftKvJournal,
}

#[derive(Clone, Copy)]
struct DraftKvOverwrite {
    start_pos: usize,
    rows: usize,
    first_saved: usize,
    retained_limit: usize,
}

struct DraftKvJournal {
    k: Box<dyn Buffer>,
    v: Box<dyn Buffer>,
    capacity: usize,
    row_bytes: usize,
    pending: Option<DraftKvOverwrite>,
}

impl DraftKvJournal {
    fn new(be: &dyn Backend, capacity: usize, row_bytes: usize) -> Result<Self> {
        anyhow::ensure!(
            capacity > 0 && row_bytes > 0,
            "invalid MTP draft journal geometry"
        );
        capacity
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow!("MTP draft ring byte count overflow"))?;
        let bytes = capacity.min(DRAFT_TOKENS) * row_bytes;
        Ok(Self {
            k: be
                .alloc(bytes, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
            v: be
                .alloc(bytes, BufferUsage::Activations)
                .map_err(|e| anyhow!("{e}"))?,
            capacity,
            row_bytes,
            pending: None,
        })
    }

    fn offsets(&self, start_pos: usize, rows: usize, first: usize) -> Result<Vec<(usize, usize)>> {
        anyhow::ensure!(
            rows > 0 && rows <= DRAFT_TOKENS && rows <= self.capacity && first <= rows,
            "MTP draft journal needs distinct ring rows and a valid retained prefix"
        );
        start_pos
            .checked_add(rows)
            .ok_or_else(|| anyhow!("MTP draft position overflow"))?;
        Ok((first..rows)
            .map(|step| {
                (
                    ((start_pos + step) % self.capacity) * self.row_bytes,
                    step * self.row_bytes,
                )
            })
            .collect())
    }

    fn begin(
        &mut self,
        be: &dyn Backend,
        k_cache: &dyn Buffer,
        v_cache: &dyn Buffer,
        start_pos: usize,
        rows: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            self.pending.is_none(),
            "MTP head draft must be settled before another draft"
        );
        // Before the first wrap, the leading slots have no historical rows to preserve.
        let first_saved = rows.min(self.capacity.saturating_sub(start_pos));
        let offsets = self.offsets(start_pos, rows, first_saved)?;
        let copies = offsets
            .iter()
            .flat_map(|&(cache, saved)| {
                [
                    (k_cache, cache, self.k.as_ref(), saved, self.row_bytes),
                    (v_cache, cache, self.v.as_ref(), saved, self.row_bytes),
                ]
            })
            .collect::<Vec<_>>();
        if !copies.is_empty() {
            be.copy_buffer_ranges(&copies)
                .map_err(|e| anyhow!("save MTP draft overwritten KV rows: {e}"))?;
        }
        self.pending = Some(DraftKvOverwrite {
            start_pos,
            rows,
            first_saved,
            retained_limit: rows,
        });
        Ok(())
    }

    fn settle(
        &mut self,
        be: &dyn Backend,
        k_cache: &dyn Buffer,
        v_cache: &dyn Buffer,
        retained_rows: usize,
    ) -> Result<()> {
        let Some(pending) = self.pending else {
            anyhow::ensure!(retained_rows == 0, "MTP head has no draft to retain");
            return Ok(());
        };
        anyhow::ensure!(
            retained_rows <= pending.retained_limit,
            "MTP draft retained prefix exceeds saved rows or a previous restore attempt"
        );
        let offsets = self.offsets(
            pending.start_pos,
            pending.rows,
            retained_rows.max(pending.first_saved),
        )?;
        // A backend may fail after copying part of the batch. Keep all original bytes for retry
        // or full abort, and forbid increasing the retained prefix after restoration has begun.
        self.pending.as_mut().unwrap().retained_limit = retained_rows;
        let copies = offsets
            .iter()
            .flat_map(|&(cache, saved)| {
                [
                    (self.k.as_ref(), saved, k_cache, cache, self.row_bytes),
                    (self.v.as_ref(), saved, v_cache, cache, self.row_bytes),
                ]
            })
            .collect::<Vec<_>>();
        if !copies.is_empty() {
            be.copy_buffer_ranges(&copies)
                .map_err(|e| anyhow!("restore MTP draft overwritten KV rows: {e}"))?;
        }
        self.pending = None;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen4MtpCheckpointState {
    pub(crate) tokens: Vec<u32>,
    pub(crate) last_h: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen4MtpCacheState {
    pub(crate) cached: Vec<u32>,
    pub(crate) last_h: Vec<f32>,
    pub(crate) turn_checkpoints:
        [Option<Qwen4MtpCheckpointState>; crate::seam::TURN_CHECKPOINT_COUNT],
    pub(crate) pending_id: Option<u32>,
    pub(crate) pending_emitted: bool,
}

pub(crate) struct Qwen4MtpKvView<'a> {
    kv: MutexGuard<'a, Qwen4MtpKvCache>,
    bytes_per_side: usize,
}

impl Qwen4MtpKvView<'_> {
    pub(crate) fn k(&self) -> &dyn Buffer {
        self.kv.k.as_ref()
    }

    pub(crate) fn v(&self) -> &dyn Buffer {
        self.kv.v.as_ref()
    }

    pub(crate) fn bytes_per_side(&self) -> usize {
        self.bytes_per_side
    }
}

pub(crate) struct Qwen4MtpSession {
    cfg: crate::Config,
    max_ctx: usize,
    attention_window: usize,
    kv_capacity: usize,
    fixed: Arc<Qwen4MtpFixed>,
    kv_spec: SegmentedKvSpec,
    kv: Mutex<Qwen4MtpKvCache>,
    catch: Arc<Qwen4MtpCatchWorkspace>,
    prime_last_h: Box<dyn Buffer>,
    draft_id: Box<dyn Buffer>,
    draft_h: Box<dyn Buffer>,
    draft_positions: [Box<dyn Buffer>; DRAFT_TOKENS],
    draft_ids: [Box<dyn Buffer>; DRAFT_TOKENS],
}

pub(crate) struct Qwen4MtpPrimeCapture {
    pub(crate) last_h: Vec<f32>,
    pub(crate) checkpoint_h: [Option<Vec<f32>>; crate::seam::TURN_CHECKPOINT_COUNT],
}

/// Request-scoped bridge from the trunk's ordinary Prefill path into one detached MTP head.
/// Only the previous target row persists on device; host readback is limited to the final row and
/// the two optional conversation checkpoints.
pub(crate) struct Qwen4MtpPrimeSink<'a> {
    head: &'a Qwen4MtpSession,
    initial_h: Vec<f32>,
    checkpoint_boundaries: [Option<usize>; crate::seam::TURN_CHECKPOINT_COUNT],
    checkpoint_h: [Option<Vec<f32>>; crate::seam::TURN_CHECKPOINT_COUNT],
    progress: Option<&'a dyn Fn(usize)>,
    next_pos: Option<usize>,
}

impl<'a> Qwen4MtpPrimeSink<'a> {
    pub(crate) fn new(
        head: &'a Qwen4MtpSession,
        initial_h: &[f32],
        checkpoint_boundaries: [Option<usize>; crate::seam::TURN_CHECKPOINT_COUNT],
        progress: Option<&'a dyn Fn(usize)>,
    ) -> Result<Self> {
        anyhow::ensure!(
            initial_h.len() == head.h_width(),
            "Qwen3.8 MTP prime starts with {} hidden values, expected {}",
            initial_h.len(),
            head.h_width(),
        );
        Ok(Self {
            head,
            initial_h: initial_h.to_vec(),
            checkpoint_boundaries,
            checkpoint_h: std::array::from_fn(|_| None),
            progress,
            next_pos: None,
        })
    }

    pub(crate) fn finish(self, be: &dyn Backend) -> Result<Qwen4MtpPrimeCapture> {
        anyhow::ensure!(
            self.next_pos.is_some(),
            "Qwen3.8 MTP prime produced no hidden rows"
        );
        Ok(Qwen4MtpPrimeCapture {
            last_h: self.head.device_prime_last_h(be)?,
            checkpoint_h: self.checkpoint_h,
        })
    }
}

impl crate::seam::MtpPrefillHiddenSink for Qwen4MtpPrimeSink<'_> {
    fn consume(
        &mut self,
        be: &dyn Backend,
        tokens: &[u32],
        target_hidden: &dyn Buffer,
        start_pos: usize,
        mrope: Option<&crate::seam::MropePlan>,
        shared: SharedWeights<'_>,
    ) -> Result<()> {
        anyhow::ensure!(
            !tokens.is_empty(),
            "Qwen3.8 MTP prime received an empty hidden chunk"
        );
        if let Some(next_pos) = self.next_pos {
            anyhow::ensure!(
                start_pos == next_pos,
                "Qwen3.8 MTP hidden chunks are not contiguous: expected {next_pos}, got {start_pos}",
            );
        } else {
            self.head.begin_device_prime(be, &self.initial_h)?;
        }
        self.head
            .catch_up_from_target(be, tokens, target_hidden, start_pos, mrope, shared)?;
        let end = start_pos + tokens.len();
        for (index, boundary) in self.checkpoint_boundaries.iter().copied().enumerate() {
            if boundary == Some(end) {
                self.checkpoint_h[index] = Some(self.head.device_prime_last_h(be)?);
            }
        }
        if let Some(progress) = self.progress {
            progress(end);
        }
        self.next_pos = Some(end);
        Ok(())
    }
}

pub(crate) fn qwen4_mtp_kv_spec(
    cfg: &crate::Config,
    kv_capacity: usize,
) -> Result<SegmentedKvSpec> {
    let row_elements = cfg
        .n_kv
        .checked_mul(cfg.head_dim)
        .ok_or_else(|| anyhow!("Qwen3.8 MTP KV row width overflow"))?;
    mtp_kv_spec(row_elements, kv_capacity)
}

fn mtp_kv_spec(row_elements: usize, kv_capacity: usize) -> Result<SegmentedKvSpec> {
    anyhow::ensure!(kv_capacity > 0, "Qwen3.8 MTP needs a non-zero KV capacity");
    let segment_elements = crate::seam::KV_GROW_ROWS
        .checked_mul(row_elements)
        .ok_or_else(|| anyhow!("Qwen3.8 MTP KV segment element count overflow"))?;
    anyhow::ensure!(
        segment_elements.is_power_of_two(),
        "Qwen3.8 MTP segmented KV needs a power-of-two segment width; got {segment_elements} elements"
    );
    Ok(SegmentedKvSpec {
        logical_bytes: kv_capacity
            .checked_mul(row_elements)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| anyhow!("Qwen3.8 MTP logical KV byte count overflow"))?,
        segment_bytes: segment_elements
            .checked_mul(2)
            .ok_or_else(|| anyhow!("Qwen3.8 MTP KV segment byte count overflow"))?,
        segment_elements,
        max_segments: kv_capacity.div_ceil(crate::seam::KV_GROW_ROWS),
    })
}

impl Qwen4MtpSession {
    pub(crate) fn h_width(&self) -> usize {
        self.cfg.hc_mult * self.cfg.n_embd
    }

    pub(crate) fn load_fixed_vulkan(
        vk: &infr_vulkan::VulkanBackend,
        sidecar_path: &Path,
        cfg: &crate::Config,
    ) -> Result<Arc<Qwen4MtpFixed>> {
        let bind: &BindWeightFn = &|_name, bytes, dtype, _numel| {
            let materialized = bytes.materialize();
            let padded = infr_vulkan::linear::pad_to_u32_align(&materialized);
            let buffer = vk
                .alloc(padded.len(), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            vk.upload(buffer.as_ref(), &padded)
                .map_err(|e| anyhow!("{e}"))?;
            Ok((buffer, dtype))
        };
        let mut fixed = Self::load_fixed(
            bind,
            sidecar_path,
            cfg,
            vk.capabilities().qwen_hc_down_inject && vk.cfg().kernels.qwen_hc_down_inject,
        )?;
        Arc::get_mut(&mut fixed)
            .ok_or_else(|| anyhow!("MTP fixed head was shared before ID-map initialization"))?
            .draft_identity_map = Some(draft_vocab::load_identity_map(vk, cfg.vocab)?);
        Ok(fixed)
    }

    pub(crate) fn load_fixed_vulkan_with_vocab(
        vk: &infr_vulkan::VulkanBackend,
        sidecar_path: &Path,
        cfg: &crate::Config,
        target: &Gguf,
        vocab_ids: Option<&Path>,
    ) -> Result<Arc<Qwen4MtpFixed>> {
        // Vulkan loaders bind raw-integer safety maps. Generic loading remains available for
        // CPU structural tests; CPU Argmax and GatherI32 use different id representations.
        let vocab = vocab_ids
            .map(|path| draft_vocab::DraftVocab::load_vulkan(vk, target, cfg, path))
            .transpose()?;
        let mut fixed = Self::load_fixed_vulkan(vk, sidecar_path, cfg)?;
        Arc::get_mut(&mut fixed)
            .ok_or_else(|| anyhow!("MTP fixed head was shared before vocabulary initialization"))?
            .draft_vocab = vocab;
        Ok(fixed)
    }

    pub(crate) fn load_fixed(
        bind: &BindWeightFn,
        sidecar_path: &Path,
        cfg: &crate::Config,
        fuse_down_inject: bool,
    ) -> Result<Arc<Qwen4MtpFixed>> {
        let sidecar = Gguf::open(sidecar_path)
            .with_context(|| format!("open Qwen3.8 MTP sidecar {}", sidecar_path.display()))?;
        let found = Qwen4MtpWeights::load(&sidecar, cfg)?;
        let mut weights = Vec::with_capacity(28);
        let mut specs = Vec::with_capacity(28);
        for tensor in found.ordered() {
            let bytes = sidecar
                .tensor_bytes_arc(&tensor.name)
                .map_err(|e| anyhow!("{e}"))?;
            let numel = tensor.shape.iter().product();
            let (buffer, dtype) = bind(
                &tensor.name,
                crate::seam::WBytes::Mmap(bytes),
                tensor.dtype,
                numel,
            )?;
            weights.push(buffer);
            specs.push((dtype, numel));
        }
        Ok(Arc::new(Qwen4MtpFixed {
            cfg: cfg.clone(),
            weights,
            specs,
            fuse_down_inject,
            draft_vocab: None,
            draft_identity_map: None,
            device_catch_plans: Mutex::new(VecDeque::new()),
        }))
    }

    pub(crate) fn with_fixed(
        be: &dyn Backend,
        fixed: Arc<Qwen4MtpFixed>,
        max_ctx: usize,
        max_batch: usize,
        attention_window: usize,
    ) -> Result<Self> {
        let catch = Self::shared_catch_workspace(be, &fixed, max_ctx, max_batch)?;
        Self::with_fixed_and_catch(be, fixed, max_ctx, catch, attention_window)
    }

    pub(crate) fn shared_catch_workspace(
        be: &dyn Backend,
        fixed: &Qwen4MtpFixed,
        max_ctx: usize,
        max_batch: usize,
    ) -> Result<Arc<Qwen4MtpCatchWorkspace>> {
        let cfg = &fixed.cfg;
        let hcw = cfg.hc_mult * cfg.n_embd;
        let max_batch = max_batch.max(DRAFT_TOKENS).min(max_ctx.max(1));
        let alloc = |bytes, usage| be.alloc(bytes, usage).map_err(|e| anyhow!("{e}"));
        Ok(Arc::new(Qwen4MtpCatchWorkspace {
            h_width: hcw,
            n_embd: cfg.n_embd,
            buffers: Mutex::new(Qwen4MtpCatchBuffers {
                max_batch,
                ids: alloc(max_batch * 4, BufferUsage::Staging)?,
                h: alloc(max_batch * hcw * 4, BufferUsage::Staging)?,
                positions: alloc(max_batch * 4, BufferUsage::Staging)?,
                positions4: alloc(max_batch * 4 * 4, BufferUsage::Staging)?,
                embedding_overrides: alloc(max_batch * cfg.n_embd * 4, BufferUsage::Staging)?,
            }),
        }))
    }

    pub(crate) fn with_fixed_and_catch(
        be: &dyn Backend,
        fixed: Arc<Qwen4MtpFixed>,
        max_ctx: usize,
        catch: Arc<Qwen4MtpCatchWorkspace>,
        attention_window: usize,
    ) -> Result<Self> {
        let cfg = &fixed.cfg;
        let hcw = cfg.hc_mult * cfg.n_embd;
        anyhow::ensure!(
            catch.h_width == hcw && catch.n_embd == cfg.n_embd,
            "Qwen3.8 MTP catch workspace belongs to an incompatible model"
        );
        anyhow::ensure!(
            attention_window > 0,
            "Qwen3.8 MTP attention context must be non-zero"
        );
        let attention_window = attention_window.min(max_ctx);
        let kv_capacity = attention_window;
        let alloc = |bytes, usage| be.alloc(bytes, usage).map_err(|e| anyhow!("{e}"));
        let kv_spec = qwen4_mtp_kv_spec(cfg, kv_capacity)?;
        let (k_cache, v_cache, segmented_kv) = match be
            .alloc_segmented_kv(kv_spec)
            .map_err(|e| anyhow!("{e}"))?
        {
            Some(k_cache) => {
                let v_cache = be
                    .alloc_segmented_kv(kv_spec)
                    .map_err(|e| anyhow!("{e}"))?
                    .ok_or_else(|| {
                        anyhow!("Qwen3.8 MTP segmented KV support disappeared during allocation")
                    })?;
                (k_cache, v_cache, true)
            }
            None => (
                alloc(kv_spec.logical_bytes, BufferUsage::KvCache)?,
                alloc(kv_spec.logical_bytes, BufferUsage::KvCache)?,
                false,
            ),
        };
        let draft_id = alloc(4, BufferUsage::Staging)?;
        let draft_h = alloc(hcw * 4, BufferUsage::Staging)?;
        let prime_last_h = alloc(hcw * 4, BufferUsage::Activations)?;
        let draft_positions = [
            alloc(4, BufferUsage::Staging)?,
            alloc(4, BufferUsage::Staging)?,
            alloc(4, BufferUsage::Staging)?,
            alloc(4, BufferUsage::Staging)?,
        ];
        let draft_ids = [
            alloc(4, BufferUsage::Readback)?,
            alloc(4, BufferUsage::Readback)?,
            alloc(4, BufferUsage::Readback)?,
            alloc(4, BufferUsage::Readback)?,
        ];
        let catch_batch = catch
            .buffers
            .lock()
            .map_err(|_| anyhow!("Qwen3.8 MTP catch workspace poisoned"))?
            .max_batch;
        tracing::info!(
            draft_tokens = DRAFT_TOKENS,
            max_ctx,
            attention_window,
            kv_capacity,
            catch_batch,
            kv_mode = if segmented_kv {
                "segmented-ring"
            } else {
                "flat"
            },
            "loaded fixed Qwen3.8 MTP sidecar runtime"
        );
        Ok(Self {
            cfg: cfg.clone(),
            max_ctx,
            attention_window,
            kv_capacity,
            fixed,
            kv_spec,
            kv: Mutex::new(Qwen4MtpKvCache {
                k: k_cache,
                v: v_cache,
                segmented: segmented_kv,
                committed_tokens: 0,
                draft_journal: DraftKvJournal::new(
                    be,
                    kv_capacity,
                    kv_spec.logical_bytes / kv_capacity,
                )?,
            }),
            catch,
            prime_last_h,
            draft_id,
            draft_h,
            draft_positions,
            draft_ids,
        })
    }

    fn kv_for_depth<'a>(
        &'a self,
        be: &dyn Backend,
        tokens: usize,
    ) -> Result<MutexGuard<'a, Qwen4MtpKvCache>> {
        anyhow::ensure!(
            tokens <= self.max_ctx,
            "Qwen3.8 MTP KV depth {tokens} exceeds the session capacity {}",
            self.max_ctx
        );
        let mut kv = self
            .kv
            .lock()
            .map_err(|_| anyhow!("Qwen3.8 MTP KV state poisoned"))?;
        anyhow::ensure!(
            kv.draft_journal.pending.is_none(),
            "settle the MTP head draft before accessing its KV state"
        );
        if kv.segmented && !be.segmented_kv_available() {
            let alloc = || {
                be.alloc(self.kv_spec.logical_bytes, BufferUsage::KvCache)
                    .map_err(|e| anyhow!("{e}"))
            };
            let k = alloc()?;
            let v = alloc()?;
            kv.k = k;
            kv.v = v;
            kv.segmented = false;
            tracing::info!(
                bytes_per_side = self.kv_spec.logical_bytes,
                "Qwen3.8 MTP head KV uses flat storage because no unified VRAM arena is active"
            );
        }
        let physical_tokens = tokens.min(self.kv_capacity);
        if !kv.segmented || physical_tokens <= kv.committed_tokens {
            return Ok(kv);
        }
        let segments = physical_tokens.div_ceil(crate::seam::KV_GROW_ROWS);
        be.ensure_segmented_kv_batch(&[kv.k.as_ref(), kv.v.as_ref()], segments)
            .map_err(|e| anyhow!("commit Qwen3.8 MTP segmented KV growth: {e}"))?;
        let committed_tokens = (segments * crate::seam::KV_GROW_ROWS).min(self.kv_capacity);
        kv.committed_tokens = committed_tokens;
        tracing::info!(
            requested_tokens = tokens,
            resident_tokens = physical_tokens,
            committed_tokens,
            segments,
            "expanded Qwen3.8 MTP head KV cache"
        );
        Ok(kv)
    }

    pub(crate) fn kv_prefix_bytes(&self, tokens: usize) -> Result<usize> {
        anyhow::ensure!(
            tokens <= self.max_ctx,
            "Qwen3.8 MTP KV depth {tokens} exceeds the session capacity {}",
            self.max_ctx
        );
        let row_bytes = self
            .kv_spec
            .logical_bytes
            .checked_div(self.kv_capacity)
            .ok_or_else(|| anyhow!("Qwen3.8 MTP KV row byte count is undefined"))?;
        tokens
            .min(self.kv_capacity)
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow!("Qwen3.8 MTP KV prefix byte count overflow"))
    }

    pub(crate) fn retains_rewind_prefix(&self, current: usize, target: usize) -> bool {
        target == current || (current <= self.kv_capacity && target <= current)
    }

    pub(crate) fn kv_prefix<'a>(
        &'a self,
        be: &dyn Backend,
        tokens: usize,
    ) -> Result<Qwen4MtpKvView<'a>> {
        let bytes_per_side = self.kv_prefix_bytes(tokens)?;
        let kv = self.kv_for_depth(be, tokens)?;
        Ok(Qwen4MtpKvView { kv, bytes_per_side })
    }

    pub(crate) fn copy_prefix_from(
        &self,
        be: &dyn Backend,
        src: &Qwen4MtpSession,
        tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            self.max_ctx == src.max_ctx && self.kv_spec == src.kv_spec,
            "Qwen3.8 MTP seed source has incompatible KV geometry"
        );
        anyhow::ensure!(
            !std::ptr::eq(self, src),
            "Qwen3.8 MTP cannot seed a session from itself"
        );
        let src = src.kv_prefix(be, tokens)?;
        let dst = self.kv_prefix(be, tokens)?;
        let bytes = src.bytes_per_side();
        anyhow::ensure!(
            bytes == dst.bytes_per_side(),
            "Qwen3.8 MTP seed byte count changed between slots"
        );
        be.copy_buffers(&[(src.k(), dst.k(), bytes), (src.v(), dst.v(), bytes)])
            .map_err(|error| anyhow!("copy Qwen3.8 MTP head KV prefix: {error}"))
    }

    pub(crate) fn release_kv(&self, be: &dyn Backend) -> Result<()> {
        self.settle_draft(be, 0)?;
        be.sync()
            .map_err(|error| anyhow!("sync before Qwen3.8 MTP KV release: {error}"))?;
        let mut kv = self
            .kv
            .lock()
            .map_err(|_| anyhow!("Qwen3.8 MTP KV state poisoned"))?;
        if !kv.segmented || kv.committed_tokens == 0 {
            return Ok(());
        }
        be.release_segmented_kv(kv.k.as_ref())
            .map_err(|error| anyhow!("release Qwen3.8 MTP K cache: {error}"))?;
        be.release_segmented_kv(kv.v.as_ref())
            .map_err(|error| anyhow!("release Qwen3.8 MTP V cache: {error}"))?;
        kv.committed_tokens = 0;
        Ok(())
    }

    fn bind_common<'a>(
        &'a self,
        bindings: &mut Bindings<'a>,
        weights: &[TensorId],
        embd_id: TensorId,
        lm_id: TensorId,
        embd: &'a dyn Buffer,
        lm_head: &'a dyn Buffer,
    ) {
        for (id, buffer) in weights.iter().copied().zip(&self.fixed.weights) {
            bindings.bind(id, buffer.as_ref());
        }
        bindings.bind(embd_id, embd);
        bindings.bind(lm_id, lm_head);
    }

    pub(crate) fn catch_up(
        &self,
        be: &dyn Backend,
        tokens: &[u32],
        h: &[f32],
        start_pos: usize,
        mrope: Option<&crate::seam::MropePlan>,
        shared: SharedWeights<'_>,
    ) -> Result<()> {
        let hcw = self.cfg.hc_mult * self.cfg.n_embd;
        if h.len() != tokens.len() * hcw {
            bail!(
                "Qwen3.8 MTP catch-up got {} hidden values for {} tokens (width {hcw})",
                h.len(),
                tokens.len()
            );
        }
        if let Some(plan) = mrope {
            anyhow::ensure!(
                plan.prompt_pos4.len().is_multiple_of(4),
                "Qwen3.8 MTP multimodal position table has {} values",
                plan.prompt_pos4.len(),
            );
            anyhow::ensure!(
                self.cfg.rope_sections.iter().sum::<u32>() > 0,
                "Qwen3.8 MTP multimodal RoPE sections are empty"
            );
        }
        let catch = self
            .catch
            .buffers
            .lock()
            .map_err(|_| anyhow!("Qwen3.8 MTP catch workspace poisoned"))?;
        for (chunk, token_rows) in tokens.chunks(catch.max_batch).enumerate() {
            let off = chunk * catch.max_batch;
            let rows = token_rows.len();
            let pos = start_pos + off;
            if pos + rows > self.max_ctx {
                bail!("Qwen3.8 MTP catch-up exceeds {} token cache", self.max_ctx);
            }
            let kv = self.kv_for_depth(be, pos + rows)?;
            let ids = token_rows
                .iter()
                .map(|&token| token as i32)
                .collect::<Vec<_>>();
            let positions4 = if let Some(plan) = mrope {
                let prompt_rows = plan.prompt_pos4.len() / 4;
                (pos..pos + rows)
                    .map(|position| {
                        if position < prompt_rows {
                            Ok(plan.prompt_pos4[position * 4..position * 4 + 4]
                                .try_into()
                                .expect("MRoPE row has four positions"))
                        } else {
                            let offset = i32::try_from(position - prompt_rows)
                                .map_err(|_| anyhow!("Qwen3.8 MTP decode position exceeds i32"))?;
                            let position = plan
                                .decode_base
                                .checked_add(offset)
                                .ok_or_else(|| anyhow!("Qwen3.8 MTP decode position overflow"))?;
                            Ok([position, position, position, 0])
                        }
                    })
                    .collect::<Result<Vec<[i32; 4]>>>()?
            } else {
                Vec::new()
            };
            let positions = if positions4.is_empty() {
                (pos as i32..(pos + rows) as i32).collect::<Vec<_>>()
            } else {
                positions4.iter().map(|row| row[0]).collect::<Vec<_>>()
            };
            let mut override_ranges = Vec::new();
            let mut override_values = Vec::new();
            if let Some(plan) = mrope {
                for (index, span) in plan.spans.iter().enumerate() {
                    let span_end = span.start.checked_add(span.n_tokens).ok_or_else(|| {
                        anyhow!("Qwen3.8 MTP image span #{index} overflows token indices")
                    })?;
                    anyhow::ensure!(
                        span.embeds.len() == span.n_tokens * self.cfg.n_embd,
                        "Qwen3.8 MTP image span #{index} has {} embedding values, expected {}",
                        span.embeds.len(),
                        span.n_tokens * self.cfg.n_embd,
                    );
                    let lo = span.start.max(pos);
                    let hi = span_end.min(pos + rows);
                    if lo >= hi {
                        continue;
                    }
                    let source_start = (lo - span.start) * self.cfg.n_embd;
                    let source_end = (hi - span.start) * self.cfg.n_embd;
                    override_ranges.push((lo - pos, hi - lo));
                    override_values.extend_from_slice(&span.embeds[source_start..source_end]);
                }
            }
            be.upload(catch.ids.as_ref(), bytemuck::cast_slice(&ids))
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(
                catch.h.as_ref(),
                bytemuck::cast_slice(&h[off * hcw..(off + rows) * hcw]),
            )
            .map_err(|e| anyhow!("{e}"))?;
            be.upload(catch.positions.as_ref(), bytemuck::cast_slice(&positions))
                .map_err(|e| anyhow!("{e}"))?;
            if mrope.is_some() {
                be.upload(catch.positions4.as_ref(), bytemuck::cast_slice(&positions4))
                    .map_err(|e| anyhow!("{e}"))?;
            }
            if !override_values.is_empty() {
                be.upload(
                    catch.embedding_overrides.as_ref(),
                    bytemuck::cast_slice(&override_values),
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
            let specs = [(shared.0 .1, shared.0 .2), (shared.1 .1, shared.1 .2)];
            let (graph, handles) = build_catch_graph(
                &self.cfg,
                &self.fixed.specs,
                specs,
                self.kv_capacity,
                rows,
                pos,
                &override_ranges,
                mrope.is_some(),
                self.fixed.fuse_down_inject,
            );
            let plan = be.compile(&graph).map_err(|e| anyhow!("{e}"))?;
            let mut bindings = Bindings::new();
            bindings.bind(handles.ids, catch.ids.as_ref());
            bindings.bind(handles.h, catch.h.as_ref());
            bindings.bind(handles.positions, catch.positions.as_ref());
            if let Some(id) = handles.positions4 {
                bindings.bind(id, catch.positions4.as_ref());
            }
            if let Some(id) = handles.embedding_overrides {
                bindings.bind(id, catch.embedding_overrides.as_ref());
            }
            bindings.bind(handles.k_cache, kv.k.as_ref());
            bindings.bind(handles.v_cache, kv.v.as_ref());
            self.bind_common(
                &mut bindings,
                &handles.weights,
                handles.embd,
                handles.lm_head,
                shared.0 .0,
                shared.1 .0,
            );
            be.execute(plan.as_ref(), &bindings)
                .map_err(|e| anyhow!("{e}"))?;
        }
        Ok(())
    }

    fn begin_device_prime(&self, be: &dyn Backend, previous_h: &[f32]) -> Result<()> {
        anyhow::ensure!(
            previous_h.len() == self.h_width(),
            "Qwen3.8 MTP device prime got {} previous-hidden values, expected {}",
            previous_h.len(),
            self.h_width(),
        );
        be.upload(self.prime_last_h.as_ref(), bytemuck::cast_slice(previous_h))
            .map_err(|e| anyhow!("{e}"))
    }

    fn device_prime_last_h(&self, be: &dyn Backend) -> Result<Vec<f32>> {
        let mut last_h = vec![0.0f32; self.h_width()];
        be.download(
            self.prime_last_h.as_ref(),
            bytemuck::cast_slice_mut(&mut last_h),
        )
        .map_err(|e| anyhow!("{e}"))?;
        Ok(last_h)
    }

    #[allow(clippy::too_many_arguments)]
    fn device_catch_plan(
        &self,
        be: &dyn Backend,
        rows: usize,
        start_pos: usize,
        embedding_overrides: &[(usize, usize)],
        mrope: bool,
        shared: [(DType, usize); 2],
    ) -> Result<Arc<DeviceCatchPlan>> {
        let key = DeviceCatchPlanKey {
            rows,
            start_pos,
            kv_capacity: self.kv_capacity,
            mrope,
            embedding_overrides: embedding_overrides.to_vec(),
            shared,
        };
        {
            let plans = self
                .fixed
                .device_catch_plans
                .lock()
                .map_err(|_| anyhow!("Qwen3.8 MTP device catch plan cache poisoned"))?;
            if let Some(plan) = plans.iter().find(|plan| plan.key == key) {
                return Ok(Arc::clone(plan));
            }
        }

        let (graph, handles) = build_device_catch_graph(
            &self.cfg,
            &self.fixed.specs,
            shared,
            self.kv_capacity,
            rows,
            start_pos,
            embedding_overrides,
            mrope,
            self.fixed.fuse_down_inject,
        );
        let compiled = Arc::new(DeviceCatchPlan {
            key,
            plan: Arc::from(be.compile(&graph).map_err(|e| anyhow!("{e}"))?),
            handles,
        });

        let mut plans = self
            .fixed
            .device_catch_plans
            .lock()
            .map_err(|_| anyhow!("Qwen3.8 MTP device catch plan cache poisoned"))?;
        if let Some(plan) = plans.iter().find(|plan| plan.key == compiled.key) {
            return Ok(Arc::clone(plan));
        }
        push_bounded(&mut plans, Arc::clone(&compiled), DEVICE_CATCH_PLAN_CACHE);
        Ok(compiled)
    }

    #[allow(clippy::too_many_arguments)]
    fn catch_up_from_target(
        &self,
        be: &dyn Backend,
        tokens: &[u32],
        target_h: &dyn Buffer,
        start_pos: usize,
        mrope: Option<&crate::seam::MropePlan>,
        shared: SharedWeights<'_>,
    ) -> Result<()> {
        let rows = tokens.len();
        anyhow::ensure!(
            rows > 0,
            "Qwen3.8 MTP device catch-up needs at least one row"
        );
        let hcw = self.h_width();
        let hidden_bytes = rows
            .checked_mul(hcw)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| anyhow!("Qwen3.8 MTP device hidden byte count overflow"))?;
        anyhow::ensure!(
            target_h.len_bytes() >= hidden_bytes,
            "Qwen3.8 MTP device catch-up needs {hidden_bytes} hidden bytes, buffer has {}",
            target_h.len_bytes(),
        );
        anyhow::ensure!(
            start_pos + rows <= self.max_ctx,
            "Qwen3.8 MTP catch-up exceeds {} token cache",
            self.max_ctx,
        );
        if let Some(plan) = mrope {
            anyhow::ensure!(
                plan.prompt_pos4.len().is_multiple_of(4),
                "Qwen3.8 MTP multimodal position table has {} values",
                plan.prompt_pos4.len(),
            );
        }

        let catch = self
            .catch
            .buffers
            .lock()
            .map_err(|_| anyhow!("Qwen3.8 MTP catch workspace poisoned"))?;
        anyhow::ensure!(
            rows <= catch.max_batch,
            "Qwen3.8 MTP device catch-up batch {rows} exceeds workspace {}",
            catch.max_batch,
        );
        let positions4 = if let Some(plan) = mrope {
            let prompt_rows = plan.prompt_pos4.len() / 4;
            (start_pos..start_pos + rows)
                .map(|position| {
                    if position < prompt_rows {
                        Ok(plan.prompt_pos4[position * 4..position * 4 + 4]
                            .try_into()
                            .expect("MRoPE row has four positions"))
                    } else {
                        let offset = i32::try_from(position - prompt_rows)
                            .map_err(|_| anyhow!("Qwen3.8 MTP decode position exceeds i32"))?;
                        let position = plan
                            .decode_base
                            .checked_add(offset)
                            .ok_or_else(|| anyhow!("Qwen3.8 MTP decode position overflow"))?;
                        Ok([position, position, position, 0])
                    }
                })
                .collect::<Result<Vec<[i32; 4]>>>()?
        } else {
            Vec::new()
        };
        let positions = if positions4.is_empty() {
            (start_pos as i32..(start_pos + rows) as i32).collect::<Vec<_>>()
        } else {
            positions4.iter().map(|row| row[0]).collect::<Vec<_>>()
        };
        let mut override_ranges = Vec::new();
        let mut override_values = Vec::new();
        if let Some(plan) = mrope {
            for (index, span) in plan.spans.iter().enumerate() {
                let span_end = span.start.checked_add(span.n_tokens).ok_or_else(|| {
                    anyhow!("Qwen3.8 MTP image span #{index} overflows token indices")
                })?;
                anyhow::ensure!(
                    span.embeds.len() == span.n_tokens * self.cfg.n_embd,
                    "Qwen3.8 MTP image span #{index} has {} embedding values, expected {}",
                    span.embeds.len(),
                    span.n_tokens * self.cfg.n_embd,
                );
                let lo = span.start.max(start_pos);
                let hi = span_end.min(start_pos + rows);
                if lo >= hi {
                    continue;
                }
                let source_start = (lo - span.start) * self.cfg.n_embd;
                let source_end = (hi - span.start) * self.cfg.n_embd;
                override_ranges.push((lo - start_pos, hi - lo));
                override_values.extend_from_slice(&span.embeds[source_start..source_end]);
            }
        }
        let ids = tokens.iter().map(|&token| token as i32).collect::<Vec<_>>();
        be.upload(catch.ids.as_ref(), bytemuck::cast_slice(&ids))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(catch.positions.as_ref(), bytemuck::cast_slice(&positions))
            .map_err(|e| anyhow!("{e}"))?;
        if mrope.is_some() {
            be.upload(catch.positions4.as_ref(), bytemuck::cast_slice(&positions4))
                .map_err(|e| anyhow!("{e}"))?;
        }
        if !override_values.is_empty() {
            be.upload(
                catch.embedding_overrides.as_ref(),
                bytemuck::cast_slice(&override_values),
            )
            .map_err(|e| anyhow!("{e}"))?;
        }

        let kv = self.kv_for_depth(be, start_pos + rows)?;
        let specs = [(shared.0 .1, shared.0 .2), (shared.1 .1, shared.1 .2)];
        let plan = self.device_catch_plan(
            be,
            rows,
            start_pos,
            &override_ranges,
            mrope.is_some(),
            specs,
        )?;
        let handles = &plan.handles;
        let mut bindings = Bindings::new();
        bindings.bind(handles.ids, catch.ids.as_ref());
        bindings.bind(handles.target_h, target_h);
        bindings.bind(handles.previous_h, self.prime_last_h.as_ref());
        bindings.bind(handles.last_h, self.prime_last_h.as_ref());
        bindings.bind(handles.positions, catch.positions.as_ref());
        if let Some(id) = handles.positions4 {
            bindings.bind(id, catch.positions4.as_ref());
        }
        if let Some(id) = handles.embedding_overrides {
            bindings.bind(id, catch.embedding_overrides.as_ref());
        }
        bindings.bind(handles.k_cache, kv.k.as_ref());
        bindings.bind(handles.v_cache, kv.v.as_ref());
        self.bind_common(
            &mut bindings,
            &handles.weights,
            handles.embd,
            handles.lm_head,
            shared.0 .0,
            shared.1 .0,
        );
        be.execute(plan.plan.as_ref(), &bindings)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Finish one head draft after target verification and delivery. `retained_rows` counts the
    /// actual committed target rows, INCLUDING the pending row (not just accepted candidates).
    /// Use zero on abort/error. Previously-live rows overwritten by the rejected suffix regain
    /// their original K/V bytes even at ring/segment boundaries; unused slots need no restore.
    /// Settle before another draft, catch-up, cache export or release, even with no saved rows.
    pub(crate) fn settle_draft(&self, be: &dyn Backend, retained_rows: usize) -> Result<()> {
        let mut kv = self
            .kv
            .lock()
            .map_err(|_| anyhow!("Qwen3.8 MTP KV state poisoned"))?;
        let Qwen4MtpKvCache {
            k,
            v,
            draft_journal,
            ..
        } = &mut *kv;
        draft_journal.settle(be, k.as_ref(), v.as_ref(), retained_rows)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draft(
        &self,
        be: &dyn Backend,
        token: u32,
        h: &[f32],
        start_pos: usize,
        rope_start_pos: i32,
        verify_tokens: usize,
        shared: SharedWeights<'_>,
    ) -> Result<Vec<u32>> {
        self.draft_with_tail_mode(
            be,
            token,
            h,
            start_pos,
            rope_start_pos,
            verify_tokens,
            shared,
            true,
        )
    }

    /// Same execution/transaction path for runtime A/B; only the final graph step differs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draft_with_tail_mode(
        &self,
        be: &dyn Backend,
        token: u32,
        h: &[f32],
        start_pos: usize,
        rope_start_pos: i32,
        verify_tokens: usize,
        shared: SharedWeights<'_>,
        tail_kv_only: bool,
    ) -> Result<Vec<u32>> {
        self.draft_with_projection(
            be,
            token,
            h,
            start_pos,
            rope_start_pos,
            verify_tokens,
            shared,
            tail_kv_only,
            self.fixed.draft_vocab.as_ref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_with_projection(
        &self,
        be: &dyn Backend,
        token: u32,
        h: &[f32],
        start_pos: usize,
        rope_start_pos: i32,
        verify_tokens: usize,
        shared: SharedWeights<'_>,
        tail_kv_only: bool,
        draft_vocab: Option<&draft_vocab::DraftVocab>,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            (2..=DRAFT_TOKENS).contains(&verify_tokens),
            "Qwen3.8 MTP verify width {verify_tokens} is outside 2..={DRAFT_TOKENS}"
        );
        let hcw = self.cfg.hc_mult * self.cfg.n_embd;
        if h.len() != hcw {
            bail!(
                "Qwen3.8 MTP draft hidden width is {}, expected {hcw}",
                h.len()
            );
        }
        let draft_end = start_pos
            .checked_add(verify_tokens)
            .ok_or_else(|| anyhow!("Qwen3.8 MTP draft position overflow"))?;
        if draft_end > self.max_ctx {
            bail!("Qwen3.8 MTP draft exceeds {} token cache", self.max_ctx);
        }
        let mut kv = self.kv_for_depth(be, draft_end)?;
        be.upload(self.draft_id.as_ref(), bytemuck::bytes_of(&(token as i32)))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(self.draft_h.as_ref(), bytemuck::cast_slice(h))
            .map_err(|e| anyhow!("{e}"))?;
        for (step, buffer) in self.draft_positions[..verify_tokens].iter().enumerate() {
            let step = i32::try_from(step)
                .map_err(|_| anyhow!("Qwen3.8 MTP draft position exceeds i32"))?;
            let position = rope_start_pos
                .checked_add(step)
                .ok_or_else(|| anyhow!("Qwen3.8 MTP draft position overflow"))?;
            be.upload(buffer.as_ref(), bytemuck::bytes_of(&position))
                .map_err(|e| anyhow!("{e}"))?;
        }
        let output_rows = draft_vocab.map_or(self.cfg.vocab, draft_vocab::DraftVocab::rows);
        let id_map = if let Some(vocab) = draft_vocab {
            vocab.id_map.as_ref()
        } else {
            self.fixed
                .draft_identity_map
                .as_deref()
                .ok_or_else(|| anyhow!("MTP draft requires a Vulkan safety identity map"))?
        };
        let specs = [
            (shared.0 .1, shared.0 .2),
            draft_vocab.map_or((shared.1 .1, shared.1 .2), |vocab| {
                (vocab.dtype, vocab.numel)
            }),
        ];
        let build_graph = if tail_kv_only {
            build_draft_graph_vocab_impl::<true>
        } else {
            build_draft_graph_vocab_impl::<false>
        };
        let (graph, handles) = build_graph(
            &self.cfg,
            &self.fixed.specs,
            specs,
            self.kv_capacity,
            self.attention_window,
            start_pos,
            verify_tokens,
            self.fixed.fuse_down_inject,
            Some(output_rows),
        );
        let plan = be.compile(&graph).map_err(|e| anyhow!("{e}"))?;
        {
            let Qwen4MtpKvCache {
                k,
                v,
                draft_journal,
                ..
            } = &mut *kv;
            draft_journal.begin(be, k.as_ref(), v.as_ref(), start_pos, verify_tokens)?;
        }
        let mut bindings = Bindings::new();
        bindings.bind(handles.id, self.draft_id.as_ref());
        bindings.bind(handles.h, self.draft_h.as_ref());
        bindings.bind(handles.k_cache, kv.k.as_ref());
        bindings.bind(handles.v_cache, kv.v.as_ref());
        for (id, buffer) in handles.positions.into_iter().zip(&self.draft_positions) {
            bindings.bind(id, buffer.as_ref());
        }
        for (id, buffer) in handles.raw_ids.into_iter().zip(&self.draft_ids) {
            bindings.bind(id, buffer.as_ref());
        }
        self.bind_common(
            &mut bindings,
            &handles.weights,
            handles.embd,
            handles.lm_head,
            shared.0 .0,
            draft_vocab.map_or(shared.1 .0, |vocab| vocab.weights.as_ref()),
        );
        if let Some(id) = handles.vocab_map {
            bindings.bind(id, id_map);
        }
        let result = (|| -> Result<Vec<u32>> {
            be.execute(plan.as_ref(), &bindings)
                .map_err(|e| anyhow!("{e}"))?;
            let mut result = vec![0u32; verify_tokens - 1];
            for (dst, buffer) in result.iter_mut().zip(&self.draft_ids) {
                be.download(buffer.as_ref(), bytemuck::bytes_of_mut(dst))
                    .map_err(|e| anyhow!("{e}"))?;
            }
            draft_vocab::validate_raw_ids(&result, output_rows)?;
            if let Some(vocab) = draft_vocab {
                for id in &mut result {
                    *id = vocab.original_id(*id);
                }
            }
            Ok(result)
        })();
        drop(bindings);
        if let Err(error) = result {
            let Qwen4MtpKvCache {
                k,
                v,
                draft_journal,
                ..
            } = &mut *kv;
            return match draft_journal.settle(be, k.as_ref(), v.as_ref(), 0) {
                Ok(()) => Err(error),
                Err(restore) => {
                    Err(error.context(format!("MTP draft abort also failed: {restore:#}")))
                }
            };
        }
        result
    }
}

/// Persistent Qwen3.8 MTP state. The detached head is built before the target trunk's first
/// forward, so its weights, KV and staging buffers are committed before the target finalizes its
/// unified expert arena. The target slot is retained too; generation resets only its logical token
/// history, not the uploaded weights or fixed runtime buffers.
pub(crate) struct Qwen4MtpRuntime {
    head: Qwen4MtpSession,
    trunk: Option<crate::seam::SeamKv>,
    max_ctx: usize,
}

impl Qwen4MtpRuntime {
    pub(crate) fn new_vulkan(
        vk: &infr_vulkan::VulkanBackend,
        model: &crate::SeamModel,
        sidecar_path: &Path,
        max_ctx: usize,
    ) -> Result<Self> {
        let catch_batch = crate::seam::ubatch_rows(model.engine_cfg());
        let fixed = Qwen4MtpSession::load_fixed_vulkan_with_vocab(
            vk,
            sidecar_path,
            model.config(),
            model.gguf(),
            model.engine_cfg().spec.mtp_vocab.as_deref(),
        )?;
        let head = Qwen4MtpSession::with_fixed(
            vk,
            fixed,
            max_ctx,
            catch_batch,
            model.engine_cfg().spec.mtp_context,
        )?;
        Ok(Self {
            head,
            trunk: None,
            max_ctx,
        })
    }

    pub(crate) fn reset(&mut self) {
        if let Some(trunk) = self.trunk.as_mut() {
            trunk.reset();
        }
    }

    #[cfg_attr(infr_profile, infr_prof::instrument)]
    pub(crate) fn generate_vulkan(
        &mut self,
        vk: &infr_vulkan::VulkanBackend,
        model: &crate::SeamModel,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: impl FnMut(&str),
    ) -> Result<(crate::GenStats, super::MtpTiming)> {
        self.head.settle_draft(vk, 0)?;
        let result = self.generate_vulkan_inner(vk, model, prompt, max_new, req, on_piece);
        if let Err(error) = result {
            return match self.head.settle_draft(vk, 0) {
                Ok(()) => Err(error),
                Err(restore) => {
                    Err(error.context(format!("MTP generation abort also failed: {restore:#}")))
                }
            };
        }
        result
    }

    fn generate_vulkan_inner(
        &mut self,
        vk: &infr_vulkan::VulkanBackend,
        model: &crate::SeamModel,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        mut on_piece: impl FnMut(&str),
    ) -> Result<(crate::GenStats, super::MtpTiming)> {
        let cfg = model.config();
        let ec = model.engine_cfg();
        // The fixed MTP runtime supports two through four VERIFY rows. Keep the configured width
        // effective so kernel improvements can change the best batch size without a hidden cap.
        let verify_tokens = ec.spec.k.clamp(2, DRAFT_TOKENS);
        let h_width = cfg.hc_mult * cfg.n_embd;
        let prompt_tokens = model.encode(prompt)?;
        if prompt_tokens.is_empty() {
            bail!("Qwen3.8 MTP received an empty prompt");
        }
        // `max_new` is a ceiling, not a capacity demand. Match ordinary decode by clipping it to
        // the remaining window while retaining the four rows a full VERIFY cycle may touch.
        let max_new = generation_budget(prompt_tokens.len(), max_new, self.max_ctx)?;
        self.reset();

        let t_prime = std::time::Instant::now();
        let (target_bind, finish_fixed_allocations) = crate::seam::vulkan_moe_binder(
            vk,
            model.gguf(),
            cfg,
            ec,
            self.trunk.is_none(),
            self.max_ctx,
        )?;
        let p = prompt_tokens.len();
        let mut reasoning_eos_guard =
            crate::sampling::Qwen4ReasoningEosGuard::from_prompt(cfg, &prompt_tokens);
        let initial_h = vec![0.0f32; h_width];
        let mut prime_sink = Qwen4MtpPrimeSink::new(
            &self.head,
            &initial_h,
            [None; crate::seam::TURN_CHECKPOINT_COUNT],
            None,
        )?;
        let (mut pending_token, mut pending_logits) = super::run_qwen4_prime_frontier_with_finish(
            vk,
            &*target_bind,
            model.gguf(),
            cfg,
            ec,
            model.embd(),
            &prompt_tokens,
            &mut self.trunk,
            self.max_ctx,
            None,
            finish_fixed_allocations.as_deref(),
            &mut prime_sink,
            None,
        )?;
        let prime_capture = prime_sink.finish(vk)?;
        if reasoning_eos_guard.blocks(cfg, pending_token, ec.sampling.ignore_eos) {
            anyhow::ensure!(
                pending_logits.len() == cfg.vocab,
                "Qwen3.8 MTP prime EOS repair expected {} logits, got {}",
                cfg.vocab,
                pending_logits.len()
            );
            reasoning_eos_guard.mask_eos(cfg, &mut pending_logits);
            let blocked = pending_token;
            pending_token = super::argmax_row(&pending_logits)?;
            tracing::warn!(
                blocked_token = blocked,
                replacement_token = pending_token,
                "Qwen3.8 MTP suppressed premature EOS inside an open reasoning block"
            );
        }
        self.trunk
            .as_mut()
            .expect("target prime initialized a trunk")
            .mtp_snapshot_delta(vk, cfg)?;
        let prompt_secs = t_prime.elapsed().as_secs_f64();

        let mut committed = prompt_tokens;
        // Keep the target's already-known next token one row ahead of the committed trunk state.
        // The next VERIFY consumes it as row zero, so every batch commits at least one token and
        // rejection never needs a separate correction forward or MTP-head catch-up.
        let mut pending_h = prime_capture.last_h;
        let mut feed = Vec::new();
        let mut acc = Vec::new();
        let mut printed = 0usize;
        let mut generated = 0usize;
        let mut timing = super::MtpTiming::default();
        let mut cycle = 0usize;
        let t_decode = std::time::Instant::now();
        let hit_eos = |token: u32| {
            !ec.sampling.ignore_eos && (cfg.eos_ids.contains(&token) || token == cfg.eos)
        };

        'decode: while generated < max_new && !req.is_some_and(crate::sampling::RequestCtx::aborted)
        {
            cycle += 1;
            let n_past = committed.len();

            let t_draft = std::time::Instant::now();
            let candidates = {
                let _profile = PhaseProfile::new("draft");
                let shared = self
                    .trunk
                    .as_ref()
                    .expect("target trunk remains initialized")
                    .mtp_shared_weights();
                self.head.draft_with_tail_mode(
                    vk,
                    pending_token,
                    &pending_h,
                    n_past,
                    i32::try_from(n_past)
                        .map_err(|_| anyhow!("Qwen3.8 MTP position exceeds i32"))?,
                    verify_tokens,
                    shared,
                    ec.spec.mtp_tail_kv_only,
                )?
            };
            let draft_secs = t_draft.elapsed().as_secs_f64();
            timing.draft_secs += draft_secs;
            timing.total_drafted += verify_tokens - 1;

            refill_verify_feed(
                &mut feed,
                &committed,
                pending_token,
                &candidates[..verify_tokens - 1],
            );
            self.trunk
                .as_mut()
                .expect("target trunk remains initialized")
                .mtp_arm_delta_trace(verify_tokens)?;
            let t_verify = std::time::Instant::now();
            let verify_profile = PhaseProfile::new("verify");
            let (target_bind, finish_fixed_allocations) = crate::seam::vulkan_moe_binder(
                vk,
                model.gguf(),
                cfg,
                ec,
                self.trunk.is_none(),
                self.max_ctx,
            )?;
            let (mut verify_ids, mut verify_logits, verify_h) = super::run_verify_with_finish(
                vk,
                &*target_bind,
                model.gguf(),
                cfg,
                ec,
                model.embd(),
                &feed,
                &mut self.trunk,
                self.max_ctx,
                finish_fixed_allocations.as_deref(),
            )?;
            let verify_secs = t_verify.elapsed().as_secs_f64();
            drop(verify_profile);
            timing.verify_secs += verify_secs;
            anyhow::ensure!(
                verify_ids.len() == verify_tokens
                    && verify_h.len() == verify_tokens * h_width,
                "Qwen3.8 MTP VERIFY must return exactly {verify_tokens} rows; got {} ids and {} hidden values",
                verify_ids.len(),
                verify_h.len()
            );

            let mut verify_guard = reasoning_eos_guard;
            for row in 0..verify_tokens {
                verify_guard.observe(cfg, feed[n_past + row]);
                if !verify_guard.blocks(cfg, verify_ids[row], ec.sampling.ignore_eos) {
                    continue;
                }
                anyhow::ensure!(
                    verify_logits.len() == verify_tokens * cfg.vocab,
                    "Qwen3.8 MTP EOS repair expected {} logits, got {}",
                    verify_tokens * cfg.vocab,
                    verify_logits.len()
                );
                let logits = &mut verify_logits[row * cfg.vocab..(row + 1) * cfg.vocab];
                verify_guard.mask_eos(cfg, logits);
                let blocked = verify_ids[row];
                verify_ids[row] = super::argmax_row(logits)?;
                tracing::warn!(
                    row,
                    blocked_token = blocked,
                    replacement_token = verify_ids[row],
                    "Qwen3.8 MTP suppressed premature EOS inside an open reasoning block"
                );
            }

            let accepted_spec = (0..verify_tokens - 1)
                .take_while(|&i| candidates[i] == verify_ids[i])
                .count();
            let accepted = accepted_spec + 1;
            timing.total_accepted += accepted_spec;
            // Retain only the deliverable prefix; EOS itself occupies one committed row.
            let retained = retained_draft_rows(
                &feed[n_past..n_past + accepted],
                max_new - generated,
                &hit_eos,
            );

            let t_catchup = std::time::Instant::now();
            if retained < verify_tokens {
                let trunk = self
                    .trunk
                    .as_mut()
                    .expect("target trunk remains initialized");
                let restore_profile = PhaseProfile::new("restore");
                trunk.mtp_restore_delta_row(vk, retained)?;
                drop(restore_profile);
            }
            let new_tokens = &feed[n_past..n_past + retained];
            self.head.settle_draft(vk, retained)?;
            committed.extend_from_slice(new_tokens);
            pending_token = verify_ids[retained - 1];
            pending_h.copy_from_slice(&verify_h[(retained - 1) * h_width..retained * h_width]);
            let snapshot_profile = PhaseProfile::new("snapshot");
            self.trunk
                .as_mut()
                .expect("target trunk remains initialized")
                .mtp_snapshot_delta(vk, cfg)?;
            drop(snapshot_profile);
            let emitted = new_tokens.to_vec();
            let catchup_secs = t_catchup.elapsed().as_secs_f64();
            timing.catchup_secs += catchup_secs;

            if ec.prof.stages || infr_core::pager_profile::active() {
                tracing::info!(
                    "[qwen4 mtp cycle {cycle}] drafted={} accepted={accepted_spec} committed={retained} draft={:.1}ms verify={:.1}ms catchup={:.1}ms",
                    verify_tokens - 1,
                    draft_secs * 1e3,
                    verify_secs * 1e3,
                    catchup_secs * 1e3,
                );
            }
            for token in emitted {
                let eos = hit_eos(token);
                generated += 1;
                reasoning_eos_guard.observe(cfg, token);
                if !eos {
                    crate::stream_token(
                        model.tokenizer(),
                        &mut acc,
                        &mut printed,
                        token,
                        &mut on_piece,
                    );
                }
                if eos || generated >= max_new {
                    break 'decode;
                }
            }
        }

        if ec.prof.stages || infr_core::pager_profile::active() {
            let (draft_pct, verify_pct, catchup_pct) = timing.phase_shares();
            tracing::info!(
                "[qwen4 mtp summary] {cycle} cycles, {}/{} accepted (alpha={:.3}), {generated} tokens generated, phase share: draft {:.0}% verify {:.0}% catchup {:.0}%",
                timing.total_accepted,
                timing.total_drafted,
                timing.alpha(),
                draft_pct,
                verify_pct,
                catchup_pct,
            );
        }

        Ok((
            crate::GenStats {
                n_prompt: p,
                n_cached: 0,
                prompt_secs,
                n_gen: generated,
                decode_secs: t_decode.elapsed().as_secs_f64(),
            },
            timing,
        ))
    }
}

#[cfg(test)]
#[path = "qwen4_draft_kv_tests.rs"]
mod draft_kv_tests;

#[cfg(test)]
#[path = "qwen4_draft_gpu_tests.rs"]
mod draft_gpu_tests;

#[cfg(test)]
mod tests {
    use super::{
        generation_budget, hidden_shift_copy_lengths, mtp_kv_spec, push_bounded, DeviceCatchPlanKey,
    };
    use infr_core::tensor::DType;
    use std::collections::VecDeque;

    #[test]
    fn generation_budget_clips_a_large_reply_to_the_remaining_context() {
        assert_eq!(
            generation_budget(200_000, 102_400, 262_144).unwrap(),
            62_140
        );
    }

    #[test]
    fn generation_budget_preserves_a_reply_that_fits() {
        assert_eq!(generation_budget(150_000, 512, 262_144).unwrap(), 512);
    }

    #[test]
    fn generation_budget_rejects_a_prompt_without_verify_room() {
        assert!(generation_budget(262_141, 1, 262_144).is_err());
    }

    #[test]
    fn head_kv_ring_uses_independent_32k_f16_segments() {
        let spec = mtp_kv_spec(2 * 256, 32_768).unwrap();
        assert_eq!(spec.logical_bytes, 32_768 * 512 * 2);
        assert_eq!(spec.segment_bytes, 32 * 1024 * 512 * 2);
        assert_eq!(spec.segment_elements, 32 * 1024 * 512);
        assert_eq!(spec.max_segments, 1);
    }

    #[test]
    fn device_prime_shift_handles_single_and_multirow_chunks() {
        assert_eq!(hidden_shift_copy_lengths(1, 10_240), (10_240, 0));
        assert_eq!(hidden_shift_copy_lengths(4, 10_240), (10_240, 3 * 10_240));
    }

    #[test]
    fn device_catch_plan_key_separates_graph_shapes_and_positions() {
        let base = DeviceCatchPlanKey {
            rows: 2_048,
            start_pos: 32_768,
            kv_capacity: 65_536,
            mrope: false,
            embedding_overrides: Vec::new(),
            shared: [(DType::F16, 2_560), (DType::Q8_0, 2_560)],
        };
        let mut changed = base.clone();
        changed.rows = 1;
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.start_pos += 2_048;
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.mrope = true;
        changed.embedding_overrides = vec![(5, 32)];
        assert_ne!(base, changed);
        let mut changed = base.clone();
        changed.shared[1].0 = DType::F16;
        assert_ne!(base, changed);
    }

    #[test]
    fn bounded_plan_cache_evicts_the_oldest_entry() {
        let mut cache = VecDeque::from([1, 2, 3]);
        push_bounded(&mut cache, 4, 3);
        assert_eq!(cache, VecDeque::from([2, 3, 4]));
        push_bounded(&mut cache, 5, 0);
        assert_eq!(cache, VecDeque::from([2, 3, 4]));
    }
}

#[cfg(test)]
mod m01_local_tests {
    use super::refill_verify_feed;
    use std::hint::black_box;

    fn hidden_rows(width: usize) -> Vec<f32> {
        let bits = [
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7fc1_2345,
            0xffc5_4321,
        ];
        (0..4 * width)
            .map(|i| f32::from_bits(bits[i % bits.len()] ^ ((i / width) as u32 * 0x100)))
            .collect()
    }

    fn update_hidden(
        pending_h: &mut Vec<f32>,
        verify_h: &[f32],
        accepted: usize,
        h_width: usize,
        reuse: bool,
    ) {
        if reuse {
            pending_h.copy_from_slice(&verify_h[(accepted - 1) * h_width..accepted * h_width]);
        } else {
            *pending_h = verify_h[(accepted - 1) * h_width..accepted * h_width].to_vec();
        }
    }

    #[test]
    fn m01_feed_matches_original_for_widths_growth_and_shrink() {
        let tokens: Vec<u32> = (0..150_004).collect();
        let candidates = [17, 29, 43];
        let mut feed = Vec::new();
        for n in [0, 1, 30_000, 150_000, 1, 0, 150_000] {
            for verify_tokens in 2..=4 {
                let tail = &candidates[..verify_tokens - 1];
                let mut expected = Vec::with_capacity(n + verify_tokens);
                expected.extend_from_slice(&tokens[..n]);
                expected.push(11);
                expected.extend_from_slice(tail);
                let (ptr, cap) = (feed.as_ptr(), feed.capacity());
                refill_verify_feed(&mut feed, &tokens[..n], 11, tail);
                assert_eq!(feed, expected);
                assert!(feed.capacity() >= n + verify_tokens);
                if cap >= n + verify_tokens {
                    assert_eq!(feed.capacity(), cap);
                    assert_eq!(feed.as_ptr(), ptr);
                }
            }
        }
    }

    #[test]
    fn m01_hidden_copy_preserves_every_bit_and_allocation_for_accepted_rows() {
        for width in [1, 7, 10_240] {
            let rows = hidden_rows(width);
            let mut pending_h = vec![0.0; width];
            let (ptr, cap) = (pending_h.as_ptr(), pending_h.capacity());
            for verify_tokens in 2..=4 {
                for accepted in 1..=verify_tokens {
                    let expected = rows[(accepted - 1) * width..accepted * width].to_vec();
                    update_hidden(&mut pending_h, &rows, accepted, width, true);
                    assert!(pending_h
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()));
                    assert_eq!(pending_h.as_ptr(), ptr);
                    assert_eq!(pending_h.capacity(), cap);
                    assert_eq!(pending_h.len(), width);
                }
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum Event {
        Restore(usize),
        Commit,
        Select,
        Snapshot,
        Observe(u32),
        Emit(u32),
        Stop,
    }

    #[derive(Clone, Copy)]
    enum Abort {
        Never,
        BeforeCycle,
        AfterFirstEmission,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Cycle {
        committed: Vec<u32>,
        pending: u32,
        hidden_bits: Vec<u32>,
        generated: usize,
        aborted: bool,
        events: Vec<Event>,
    }

    // A selector/order contract, not an implementation of the real EOS guard or GPU rollback.
    fn cycle_contract(
        reuse: bool,
        verify_tokens: usize,
        verify_ids: &[u32],
        eos: Option<u32>,
        ignore_eos: bool,
        limit: usize,
        abort: Abort,
    ) -> Cycle {
        let mut committed = vec![1, 2, 3];
        let mut pending_token = 10;
        let mut pending_h = vec![0.0; 7];
        let verify_h = hidden_rows(7);
        let h_width = 7;
        let mut events = Vec::new();
        let mut generated = 0;
        let mut aborted = matches!(abort, Abort::BeforeCycle);
        if limit > 0 && !aborted {
            let candidates = [11, 12, 13];
            let n_past = committed.len();
            let mut feed = Vec::new();
            if reuse {
                refill_verify_feed(
                    &mut feed,
                    &committed,
                    pending_token,
                    &candidates[..verify_tokens - 1],
                );
            } else {
                feed = Vec::with_capacity(committed.len() + verify_tokens);
                feed.extend_from_slice(&committed);
                feed.push(pending_token);
                feed.extend_from_slice(&candidates[..verify_tokens - 1]);
            }
            // verify_ids is the already-repaired target selector, just like the production loop.
            let accepted_spec = (0..verify_tokens - 1)
                .take_while(|&i| candidates[i] == verify_ids[i])
                .count();
            let accepted = accepted_spec + 1;
            if accepted < verify_tokens {
                events.push(Event::Restore(accepted));
            }
            let new_tokens = &feed[n_past..n_past + accepted];
            committed.extend_from_slice(new_tokens);
            events.push(Event::Commit);
            pending_token = verify_ids[accepted - 1];
            update_hidden(&mut pending_h, &verify_h, accepted, h_width, reuse);
            events.push(Event::Select);
            events.push(Event::Snapshot);
            let emitted = new_tokens.to_vec();
            for token in emitted {
                let stop_eos = !ignore_eos && eos == Some(token);
                generated += 1;
                events.push(Event::Observe(token));
                if !stop_eos {
                    events.push(Event::Emit(token));
                    if matches!(abort, Abort::AfterFirstEmission) {
                        aborted = true;
                    }
                }
                // Abort is checked at the NEXT cycle, not midway through this accepted batch.
                if stop_eos || generated >= limit {
                    events.push(Event::Stop);
                    break;
                }
            }
        }
        Cycle {
            committed,
            pending: pending_token,
            hidden_bits: pending_h.iter().map(|v| v.to_bits()).collect(),
            generated,
            aborted,
            events,
        }
    }

    #[test]
    fn m01_rejection_eos_abort_and_limit_keep_original_selectors_and_order() {
        for verify_tokens in 2..=4 {
            for accepted in 1..=verify_tokens {
                let mut ids = [11, 12, 13, 77];
                if accepted < verify_tokens {
                    ids[accepted - 1] = 90 + accepted as u32;
                }
                for eos in [None, Some(10), Some(11), Some(12), Some(13)] {
                    for ignore_eos in [false, true] {
                        for limit in [0, 1, 2, 3, 4, 8] {
                            for abort in
                                [Abort::Never, Abort::BeforeCycle, Abort::AfterFirstEmission]
                            {
                                let old = cycle_contract(
                                    false,
                                    verify_tokens,
                                    &ids,
                                    eos,
                                    ignore_eos,
                                    limit,
                                    abort,
                                );
                                let new = cycle_contract(
                                    true,
                                    verify_tokens,
                                    &ids,
                                    eos,
                                    ignore_eos,
                                    limit,
                                    abort,
                                );
                                assert_eq!(new, old);
                                if limit > 0 && !matches!(abort, Abort::BeforeCycle) {
                                    assert_eq!(new.pending, ids[accepted - 1]);
                                    assert_eq!(new.committed.len(), 3 + accepted);
                                    let first = usize::from(accepted < verify_tokens);
                                    assert_eq!(
                                        new.events[first..first + 3],
                                        [Event::Commit, Event::Select, Event::Snapshot]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn m01_repaired_target_ids_and_deferred_pending_eos_are_selected_before_emission() {
        let raw = cycle_contract(true, 4, &[11, 99, 13, 77], None, false, 8, Abort::Never);
        let repaired = cycle_contract(true, 4, &[11, 12, 13, 99], Some(99), false, 8, Abort::Never);
        assert_eq!(raw.committed.len(), 5);
        assert_eq!(repaired.committed.len(), 7);
        assert_eq!(repaired.pending, 99);
        assert_eq!(repaired.generated, 4); // Pending EOS is not yet an emitted token.
        assert!(!repaired.events.contains(&Event::Stop));
        let aborted = cycle_contract(
            true,
            4,
            &[11, 12, 13, 77],
            None,
            false,
            8,
            Abort::AfterFirstEmission,
        );
        assert!(aborted.aborted);
        assert_eq!(aborted.generated, 4);
    }

    pub(super) struct AllocationCase {
        tokens: Vec<u32>,
        rows: Vec<f32>,
        pending_h: Vec<f32>,
        feed: Vec<u32>,
        prefix: usize,
        cycles: usize,
    }

    impl AllocationCase {
        pub(super) fn new(prefix: usize, cycles: usize) -> Self {
            Self {
                tokens: (0..prefix + cycles + 4).map(|i| i as u32).collect(),
                rows: hidden_rows(10_240),
                pending_h: vec![0.0; 10_240],
                feed: Vec::new(),
                prefix,
                cycles,
            }
        }

        pub(super) fn run(&mut self, reuse: bool) -> usize {
            let mut growth_calls = 0;
            let candidates = [11, 12, 13];
            for i in 0..self.cycles {
                let committed = &self.tokens[..self.prefix + i];
                let accepted = i % 4 + 1;
                if reuse {
                    let cap = self.feed.capacity();
                    refill_verify_feed(&mut self.feed, committed, 10, &candidates);
                    growth_calls += usize::from(self.feed.capacity() != cap);
                    update_hidden(&mut self.pending_h, &self.rows, accepted, 10_240, true);
                    black_box((&self.feed[..], &self.pending_h[..]));
                } else {
                    let mut feed = Vec::with_capacity(committed.len() + 4);
                    feed.extend_from_slice(committed);
                    feed.push(10);
                    feed.extend_from_slice(&candidates);
                    update_hidden(&mut self.pending_h, &self.rows, accepted, 10_240, false);
                    growth_calls += 2; // One nonempty feed allocation and one hidden allocation.
                    black_box((&feed[..], &self.pending_h[..]));
                }
            }
            growth_calls
        }
    }

    #[test]
    fn m01_capacity_growth_counts_exclude_setup_and_hidden_copy_allocations() {
        for prefix in [0, 30_000, 150_000] {
            let cycles = 4096;
            let mut old = AllocationCase::new(prefix, cycles);
            let mut new = AllocationCase::new(prefix, cycles);
            assert_eq!(old.run(false), 2 * cycles);
            let (ptr, cap) = (new.pending_h.as_ptr(), new.pending_h.capacity());
            let growth = new.run(true);
            assert!(growth > 0 && growth < 32);
            assert_eq!(new.pending_h.as_ptr(), ptr);
            assert_eq!(new.pending_h.capacity(), cap);
            assert!(old
                .pending_h
                .iter()
                .zip(&new.pending_h)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_eq!(new.feed.len(), prefix + cycles - 1 + 4);
            // A second run is fully warmed and must not grow either allocation.
            assert_eq!(new.run(true), 0);
        }
    }
}
