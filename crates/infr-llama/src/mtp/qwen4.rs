use anyhow::{anyhow, bail, Context, Result};
use infr_core::backend::{Backend, Bindings, Buffer, BufferUsage};
use infr_core::graph::{Activation, AttnMask, Graph, Op};
use infr_core::tensor::{DType, TensorDesc, TensorId};
use infr_core::{TensorInfo, WeightSource};
use infr_gguf::Gguf;
use std::path::Path;

use super::{BindWeightFn, MtpTensor};

pub const DRAFT_TOKENS: usize = 4;

struct Qwen4MtpWeights {
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
    ones: TensorId,
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
    let ones = next();
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
            ones,
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

fn emit_hc_mix(
    g: &mut Graph,
    cfg: &crate::Config,
    rows: usize,
    weights: HcW,
    ones: TensorId,
    residual: TensorId,
    dst: TensorId,
    scratch: HcScratch,
) {
    let ne = cfg.n_embd;
    let hcw = cfg.hc_mult * ne;
    g.push(Op::RmsNorm {
        x: residual,
        weight: ones,
        dst: scratch.normed,
        rows: (rows * cfg.hc_mult) as u32,
        dim: ne as u32,
        eps: cfg.rms_eps,
    });
    g.push(Op::MulVec {
        x: scratch.normed,
        vec: weights.norm,
        dst: scratch.normed,
        rows: rows as u32,
        n: hcw as u32,
    });
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
    if let Some(inject) = weights.inject {
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

fn emit_bridge(
    g: &mut Graph,
    cfg: &crate::Config,
    rows: usize,
    ids: TensorId,
    h_in: TensorId,
    weights: &GraphW,
    scratch: &StepScratch,
) {
    let ne = cfg.n_embd;
    let hcw = cfg.hc_mult * ne;
    g.push(Op::EmbedGather {
        ids,
        table: weights.embd,
        dst: scratch.e,
        rows: rows as u32,
        ne: ne as u32,
        scale: 1.0,
    });
    g.push(Op::RmsNorm {
        x: scratch.e,
        weight: weights.enorm,
        dst: scratch.e_norm,
        rows: rows as u32,
        dim: ne as u32,
        eps: cfg.rms_eps,
    });
    g.push(Op::RmsNorm {
        x: h_in,
        weight: weights.ones,
        dst: scratch.h_norm,
        rows: (rows * cfg.hc_mult) as u32,
        dim: ne as u32,
        eps: cfg.rms_eps,
    });
    g.push(Op::MulVec {
        x: scratch.h_norm,
        vec: weights.hnorm,
        dst: scratch.h_norm,
        rows: rows as u32,
        n: hcw as u32,
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

fn emit_attention_kv(
    g: &mut Graph,
    cfg: &crate::Config,
    rows: usize,
    start_pos: usize,
    positions: TensorId,
    k_cache: TensorId,
    v_cache: TensorId,
    weights: &GraphW,
    scratch: &StepScratch,
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
        weights.ones,
        scratch.residual,
        scratch.hidden,
        HcScratch {
            normed: scratch.hc.normed,
            low: scratch.hc.low,
            gate: scratch.hc.gate,
            inject: scratch.hc.inject,
        },
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

fn emit_full_step(
    g: &mut Graph,
    cfg: &crate::Config,
    start_pos: usize,
    positions: TensorId,
    k_cache: TensorId,
    v_cache: TensorId,
    weights: &GraphW,
    scratch: &StepScratch,
    h_next: TensorId,
) {
    let ne = cfg.n_embd;
    let hcw = cfg.hc_mult * ne;
    let qrow = cfg.n_head * cfg.head_dim;
    let moe = cfg.moe.expect("validated Qwen3.8 MoE config");
    emit_attention_kv(
        g, cfg, 1, start_pos, positions, k_cache, v_cache, weights, scratch, true,
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
        mask: AttnMask::Causal,
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
    g.push(Op::QwenHcInject {
        residual: scratch.residual,
        block: scratch.block,
        gate: scratch.hc.inject,
        dst: scratch.alt,
        rows: 1,
        hc: cfg.hc_mult as u32,
        n_embd: ne as u32,
    });
    emit_hc_mix(
        g,
        cfg,
        1,
        weights.ffn_hc,
        weights.ones,
        scratch.alt,
        scratch.hidden,
        HcScratch {
            normed: scratch.hc.normed,
            low: scratch.hc.low,
            gate: scratch.hc.gate,
            inject: scratch.hc.inject,
        },
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
    k_cache: TensorId,
    v_cache: TensorId,
    weights: Vec<TensorId>,
    embd: TensorId,
    lm_head: TensorId,
}

fn build_catch_graph(
    cfg: &crate::Config,
    specs: &[(DType, usize)],
    shared: [(DType, usize); 2],
    max_ctx: usize,
    rows: usize,
    start_pos: usize,
) -> (Graph, CatchHandles) {
    let mut g = Graph::new();
    let hcw = cfg.hc_mult * cfg.n_embd;
    let kvrow = cfg.n_kv * cfg.head_dim;
    let ids = g.input(TensorDesc::new(vec![rows], DType::I32));
    let h = g.input(TensorDesc::new(vec![rows * hcw], DType::F32));
    let positions = g.input(TensorDesc::new(vec![rows], DType::I32));
    let k_cache = g.input(TensorDesc::new(vec![max_ctx * kvrow], DType::F16));
    let v_cache = g.input(TensorDesc::new(vec![max_ctx * kvrow], DType::F16));
    let (weights, handles) = declare_weights(&mut g, specs, shared[0], shared[1]);
    let scratch = step_scratch(&mut g, cfg, rows);
    emit_bridge(&mut g, cfg, rows, ids, h, &weights, &scratch);
    emit_attention_kv(
        &mut g, cfg, rows, start_pos, positions, k_cache, v_cache, &weights, &scratch, false,
    );
    (
        g,
        CatchHandles {
            ids,
            h,
            positions,
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
    positions: [TensorId; DRAFT_TOKENS],
    k_cache: TensorId,
    v_cache: TensorId,
    weights: Vec<TensorId>,
    embd: TensorId,
    lm_head: TensorId,
    ids: [TensorId; DRAFT_TOKENS],
}

fn build_draft_graph(
    cfg: &crate::Config,
    specs: &[(DType, usize)],
    shared: [(DType, usize); 2],
    max_ctx: usize,
    start_pos: usize,
) -> (Graph, DraftHandles) {
    let mut g = Graph::new();
    let f32d = |n| TensorDesc::new(vec![n], DType::F32);
    let hcw = cfg.hc_mult * cfg.n_embd;
    let kvrow = cfg.n_kv * cfg.head_dim;
    let id = g.input(TensorDesc::new(vec![1], DType::I32));
    let h = g.input(f32d(hcw));
    let positions = std::array::from_fn(|_| g.input(TensorDesc::new(vec![1], DType::I32)));
    let k_cache = g.input(TensorDesc::new(vec![max_ctx * kvrow], DType::F16));
    let v_cache = g.input(TensorDesc::new(vec![max_ctx * kvrow], DType::F16));
    let (weights, handles) = declare_weights(&mut g, specs, shared[0], shared[1]);
    let scratch = step_scratch(&mut g, cfg, 1);
    let mut prev_id = id;
    let mut prev_h = h;
    let mut out_ids = Vec::with_capacity(DRAFT_TOKENS);
    for step in 0..DRAFT_TOKENS {
        emit_bridge(&mut g, cfg, 1, prev_id, prev_h, &weights, &scratch);
        let h_next = g.internal(f32d(hcw));
        emit_full_step(
            &mut g,
            cfg,
            start_pos + step,
            positions[step],
            k_cache,
            v_cache,
            &weights,
            &scratch,
            h_next,
        );
        let head_hidden = g.internal(f32d(cfg.n_embd));
        emit_hc_mix(
            &mut g,
            cfg,
            1,
            weights.head_hc,
            weights.ones,
            h_next,
            head_hidden,
            HcScratch {
                normed: scratch.hc.normed,
                low: scratch.hc.low,
                gate: scratch.hc.gate,
                inject: scratch.hc.inject,
            },
        );
        let logits = g.internal(f32d(cfg.vocab));
        g.push(Op::Linear {
            x: head_hidden,
            weight: weights.lm_head,
            dst: logits,
            m: 1,
            in_f: cfg.n_embd as u32,
            out_f: cfg.vocab as u32,
            w_off: 0,
        });
        let next_id = g.output(f32d(1));
        g.push(Op::Argmax {
            x: logits,
            dst: next_id,
            n: cfg.vocab as u32,
            rows: 1,
        });
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
            ids: out_ids.try_into().expect("fixed four draft ids"),
        },
    )
}

pub(crate) struct Qwen4MtpSession {
    cfg: crate::Config,
    max_ctx: usize,
    max_batch: usize,
    weights: Vec<Box<dyn Buffer>>,
    specs: Vec<(DType, usize)>,
    k_cache: Box<dyn Buffer>,
    v_cache: Box<dyn Buffer>,
    catch_ids: Box<dyn Buffer>,
    catch_h: Box<dyn Buffer>,
    catch_positions: Box<dyn Buffer>,
    draft_id: Box<dyn Buffer>,
    draft_h: Box<dyn Buffer>,
    draft_positions: [Box<dyn Buffer>; DRAFT_TOKENS],
    draft_ids: [Box<dyn Buffer>; DRAFT_TOKENS],
}

impl Qwen4MtpSession {
    pub(crate) fn new(
        be: &dyn Backend,
        bind: &BindWeightFn,
        sidecar_path: &Path,
        cfg: &crate::Config,
        max_ctx: usize,
        max_batch: usize,
    ) -> Result<Self> {
        let sidecar = Gguf::open(sidecar_path)
            .with_context(|| format!("open Qwen3.8 MTP sidecar {}", sidecar_path.display()))?;
        let found = Qwen4MtpWeights::load(&sidecar, cfg)?;
        let mut weights = Vec::with_capacity(29);
        let mut specs = Vec::with_capacity(29);
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
        let ones = vec![1.0f32; cfg.n_embd];
        let ones_buf = be
            .alloc(ones.len() * 4, BufferUsage::Weights)
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(ones_buf.as_ref(), bytemuck::cast_slice(&ones))
            .map_err(|e| anyhow!("{e}"))?;
        weights.push(ones_buf);
        specs.push((DType::F32, ones.len()));

        let kvrow = cfg.n_kv * cfg.head_dim;
        let hcw = cfg.hc_mult * cfg.n_embd;
        let max_batch = max_batch.max(DRAFT_TOKENS).min(max_ctx.max(1));
        let alloc = |bytes, usage| be.alloc(bytes, usage).map_err(|e| anyhow!("{e}"));
        let k_cache = alloc(max_ctx * kvrow * 2, BufferUsage::KvCache)?;
        let v_cache = alloc(max_ctx * kvrow * 2, BufferUsage::KvCache)?;
        let catch_ids = alloc(max_batch * 4, BufferUsage::Staging)?;
        let catch_h = alloc(max_batch * hcw * 4, BufferUsage::Staging)?;
        let catch_positions = alloc(max_batch * 4, BufferUsage::Staging)?;
        let draft_id = alloc(4, BufferUsage::Staging)?;
        let draft_h = alloc(hcw * 4, BufferUsage::Staging)?;
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
        tracing::info!(
            path = %sidecar_path.display(),
            draft_tokens = DRAFT_TOKENS,
            max_ctx,
            catch_batch = max_batch,
            "loaded fixed Qwen3.8 MTP sidecar runtime"
        );
        Ok(Self {
            cfg: cfg.clone(),
            max_ctx,
            max_batch,
            weights,
            specs,
            k_cache,
            v_cache,
            catch_ids,
            catch_h,
            catch_positions,
            draft_id,
            draft_h,
            draft_positions,
            draft_ids,
        })
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
        for (id, buffer) in weights.iter().copied().zip(&self.weights) {
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
        shared: ((&dyn Buffer, DType, usize), (&dyn Buffer, DType, usize)),
    ) -> Result<()> {
        let hcw = self.cfg.hc_mult * self.cfg.n_embd;
        if h.len() != tokens.len() * hcw {
            bail!(
                "Qwen3.8 MTP catch-up got {} hidden values for {} tokens (width {hcw})",
                h.len(),
                tokens.len()
            );
        }
        for (chunk, token_rows) in tokens.chunks(self.max_batch).enumerate() {
            let off = chunk * self.max_batch;
            let rows = token_rows.len();
            let pos = start_pos + off;
            if pos + rows > self.max_ctx {
                bail!("Qwen3.8 MTP catch-up exceeds {} token cache", self.max_ctx);
            }
            let ids = token_rows
                .iter()
                .map(|&token| token as i32)
                .collect::<Vec<_>>();
            let positions = (pos as i32..(pos + rows) as i32).collect::<Vec<_>>();
            be.upload(self.catch_ids.as_ref(), bytemuck::cast_slice(&ids))
                .map_err(|e| anyhow!("{e}"))?;
            be.upload(
                self.catch_h.as_ref(),
                bytemuck::cast_slice(&h[off * hcw..(off + rows) * hcw]),
            )
            .map_err(|e| anyhow!("{e}"))?;
            be.upload(
                self.catch_positions.as_ref(),
                bytemuck::cast_slice(&positions),
            )
            .map_err(|e| anyhow!("{e}"))?;
            let specs = [(shared.0 .1, shared.0 .2), (shared.1 .1, shared.1 .2)];
            let (graph, handles) =
                build_catch_graph(&self.cfg, &self.specs, specs, self.max_ctx, rows, pos);
            let plan = be.compile(&graph).map_err(|e| anyhow!("{e}"))?;
            let mut bindings = Bindings::new();
            bindings.bind(handles.ids, self.catch_ids.as_ref());
            bindings.bind(handles.h, self.catch_h.as_ref());
            bindings.bind(handles.positions, self.catch_positions.as_ref());
            bindings.bind(handles.k_cache, self.k_cache.as_ref());
            bindings.bind(handles.v_cache, self.v_cache.as_ref());
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

    pub(crate) fn draft4(
        &self,
        be: &dyn Backend,
        token: u32,
        h: &[f32],
        start_pos: usize,
        shared: ((&dyn Buffer, DType, usize), (&dyn Buffer, DType, usize)),
    ) -> Result<[u32; DRAFT_TOKENS]> {
        let hcw = self.cfg.hc_mult * self.cfg.n_embd;
        if h.len() != hcw {
            bail!(
                "Qwen3.8 MTP draft hidden width is {}, expected {hcw}",
                h.len()
            );
        }
        if start_pos + DRAFT_TOKENS > self.max_ctx {
            bail!("Qwen3.8 MTP draft exceeds {} token cache", self.max_ctx);
        }
        be.upload(self.draft_id.as_ref(), bytemuck::bytes_of(&(token as i32)))
            .map_err(|e| anyhow!("{e}"))?;
        be.upload(self.draft_h.as_ref(), bytemuck::cast_slice(h))
            .map_err(|e| anyhow!("{e}"))?;
        for (step, buffer) in self.draft_positions.iter().enumerate() {
            be.upload(
                buffer.as_ref(),
                bytemuck::bytes_of(&((start_pos + step) as i32)),
            )
            .map_err(|e| anyhow!("{e}"))?;
        }
        let specs = [(shared.0 .1, shared.0 .2), (shared.1 .1, shared.1 .2)];
        let (graph, handles) =
            build_draft_graph(&self.cfg, &self.specs, specs, self.max_ctx, start_pos);
        let plan = be.compile(&graph).map_err(|e| anyhow!("{e}"))?;
        let mut bindings = Bindings::new();
        bindings.bind(handles.id, self.draft_id.as_ref());
        bindings.bind(handles.h, self.draft_h.as_ref());
        bindings.bind(handles.k_cache, self.k_cache.as_ref());
        bindings.bind(handles.v_cache, self.v_cache.as_ref());
        for (id, buffer) in handles.positions.into_iter().zip(&self.draft_positions) {
            bindings.bind(id, buffer.as_ref());
        }
        for (id, buffer) in handles.ids.into_iter().zip(&self.draft_ids) {
            bindings.bind(id, buffer.as_ref());
        }
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
        let mut result = [0u32; DRAFT_TOKENS];
        for (dst, buffer) in result.iter_mut().zip(&self.draft_ids) {
            be.download(buffer.as_ref(), bytemuck::bytes_of_mut(dst))
                .map_err(|e| anyhow!("{e}"))?;
        }
        Ok(result)
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
        let bind: &BindWeightFn = &|_name, bytes, dtype, _numel| {
            let bytes = bytes.materialize();
            let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
            let buffer = vk
                .alloc(padded.len(), BufferUsage::Weights)
                .map_err(|e| anyhow!("{e}"))?;
            vk.upload(buffer.as_ref(), &padded)
                .map_err(|e| anyhow!("{e}"))?;
            Ok((buffer, dtype))
        };
        let catch_batch = crate::seam::ubatch_rows(model.engine_cfg());
        let head =
            Qwen4MtpSession::new(vk, bind, sidecar_path, model.config(), max_ctx, catch_batch)?;
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
        mut on_piece: impl FnMut(&str),
    ) -> Result<(crate::GenStats, super::MtpTiming)> {
        let cfg = model.config();
        let ec = model.engine_cfg();
        let h_width = cfg.hc_mult * cfg.n_embd;
        let prompt_tokens = model.encode(prompt)?;
        if prompt_tokens.is_empty() {
            bail!("Qwen3.8 MTP received an empty prompt");
        }
        if prompt_tokens.len() + max_new + DRAFT_TOKENS > self.max_ctx {
            bail!(
                "Qwen3.8 MTP needs {} context rows, but its fixed runtime has {}",
                prompt_tokens.len() + max_new + DRAFT_TOKENS,
                self.max_ctx
            );
        }
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
        let (prime_ids, prime_h) = super::run_verify_with_finish(
            vk,
            &*target_bind,
            model.gguf(),
            cfg,
            ec,
            model.embd(),
            &prompt_tokens,
            &mut self.trunk,
            self.max_ctx,
            finish_fixed_allocations.as_deref(),
        )?;
        let p = prompt_tokens.len();
        anyhow::ensure!(
            prime_ids.len() == p && prime_h.len() == p * h_width,
            "Qwen3.8 MTP prime returned {} ids and {} hidden values for {p} rows",
            prime_ids.len(),
            prime_h.len()
        );
        let mut shifted_h = vec![0.0f32; p * h_width];
        if p > 1 {
            shifted_h[h_width..].copy_from_slice(&prime_h[..(p - 1) * h_width]);
        }
        {
            let shared = self
                .trunk
                .as_ref()
                .expect("target prime initialized a trunk")
                .mtp_shared_weights();
            self.head
                .catch_up(vk, &prompt_tokens, &shifted_h, 0, shared)?;
        }
        self.trunk
            .as_mut()
            .expect("target prime initialized a trunk")
            .mtp_snapshot_delta(vk, cfg)?;
        let prompt_secs = t_prime.elapsed().as_secs_f64();

        let mut committed = prompt_tokens;
        let mut last_token = *committed.last().expect("non-empty prompt");
        // The NextN block is trained on (token_i, target_hidden_{i-1}) at position i. Keep both
        // sides of that one-row offset: `last_h` seeds catch-up for the next committed token,
        // while `draft_h` pairs with `last_token` to predict that next token.
        let mut last_h = prime_h[(p - 1) * h_width..].to_vec();
        let mut draft_h = shifted_h[(p - 1) * h_width..].to_vec();
        let mut target_prediction = prime_ids[p - 1];
        let mut acc = Vec::new();
        let mut printed = 0usize;
        let mut generated = 0usize;
        let mut timing = super::MtpTiming::default();
        let mut cycle = 0usize;
        let t_decode = std::time::Instant::now();
        let hit_eos = |token: u32| {
            !ec.sampling.ignore_eos && (cfg.eos_ids.contains(&token) || token == cfg.eos)
        };

        while generated < max_new && !req.is_some_and(crate::sampling::RequestCtx::aborted) {
            cycle += 1;
            let n_past = committed.len();

            let t_draft = std::time::Instant::now();
            let candidates = {
                let shared = self
                    .trunk
                    .as_ref()
                    .expect("target trunk remains initialized")
                    .mtp_shared_weights();
                self.head
                    .draft4(vk, last_token, &draft_h, n_past - 1, shared)?
            };
            let draft_secs = t_draft.elapsed().as_secs_f64();
            timing.draft_secs += draft_secs;
            timing.total_drafted += DRAFT_TOKENS;

            let mut feed = Vec::with_capacity(committed.len() + DRAFT_TOKENS);
            feed.extend_from_slice(&committed);
            feed.extend_from_slice(&candidates);
            let t_verify = std::time::Instant::now();
            let (target_bind, finish_fixed_allocations) = crate::seam::vulkan_moe_binder(
                vk,
                model.gguf(),
                cfg,
                ec,
                self.trunk.is_none(),
                self.max_ctx,
            )?;
            let (verify_ids, verify_h) = super::run_verify_with_finish(
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
            timing.verify_secs += verify_secs;
            anyhow::ensure!(
                verify_ids.len() == DRAFT_TOKENS
                    && verify_h.len() == DRAFT_TOKENS * h_width,
                "Qwen3.8 MTP VERIFY must return exactly {DRAFT_TOKENS} rows; got {} ids and {} hidden values",
                verify_ids.len(),
                verify_h.len()
            );

            let accepted = (0..DRAFT_TOKENS)
                .take_while(|&i| {
                    let expected = if i == 0 {
                        target_prediction
                    } else {
                        verify_ids[i - 1]
                    };
                    candidates[i] == expected
                })
                .count();
            timing.total_accepted += accepted;

            let t_catchup = std::time::Instant::now();
            let emitted: Vec<u32>;
            if accepted == DRAFT_TOKENS {
                let mut catch_h = Vec::with_capacity(DRAFT_TOKENS * h_width);
                catch_h.extend_from_slice(&last_h);
                catch_h.extend_from_slice(&verify_h[..(DRAFT_TOKENS - 1) * h_width]);
                {
                    let shared = self
                        .trunk
                        .as_ref()
                        .expect("target trunk remains initialized")
                        .mtp_shared_weights();
                    self.head
                        .catch_up(vk, &candidates, &catch_h, n_past, shared)?;
                }
                committed.extend_from_slice(&candidates);
                last_token = candidates[DRAFT_TOKENS - 1];
                draft_h =
                    verify_h[(DRAFT_TOKENS - 2) * h_width..(DRAFT_TOKENS - 1) * h_width].to_vec();
                last_h = verify_h[(DRAFT_TOKENS - 1) * h_width..].to_vec();
                target_prediction = verify_ids[DRAFT_TOKENS - 1];
                self.trunk
                    .as_mut()
                    .expect("target trunk remains initialized")
                    .mtp_snapshot_delta(vk, cfg)?;
                emitted = candidates.to_vec();
            } else {
                let correction = if accepted == 0 {
                    target_prediction
                } else {
                    verify_ids[accepted - 1]
                };
                let mut catch_tokens = candidates[..accepted].to_vec();
                catch_tokens.push(correction);
                let mut catch_h = Vec::with_capacity(catch_tokens.len() * h_width);
                catch_h.extend_from_slice(&last_h);
                catch_h.extend_from_slice(&verify_h[..accepted * h_width]);
                let next_draft_h = if accepted == 0 {
                    last_h.clone()
                } else {
                    verify_h[(accepted - 1) * h_width..accepted * h_width].to_vec()
                };
                {
                    let shared = self
                        .trunk
                        .as_ref()
                        .expect("target trunk remains initialized")
                        .mtp_shared_weights();
                    self.head
                        .catch_up(vk, &catch_tokens, &catch_h, n_past, shared)?;
                }

                let trunk = self
                    .trunk
                    .as_mut()
                    .expect("target trunk remains initialized");
                trunk.mtp_restore_delta(vk)?;
                committed.extend_from_slice(&candidates[..accepted]);
                committed.push(correction);
                let (next_prediction, _, correction_h) = super::run_prime_last(
                    vk,
                    &|_name, bytes, dtype, _numel| {
                        let bytes = bytes.materialize();
                        let padded = infr_vulkan::linear::pad_to_u32_align(&bytes);
                        let buffer = vk
                            .alloc(padded.len(), BufferUsage::Weights)
                            .map_err(|e| anyhow!("{e}"))?;
                        vk.upload(buffer.as_ref(), &padded)
                            .map_err(|e| anyhow!("{e}"))?;
                        Ok((buffer, dtype))
                    },
                    model.gguf(),
                    cfg,
                    ec,
                    model.embd(),
                    &committed,
                    &mut self.trunk,
                    self.max_ctx,
                    false,
                )?;
                self.trunk
                    .as_mut()
                    .expect("target trunk remains initialized")
                    .mtp_snapshot_delta(vk, cfg)?;
                last_token = correction;
                draft_h = next_draft_h;
                last_h = correction_h;
                target_prediction = next_prediction;
                emitted = catch_tokens;
            }
            let catchup_secs = t_catchup.elapsed().as_secs_f64();
            timing.catchup_secs += catchup_secs;

            if ec.prof.stages {
                tracing::info!(
                    "[qwen4 mtp cycle {cycle}] drafted={DRAFT_TOKENS} accepted={accepted} draft={:.1}ms verify={:.1}ms catchup={:.1}ms",
                    draft_secs * 1e3,
                    verify_secs * 1e3,
                    catchup_secs * 1e3,
                );
            }
            for token in emitted {
                let eos = hit_eos(token);
                generated += 1;
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
                    break;
                }
            }
        }

        if ec.prof.stages {
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
