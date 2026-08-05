//! Shared peephole graph-rewrite fusion pass.
//!
//! Every GPU backend (Vulkan, Metal) independently re-implemented the SAME device-agnostic
//! peephole rewrites over the [`Graph`] IR — matching adjacent ops and folding them into one fused
//! kernel dispatch. This module hosts that logic ONCE; each backend supplies a [`FusionCfg`] naming
//! which patterns it can fuse and, per pattern, a `fn(DType) -> bool` predicate intersecting the
//! rewrite with the backend's own fused-kernel coverage. The result is a [`FusionPlan`] the backend
//! consumes exactly as it consumed its private `(fused, skip)` pair before.
//!
//! Four patterns are covered (the union of what the three backends did, plus the MoE analogue of
//! the first two):
//!
//! 1. **`Linear(m==1) → Add(residual)`** ([`FusionCfg::linear_add`]) — fold a decode projection's
//!    following residual `Add` into the GEMV epilogue (`dst = gemv + residual`), killing the
//!    standalone `Add` kernel and the round-trip of the un-added projection. The decode
//!    `o_proj`/`down_proj` shape. Vulkan and Metal both did this.
//! 2. **`RmsNorm → Linear(m==1)`** ([`FusionCfg::rmsnorm_linear`]) — elide a standalone `RmsNorm`
//!    whose normalized output feeds ONLY fusable decode `Linear`s, which normalize their raw input
//!    row in-kernel instead.
//!    Optionally ([`RmsNormLinearCfg::moe_ok`]) a single-row `Op::MoeFfn` also counts as a fusable
//!    consumer — the MoE arm then normalizes in its own expert-quantize pass.
//! 3. **`MoeFfn → Add(residual)`** ([`FusionCfg::moe_add`]) — the exact analogue of (1) for the MoE
//!    sublayer: a pure-MoE FFN's residual `Add` never matched `linear_add` (the producer is
//!    `Op::MoeFfn`, not `Op::Linear`), so a 48-layer MoE decode paid 48 standalone `add`
//!    dispatches. Folded into the backend's expert-accumulate epilogue.
//! 4. **`Rope/QkNormRope → WriteKv`** ([`FusionCfg::kv_write`]) — redirect a fused rope kernel's
//!    f16 K-row write straight into an f16 KV cache, absorbing the standalone `WriteKv`. Vulkan's
//!    `kv_write_peephole`, including the SWA ring-wrap guard.
//!
//! ## Live-range bounding (a correctness fix, applied to ALL backends)
//!
//! The live-range bound guards that the fused `dst`'s scratch handle is not consumed
//! by anything other than the absorbing op before it is next rewritten.
//! On the graphs the seam emits a fused Linear/RmsNorm `dst` is
//! single-use, so it never un-fuses a real pair — it only prevents an unsafe fold. Keeping it
//! everywhere means the shared pass is correct for any future graph, not just today's.

use crate::graph::{Graph, Op, TensorKind};
use crate::tensor::{DType, TensorId};
use std::collections::{HashMap, HashSet};

/// Per-pattern config for [`FusionCfg::linear_add`].
pub struct LinearAddCfg<'a> {
    /// Fuse only when the `Linear` weight's dtype passes this predicate — the backend's
    /// fused-residual-GEMV kernel coverage (e.g. Vulkan `native_dense_supported || F16`, Metal's
    /// legacy+Q4K/Q6K list).
    pub weight_ok: &'a dyn Fn(DType) -> bool,
    /// `false` disables this pass entirely — the backend's escape hatch, taken off ITS config
    /// (Vulkan `kernels.vulkan.fuse_add` / `INFR_NO_FUSE_ADD`). A backend with no hatch passes `true`.
    pub enabled: bool,
}

/// Predicate over an `Op::MoeFfn`'s three expert banks `(gate, up, down)` — a backend's fused-MoE
/// kernel coverage. Used by both MoE patterns ([`FusionCfg::moe_add`] and
/// [`RmsNormLinearCfg::moe_ok`]).
pub type ExpertsOk<'a> = &'a dyn Fn(DType, DType, DType) -> bool;

/// Per-pattern config for [`FusionCfg::rmsnorm_linear`].
pub struct RmsNormLinearCfg<'a> {
    /// Fuse only when each consuming `Linear`'s weight dtype passes this predicate.
    pub weight_ok: &'a dyn Fn(DType) -> bool,
    /// When `Some`, a SINGLE-ROW `Op::MoeFfn` reading the normalized row as BOTH its expert input
    /// `x` and its router input `router_x` also counts as a fusable consumer, provided its expert
    /// banks pass the predicate — the backend's MoE arm then produces the normalized row itself.
    /// what every backend did before F1c.
    ///
    /// Single-row only, deliberately: the multi-row (prefill) MoE arm chunks its rows, so one
    /// normalize pass per chunk could not serve a router GEMV that spans every row — and prefill is
    /// weight-bandwidth-bound, where a saved launch does not show up anyway.
    pub moe_ok: Option<ExpertsOk<'a>>,
    /// `false` disables this pass entirely.
    pub enabled: bool,
}

/// Per-pattern config for [`FusionCfg::moe_add`].
pub struct MoeAddCfg<'a> {
    /// Fuse only when the `MoeFfn`'s expert banks pass this predicate — the backend's fused-residual
    /// expert-accumulate coverage.
    pub experts_ok: ExpertsOk<'a>,
    /// `false` disables this pass entirely.
    pub enabled: bool,
}

/// Which peephole rewrites a backend wants planned, and the per-pattern predicates/hatches.
/// A `None`/`false` field leaves that pattern un-planned (its ops stay split).
pub struct FusionCfg<'a> {
    /// `Linear(m==1) → Add` residual fusion.
    pub linear_add: Option<LinearAddCfg<'a>>,
    /// `RmsNorm → Linear(m==1)` fusion.
    pub rmsnorm_linear: Option<RmsNormLinearCfg<'a>>,
    /// `MoeFfn → Add` residual fusion.
    pub moe_add: Option<MoeAddCfg<'a>>,
    /// `Rope/QkNormRope → WriteKv` fusion (f16 cache only; predicate is fixed f16, so a plain flag).
    pub kv_write: bool,
}

/// The planned rewrites for one graph. Each map is keyed by the op index of the op that ABSORBS
/// its neighbour; `skip` holds every op index the executor must NOT dispatch (the absorbed ops).
#[derive(Default)]
pub struct FusionPlan {
    /// `Linear` op idx → (residual operand, final `Add` dst). The absorbed `Add` is at `idx + 1`.
    pub linear_add: HashMap<usize, (TensorId, TensorId)>,
    /// Consuming `Linear` (or, under [`RmsNormLinearCfg::moe_ok`], `MoeFfn`) op idx → (raw pre-norm
    /// `x`, norm `weight`, `eps`). The elided `RmsNorm` op index is in `skip` (it is NOT `idx - 1`
    /// in general — one norm can feed several consumers).
    pub rmsnorm_linear: HashMap<usize, (TensorId, TensorId, f32)>,
    /// `MoeFfn` op idx → (residual operand, final `Add` dst). The absorbed `Add` is at `idx + 1`.
    pub moe_add: HashMap<usize, (TensorId, TensorId)>,
    /// `Rope`/`QkNormRope` op idx → (KV cache tensor, write row). The absorbed `WriteKv` is at
    /// `idx + 1`.
    pub kv_write: HashMap<usize, (TensorId, usize)>,
    /// Op indices the executor elides (absorbed `Add`s, absorbed `WriteKv`s, elided `RmsNorm`s).
    pub skip: HashSet<usize>,
}

/// Plan the peephole fusions `cfg` enables over `graph`. Pure host logic over the IR — no device
/// types, and (since the config campaign's S2) no environment: each pattern's escape hatch arrives
/// as the `enabled` flag its backend read off its own `Config`.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn plan_fusions(graph: &Graph, cfg: &FusionCfg) -> FusionPlan {
    let mut plan = FusionPlan::default();
    if cfg.kv_write {
        plan_kv_write(graph, &mut plan);
    }
    if let Some(c) = cfg.rmsnorm_linear.as_ref().filter(|c| c.enabled) {
        plan_rmsnorm_linear(graph, c, &mut plan);
    }
    if let Some(c) = cfg.linear_add.as_ref().filter(|c| c.enabled) {
        plan_linear_add(graph, c, &mut plan);
    }
    if let Some(c) = cfg.moe_add.as_ref().filter(|c| c.enabled) {
        plan_moe_add(graph, c, &mut plan);
    }
    plan
}

/// The `(gate, up, down)` expert-bank dtypes of an `Op::MoeFfn`, for the two `ExpertsOk` predicates.
fn expert_dtypes(graph: &Graph, op: &Op) -> Option<(DType, DType, DType)> {
    let Op::MoeFfn {
        gate_exps,
        up_exps,
        down_exps,
        ..
    } = *op
    else {
        return None;
    };
    Some((
        graph.desc(gate_exps).dtype,
        graph.desc(up_exps).dtype,
        graph.desc(down_exps).dtype,
    ))
}

/// `MoeFfn(Internal dst, covered expert banks) → Add(residual)`: fold the MoE sublayer's residual
/// `Add` into the expert-accumulate epilogue. Structurally identical to [`plan_linear_add`] — same
/// Internal-dst requirement, same immediately-following-op rule (a gemma sandwich `post_ffw` norm
/// between the two correctly blocks it), same live-range bound — only the producer differs.
fn plan_moe_add(graph: &Graph, cfg: &MoeAddCfg, plan: &mut FusionPlan) {
    for (i, op) in graph.ops.iter().enumerate() {
        let Op::MoeFfn { dst, .. } = *op else {
            continue;
        };
        if !matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal) {
            continue;
        }
        let Some((gd, ud, dd)) = expert_dtypes(graph, op) else {
            continue;
        };
        if !(cfg.experts_ok)(gd, ud, dd) {
            continue;
        }
        let Some(Op::Add {
            a,
            b,
            dst: add_dst,
            n,
        }) = graph.ops.get(i + 1)
        else {
            continue;
        };
        // The `Add` must span the WHOLE MoE output, not a slice of it.
        if *n as usize != graph.desc(dst).numel() {
            continue;
        }
        let residual = if *b == dst && *a != dst {
            *a
        } else if *a == dst && *b != dst {
            *b
        } else {
            continue;
        };
        // Live-range bound (see module docs): nothing but this `Add` may read the un-added MoE
        // output before its scratch handle is next rewritten.
        if !dst_only_read_by_next(graph, i + 2, dst) {
            continue;
        }
        plan.moe_add.insert(i, (residual, *add_dst));
        plan.skip.insert(i + 1);
    }
}

/// `Linear(m==1, Internal dst, covered weight) → Add(residual)`: fold the following residual `Add`
/// into the GEMV epilogue. Only the IMMEDIATELY following op fuses (the seam emits the pair
/// adjacent for non-gemma models; gemma's sandwich norm sits between and correctly blocks it).
fn plan_linear_add(graph: &Graph, cfg: &LinearAddCfg, plan: &mut FusionPlan) {
    for (i, op) in graph.ops.iter().enumerate() {
        let Op::Linear {
            dst,
            m: 1,
            weight,
            out_f,
            ..
        } = op
        else {
            continue;
        };
        if !matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal) {
            continue;
        }
        if !(cfg.weight_ok)(graph.desc(*weight).dtype) {
            continue;
        }
        let Some(Op::Add {
            a,
            b,
            dst: add_dst,
            n,
        }) = graph.ops.get(i + 1)
        else {
            continue;
        };
        if *n != *out_f {
            continue;
        }
        let residual = if b == dst && a != dst {
            *a
        } else if a == dst && b != dst {
            *b
        } else {
            continue;
        };
        // Live-range bound (see module docs): the Linear's `dst` must be consumed ONLY by this Add
        // before it is next rewritten — eliding the standalone write is unsafe if anything else
        // reads the un-added projection (the `dst` scratch may be recycled by a later layer).
        if !dst_only_read_by_next(graph, i + 2, *dst) {
            continue;
        }
        plan.linear_add.insert(i, (residual, *add_dst));
        plan.skip.insert(i + 1);
    }
}

/// `RmsNorm(Internal dst) → Linear(m==1)`s: elide the standalone `RmsNorm` when its normalized
/// output feeds ONLY covered decode Linears (each normalizes its raw input in-kernel instead).
fn plan_rmsnorm_linear(graph: &Graph, cfg: &RmsNormLinearCfg, plan: &mut FusionPlan) {
    for (i, op) in graph.ops.iter().enumerate() {
        let Op::RmsNorm {
            x,
            weight,
            dst,
            dim,
            eps,
            ..
        } = *op
        else {
            continue;
        };
        if !matches!(graph.tensors[dst.0 as usize].kind, TensorKind::Internal) {
            continue;
        }
        // The normalized-output tensor is a scratch handle RECYCLED across layers, so a whole-graph
        // reader scan would wrongly match every layer's q/k/v against ONE norm. Only readers in THIS
        // norm's live range — from here until the next op that rewrites `dst` — are its consumers.
        // Each must be a fusable decode GEMV whose `in_f` matches the norm `dim`.
        let mut consumers: Vec<usize> = Vec::new();
        let mut ok = true;
        for j in (i + 1)..graph.ops.len() {
            let (ins, outs) = graph.ops[j].io();
            if ins.contains(&dst) {
                match &graph.ops[j] {
                    Op::Linear {
                        x: lx,
                        weight: lw,
                        m: 1,
                        in_f,
                        ..
                    } if *lx == dst && *in_f == dim && (cfg.weight_ok)(graph.desc(*lw).dtype) => {
                        consumers.push(j);
                    }
                    // F1c: a single-row `MoeFfn` that reads the normalized row as BOTH its expert
                    // input and its router input. Requiring `x == router_x == dst` is not a
                    // convenience: a backend's MoE arm produces ONE normalized row, so a router
                    // reading some OTHER tensor would still need the standalone norm's output.
                    Op::MoeFfn {
                        x: mx,
                        router_x,
                        dst: mdst,
                        ne,
                        ..
                    } if cfg.moe_ok.is_some()
                        && *mx == dst
                        && *router_x == dst
                        && *ne == dim
                        && graph.desc(*mdst).numel() == dim as usize
                        && expert_dtypes(graph, &graph.ops[j])
                            .is_some_and(|(g_, u, d)| (cfg.moe_ok.unwrap())(g_, u, d)) =>
                    {
                        consumers.push(j);
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if outs.contains(&dst) {
                break; // `dst` rewritten — live range ends (a fusable Linear never writes it).
            }
        }
        if ok && !consumers.is_empty() {
            plan.skip.insert(i);
            for j in consumers {
                plan.rmsnorm_linear.insert(j, (x, weight, eps));
            }
        }
    }
}

/// `Rope`/`QkNormRope(Internal f16 dst) → WriteKv(f16 cache)`: redirect the fused rope kernel's f16
/// K-row write straight into the KV cache, absorbing the standalone `WriteKv`.
fn plan_kv_write(graph: &Graph, plan: &mut FusionPlan) {
    for (i, op) in graph.ops.iter().enumerate() {
        // QkNormRope (qwen/gemma) or f16-out Rope (llama) — both write the f16 K row the peephole
        // can redirect straight into the KV cache.
        let kxx = match op {
            Op::QkNormRope { dst, .. } | Op::Rope { dst, .. } => dst,
            _ => continue,
        };
        // Only fuse an Internal (scratch) dst (we redirect the write into the KV cache). The output
        // must be f16 (the shader casts f32→f16); WriteKv of an f16 src is a plain copy.
        if !matches!(graph.tensors[kxx.0 as usize].kind, TensorKind::Internal) {
            continue;
        }
        if !matches!(graph.desc(*kxx).dtype, DType::F16) {
            continue;
        }
        let Some(Op::WriteKv {
            src,
            cache,
            pos,
            rows,
            row_stride,
        }) = graph.ops.get(i + 1)
        else {
            continue;
        };
        // A Q8_0 cache needs a real quantizing WriteKv (store_q8), so DON'T fuse the f16 rope write
        // into it — leave the standalone WriteKv to run.
        if src == kxx && matches!(graph.desc(*cache).dtype, DType::F16) {
            // SWA ring cache (row capacity < full context): the write row is pos % cap_rows. The
            // fused rope kernels write `rows` CONTIGUOUS rows from out_base, so a batched prefill
            // write that would cross the ring's wrap boundary can't fuse — leave the standalone
            // WriteKv, whose lowering splits the write at the wrap. Decode (rows == 1) always fuses;
            // a full-context cache never wraps (pos < cap_rows).
            let cap_rows = graph.desc(*cache).numel() / (*row_stride as usize).max(1);
            let pos_r = if cap_rows > 0 {
                *pos as usize % cap_rows
            } else {
                *pos as usize
            };
            if cap_rows == 0 || pos_r + *rows as usize <= cap_rows {
                plan.kv_write.insert(i, (*cache, pos_r));
                plan.skip.insert(i + 1);
            }
        }
    }
}

/// Live-range check: from op index `start`, is `dst` read by NOTHING before it is next rewritten?
/// Returns `true` if `dst` is dead until its next write (or graph end) — the fold is safe.
///
/// `pub` because a backend post-filtering [`FusionPlan::kv_write`] needs the SAME bound: that plan
/// carries no live-range guard (Vulkan's record-once decode path REQUIRES its K write fused, so it
/// cannot be un-fused by a graph-shape test), and a backend that redirects the rope's write into the
/// cache leaves the rope's `dst` scratch unwritten — safe only if nothing reads it. Sharing this
/// function keeps that check the same one every other pattern uses.
pub fn dst_only_read_by_next(graph: &Graph, start: usize, dst: TensorId) -> bool {
    for j in start..graph.ops.len() {
        let (ins, outs) = graph.ops[j].io();
        if ins.contains(&dst) {
            return false;
        }
        if outs.contains(&dst) {
            break;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::graph::Op;
    use crate::tensor::TensorDesc;

    /// A minimal decode `Linear(m==1) -> Add(residual)` pair, the shape both `linear_add` backends
    /// fuse.
    fn linear_add_graph() -> Graph {
        let mut g = Graph::new();
        let x = g.input(TensorDesc::new(vec![1, 4], DType::F32));
        let w = g.weight(TensorDesc::new(vec![4, 4], DType::F16));
        let dst = g.internal(TensorDesc::new(vec![1, 4], DType::F32));
        let res = g.input(TensorDesc::new(vec![1, 4], DType::F32));
        let out = g.output(TensorDesc::new(vec![1, 4], DType::F32));
        g.push(Op::Linear {
            x,
            weight: w,
            dst,
            m: 1,
            in_f: 4,
            out_f: 4,
            w_off: 0,
        });
        g.push(Op::Add {
            a: dst,
            b: res,
            dst: out,
            n: 4,
        });
        g
    }

    /// The escape hatch, driven through a `Config` rather than the process environment: the
    /// backend hands `plan_fusions` the POSITIVE `fuse_add` field, and a cleared field leaves the
    /// pair split (both ops dispatched) exactly as `INFR_NO_FUSE_ADD` used to.
    #[test]
    fn linear_add_hatch_comes_from_the_config() {
        let g = linear_add_graph();
        let all: &dyn Fn(DType) -> bool = &|_| true;
        let plan_with = |enabled: bool| {
            plan_fusions(
                &g,
                &FusionCfg {
                    linear_add: Some(LinearAddCfg {
                        weight_ok: all,
                        enabled,
                    }),
                    rmsnorm_linear: None,
                    moe_add: None,
                    kv_write: false,
                },
            )
        };

        // Shipped default: the fold is planned and the standalone `Add` is elided.
        let d = Config::default();
        assert!(d.kernels.vulkan.fuse_add);
        let on = plan_with(d.kernels.vulkan.fuse_add);
        assert_eq!(on.linear_add.len(), 1);
        assert!(on.skip.contains(&1));

        // The `NO_FUSE` keys clear their POSITIVE field on PRESENCE — value irrelevant,
        // including `=0` (the presence-inv truth table).
        for raw in ["", "0", "1"] {
            let layer =
                crate::config::env::parse(&|k| (k == "INFR_NO_FUSE_ADD").then(|| raw.to_string()))
                    .expect("INFR_NO_FUSE_ADD");
            let cfg = Config::load_from_layers(&[layer]);
            assert!(
                !cfg.kernels.vulkan.fuse_add,
                "INFR_NO_FUSE_ADD={raw:?} must clear its fusion field"
            );
        }

        // Cleared field => nothing planned, nothing skipped: both ops dispatch.
        let off = plan_with(false);
        assert!(off.linear_add.is_empty());
        assert!(off.skip.is_empty());
    }

    // ── F1c: the two MoE patterns ────────────────────────────────────────────

    /// The MoE decode shape the seam emits: `RmsNorm(hidden) -> MoeFfn(x = router_x = hn)
    /// -> Add(hidden, sub)`. `tail` appends an extra reader of the MoE dst, to exercise the
    /// live-range bound.
    fn moe_graph(tail_reads_moe_dst: bool) -> Graph {
        let mut g = Graph::new();
        let hidden = g.input(TensorDesc::new(vec![1, 4], DType::F32));
        let nw = g.weight(TensorDesc::new(vec![4], DType::F16));
        let hn = g.internal(TensorDesc::new(vec![1, 4], DType::F32));
        let router = g.weight(TensorDesc::new(vec![2, 4], DType::F32));
        let gate = g.weight(TensorDesc::new(vec![2, 8, 4], DType::Q4K));
        let up = g.weight(TensorDesc::new(vec![2, 8, 4], DType::Q4K));
        let down = g.weight(TensorDesc::new(vec![2, 4, 8], DType::Q6K));
        let sub = g.internal(TensorDesc::new(vec![1, 4], DType::F32));
        let out = g.output(TensorDesc::new(vec![1, 4], DType::F32));
        g.push(Op::RmsNorm {
            x: hidden,
            weight: nw,
            dst: hn,
            rows: 1,
            dim: 4,
            eps: 1e-6,
        });
        g.push(Op::MoeFfn {
            x: hn,
            router_x: hn,
            router,
            gate_exps: gate,
            up_exps: up,
            down_exps: down,
            down_scale: None,
            dst: sub,
            ne: 4,
            n_expert: 2,
            n_used: 2,
            n_ff_exp: 8,
            scale: 1.0,
            act: crate::graph::Activation::Silu,
            gating: crate::graph::MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            fused_gate_up: false,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
        });
        g.push(Op::Add {
            a: hidden,
            b: sub,
            dst: out,
            n: 4,
        });
        if tail_reads_moe_dst {
            // A second consumer of the un-added MoE output: the fold must NOT fire.
            g.push(Op::Scale {
                x: sub,
                dst: out,
                s: 2.0,
                n: 4,
            });
        }
        g
    }

    fn moe_cfg<'a>(
        experts_ok: ExpertsOk<'a>,
        weight_ok: &'a dyn Fn(DType) -> bool,
        moe_norm: bool,
        moe_add: bool,
    ) -> FusionCfg<'a> {
        FusionCfg {
            linear_add: None,
            rmsnorm_linear: Some(RmsNormLinearCfg {
                weight_ok,
                moe_ok: moe_norm.then_some(experts_ok),
                enabled: true,
            }),
            moe_add: moe_add.then_some(MoeAddCfg {
                experts_ok,
                enabled: true,
            }),
            kv_write: false,
        }
    }

    /// Both F1c patterns fire on the seam's MoE decode shape, and BOTH are opt-in: a backend that
    /// leaves `moe_ok`/`moe_add` unset gets exactly the pre-F1c plan (nothing), which is what keeps
    /// Vulkan and Metal on their existing dispatch sequence.
    #[test]
    fn moe_patterns_are_opt_in_and_fire_on_the_seam_shape() {
        let g = moe_graph(false);
        let all: &dyn Fn(DType) -> bool = &|_| true;
        let experts: ExpertsOk = &|_, _, _| true;

        // Off (Vulkan/Metal today): the `MoeFfn` consumer disqualifies the norm fold and the
        // residual `Add` is not planned — all three ops dispatch.
        let off = plan_fusions(&g, &moe_cfg(experts, all, false, false));
        assert!(off.rmsnorm_linear.is_empty());
        assert!(off.moe_add.is_empty());
        assert!(off.skip.is_empty());

        // On: the `RmsNorm` (op 0) and the `Add` (op 2) are both elided, and the `MoeFfn` (op 1)
        // absorbs both.
        let on = plan_fusions(&g, &moe_cfg(experts, all, true, true));
        assert_eq!(on.rmsnorm_linear.len(), 1);
        assert!(on.rmsnorm_linear.contains_key(&1));
        assert_eq!(on.moe_add.len(), 1);
        assert!(on.moe_add.contains_key(&1));
        assert_eq!(on.skip, [0usize, 2].into_iter().collect::<HashSet<_>>());
    }

    /// The expert-bank predicate gates BOTH MoE patterns — a backend whose fused-MoE kernels do not
    /// cover this quant gets the split ops, not a fold it cannot execute.
    #[test]
    fn moe_patterns_respect_the_expert_predicate() {
        let g = moe_graph(false);
        let all: &dyn Fn(DType) -> bool = &|_| true;
        // Rejects the Q6_K down bank (the shape Q4_K_M actually ships).
        let experts: ExpertsOk = &|gd, _, dd| gd == DType::Q4K && dd == DType::Q4K;
        let p = plan_fusions(&g, &moe_cfg(experts, all, true, true));
        assert!(p.rmsnorm_linear.is_empty());
        assert!(p.moe_add.is_empty());
        assert!(p.skip.is_empty());
    }

    /// The live-range bound applies to `MoeFfn → Add` exactly as it does to `Linear → Add`: a second
    /// reader of the un-added MoE output blocks the fold, because the fold never writes it.
    #[test]
    fn moe_add_respects_the_live_range_bound() {
        let g = moe_graph(true);
        let all: &dyn Fn(DType) -> bool = &|_| true;
        let experts: ExpertsOk = &|_, _, _| true;
        let p = plan_fusions(&g, &moe_cfg(experts, all, true, true));
        assert!(
            p.moe_add.is_empty(),
            "the un-added MoE output has a second reader — folding the Add would leave it unwritten"
        );
        assert!(!p.skip.contains(&2), "the Add must still dispatch");
        // The norm fold is independent and still fires (its own live range is unaffected).
        assert!(p.rmsnorm_linear.contains_key(&1));
    }

    /// A multi-row (prefill) `MoeFfn` is NOT a fusable norm consumer — the backends' MoE arms chunk
    /// their rows, so one normalize pass cannot serve a router GEMV spanning every row. The residual
    /// fold has no such restriction and still fires.
    #[test]
    fn rmsnorm_moe_fold_is_single_row_only() {
        let mut g = Graph::new();
        let hidden = g.input(TensorDesc::new(vec![4, 4], DType::F32));
        let nw = g.weight(TensorDesc::new(vec![4], DType::F16));
        let hn = g.internal(TensorDesc::new(vec![4, 4], DType::F32));
        let router = g.weight(TensorDesc::new(vec![2, 4], DType::F32));
        let gate = g.weight(TensorDesc::new(vec![2, 8, 4], DType::Q4K));
        let down = g.weight(TensorDesc::new(vec![2, 4, 8], DType::Q4K));
        let sub = g.internal(TensorDesc::new(vec![4, 4], DType::F32));
        let out = g.output(TensorDesc::new(vec![4, 4], DType::F32));
        g.push(Op::RmsNorm {
            x: hidden,
            weight: nw,
            dst: hn,
            rows: 4,
            dim: 4,
            eps: 1e-6,
        });
        g.push(Op::MoeFfn {
            x: hn,
            router_x: hn,
            router,
            gate_exps: gate,
            up_exps: gate,
            down_exps: down,
            down_scale: None,
            dst: sub,
            ne: 4,
            n_expert: 2,
            n_used: 2,
            n_ff_exp: 8,
            scale: 1.0,
            act: crate::graph::Activation::Silu,
            gating: crate::graph::MoeGating::Softmax,
            norm_w: true,
            weight_before: false,
            fused_gate_up: true,
            ep_band: None,
            exp_probs_b: None,
            n_expert_groups: 0,
            n_expert_groups_used: 0,
        });
        g.push(Op::Add {
            a: hidden,
            b: sub,
            dst: out,
            n: 16,
        });
        let all: &dyn Fn(DType) -> bool = &|_| true;
        let experts: ExpertsOk = &|_, _, _| true;
        let p = plan_fusions(&g, &moe_cfg(experts, all, true, true));
        assert!(
            p.rmsnorm_linear.is_empty() && !p.skip.contains(&0),
            "a 4-row MoeFfn must not elide its input norm"
        );
        assert!(p.moe_add.contains_key(&1) && p.skip.contains(&2));
    }
}
