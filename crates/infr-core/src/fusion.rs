//! Shared peephole graph-rewrite fusion pass.
//!
//! Every GPU backend (Vulkan, Metal, ROCm) independently re-implemented the SAME device-agnostic
//! peephole rewrites over the [`Graph`] IR — matching adjacent ops and folding them into one fused
//! kernel dispatch. This module hosts that logic ONCE; each backend supplies a [`FusionCfg`] naming
//! which patterns it can fuse and, per pattern, a `fn(DType) -> bool` predicate intersecting the
//! rewrite with the backend's own fused-kernel coverage. The result is a [`FusionPlan`] the backend
//! consumes exactly as it consumed its private `(fused, skip)` pair before.
//!
//! Three patterns are covered (the union of what the three backends did):
//!
//! 1. **`Linear(m==1) → Add(residual)`** ([`FusionCfg::linear_add`]) — fold a decode projection's
//!    following residual `Add` into the GEMV epilogue (`dst = gemv + residual`), killing the
//!    standalone `Add` kernel and the round-trip of the un-added projection. The decode
//!    `o_proj`/`down_proj` shape. Vulkan/Metal/ROCm all did this.
//! 2. **`RmsNorm → Linear(m==1)`** ([`FusionCfg::rmsnorm_linear`]) — elide a standalone `RmsNorm`
//!    whose normalized output feeds ONLY fusable decode `Linear`s, which normalize their raw input
//!    row in-kernel instead. ROCm's `input_norm→qkv` / `post_attn_norm→gate/up` int8 fusion.
//! 3. **`Rope/QkNormRope → WriteKv`** ([`FusionCfg::kv_write`]) — redirect a fused rope kernel's
//!    f16 K-row write straight into an f16 KV cache, absorbing the standalone `WriteKv`. Vulkan's
//!    `kv_write_peephole`, including the SWA ring-wrap guard.
//!
//! ## Live-range bounding (a correctness fix, applied to ALL backends)
//!
//! ROCm's copy bounds every candidate to the fused `dst`'s LIVE RANGE — the scratch `dst` handle
//! is recycled across layers, so eliding a standalone write is only safe if nothing OTHER than the
//! absorbing op reads `dst` before it is next rewritten. This guard is folded in for every backend
//! (per the unification plan): on the graphs the seam emits a fused Linear/RmsNorm `dst` is
//! single-use, so it never un-fuses a real pair — it only prevents an unsafe fold. Keeping it
//! everywhere means the shared pass is correct for any future graph, not just today's.

use crate::graph::{Graph, Op, TensorKind};
use crate::tensor::{DType, TensorId};
use std::collections::{HashMap, HashSet};

/// Per-pattern config for [`FusionCfg::linear_add`].
pub struct LinearAddCfg<'a> {
    /// Fuse only when the `Linear` weight's dtype passes this predicate — the backend's
    /// fused-residual-GEMV kernel coverage (e.g. Vulkan `native_dense_supported || F16`, Metal's
    /// legacy+Q4K/Q6K list, ROCm's int8-decode set).
    pub weight_ok: &'a dyn Fn(DType) -> bool,
    /// If set and the named env var is present, this pass is disabled entirely (the escape hatch
    /// each backend already exposed — Vulkan `INFR_NO_FUSE_ADD`, ROCm `INFR_ROCM_NO_FUSE_ADD`).
    pub disable_env: Option<&'static str>,
}

/// Per-pattern config for [`FusionCfg::rmsnorm_linear`].
pub struct RmsNormLinearCfg<'a> {
    /// Fuse only when each consuming `Linear`'s weight dtype passes this predicate.
    pub weight_ok: &'a dyn Fn(DType) -> bool,
    /// Env escape hatch (ROCm `INFR_ROCM_NO_FUSE_NORM`).
    pub disable_env: Option<&'static str>,
}

/// Which peephole rewrites a backend wants planned, and the per-pattern predicates/hatches.
/// A `None`/`false` field leaves that pattern un-planned (its ops stay split).
pub struct FusionCfg<'a> {
    /// `Linear(m==1) → Add` residual fusion.
    pub linear_add: Option<LinearAddCfg<'a>>,
    /// `RmsNorm → Linear(m==1)` fusion.
    pub rmsnorm_linear: Option<RmsNormLinearCfg<'a>>,
    /// `Rope/QkNormRope → WriteKv` fusion (f16 cache only; predicate is fixed f16, so a plain flag).
    pub kv_write: bool,
}

/// The planned rewrites for one graph. Each map is keyed by the op index of the op that ABSORBS
/// its neighbour; `skip` holds every op index the executor must NOT dispatch (the absorbed ops).
#[derive(Default)]
pub struct FusionPlan {
    /// `Linear` op idx → (residual operand, final `Add` dst). The absorbed `Add` is at `idx + 1`.
    pub linear_add: HashMap<usize, (TensorId, TensorId)>,
    /// Consuming `Linear` op idx → (raw pre-norm `x`, norm `weight`, `eps`). The elided `RmsNorm`
    /// op index is in `skip` (it is NOT `idx - 1` in general — one norm can feed several Linears).
    pub rmsnorm_linear: HashMap<usize, (TensorId, TensorId, f32)>,
    /// `Rope`/`QkNormRope` op idx → (KV cache tensor, write row). The absorbed `WriteKv` is at
    /// `idx + 1`.
    pub kv_write: HashMap<usize, (TensorId, usize)>,
    /// Op indices the executor elides (absorbed `Add`s, absorbed `WriteKv`s, elided `RmsNorm`s).
    pub skip: HashSet<usize>,
}

fn env_disabled(disable_env: Option<&'static str>) -> bool {
    disable_env.is_some_and(|name| std::env::var_os(name).is_some())
}

/// Plan the peephole fusions `cfg` enables over `graph`. Pure host logic over the IR — no device
/// types. The returned [`FusionPlan`] is what the backend feeds its executor.
#[cfg_attr(infr_profile, infr_prof::instrument)]
pub fn plan_fusions(graph: &Graph, cfg: &FusionCfg) -> FusionPlan {
    let mut plan = FusionPlan::default();
    if cfg.kv_write {
        plan_kv_write(graph, &mut plan);
    }
    if let Some(c) = &cfg.rmsnorm_linear {
        if !env_disabled(c.disable_env) {
            plan_rmsnorm_linear(graph, c, &mut plan);
        }
    }
    if let Some(c) = &cfg.linear_add {
        if !env_disabled(c.disable_env) {
            plan_linear_add(graph, c, &mut plan);
        }
    }
    plan
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
fn dst_only_read_by_next(graph: &Graph, start: usize, dst: TensorId) -> bool {
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
