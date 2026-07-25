//! Shared graph-executor skeleton: the residency contract + the op walk.
//!
//! Every backend runs the same *shape* around its per-op bodies — provision a working slot for each
//! [`TensorId`], walk `plan.graph.ops` in order eliding the fusion-skipped indices, then copy the
//! handles the caller reads back to their bound buffers. That structure was re-derived four times
//! (cpu `infr-cpu/src/lib.rs`, metal `Resident`/`run_graph`, rocm `ExecCtx`/`execute_graph`, vulkan
//! `execute_static`/`record_decode_replay`), and the pieces that MUST agree between them — which
//! [`TensorKind`]s are materialized, which are mutated in place, which are written back, and what
//! "the ops the executor actually dispatches" means — were four independent copies of the same
//! three predicates. This module hosts those ONCE.
//!
//! ## What is shared and what deliberately is not
//!
//! Shared here (pure host logic over the IR, no device types):
//!
//! * [`Provision`]/[`provisions`] — the pre-walk per-handle action (materialize zeroed scratch,
//!   load a bound `Input`, or leave alone).
//! * [`writes_back`]/[`write_back_targets`] — the post-walk write-back predicate.
//! * [`live_ops`] — THE definition of the executor's walk: graph order, fusion-skipped indices
//!   elided. Anything that must stay in lockstep with the walk (rocm's dense-weight prefetch
//!   schedule) iterates the same function rather than re-spelling the filter.
//! * [`OpDispatch`]/[`run_ops`] — the walk as a driver, for the backends whose per-op body is
//!   already one function. Generic (`impl OpDispatch`), so the per-op call stays static dispatch.
//!
//! NOT shared, on purpose:
//!
//! * **The residency CONTAINER.** The four disagree on it for real reasons, not drift: cpu keeps a
//!   host-only `Vec<Vec<f32>>`; rocm keeps `dev: Vec<Option<RocmBuffer>>` + a lazily-filled
//!   `vals: Vec<Option<Vec<f32>>>`; metal keeps both PLUS a `loc: Vec<Loc>` host/device tracker
//!   (43 sites) that rocm has no analogue for, because metal's `ensure_device` uploads the host
//!   mirror and rocm's never does; vulkan has no per-op residency at all (its `Internal` scratch is
//!   allocated up front by `alloc_scratch` and the walk only records into a command buffer). A
//!   common container would have to be either the LCD (dropping metal's `loc`) or the union (an
//!   unused `loc` on rocm) — and either way every one of the ~240 `vals`/`dev`/`loc` accesses in
//!   the per-op bodies would be rewritten. Behavior-neutral in principle, a rewrite in fact.
//! * **The per-op bodies / a per-op trait method.** The `match Op::` arm sets are parallel, but the
//!   arms need wildly different ambient state (vulkan's `lower_op` threads 15 extra parameters —
//!   recorder, scratch, pool, dyn-attn contexts, mmv memo, streamed-weight substitution; rocm's
//!   takes the two per-op fusion payloads). A 27-method trait would move ~10k lines for no line
//!   saved and would have to carry every backend's ambient state in its `Self`.
//! * **Vulkan's walk.** It is a RECORDER, not an interpreter: no host values, no residency
//!   transitions, and its loop carries five device concerns between ops (submit splitter, shutdown
//!   poll, paged-MoE hand-off, dense-streaming stage, an E2B peephole that MUTATES the skip set
//!   mid-walk). It uses [`live_ops`] where the skip set is immutable and keeps its own loop.

use crate::error::Result;
use crate::graph::{Graph, Op, TensorDecl, TensorKind};
use crate::tensor::{DType, TensorId};
use std::collections::HashSet;

/// What an interpreter-style backend must do with one tensor handle BEFORE the op walk.
///
/// This is the classification the cpu and metal interpreters each spelled inline as a
/// `match decl.kind` with a `direct.contains(..)` guard. Kept an enum (rather than a bool pair) so
/// a new [`TensorKind`] forces a decision in one place instead of drifting between backends.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provision {
    /// Backend-allocated scratch or a read-back output: materialize `desc.numel()` zeroed slots.
    /// (`TensorKind::Internal` / [`TensorKind::Output`].)
    Zero,
    /// A per-step [`TensorKind::Input`] the ops only READ: load its bound buffer into the working
    /// store.
    Load,
    /// Nothing to do. Either a [`TensorKind::Weight`] (dequantized lazily at its use site, never
    /// mirrored whole) or an `Input` the ops mutate IN PLACE through its bound buffer — the KV
    /// cache, per [`Graph::in_place_inputs`]. Loading a KV cache would cost O(max_ctx) traffic per
    /// token instead of O(kv_len); that skip is the reason this classification exists.
    Skip,
}

/// The pre-walk action for one handle. `direct` is [`Graph::in_place_inputs`].
#[inline]
pub fn provision(kind: TensorKind, id: TensorId, direct: &HashSet<TensorId>) -> Provision {
    match kind {
        TensorKind::Internal | TensorKind::Output => Provision::Zero,
        TensorKind::Input if direct.contains(&id) => Provision::Skip,
        TensorKind::Input => Provision::Load,
        TensorKind::Weight => Provision::Skip,
    }
}

/// Every handle in declaration order with its [`Provision`] — the shared form of the pre-walk
/// working-store setup loop. Yields the decl too, since the caller needs `desc.numel()`/`dtype`.
pub fn provisions(graph: &Graph) -> impl Iterator<Item = (TensorId, &TensorDecl, Provision)> + '_ {
    let direct = graph.in_place_inputs();
    graph.tensors.iter().enumerate().map(move |(i, decl)| {
        let id = TensorId(i as u32);
        (id, decl, provision(decl.kind, id, direct))
    })
}

/// Must this handle's post-walk value be copied back to its bound buffer?
///
/// `true` for every [`TensorKind::Output`] (the logits the caller reads) and for an f32
/// [`TensorKind::Input`] the ops MUTATED (conv/recurrent state) — but never for one the ops wrote
/// IN PLACE (the KV cache: `direct`, per [`Graph::in_place_inputs`]), which is already current in
/// its bound buffer and whose copy would be O(max_ctx). Non-f32 `Input`s (the i32 positions) are
/// read-only by construction.
///
/// Identical in the cpu, metal and rocm executors; hoisted so a change can't land in one of them.
#[inline]
pub fn writes_back(
    kind: TensorKind,
    dtype: DType,
    id: TensorId,
    direct: &HashSet<TensorId>,
) -> bool {
    matches!(kind, TensorKind::Output)
        || (kind == TensorKind::Input && dtype == DType::F32 && !direct.contains(&id))
}

/// The handles [`writes_back`] selects, in declaration order — the shared form of the post-walk
/// write-back loop's header. The COPY itself stays per-backend (host memcpy / device DtoD /
/// unified-memory store).
pub fn write_back_targets(graph: &Graph) -> impl Iterator<Item = TensorId> + '_ {
    let direct = graph.in_place_inputs();
    graph
        .tensors
        .iter()
        .enumerate()
        .filter_map(move |(i, decl)| {
            let id = TensorId(i as u32);
            writes_back(decl.kind, decl.desc.dtype, id, direct).then_some(id)
        })
}

/// The ops the executor actually dispatches: `(index, op)` in graph order, with every index in
/// `skip` elided.
///
/// This is THE definition of "the walk" — the fusion pass ([`crate::fusion`]) hands back a `skip`
/// set of absorbed ops, and anything that must stay in lockstep with the dispatch order iterates
/// this, not its own `enumerate().filter()`. (rocm's dense-weight prefetch ring is the reason that
/// matters: its schedule is built by a SEPARATE pass whose cursor has to track the op walk exactly,
/// or a staged bank is handed to the wrong `Linear`.)
///
/// Zero-cost: a borrowed iterator adapter, monomorphized and inlined, no allocation — the same
/// `HashSet::contains` per op the hand-written loops did.
#[inline]
pub fn live_ops<'a>(
    ops: &'a [Op],
    skip: &'a HashSet<usize>,
) -> impl Iterator<Item = (usize, &'a Op)> + 'a {
    ops.iter()
        .enumerate()
        .filter(move |(i, _)| !skip.contains(i))
}

/// A backend's per-op body, as the shared walk sees it: run (or record) the op at `idx`.
///
/// Deliberately ONE method taking the whole `&Op`, not one method per variant — the per-op `match`
/// and every device decision inside it stay with the backend (see this module's header). The
/// implementor is a small per-forward struct holding the backend's ambient state (device handle,
/// graph, bindings, residency), so `run_ops` can be generic and the call monomorphizes to the same
/// direct call the inline loop made.
pub trait OpDispatch {
    fn dispatch(&mut self, idx: usize, op: &Op) -> Result<()>;
}

/// Walk `ops` in [`live_ops`] order, dispatching each through `d`. Generic (NOT `&dyn`), so the
/// per-op call is static dispatch and inlines exactly as the hand-written loop did.
///
/// Per-op profiling / error cleanup belongs in the implementor's [`OpDispatch::dispatch`] — the
/// backends differ there (metal seals its open encoder before unwinding; rocm does not), and
/// putting it here would either flatten that or cost a per-op branch.
#[inline]
pub fn run_ops<D: OpDispatch>(ops: &[Op], skip: &HashSet<usize>, d: &mut D) -> Result<()> {
    for (idx, op) in live_ops(ops, skip) {
        d.dispatch(idx, op)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{AttnMask, Op};
    use crate::tensor::TensorDesc;

    /// Build a graph with one of each kind plus a KV cache the ops write in place.
    fn kinds_graph() -> Graph {
        let mut g = Graph::new();
        let hidden = g.input(TensorDesc::new(vec![4], DType::F32)); // 0: plain f32 Input
        let positions = g.input(TensorDesc::new(vec![1], DType::I32)); // 1: non-f32 Input
        let w = g.weight(TensorDesc::new(vec![4], DType::Q4K)); // 2: Weight
        let scratch = g.internal(TensorDesc::new(vec![4], DType::F32)); // 3: Internal
        let kcache = g.input(TensorDesc::new(vec![8], DType::F16)); // 4: in-place Input
        let vcache = g.input(TensorDesc::new(vec![8], DType::F32)); // 5: in-place f32 Input
        let logits = g.output(TensorDesc::new(vec![4], DType::F32)); // 6: Output
        g.push(Op::WriteKv {
            src: hidden,
            cache: kcache,
            rows: 1,
            row_stride: 8,
            pos: 0,
        });
        g.push(Op::Attention {
            q: scratch,
            k_cache: kcache,
            v_cache: vcache,
            dst: logits,
            rows: 1,
            kv_len: 1,
            n_head: 1,
            n_kv: 1,
            head_dim: 8,
            scale: 1.0,
            mask: AttnMask::Causal,
            pos: 0,
        });
        g.push(Op::RmsNorm {
            x: hidden,
            weight: w,
            dst: scratch,
            rows: 1,
            dim: 4,
            eps: 1e-6,
        });
        let _ = positions;
        g
    }

    /// `provision` reproduces the classification the cpu/metal setup loops spelled inline:
    /// Internal/Output materialize, a plain Input loads, a Weight and an in-place (KV) Input do
    /// nothing.
    #[test]
    fn provision_matches_inline_classification() {
        let g = kinds_graph();
        let got: Vec<Provision> = provisions(&g).map(|(_, _, p)| p).collect();
        assert_eq!(
            got,
            vec![
                Provision::Load, // 0 hidden: plain f32 Input
                Provision::Load, // 1 positions: plain i32 Input (loaded, never written back)
                Provision::Skip, // 2 weight: lazily dequantized
                Provision::Zero, // 3 scratch: Internal
                Provision::Skip, // 4 k_cache: in place
                Provision::Skip, // 5 v_cache: in place
                Provision::Zero, // 6 logits: Output
            ]
        );
        // The direct set is exactly the KV handles, so the two Skips at 4/5 are the in-place ones.
        let direct = g.in_place_inputs();
        assert!(direct.contains(&TensorId(4)) && direct.contains(&TensorId(5)));
        assert!(!direct.contains(&TensorId(0)));
    }

    /// `writes_back` reproduces the inline predicate
    /// `Output || (Input && F32 && !direct)` — the byte-identical copy in cpu/metal/rocm.
    #[test]
    fn write_back_targets_match_inline_predicate() {
        let g = kinds_graph();
        let direct = g.in_place_inputs();
        let want: Vec<TensorId> = g
            .tensors
            .iter()
            .enumerate()
            .filter(|(i, decl)| {
                matches!(decl.kind, TensorKind::Output)
                    || (decl.kind == TensorKind::Input
                        && decl.desc.dtype == DType::F32
                        && !direct.contains(&TensorId(*i as u32)))
            })
            .map(|(i, _)| TensorId(i as u32))
            .collect();
        assert_eq!(want, vec![TensorId(0), TensorId(6)]);
        assert_eq!(write_back_targets(&g).collect::<Vec<_>>(), want);
        // The f32 KV cache (5) is f32 and an Input but written IN PLACE — never copied back.
        assert!(!write_back_targets(&g).any(|id| id == TensorId(5)));
        // The i32 positions Input (1) is read-only.
        assert!(!write_back_targets(&g).any(|id| id == TensorId(1)));
    }

    /// `live_ops` yields graph order with the skipped indices elided, and keeps the ORIGINAL index
    /// (the key the fusion payload maps are keyed by).
    #[test]
    fn live_ops_elides_skipped_indices_keeping_original_index() {
        let g = kinds_graph();
        let empty = HashSet::new();
        assert_eq!(
            live_ops(&g.ops, &empty).map(|(i, _)| i).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let skip: HashSet<usize> = [1].into_iter().collect();
        let live: Vec<(usize, &'static str)> = live_ops(&g.ops, &skip)
            .map(|(i, op)| (i, op.kind()))
            .collect();
        assert_eq!(live, vec![(0, "WriteKv"), (2, "RmsNorm")]);
    }

    /// `run_ops` visits exactly `live_ops`, in order, and propagates the first error without
    /// visiting anything after it.
    #[test]
    fn run_ops_visits_live_ops_in_order_and_stops_on_error() {
        struct Rec {
            seen: Vec<usize>,
            fail_at: Option<usize>,
        }
        impl OpDispatch for Rec {
            fn dispatch(&mut self, idx: usize, _op: &Op) -> Result<()> {
                self.seen.push(idx);
                if self.fail_at == Some(idx) {
                    return Err(crate::error::backend("boom"));
                }
                Ok(())
            }
        }
        let g = kinds_graph();
        let skip: HashSet<usize> = [1].into_iter().collect();
        let mut r = Rec {
            seen: Vec::new(),
            fail_at: None,
        };
        run_ops(&g.ops, &skip, &mut r).unwrap();
        assert_eq!(r.seen, vec![0, 2]);

        let mut r = Rec {
            seen: Vec::new(),
            fail_at: Some(0),
        };
        let err = run_ops(&g.ops, &HashSet::new(), &mut r).unwrap_err();
        assert!(err.to_string().contains("boom"));
        assert_eq!(r.seen, vec![0], "the walk must stop at the failing op");
    }
}
