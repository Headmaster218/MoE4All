//! Phase-cached pair admission. Cache misses allocate before GPU-copy candidate begin.
use super::*;
use verify_pair::{Cohort, Layer, LayerPlan, Role, VerifyPairWindow};

pub(super) fn allocate(
    backend: &VulkanBackend,
    graph: &Graph,
    shared: &HashMap<usize, PagedMoeShared>,
    cache: &mut verify_pair::WindowCache,
) -> Result<Option<Arc<VerifyPairWindow>>> {
    if !backend.cfg().kernels.vulkan.verify_pair
        || !graph.mtp_verify
        || graph.independent_rows
        || preparing_commands()
    {
        return Ok(None);
    }
    let plans: Vec<_> = graph
        .ops
        .iter()
        .enumerate()
        .map(|(op_idx, op)| {
            let Op::MoeFfn {
                x,
                gate_exps,
                up_exps,
                fused_gate_up,
                weight_before,
                ne,
                n_ff_exp,
                n_expert,
                n_used,
                ..
            } = op
            else {
                return None;
            };
            if *fused_gate_up || *weight_before || *ne == 0 {
                return None;
            }
            let cohort = Cohort {
                verify: graph.mtp_verify,
                independent_rows: graph.independent_rows,
                rows: graph.desc(*x).numel() / *ne as usize,
                used: *n_used as usize,
                experts: *n_expert as usize,
                in_f: *ne as usize,
                out_f: *n_ff_exp as usize,
                shared_q8: shared.contains_key(&op_idx),
            };
            let roles = [*gate_exps, *up_exps].map(|id| {
                crate::recorder::verify_pair_baseline_policy(backend, graph.desc(id).dtype, cohort)
            });
            LayerPlan::new(cohort, roles)
        })
        .collect();
    cache.acquire(&plans, false, || {
        VerifyPairWindow::allocate(backend, &plans).map(|window| window.map(Arc::new))
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn linear(
    causal_verify: bool,
    pair: Option<&Arc<Layer>>,
    role: Role,
    rec: &Recorder<'_>,
    dtype: infr_core::DType,
    arena_addr: u64,
    slot_bytes: u32,
    lut: &dyn Buffer,
    ids: &dyn Buffer,
    n_used: usize,
    lut_base: usize,
    shared: Option<&dyn Buffer>,
    x: &dyn Buffer,
    x_per_slot: bool,
    y: &dyn Buffer,
    in_f: usize,
    out_f: usize,
    rows: usize,
    active_mask: u32,
    row_mask_bank: u32,
) {
    if let Some(pair) = pair {
        let c = pair.cohort;
        if !x_per_slot
            && c.used == n_used
            && c.rows == rows
            && c.in_f == in_f
            && c.out_f == out_f
            && c.shared_q8 == shared.is_some()
            && rec.linear_verify_pair(
                pair,
                role,
                dtype,
                arena_addr,
                slot_bytes,
                lut,
                ids,
                lut_base,
                shared,
                x,
                y,
                active_mask,
                row_mask_bank,
            )
        {
            return;
        }
    }
    linear_paged_maybe_shared(
        causal_verify,
        rec,
        dtype,
        arena_addr,
        slot_bytes,
        lut,
        ids,
        n_used,
        lut_base,
        shared,
        x,
        x_per_slot,
        y,
        in_f,
        out_f,
        rows,
        active_mask,
        row_mask_bank,
    );
}
