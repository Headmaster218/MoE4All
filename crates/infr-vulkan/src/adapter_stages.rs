//! Consecutive static plans under one forward owner. Included as `adapter::stages` so the
//! adapter retains ownership of workspace admission, topology switching and submission.
use super::*;
use infr_core::backend::{ExecutionLookahead, ExecutionStage};

thread_local! {
    static FORWARD_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn staged_forward_active() -> bool {
    FORWARD_DEPTH.with(|depth| depth.get() != 0)
}

#[derive(Clone, Copy)]
struct InputRange {
    buffer: ash::vk::Buffer,
    start: usize,
    end: usize,
    mapped: Option<(usize, usize)>,
}

impl InputRange {
    fn of(buffer: &dyn Buffer) -> Option<Self> {
        let buffer = crate::as_vk_buf(buffer).ok()?;
        let start = buffer.sub_offset;
        let end = start.checked_add(buffer.size)?;
        let mapped = match buffer.mapped_ptr() {
            Some(ptr) => {
                let start = ptr as usize;
                Some((start, start.checked_add(buffer.size)?))
            }
            None => None,
        };
        Some(Self {
            buffer: buffer.buffer,
            start,
            end,
            mapped,
        })
    }

    fn overlaps(self, other: Self) -> bool {
        (self.buffer == other.buffer && self.start < other.end && other.start < self.end)
            || matches!((self.mapped, other.mapped), (Some((a, b)), Some((c, d))) if a < d && c < b)
    }

    fn disjoint_from(self, buffer: &dyn Buffer) -> bool {
        if let Some(segmented) = crate::as_segmented_kv(buffer) {
            // The logical KV handle is not a VkBuffer. Prove every physical allocation,
            // including its host table, instead of treating an unknown handle as disjoint.
            return self.disjoint_from(&segmented.table)
                && segmented.segments.lock().is_ok_and(|segments| {
                    segments.iter().all(|segment| self.disjoint_from(segment))
                });
        }
        Self::of(buffer).is_some_and(|range| !self.overlaps(range))
    }
}

fn range_isolated_from_stage(
    target: InputRange,
    bindings: &Bindings<'_>,
    touched: &HashSet<TensorId>,
    scratch: &ScratchSet,
    pool: &ScratchPool,
) -> bool {
    touched.iter().all(|&id| {
        !bindings
            .get(id)
            .is_some_and(|buffer| !target.disjoint_from(buffer))
            && !bindings
                .get_rows(id)
                .is_some_and(|rows| rows.iter().any(|buffer| !target.disjoint_from(*buffer)))
    }) && scratch
        .iter()
        .filter_map(Option::as_deref)
        .all(|buffer| target.disjoint_from(buffer))
        && pool
            .buffers
            .values()
            .all(|buffer| target.disjoint_from(buffer.as_ref()))
}

/// The callback may overlap an earlier GPU tail only for explicitly declared, ordinary
/// coherent host inputs. Graph-local TensorIds are not an alias proof: compare real ranges,
/// including row bindings and the retained scratch/pool that the earlier stage can use.
pub(super) fn independent_stage_inputs(
    stage: &ExecutionStage<'_>,
    plan: &VkDecodePlan,
    previous: &ExecutionStage<'_>,
    previous_plan: &VkDecodePlan,
    previous_scratch: &ScratchSet,
    pool: &ScratchPool,
) -> bool {
    let Some(ids) = stage.independent_host_inputs.filter(|ids| !ids.is_empty()) else {
        return false;
    };
    let mut touched = HashSet::new();
    let mut written = HashSet::new();
    for op in &previous_plan.graph.ops {
        let (reads, writes) = op.io_inline();
        written.extend(writes.iter().copied());
        touched.extend(reads.into_iter().chain(writes));
    }
    for (index, &id) in ids.iter().enumerate() {
        if ids[..index].contains(&id)
            || !plan
                .graph
                .tensors
                .get(id.0 as usize)
                .is_some_and(|tensor| tensor.kind == infr_core::graph::TensorKind::Input)
        {
            return false;
        }
        let Some(buffer) = stage.bindings.get(id) else {
            return false;
        };
        let desc = &plan.graph.tensors[id.0 as usize].desc;
        let required = desc
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .and_then(|n| desc.dtype.dense_bytes(n));
        if !required.is_some_and(|bytes| bytes != 0 && bytes <= buffer.len_bytes()) {
            return false;
        }
        let Ok(vk_buffer) = crate::as_vk_buf(buffer) else {
            return false;
        };
        if !matches!(vk_buffer.backing, crate::Backing::Pooled(_))
            || vk_buffer.location != gpu_allocator::MemoryLocation::CpuToGpu
            || vk_buffer.mapped_ptr().is_none()
        {
            return false;
        }
        let Some(target) = InputRange::of(buffer) else {
            return false;
        };
        if !range_isolated_from_stage(target, previous.bindings, &touched, previous_scratch, pool) {
            return false;
        }
    }
    // refresh_inputs also reads positions on the CPU after the callback. Those reads
    // need their own proof; a disjoint callback target does not make every host read safe.
    for &id in &plan.setup_analysis.rope_ids {
        let Some(buffer) = stage.bindings.get(id) else {
            return false;
        };
        let Ok(vk_buffer) = crate::as_vk_buf(buffer) else {
            return false;
        };
        if vk_buffer.mapped_ptr().is_none() || buffer.len_bytes() < 4 {
            return false;
        }
        let Some(target) = InputRange::of(buffer) else {
            return false;
        };
        if !range_isolated_from_stage(target, previous.bindings, &written, previous_scratch, pool) {
            return false;
        }
    }
    true
}

pub(crate) struct ForwardGuard<'a> {
    pager: &'a crate::pager::MoePagerCell,
    _caller_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

pub(crate) fn forward_guard(be_: &VulkanBackend) -> ForwardGuard<'_> {
    FORWARD_DEPTH.with(|depth| depth.set(depth.get() + 1));
    ForwardGuard {
        pager: be_.moe_pager(),
        _caller_thread: std::marker::PhantomData,
    }
}

impl Drop for ForwardGuard<'_> {
    fn drop(&mut self) {
        FORWARD_DEPTH.with(|depth| depth.set(depth.get() - 1));
        if !staged_forward_active() {
            if let Ok(guard) = self.pager.lock() {
                if let Some(session) = guard.as_ref() {
                    session.finish_gpu_decode_forward();
                }
            }
        }
    }
}

fn linear_scratch_requests(
    be_: &VulkanBackend,
    graph: &Graph,
    requests: &mut HashMap<&'static str, usize>,
) {
    let cfg = &be_.cfg().kernels.vulkan;
    for op in &graph.ops {
        let Op::Linear {
            weight,
            m,
            in_f,
            out_f,
            ..
        } = op
        else {
            continue;
        };
        let (m, k, n) = (*m as usize, *in_f as usize, *out_f as usize);
        let dt = graph.desc(*weight).dtype;
        if m == 0 || k == 0 || n == 0 || k % 32 != 0 {
            continue;
        }
        let mut add = |tag, bytes| record_scratch_max(requests, tag, bytes);
        if m == 1 {
            if !native_dense_supported(dt) || !be_.caps().i8_dot {
                continue;
            }
            let mw = mmv_mw_choice(be_.caps(), cfg, dt, k, n);
            let legacy = mmv_decode_enabled(cfg)
                && k * n >= MMV_MIN_ELEMS
                && (crate::gemm::native_mmv_kernel_name(dt, false).is_some()
                    || crate::gemm::native_mmv_kernel_name(dt, true).is_some());
            if (mw.is_some() && !cfg.mmv_fuse_quant) || (mw.is_none() && legacy) {
                add("mmv_qa", k);
                add("mmv_dact", (k / 32) * 2);
                add("mmv_sact", (k / 32) * 2);
            }
            continue;
        }
        let mmv_gate = be_.caps().i8_dot
            && k * n >= 8 << 20
            && n < 65536
            && crate::gemm::native_mmv_mrow_kernel_name(dt).is_some()
            && mrow_int8_dtype_ok(be_.caps(), cfg, dt, graph.mtp_verify)
            && cfg.mmv;
        let tier = infr_core::tier::linear_tier(m, n, &MROW_BANDS);
        let band = |b| tier == infr_core::tier::LinearTier::MultiRow { band: b };
        let mrow16 = band(MROW16_BAND)
            && mmv_gate
            && cfg.mrow16
            && crate::gemm::native_mmv_mrow_m16_kernel_name(dt).is_some();
        if (band(MROW_BAND) || mrow16)
            && cfg.mrow
            && crate::gemm::native_mrow_kernel_name(dt).is_some()
        {
            if mmv_gate {
                add("mmvr_qa", m * k);
                add("mmvr_dact", m * (k / 32) * 2);
                add("mmvr_sact", m * (k / 32) * 2);
            }
            continue;
        }
        if n % 64 != 0 {
            continue;
        }
        let cm8 = be_.caps().f16_coopmat_8x8()
            && n % 128 == 0
            && k % 64 == 0
            && crate::gemm::native_gemm_warp_cm8_build_spv(dt).is_some();
        if cm8 {
            continue;
        }
        let coop =
            be_.caps().f16_coopmat() && (native_dense_supported(dt) || dt == infr_core::DType::F16);
        let nc_mmq = !be_.caps().f16_coopmat()
            && cfg.mmq_fallback
            && be_.caps().i8_dot
            && infr_core::tensor::moe_mmq_ok(dt);
        if !coop && !nc_mmq {
            continue;
        }
        let f8 = coop
            && dt == infr_core::DType::Q8_0
            && cfg.f8_coopmat
            && be_.caps().f8_coopmat()
            && (n % 256 == 0 || (n % 128 == 0 && k % 64 == 0));
        if f8 {
            add("f8cm_qa", m * k);
            add("f8cm_srow", m * 4);
            if cfg.f8_prepack {
                add("f8_w8", n * k);
            }
            continue;
        }
        if coop && dt == infr_core::DType::Q8_0 && be_.i8_coopmat_ready() {
            if cfg.i8_row_scale {
                add("i8cm_qa_row", m * k);
                add("i8cm_dact_row", m * 2);
            } else {
                add("mmq_qa", m * k);
                add("mmq_dact", m * (k / 32) * 2);
                add("mmq_sact", m * (k / 32) * 2);
            }
            continue;
        }
        let warp = coop
            && cfg.gemm_warp
            && n % 128 == 0
            && crate::gemm::native_gemm_warp_kernel_name(dt).is_some();
        if nc_mmq || (coop && dt == infr_core::DType::Q4K && !warp && be_.caps().i8_dot && cfg.mmq)
        {
            add("mmq_qa", m * k);
            add("mmq_dact", m * (k / 32) * 2);
            add("mmq_sact", m * (k / 32) * 2);
        }
        let mpad = m.div_ceil(64) * 64;
        if warp
            && !(cfg.gemm_direct_a
                && crate::gemm::native_gemm_warp_n128_direct_kernel_name(dt).is_some())
            && crate::gemm::native_gemm_warp_ag_kernel_name(dt).is_some()
        {
            add("lin_a16", mpad * k * 2);
        }
        let splits = split_k_plan(m, k, n);
        if coop
            && cfg.gemm_warp
            && splits > 1
            && crate::gemm::native_gemm_warp_sk_kernel_name(dt).is_some()
            && (warp || dt == infr_core::DType::F16)
        {
            add("splitk_part", splits * mpad * n * 4);
        }
    }
}

fn small_moe_scratch_requests(
    be_: &VulkanBackend,
    graph: &Graph,
    requests: &mut HashMap<&'static str, usize>,
    staging: &mut HashMap<&'static str, usize>,
) -> Result<()> {
    let shared: HashSet<_> = paged_moe_shared_candidates_with_policy(
        graph,
        std::env::var_os("INFR_NO_MOE_SHARED_SLOT").is_some(),
        false,
    )
    .map(|(index, _)| index)
    .collect();
    for (index, op) in graph.ops.iter().enumerate() {
        let Op::MoeFfn {
            x,
            ne,
            n_ff_exp,
            n_expert,
            n_used,
            fused_gate_up,
            exp_probs_b,
            expert_ids,
            ..
        } = op
        else {
            continue;
        };
        let (ne, nff, experts, used) = (
            *ne as usize,
            *n_ff_exp as usize,
            *n_expert as usize,
            *n_used as usize,
        );
        if ne == 0 {
            return Err(be("staged MoE has zero input width"));
        }
        let rows = graph.desc(*x).numel() / ne;
        if exp_probs_b.is_none() || expert_ids.is_none() {
            record_scratch_max(requests, "moe_dummy", 4);
        }
        // Binding-independent upper bound: any registered bank may use the paged path.
        // Mixed/nonresident graphs cannot start GPU ownership and keep ordinary lazy lowering.
        let geometry = PagedMoeScratchGeometry {
            rows,
            ne,
            nff,
            experts,
            used,
            fused_gate_up: *fused_gate_up,
            shared_slot: shared.contains(&index),
        };
        geometry.record(
            requests,
            staging,
            paged_mmq_scratch_requests(be_, graph, op).is_none(),
        );
    }
    Ok(())
}

struct PagedMoeScratchGeometry {
    rows: usize,
    ne: usize,
    nff: usize,
    experts: usize,
    used: usize,
    fused_gate_up: bool,
    shared_slot: bool,
}

impl PagedMoeScratchGeometry {
    fn record(
        &self,
        requests: &mut HashMap<&'static str, usize>,
        staging: &mut HashMap<&'static str, usize>,
        small: bool,
    ) {
        let slots = self.rows * self.used;
        let masks = self.rows <= MOE_HIT_FIRST_MAX_ROWS && slots <= MOE_HIT_FIRST_MAX_SLOTS;
        record_scratch_max(
            staging,
            "moe_paged_ids",
            (slots + usize::from(masks) * 2 * self.rows) * 4,
        );
        let mut add = |tag, bytes| record_scratch_max(requests, tag, bytes);
        add("moe_paged_logits", self.rows * self.experts * 4);
        add("moe_paged_wts", slots * 4);
        if small {
            let physical = self.rows * (self.used + usize::from(self.shared_slot));
            add(
                "moe_paged_g",
                physical
                    * if self.fused_gate_up {
                        2 * self.nff
                    } else {
                        self.nff
                    }
                    * 4,
            );
            if !self.fused_gate_up {
                add("moe_paged_u", physical * self.nff * 4);
            }
            add("moe_paged_a", physical * self.nff * 4);
            add("moe_paged_y", physical * self.ne * 4);
        }
    }
}

pub(super) struct WorkspaceRequests {
    pub(super) activation: Vec<(&'static str, usize)>,
    staging: Vec<(&'static str, usize)>,
}

impl WorkspaceRequests {
    pub(super) fn rolling_ready(&self, graph: &Graph, pool: &ScratchPool) -> bool {
        let Some(rows) = decode_token_rows(graph) else {
            return false;
        };
        if !complete_decode_requests(paged_static_phase(graph), graph.mtp_verify)
            || pool.needs_compaction
            || !graph.ops.iter().all(|op| match op {
                Op::MoeFfn { .. } => ResidentMoeLayout::from_op(graph, op).is_some(),
                // The complete phase planner does not cover sink-attention workspaces.
                Op::Attention { sinks: Some(_), .. } => false,
                Op::Sample { .. } | Op::Argmax { .. } | Op::ArgmaxProb { .. } => true,
                _ => command_prefix::supported_op_rows(op, rows),
            })
        {
            return false;
        }
        self.capacities_ready(pool)
    }

    fn capacities_ready(&self, pool: &ScratchPool) -> bool {
        self.activation
            .iter()
            .chain(&self.staging)
            .all(|&(tag, bytes)| {
                pool.buffers.iter().any(|(&(candidate, capacity), buffer)| {
                    candidate == tag
                        && capacity >= bytes.max(4)
                        && buffer.len_bytes() >= bytes.max(4)
                })
            })
    }

    fn from_maps(
        requests: HashMap<&'static str, usize>,
        staging: HashMap<&'static str, usize>,
    ) -> Self {
        let mut activation = requests.into_iter().collect::<Vec<_>>();
        activation.sort_unstable_by_key(|&(tag, _)| tag);
        let mut staging = staging.into_iter().collect::<Vec<_>>();
        staging.sort_unstable_by_key(|&(tag, _)| tag);
        Self {
            activation,
            staging,
        }
    }

    fn merge_into(
        &self,
        requests: &mut HashMap<&'static str, usize>,
        staging: &mut HashMap<&'static str, usize>,
    ) {
        for &(tag, bytes) in &self.activation {
            record_scratch_max(requests, tag, bytes);
        }
        for &(tag, bytes) in &self.staging {
            record_scratch_max(staging, tag, bytes);
        }
    }
}

fn complete_decode_requests(phase: Option<RuntimePhase>, mtp_verify: bool) -> bool {
    phase == Some(RuntimePhase::Decode) && !mtp_verify
}

pub(super) fn compiled_workspace_requests(
    be_: &VulkanBackend,
    graph: &Graph,
) -> Result<WorkspaceRequests> {
    let mut requests = HashMap::new();
    for (tag, bytes) in paged_phase_scratch_requests(be_, graph)? {
        record_scratch_max(&mut requests, tag, bytes);
    }
    let mut staging = HashMap::new();
    if complete_decode_requests(paged_static_phase(graph), graph.mtp_verify) {
        linear_scratch_requests(be_, graph, &mut requests);
        small_moe_scratch_requests(be_, graph, &mut requests, &mut staging)?;
    }
    Ok(WorkspaceRequests::from_maps(requests, staging))
}

pub(super) fn grow_decode_workspace(
    be_: &VulkanBackend,
    arena: &mut RuntimePhaseArena,
    layout: &ScratchLayout,
    requests: &WorkspaceRequests,
) -> Result<()> {
    arena.grow_workspace(be_, layout, &requests.activation)?;
    for &(tag, bytes) in &requests.staging {
        pooled_usage(&mut arena.pool, be_, tag, bytes, BufferUsage::Staging)?;
    }
    Ok(())
}

/// Parent preflight helper, called under the unified gate before beginning GPU ownership.
/// Warm each graph-local topology, retaining the other stages in the bounded Decode cache.
/// The parent must additionally prepare any lazy resources not represented by phase requests.
/// Unsupported phase/topology counts fail before changing the arena or submitting a stage.
pub(crate) fn prepare_decode_workspaces(
    be_: &VulkanBackend,
    stages: &[ExecutionStage<'_>],
) -> Result<()> {
    if stages.is_empty() {
        return Ok(());
    }
    if be_.unified_client.is_some() || !be_.moe_paged() || be_.dense_paged() {
        return Err(be(
            "staged workspace retention requires primary paged Decode",
        ));
    }
    let mut layouts = Vec::with_capacity(stages.len());
    let mut topologies = HashSet::new();
    let mut requests = HashMap::<&'static str, usize>::new();
    let mut staging = HashMap::<&'static str, usize>::new();
    for stage in stages {
        let plan = stage
            .plan
            .as_any()
            .downcast_ref::<VkDecodePlan>()
            .ok_or_else(|| be("staged workspace requires VkDecodePlan"))?;
        if plan.graph.mtp_verify || paged_static_phase(&plan.graph) != Some(RuntimePhase::Decode) {
            return Err(be(
                "staged workspace retention requires ordinary Decode stages",
            ));
        }
        let layout = if be_.cfg().kernels.vulkan.static_scratch_memo {
            plan.static_scratch_metadata()?.layout.clone()
        } else {
            StaticScratchMetadata::analyze(&plan.graph)?.layout
        };
        topologies.insert(layout.iter().map(Option::is_some).collect::<Vec<_>>());
        let uncached;
        let phase_requests = match plan.phase_requests.as_ref() {
            Some(requests) => requests,
            None => {
                uncached = compiled_workspace_requests(be_, &plan.graph)?;
                &uncached
            }
        };
        phase_requests.merge_into(&mut requests, &mut staging);
        layouts.push((plan, layout));
    }
    if topologies.len() > RuntimePhaseArena::MAX_DECODE_SCRATCH_TOPOLOGIES {
        return Err(be("staged Decode exceeds retained scratch topology limit"));
    }
    let mut requests = requests.into_iter().collect::<Vec<_>>();
    requests.sort_unstable_by_key(|&(tag, _)| tag);
    let mut arena = be_.runtime_phase.lock().unwrap();
    for ((plan, layout), stage) in layouts.iter().zip(stages) {
        let start = arena.begin_execute(RuntimePhase::Decode, layout);
        if pager_static_transition(
            start.previous,
            RuntimePhase::Decode,
            be_.cfg().paging.moe_layer_stream,
        ) == Some(RuntimePhase::Decode)
        {
            if let Some(session) = be_.moe_pager().lock().unwrap().as_mut() {
                session.enter_decode();
            }
        }
        arena.grow_workspace(be_, layout, &requests)?;
        // grow_workspace is Activation-only; IDs must retain their mapped Staging allocation.
        for (&tag, &bytes) in &staging {
            pooled_usage(&mut arena.pool, be_, tag, bytes, BufferUsage::Staging)?;
        }
        let resident =
            resident_mtp_graph_supported(be_, &plan.graph, &arena.scratch, stage.bindings);
        // This constructs only the retained protocol resources, not a recording or submission.
        drop(ResidentChunk::new(
            be_,
            &plan.graph,
            resident,
            &mut arena.resident_buffers,
        )?);
    }
    // Prove every graph is still retained, then leave L0 active. No allocations in this pass.
    for (_, layout) in &layouts {
        arena.begin_execute(RuntimePhase::Decode, layout);
        if !arena.scratch_satisfies(layout) {
            return Err(be("staged preflight did not retain every graph workspace"));
        }
    }
    arena.begin_execute(RuntimePhase::Decode, &layouts[0].1);
    Ok(())
}

/// Immutable graph analysis only. The worker must never resolve bound data, record commands,
/// allocate device buffers, lock the pager/runtime arena, or acquire an execution lease.
pub(crate) struct CpuPreparation {
    pub(crate) fusions: infr_core::fusion::FusionPlan,
}

impl CpuPreparation {
    fn analyze(be_: &VulkanBackend, plan: &VkDecodePlan) -> Self {
        Self {
            fusions: plan
                .setup_analysis
                .fusions(be_.cfg().kernels.vulkan.fuse_add),
        }
    }
}

/// `VulkanBackend::execute_staged` supplies the exclusive unified gate around this entire call.
///
/// Adapter integration contract:
/// - `preflight_stages(be_, stages)` reserves ALL stage topologies and pooled/lazy device
///   resources before GPU work, then returns the overall forward-owner guard. Stage execution
///   must reuse them without allocations or ownership handoffs. Scratch indexed by graph-local
///   TensorId cannot be unioned across different topologies: `resolve` prefers scratch to binds.
/// - `execute_static_prepared` consumes `CpuPreparation::fusions` in place of `plan_fusions`,
///   retaining binding-dependent validation and fusion adjustments on the execution thread.
///
/// Both hooks live in adapter.rs; no worker enters either hook.
pub(crate) fn execute(
    be_: &VulkanBackend,
    stages: &[ExecutionStage<'_>],
    input_ready: &mut dyn FnMut(usize) -> Result<()>,
    next: Option<(&dyn Plan, &Bindings, &[usize])>,
) -> Result<()> {
    execute_inner(be_, stages, input_ready, next, None)
}

pub(crate) fn execute_lookahead(
    be_: &VulkanBackend,
    stages: &[ExecutionStage<'_>],
    input_ready: &mut dyn FnMut(usize) -> Result<()>,
    next: Option<ExecutionLookahead<'_>>,
) -> Result<()> {
    let first = next.and_then(|next| {
        next.stages
            .first()
            .map(|stage| (stage.plan, stage.bindings, next.positions))
    });
    execute_inner(be_, stages, input_ready, first, next)
}

fn execute_inner(
    be_: &VulkanBackend,
    stages: &[ExecutionStage<'_>],
    input_ready: &mut dyn FnMut(usize) -> Result<()>,
    next: Option<(&dyn Plan, &Bindings, &[usize])>,
    next_stages: Option<ExecutionLookahead<'_>>,
) -> Result<()> {
    if stages.is_empty() {
        return Ok(());
    }
    // Validate every plan before admission/submission. There is deliberately no fallback after
    // starting stage zero, even if a later callback or execution fails.
    let plans = stages
        .iter()
        .map(|stage| {
            stage
                .plan
                .as_any()
                .downcast_ref::<VkDecodePlan>()
                .ok_or_else(|| be("vulkan staged execution: plan is not a VkDecodePlan"))
        })
        .collect::<Result<Vec<_>>>()?;
    let next = next.and_then(|(plan, bindings, positions)| {
        plan.as_any()
            .downcast_ref::<VkDecodePlan>()
            .map(|plan| (plan, bindings, positions))
    });
    let next_stages = next_stages
        .filter(|next| next.stages.len() == stages.len())
        .and_then(|next| {
            next.stages
                .iter()
                .map(|stage| {
                    stage
                        .plan
                        .as_any()
                        .downcast_ref::<VkDecodePlan>()
                        .map(|plan| (plan, stage.bindings, next.positions))
                })
                .collect::<Option<Vec<_>>>()
        });
    let mut next_guards = next_stages.as_ref().map_or_else(
        || {
            vec![command_prefix::PreparedOutputGuard::new(
                next.map(|(plan, _, _)| &plan.prepared_prefix),
            )]
        },
        |stages| {
            stages
                .iter()
                .map(|(plan, _, _)| {
                    command_prefix::PreparedOutputGuard::new(Some(&plan.prepared_prefix))
                })
                .collect()
        },
    );
    let result = (|| {
        let _forward = super::preflight_stages(be_, stages)?;
        let _staged_forward = forward_guard(be_);
        if super::staged_continuity_supported(be_, &plans) {
            return super::execute_static_continuation(
                be_,
                stages,
                &plans,
                input_ready,
                next,
                next_stages.as_deref(),
            );
        }
        input_ready(0)?;
        if stages.len() == 1 {
            return super::execute_static_prepared(be_, plans[0], stages[0].bindings, next, None);
        }

        let context = infr_core::timeline::context();
        let prepared = overlap_cpu_preparation(
            || {
                let _context = infr_core::timeline::scoped_context(context);
                let _span = infr_core::timeline::span("staged_tail_prepare");
                plans[1..]
                    .iter()
                    .map(|plan| CpuPreparation::analyze(be_, plan))
                    .collect::<Vec<_>>()
            },
            || super::execute_static_prepared(be_, plans[0], stages[0].bindings, None, None),
        )?;
        // The scope has joined. Callbacks/uploads run on the lease-owning caller, with no arena or
        // pager mutex held, and only after the preceding stage has drained its submitted segments.
        for (index, preparation) in (1..stages.len()).zip(prepared) {
            input_ready(index)?;
            super::execute_static_prepared(
                be_,
                plans[index],
                stages[index].bindings,
                if index + 1 == stages.len() {
                    next
                } else {
                    None
                },
                Some(preparation),
            )?;
        }
        Ok(())
    })();
    if result.is_ok() {
        for guard in &mut next_guards {
            guard.commit();
        }
    }
    result
}

/// Callbacks run between stages on the execution-lease owning thread. Only an explicitly
/// declared and range-validated independent input may overlap an earlier submitted GPU tail.
/// A failed input or partially submitted stage stops the sequence without retry or finalization.
pub(super) fn run_sequence(
    count: usize,
    ready: &mut dyn FnMut(usize) -> Result<()>,
    mut execute: impl FnMut(usize) -> Result<()>,
) -> Result<()> {
    for index in 0..count {
        ready(index)?;
        execute(index)?;
    }
    Ok(())
}

fn overlap_cpu_preparation<T: Send>(
    prepare: impl FnOnce() -> T + Send,
    execute_first: impl FnOnce() -> Result<()>,
) -> Result<T> {
    let prepared = Mutex::new(None);
    let result = rayon::in_place_scope(|scope| {
        let prepared = &prepared;
        scope.spawn(move |_| *prepared.lock().unwrap() = Some(prepare()));
        // Run on the calling thread: unified execution ownership is thread-local.
        execute_first()
    });
    result?;
    Ok(prepared
        .into_inner()
        .unwrap()
        .expect("scoped CPU preparation joined"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn independent_input_ranges_detect_suballocations_and_distinct_buffer_host_aliases() {
        use ash::vk::Handle;
        let a = InputRange {
            buffer: ash::vk::Buffer::from_raw(1),
            start: 64,
            end: 128,
            mapped: Some((1024, 1088)),
        };
        assert!(a.overlaps(a));
        assert!(!a.overlaps(InputRange {
            start: 128,
            end: 192,
            mapped: Some((1088, 1152)),
            ..a
        }));
        assert!(a.overlaps(InputRange {
            start: 127,
            end: 192,
            mapped: None,
            ..a
        }));
        assert!(a.overlaps(InputRange {
            buffer: ash::vk::Buffer::from_raw(2),
            start: 0,
            end: 64,
            mapped: Some((1087, 1151)),
        }));
        assert!(!a.overlaps(InputRange {
            buffer: ash::vk::Buffer::from_raw(2),
            start: 64,
            end: 128,
            mapped: None,
        }));
    }

    #[test]
    fn rolling_workspace_receipt_rejects_missing_and_short_capacities() {
        struct Bytes(usize);
        impl Buffer for Bytes {
            fn len_bytes(&self) -> usize {
                self.0
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let requests = WorkspaceRequests::from_maps(
            HashMap::from([("activation", 32)]),
            HashMap::from([("ids", 16)]),
        );
        let mut pool = ScratchPool::default();
        assert!(!requests.capacities_ready(&pool));
        pool.buffers.insert(("activation", 64), Box::new(Bytes(64)));
        assert!(!requests.capacities_ready(&pool));
        pool.buffers.insert(("ids", 16), Box::new(Bytes(8)));
        assert!(
            !requests.capacities_ready(&pool),
            "nominal key is not enough"
        );
        pool.buffers.insert(("ids", 16), Box::new(Bytes(16)));
        assert!(requests.capacities_ready(&pool));
        pool.buffers.remove(&("activation", 64));
        pool.buffers.insert(("activation", 16), Box::new(Bytes(16)));
        assert!(!requests.capacities_ready(&pool));
    }

    #[test]
    fn continuation_sequence_keeps_workspace_and_calls_inputs_without_its_lock() {
        use std::cell::RefCell;
        let caller = std::thread::current().id();
        let workspace = Mutex::new(0usize);
        let events = RefCell::new(Vec::new());
        let forward = RefCell::new(ForwardExecution::default());
        let mut ready = |index| {
            assert_eq!(caller, std::thread::current().id());
            assert_eq!(
                *workspace
                    .try_lock()
                    .expect("callback must be outside arena lock"),
                index
            );
            assert_eq!(forward.borrow().pstream.tape_cursor, index * 3);
            events.borrow_mut().push((index, "input"));
            Ok(())
        };
        run_sequence(3, &mut ready, |index| {
            let mut workspace = workspace.lock().unwrap();
            let mut forward = forward.borrow_mut();
            events.borrow_mut().push((index, "execute"));
            *workspace += 1;
            forward.pstream.tape_cursor += 3;
            Ok(())
        })
        .unwrap();
        assert_eq!(*workspace.lock().unwrap(), 3);
        assert_eq!(
            events.into_inner(),
            vec![
                (0, "input"),
                (0, "execute"),
                (1, "input"),
                (1, "execute"),
                (2, "input"),
                (2, "execute")
            ]
        );
    }

    #[test]
    fn continuation_failure_never_repeats_a_stage_or_calls_later_inputs() {
        use std::cell::RefCell;
        for failed in 0..3 {
            for fail_input in [false, true] {
                let events = RefCell::new(Vec::new());
                let result = run_sequence(
                    3,
                    &mut |index| {
                        events.borrow_mut().push((index, "input"));
                        if fail_input && index == failed {
                            Err(be("input failed"))
                        } else {
                            Ok(())
                        }
                    },
                    |index| {
                        events.borrow_mut().push((index, "execute"));
                        if !fail_input && index == failed {
                            Err(be("partially submitted stage"))
                        } else {
                            Ok(())
                        }
                    },
                );
                assert!(result.is_err());
                let events = events.into_inner();
                assert!(events.iter().all(|&(index, _)| index <= failed));
                for index in 0..=failed {
                    assert_eq!(
                        events
                            .iter()
                            .filter(|&&event| event == (index, "input"))
                            .count(),
                        1
                    );
                    assert_eq!(
                        events
                            .iter()
                            .filter(|&&event| event == (index, "execute"))
                            .count(),
                        usize::from(index < failed || !fail_input)
                    );
                }
            }
        }
        run_sequence(0, &mut |_| panic!("empty inputs"), |_| {
            panic!("empty execute")
        })
        .unwrap();
    }

    #[test]
    fn compiled_decode_requests_cache_all_layer_maxima_and_keep_ids_mapped() {
        let mut activation = HashMap::from([("split_pm", 4096), ("mmv_qa", 256)]);
        let mut staging = HashMap::new();
        // Cold top-1/top-2/top-1 must reserve the largest middle layer, not the first/last.
        for used in [1, 2, 1] {
            PagedMoeScratchGeometry {
                rows: 1,
                ne: 32,
                nff: 64,
                experts: 4,
                used,
                fused_gate_up: false,
                shared_slot: true,
            }
            .record(&mut activation, &mut staging, true);
        }
        assert_eq!(activation["moe_paged_g"], 3 * 64 * 4);
        assert_eq!(activation["moe_paged_u"], 3 * 64 * 4);
        assert_eq!(activation["moe_paged_a"], 3 * 64 * 4);
        assert_eq!(activation["moe_paged_y"], 3 * 32 * 4);
        assert_eq!(activation["moe_paged_wts"], 2 * 4);
        assert_eq!(staging["moe_paged_ids"], (2 + 2) * 4);
        assert!(!activation.contains_key("moe_paged_ids"));

        let expected = (activation.clone(), staging.clone());
        let cached = WorkspaceRequests::from_maps(activation, staging);
        let mut plan = VkDecodePlan::new_cpu(&Graph::new(), false, true);
        plan.phase_requests = Some(cached);
        plan.prepare().unwrap();
        let cached = plan.phase_requests.as_ref().unwrap();
        assert!(cached
            .activation
            .windows(2)
            .all(|pair| pair[0].0 < pair[1].0));
        // Both ordinary admission and staged union consume this exact immutable metadata.
        for _ in 0..2 {
            let mut activation = HashMap::new();
            let mut staging = HashMap::new();
            cached.merge_into(&mut activation, &mut staging);
            cached.merge_into(&mut activation, &mut staging);
            assert_eq!((&activation, &staging), (&expected.0, &expected.1));
        }
    }

    #[test]
    fn compiled_decode_requests_preserve_prefill_and_mtp_base_sizing() {
        assert!(complete_decode_requests(Some(RuntimePhase::Decode), false));
        for phase in [
            None,
            Some(RuntimePhase::Prefill),
            Some(RuntimePhase::Auxiliary),
        ] {
            assert!(!complete_decode_requests(phase, false));
        }
        assert!(!complete_decode_requests(Some(RuntimePhase::Decode), true));
        let base = HashMap::from([("split_pm", 64), ("prefill_workspace", 128)]);
        let cached = WorkspaceRequests::from_maps(base.clone(), HashMap::new());
        assert!(cached.staging.is_empty());
        assert_eq!(
            cached.activation.into_iter().collect::<HashMap<_, _>>(),
            base
        );
    }

    #[test]
    fn compiled_paged_geometry_keeps_fused_and_mmq_requests_bounded() {
        let geometry = PagedMoeScratchGeometry {
            rows: 2,
            ne: 32,
            nff: 64,
            experts: 8,
            used: 2,
            fused_gate_up: true,
            shared_slot: false,
        };
        let mut activation = HashMap::new();
        let mut staging = HashMap::new();
        geometry.record(&mut activation, &mut staging, false);
        assert!(!activation.contains_key("moe_paged_g"));
        assert_eq!(activation["moe_paged_logits"], 2 * 8 * 4);
        assert_eq!(staging["moe_paged_ids"], (4 + 4) * 4);
        geometry.record(&mut activation, &mut staging, true);
        assert_eq!(activation["moe_paged_g"], 4 * 2 * 64 * 4);
        assert!(!activation.contains_key("moe_paged_u"));
    }

    #[test]
    fn cpu_preparation_joins_and_execution_keeps_its_thread() {
        let caller = std::thread::current().id();
        let ready = AtomicBool::new(false);
        let value = overlap_cpu_preparation(
            || {
                ready.store(true, Ordering::Release);
                17
            },
            || {
                assert_eq!(std::thread::current().id(), caller);
                Ok(())
            },
        )
        .unwrap();
        assert!(ready.load(Ordering::Acquire));
        assert_eq!(value, 17);
    }

    #[test]
    fn partial_execution_error_joins_preparation_without_retrying() {
        let executes = AtomicUsize::new(0);
        let prepared = AtomicBool::new(false);
        let error = overlap_cpu_preparation(
            || prepared.store(true, Ordering::Release),
            || {
                executes.fetch_add(1, Ordering::Relaxed);
                Err(be("stage zero partially submitted"))
            },
        )
        .unwrap_err();
        assert!(prepared.load(Ordering::Acquire));
        assert_eq!(executes.load(Ordering::Relaxed), 1);
        assert!(error.to_string().contains("partially submitted"));
    }
}
