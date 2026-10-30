#![cfg(test)]

//! Child of recorder_owned_fused_tests; the parent adds its mod declaration separately.
//! Cold copied RAM payload, but deliberately ONE shared resident H-weight bank per case.
//! This isolates small-miss overlap, not a cold whole-model weight/cache workload.
use super::*;

#[path = "recorder_owned_copy_gu_fused_tests.rs"]
mod copy_gu_v2;

const COPIED_WORKING_SET: usize = 128 * 1024 * 1024;
const MAX_STAGES: usize = 512;
const MISS_EXPERT: usize = 0;
const NR2_VARIANTS: [(usize, usize); 2] = [(128, 2), (256, 2)];
const FOCUSED_VARIANTS: [(usize, usize); 4] = [(128, 2), (256, 2), (256, 4), (256, 8)];
const COPY_GROUPS: u32 = 8;
const SCRATCH_NAN: u32 = 0x7fc0_51ab;

fn copy_local(max_invocations: u32, max_x: u32) -> usize {
    if max_invocations >= 1024 && max_x >= 1024 {
        1024
    } else {
        256
    }
}

struct SparseCopyKernels {
    full: ComputeKernel,
    roles: ComputeKernel,
    local: usize,
}

impl SparseCopyKernels {
    fn new(be: &VulkanBackend) -> Self {
        let limits = unsafe {
            be.shared
                .instance
                .get_physical_device_properties(be.shared.physical_device)
        }
        .limits;
        let local = copy_local(
            limits.max_compute_work_group_invocations,
            limits.max_compute_work_group_size[0],
        );
        assert!(
            limits.max_compute_work_group_invocations >= local as u32
                && limits.max_compute_work_group_size[0] >= local as u32
        );
        let wide = local == 1024;
        Self {
            full: be.kernel(
                if wide {
                    "sparse_builtin_full_copy1024"
                } else {
                    "sparse_builtin_full_copy256"
                },
                crate::gemm::gpu_resident_copy_payload_spv(wide),
                2,
                4,
            ),
            roles: be.kernel(
                if wide {
                    "sparse_builtin_role_copy1024"
                } else {
                    "sparse_builtin_role_copy256"
                },
                crate::gemm::gpu_owned_role_copy_spv(wide),
                2,
                20,
            ),
            local,
        }
    }
}

fn sparse_kernels(
    be: &VulkanBackend,
    p: Profile,
    local: usize,
    nr: usize,
    copy: &SparseCopyKernels,
    grid_buffer: bool,
) -> Kernels {
    assert!(matches!(nr, 1 | 2 | 4 | 8));
    if grid_buffer {
        assert_eq!(local, 64);
        let mut kernels = Kernels::tree64_gu64_grid(be, p, nr);
        kernels.roles = copy.roles;
        return kernels;
    }
    // The parent's factory may still assert NR1/2. Reuse its bounded NR2 skeleton;
    // larger NR replaces ONLY the built-in GU, never requesting a FullHit pipeline.
    let mut kernels = Kernels::tree64_gu(be, p, local, 2);
    if nr != 2 {
        let spv = crate::gemm::gpu_owned_miss_gu_tree64_spv(p.gate_up, local as u32, nr as u32)
            .expect("requested built-in Tree64 GU must exist");
        let name = Box::leak(
            format!("sparse_builtin_tree64_{}_{local}_nr{nr}", format_name(p)).into_boxed_str(),
        );
        let gu = be.kernel_sg(name, spv, 4, 40, 32);
        kernels.gu = gu;
        kernels.hit = gu;
        kernels.gu_outputs = (local / 64) * nr;
    }
    // Explicit child-local override: do not inherit a narrow role-copy factory choice.
    kernels.roles = copy.roles;
    kernels
}

fn record_sparse_baseline(
    be: &VulkanBackend,
    fixtures: &[FusedFixture],
    copy: &SparseCopyKernels,
    query: &Timestamps<'_>,
) -> RecordedCmd {
    let rec = be.recorder_persistent().unwrap();
    assert!(
        !rec.no_barrier
            && !rec.full_barrier
            && !rec.prof_ops
            && !rec.stages
            && rec.timeline_query.borrow().is_none()
    );
    rec.seed_barrier();
    query.begin(&rec);
    for fixture in fixtures {
        assert!(fixture.shared.is_none());
        let f = &fixture.inner;
        // Exactly Fixture::record(CuPartial,false), except that the full-copy kernel
        // uses the SAME device-supported local size as the candidate's role copies.
        let independent = rec.independent_dispatch_scope();
        rec.dispatch(
            copy.full,
            &[
                Recorder::vkb(f.journal.as_ref()),
                Recorder::vkb(f.jobs.as_ref()),
            ],
            0,
            bytemuck::bytes_of(&(ORDINAL as u32)),
            COPY_GROUPS,
        );
        f.projections(&rec, 1, 0);
        drop(independent);
        f.tail(&rec, 1, 0);
        rec.global_barrier();
        if f.job_count != 0 {
            f.projections(&rec, 2, 0);
            f.tail(&rec, 2, 0);
        }
        rec.global_barrier();
    }
    query.end(&rec);
    rec.publish_host_reads();
    rec.finish_record().unwrap()
}

struct SharedBuffer(Arc<dyn Buffer>);

impl Buffer for SharedBuffer {
    fn len_bytes(&self) -> usize {
        self.0.len_bytes()
    }
    fn device_addr(&self) -> Option<u64> {
        self.0.device_addr()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self.0.as_any()
    }
}

fn shared(buffer: &Arc<dyn Buffer>) -> Box<dyn Buffer> {
    Box::new(SharedBuffer(Arc::clone(buffer)))
}

fn arena(be: &VulkanBackend, bytes: &[u8]) -> Arc<dyn Buffer> {
    let buffer = mapped(be, bytes.len());
    write(buffer.as_ref(), bytes);
    Arc::from(buffer)
}

#[derive(Clone, Copy, Debug)]
enum Miss {
    Gate,
    Down,
    ThreeRoles,
}

impl Miss {
    fn roles(self) -> &'static [usize] {
        match self {
            Self::Gate => &[0],
            Self::Down => &[2],
            Self::ThreeRoles => &[0, 1, 2],
        }
    }
}

fn variants(rows: usize, miss: Miss) -> &'static [(usize, usize)] {
    if rows == 1 && matches!(miss, Miss::Down | Miss::ThreeRoles) {
        &FOCUSED_VARIANTS
    } else {
        &NR2_VARIANTS
    }
}

fn require_builtin_variants(p: Profile, rows: usize, miss: Miss) {
    for &(local, nr) in variants(rows, miss) {
        assert!(
            crate::gemm::gpu_owned_miss_gu_tree64_spv(p.gate_up, local as u32, nr as u32).is_some(),
            "register the requested built-in before running: profile={} local={local} nr={nr}",
            p.name,
        );
    }
}

struct WeightLayout {
    ranges: [Vec<Range<usize>>; 3],
    bytes: Vec<u8>,
}

impl WeightLayout {
    fn new(p: Profile) -> Self {
        let mut bytes = vec![POISON; GUARD];
        let ranges = std::array::from_fn(|role| {
            let dtype = if role == 2 { p.down } else { p.gate_up };
            (0..p.used)
                .map(|expert| {
                    let weight = infr_testkit::synth_weight(
                        dtype,
                        p.ne * p.nff,
                        17 + (role * 31 + expert) as u32,
                    );
                    let start = bytes.len();
                    bytes.extend_from_slice(&weight);
                    let end = bytes.len();
                    assert_eq!((start | weight.len()) & 15, 0);
                    bytes.extend(std::iter::repeat_n(POISON, GUARD));
                    start..end
                })
                .collect()
        });
        Self { ranges, bytes }
    }
}

struct SparseRing {
    host: Arc<AlignedHostBuffer>,
    imported: Arc<dyn Buffer>,
    ranges: Vec<(usize, Range<usize>)>,
    span: usize,
    stages: usize,
    payload_per_stage: usize,
}

impl SparseRing {
    fn new(be: &VulkanBackend, p: Profile, miss: Miss, layout: &WeightLayout) -> Self {
        let mut span = GUARD;
        let ranges: Vec<_> = miss
            .roles()
            .iter()
            .map(|&role| {
                let start = span;
                span += layout.ranges[role][MISS_EXPERT].len() + GUARD;
                (role, start..span - GUARD)
            })
            .collect();
        let payload_per_stage = ranges.iter().map(|(_, r)| r.len()).sum::<usize>();
        let stages = COPIED_WORKING_SET / payload_per_stage + 1;
        assert!(
            (2..=MAX_STAGES).contains(&stages),
            "bounded sparse frame, do not expand the sweep"
        );
        assert!(stages * payload_per_stage > COPIED_WORKING_SET);
        let host = AlignedHostBuffer::new(span * stages).unwrap();
        let bytes = unsafe { std::slice::from_raw_parts_mut(host.as_ptr(), span * stages) };
        bytes.fill(POISON);
        for stage in 0..stages {
            for (role, range) in &ranges {
                let dtype = if *role == 2 { p.down } else { p.gate_up };
                let weight =
                    infr_testkit::synth_weight(dtype, p.ne * p.nff, Self::seed(stage, *role));
                assert_eq!(weight.len(), range.len());
                bytes[stage * span + range.start..stage * span + range.end]
                    .copy_from_slice(&weight);
            }
        }
        let imported = be
            .try_import_host_shard(
                be.shared
                    .external_memory_host
                    .as_ref()
                    .expect("real ImportedHost BDA required"),
                Arc::clone(&host),
                0,
                host.allocated_len(),
            )
            .unwrap()
            .buffer;
        assert!(matches!(
            &as_vk_buf(imported.as_ref()).unwrap().backing,
            Backing::ImportedHost { .. }
        ));
        assert!(
            imported.device_addr().is_some(),
            "no resident or staging source substitution"
        );
        Self {
            host,
            imported,
            ranges,
            span,
            stages,
            payload_per_stage,
        }
    }

    fn seed(stage: usize, role: usize) -> u32 {
        0x5000_0000 ^ ((stage as u32) << 8) ^ role as u32
    }

    fn bytes(&self, stage: usize, range: &Range<usize>) -> &[u8] {
        assert!(stage < self.stages && range.end <= self.span);
        unsafe {
            std::slice::from_raw_parts(
                self.host.as_ptr().add(stage * self.span + range.start),
                range.len(),
            )
        }
    }
}

struct Resident {
    weights: Arc<dyn Buffer>,
    tape: Arc<dyn Buffer>,
    ids: Arc<dyn Buffer>,
    input: Arc<dyn Buffer>,
    unused: Arc<dyn Buffer>,
    tape_bytes: Vec<u8>,
    ids_bytes: Vec<u8>,
    input_bytes: Vec<u8>,
    routed: Vec<u32>,
    hits: Vec<u32>,
}

impl Resident {
    fn new(be: &VulkanBackend, p: Profile, rows: usize, layout: &WeightLayout) -> Self {
        let weights = arena(be, &layout.bytes);
        let base = weights.device_addr().unwrap();
        let tape_words: Vec<u32> = layout
            .ranges
            .iter()
            .flatten()
            .flat_map(|r| {
                let address = base + r.start as u64;
                [address as u32, (address >> 32) as u32]
            })
            .collect();
        let tape_bytes = bytemuck::cast_slice(&tape_words).to_vec();
        let routed: Vec<_> = (0..rows)
            .flat_map(|_| (0..p.used).map(|e| e as u32))
            .collect();
        let all = (1u32 << p.used) - 1;
        let hits = vec![all ^ (1 << MISS_EXPERT); rows];
        let mut ids = routed.clone();
        ids.extend_from_slice(&hits);
        ids.extend(hits.iter().map(|mask| all ^ mask));
        let ids_bytes = bytemuck::cast_slice(&ids).to_vec();
        let input: Vec<f32> = (0..rows * p.ne)
            .map(|i| ((i as f32 * 0.037).sin() + (i as f32 * 0.013).cos()) * 0.005)
            .collect();
        let input_bytes = bytemuck::cast_slice(&input).to_vec();
        Self {
            weights,
            tape: arena(be, &tape_bytes),
            ids: arena(be, &ids_bytes),
            input: arena(be, &input_bytes),
            unused: arena(be, &vec![POISON; GUARD]),
            tape_bytes,
            ids_bytes,
            input_bytes,
            routed,
            hits,
        }
    }
}

fn scratch(be: &VulkanBackend, body: usize) -> Box<dyn Buffer> {
    let buffer = mapped(be, body + GUARD);
    let mut bytes = vec![POISON; body + GUARD];
    for word in bytes[..body].chunks_exact_mut(4) {
        word.copy_from_slice(&0.125f32.to_le_bytes());
    }
    write(buffer.as_ref(), &bytes);
    buffer
}

fn stage_fixture(
    be: &VulkanBackend,
    p: Profile,
    rows: usize,
    stage: usize,
    layout: &WeightLayout,
    ring: &SparseRing,
    resident: &Resident,
) -> FusedFixture {
    let mut job_words = vec![0u32; LAYERS * CAPACITY * 8];
    for (j, (role, range)) in ring.ranges.iter().enumerate() {
        let src = ring.imported.device_addr().unwrap() + (stage * ring.span + range.start) as u64;
        let target = &layout.ranges[*role][MISS_EXPERT];
        let dst = resident.weights.device_addr().unwrap() + target.start as u64;
        let at = (ORDINAL * CAPACITY + j) * 8;
        job_words[at..at + 8].copy_from_slice(&[
            src as u32,
            (src >> 32) as u32,
            dst as u32,
            (dst >> 32) as u32,
            target.len() as u32,
            (*role * p.used + MISS_EXPERT) as u32,
            j as u32,
            (*role * p.used + MISS_EXPERT) as u32,
        ]);
    }
    let jobs = mapped(be, job_words.len() * 4);
    write(jobs.as_ref(), bytemuck::cast_slice(&job_words));
    // Construct only stage-local scratch and job metadata. In particular, never call
    // Fixture::new/prepare/reference, which allocate/clone a complete weight bank per stage.
    let inner = Fixture {
        p,
        rows,
        weights: shared(&resident.weights),
        tape: shared(&resident.tape),
        ids: shared(&resident.ids),
        input: shared(&resident.input),
        gate: scratch(be, rows * p.used * p.nff * 4),
        up: scratch(be, rows * p.used * p.nff * 4),
        activation: scratch(be, rows * p.used * p.nff * 4),
        output: scratch(be, rows * p.used * p.ne * 4),
        journal: mapped(be, LAYERS * RESIDENT_JOURNAL_BYTES),
        jobs,
        commands: shared(&resident.unused),
        commands_late: shared(&resident.unused),
        early_ready: CopyReadyEvent::new(be),
        dummy: shared(&resident.unused),
        routed: resident.routed.clone(),
        hits: resident.hits.clone(),
        expected_weights: Vec::new(),
        job_words,
        job_count: ring.ranges.len(),
    };
    inner.set_status(4, inner.job_count);
    FusedFixture {
        inner,
        shared: None,
    }
}

fn assert_guards(f: &FusedFixture) {
    for buffer in [
        &f.inner.gate,
        &f.inner.up,
        &f.inner.activation,
        &f.inner.output,
    ] {
        let bytes = read(buffer.as_ref());
        assert!(
            bytes[bytes.len() - GUARD..].iter().all(|&b| b == POISON),
            "stage scratch guard"
        );
    }
}

fn reset_scratch_bytes(bytes: &mut [u8]) {
    let body = bytes.len().checked_sub(GUARD).expect("scratch guard");
    assert_eq!(body % 4, 0);
    assert!(bytes[body..].iter().all(|&b| b == POISON));
    for word in bytes[..body].chunks_exact_mut(4) {
        word.copy_from_slice(&SCRATCH_NAN.to_le_bytes());
    }
}

fn reset_candidate_state(
    fixtures: &[FusedFixture],
    resident: &Resident,
    layout: &WeightLayout,
    ring: &SparseRing,
) {
    // Call only after replay() has drained. Reset ALL H/M scratch, not just the final
    // output: the unmasked H SiLU must not reconstruct M activation from baseline G/U.
    for f in fixtures {
        for buffer in [
            &f.inner.gate,
            &f.inner.up,
            &f.inner.activation,
            &f.inner.output,
        ] {
            let ptr = as_vk_buf(buffer.as_ref()).unwrap().mapped_ptr().unwrap();
            reset_scratch_bytes(unsafe { std::slice::from_raw_parts_mut(ptr, buffer.len_bytes()) });
        }
    }
    let ptr = as_vk_buf(resident.weights.as_ref())
        .unwrap()
        .mapped_ptr()
        .unwrap();
    for (role, _) in &ring.ranges {
        let target = &layout.ranges[*role][MISS_EXPERT];
        // Only real copy destinations are invalidated. H weights and nonmissing M
        // roles must remain resident, including GU for the Down-only case.
        unsafe { std::ptr::write_bytes(ptr.add(target.start), POISON, target.len()) };
    }
}

fn assert_fresh_candidate_scratch(fixtures: &[FusedFixture]) {
    for (stage, f) in fixtures.iter().enumerate() {
        let f = &f.inner;
        let assignment_bytes = f.p.nff * 4;
        for buffer in [&f.gate, &f.up] {
            let bytes = read(buffer.as_ref());
            for row in 0..f.rows {
                for expert in 0..f.p.used {
                    if expert == MISS_EXPERT {
                        continue; // Fused M-GU writes ACT directly, not G/U scratch.
                    }
                    let at = (row * f.p.used + expert) * assignment_bytes;
                    assert!(
                        bytes[at..at + assignment_bytes]
                            .chunks_exact(4)
                            .all(|w| f32::from_le_bytes(w.try_into().unwrap()).is_finite()),
                        "candidate must rewrite H projections stage={stage} row={row} expert={expert}"
                    );
                }
            }
        }
        let activation = read(f.activation.as_ref());
        assert!(
            activation[..activation.len() - GUARD]
                .chunks_exact(4)
                .all(|w| f32::from_le_bytes(w.try_into().unwrap()).is_finite()),
            "candidate must rewrite H and M activation stage={stage}"
        );
    }
}

fn record_sparse_skipped_gu(
    be: &VulkanBackend,
    fixtures: &[FusedFixture],
    kernels: Kernels,
) -> RecordedCmd {
    let rec = be.recorder_persistent().unwrap();
    rec.seed_barrier();
    for f in fixtures {
        assert!(f.inner.job_count != 0);
        // Exact PipelinePartialHit schedule, except deliberately OMIT fused M-GU.
        let independent = rec.independent_dispatch_scope();
        f.role_copy(&rec, kernels.roles, 1);
        f.projections(&rec, 1, 0);
        drop(independent);
        f.activation(&rec);
        f.down(&rec, 1, 0);
        let independent = rec.independent_dispatch_scope();
        f.role_copy(&rec, kernels.roles, 2);
        drop(independent);
        f.down(&rec, 2, 0);
        rec.global_barrier();
    }
    rec.publish_host_reads();
    rec.finish_record().unwrap()
}

fn assert_skipped_gu_rejected(fixtures: &[FusedFixture], oracle: &[Vec<u8>]) {
    assert_eq!(fixtures.len(), oracle.len());
    for (stage, (f, expected)) in fixtures.iter().zip(oracle).enumerate() {
        let output = read(f.inner.output.as_ref());
        let activation = read(f.inner.activation.as_ref());
        for row in 0..f.inner.rows {
            for expert in 0..f.inner.p.used {
                let assignment = row * f.inner.p.used + expert;
                let at = assignment * f.inner.p.ne * 4;
                let end = at + f.inner.p.ne * 4;
                if expert == MISS_EXPERT {
                    assert_ne!(
                        &output[at..end],
                        &expected[at..end],
                        "skipped M-GU must fail stage={stage} row={row}"
                    );
                    let at = assignment * f.inner.p.nff * 4;
                    assert!(
                        activation[at..at + f.inner.p.nff * 4]
                            .chunks_exact(4)
                            .all(|w| f32::from_le_bytes(w.try_into().unwrap()).is_nan()),
                        "skipped M-GU must leave poisoned ACT stage={stage} row={row}"
                    );
                } else {
                    assert_eq!(
                        &output[at..end],
                        &expected[at..end],
                        "negative control must retain correct H output stage={stage} row={row} expert={expert}"
                    );
                }
            }
        }
        assert_guards(f);
    }
}

fn verify_final(
    fixtures: &[FusedFixture],
    oracle: &[Vec<u8>],
    resident: &Resident,
    layout: &mut WeightLayout,
    ring: &SparseRing,
) {
    for (stage, (f, reference)) in fixtures.iter().zip(oracle).enumerate() {
        assert_eq!(
            read(f.inner.output.as_ref()),
            *reference,
            "byte-exact stage output stage={stage}"
        );
        assert_guards(f);
        assert_eq!(
            read(f.inner.jobs.as_ref()),
            bytemuck::cast_slice::<u32, u8>(&f.inner.job_words),
            "immutable jobs stage={stage}"
        );
        let mut journal = vec![0u32; LAYERS * RESIDENT_JOURNAL_WORDS];
        let at = ORDINAL * RESIDENT_JOURNAL_WORDS;
        journal[at + RESIDENT_JOURNAL_STATUS_WORD] = 4;
        journal[at + RESIDENT_JOURNAL_JOB_COUNT_WORD] = f.inner.job_count as u32;
        journal[at + RESIDENT_JOURNAL_JOB_COUNT_WORD + 1] = 19;
        assert_eq!(
            read(f.inner.journal.as_ref()),
            bytemuck::cast_slice::<u32, u8>(&journal),
            "immutable journal stage={stage}"
        );
    }
    // One expected resident image for the whole case. H weights never change; only the
    // selected M-role slots contain the final stage's source blocks after a drained frame.
    for (role, range) in &ring.ranges {
        layout.bytes[layout.ranges[*role][MISS_EXPERT].clone()]
            .copy_from_slice(ring.bytes(ring.stages - 1, range));
    }
    assert_eq!(
        read(resident.weights.as_ref()),
        layout.bytes,
        "final M payload, all H weights and inter-matrix guards"
    );
    for (buffer, expected) in [
        (&resident.tape, &resident.tape_bytes),
        (&resident.ids, &resident.ids_bytes),
        (&resident.input, &resident.input_bytes),
    ] {
        assert_eq!(
            read(buffer.as_ref()),
            *expected,
            "immutable shared input/directory"
        );
    }
    assert_eq!(
        read(resident.unused.as_ref()),
        vec![POISON; GUARD],
        "unused indirect sidecars"
    );
}

fn sparse_case(be: &VulkanBackend, p: Profile, rows: usize, miss: Miss) {
    let selected: Vec<_> = variants(rows, miss)
        .iter()
        .map(|&(local, nr)| (local, nr, false, Arm::PipelinePartialHit))
        .collect();
    sparse_case_variants(be, p, rows, miss, &selected);
}

fn sparse_case_variants(
    be: &VulkanBackend,
    p: Profile,
    rows: usize,
    miss: Miss,
    selected: &[(usize, usize, bool, Arm)],
) {
    assert!(matches!(
        (p.gate_up, p.down, p.ne, p.nff, p.used),
        (DType::Q5K, DType::Q5K, 2048, 512, 8) | (DType::Iq2S, DType::Iq4Nl, 2560, 640, 10)
    ));
    assert!(rows == 1 || ((2..=8).contains(&rows) && matches!(miss, Miss::ThreeRoles)));
    assert_eq!(FRAME_PAIRS, 6);
    require_builtin_variants(p, rows, miss);
    let mut layout = WeightLayout::new(p);
    let ring = SparseRing::new(be, p, miss, &layout);
    let resident = Resident::new(be, p, rows, &layout);
    let fixtures: Vec<_> = (0..ring.stages)
        .map(|stage| stage_fixture(be, p, rows, stage, &layout, &ring, &resident))
        .collect();
    let query = Timestamps::new(be);
    // record_frame inserts a global barrier AFTER EVERY stage, so a later copy cannot
    // overwrite the shared M slots while an earlier stage still reads them for Down.
    let copy = SparseCopyKernels::new(be);
    let baseline = record_sparse_baseline(be, &fixtures, &copy, &query);
    for (role, _) in &ring.ranges {
        let target = &layout.ranges[*role][MISS_EXPERT];
        let ptr = as_vk_buf(resident.weights.as_ref())
            .unwrap()
            .mapped_ptr()
            .unwrap();
        unsafe { std::ptr::write_bytes(ptr.add(target.start), POISON, target.len()) };
    }
    // One original-CuPartial complete frame supplies each stage's byte-exact oracle.
    baseline.replay().unwrap();
    let oracle: Vec<_> = fixtures
        .iter()
        .map(|f| {
            let bytes = read(f.inner.output.as_ref());
            let values = &bytes[..bytes.len() - GUARD];
            assert!(values
                .chunks_exact(4)
                .all(|w| f32::from_le_bytes(w.try_into().unwrap()).is_finite()));
            assert!(values
                .chunks_exact(4)
                .any(|w| f32::from_le_bytes(w.try_into().unwrap()).abs() > 1e-7));
            bytes
        })
        .collect();
    let miss_output = MISS_EXPERT * p.ne * 4..(MISS_EXPERT + 1) * p.ne * 4;
    assert!(
        oracle
            .windows(2)
            .any(|pair| pair[0][miss_output.clone()] != pair[1][miss_output.clone()]),
        "stage-specific source data must produce observable M outputs"
    );
    verify_final(&fixtures, &oracle, &resident, &mut layout, &ring);
    println!(
        "OWNED_SPARSE_ID profile={} rows={rows} miss={miss:?} native_geometry=true synthetic_weights=true ne={} nff={} used={} source=ImportedHostBDA source_bda={} source_host_bytes={} source_allocated_bytes={} source_stage_span={} source_role_ranges={:?} source_seed=0x50000000_xor_stage_shift8_xor_role stages={} copied_bytes_per_stage={} copied_bytes_per_frame={} resident_weight_banks=1 resident_weight_bda={} resident_weight_bytes={} weight_ranges={:?} H_weights_shared=true whole_model_cold=false jobs_per_stage={} ordinal={ORDINAL} capacity={CAPACITY} job_stride=32",
        p.name,
        p.ne,
        p.nff,
        p.used,
        ring.imported.device_addr().unwrap(),
        ring.span * ring.stages,
        ring.host.allocated_len(),
        ring.span,
        ring.ranges,
        ring.stages,
        ring.payload_per_stage,
        ring.payload_per_stage * ring.stages,
        resident.weights.device_addr().unwrap(),
        layout.bytes.len(),
        layout.ranges,
        ring.ranges.len()
    );
    for (stage, f) in fixtures.iter().enumerate() {
        println!(
            "OWNED_SPARSE_JOB_DIRECTORY profile={} rows={rows} miss={miss:?} stage={stage} source_offset={} source_seeds={:?} journal_bda={} jobs_bda={} active_job_words={:?} output_bda={}",
            p.name,
            stage * ring.span,
            ring.ranges
                .iter()
                .map(|(role, _)| SparseRing::seed(stage, *role))
                .collect::<Vec<_>>(),
            f.inner.journal.device_addr().unwrap(),
            f.inner.jobs.device_addr().unwrap(),
            &f.inner.job_words
                [ORDINAL * CAPACITY * 8..ORDINAL * CAPACITY * 8 + f.inner.job_count * 8],
            f.inner.output.device_addr().unwrap()
        );
    }
    // Reuse the exact fixture, ring, weight bank, oracle and CuPartial recording for every variant.
    // Each comparison warms BOTH arms twice and has six balanced AB/BA pairs. No output/weight
    // downloads, fixture mutation or CPU quantization occurs inside measure_pair.
    for (variant, &(local, nr, grid_buffer, arm)) in selected.iter().enumerate() {
        let kernels = sparse_kernels(be, p, local, nr, &copy, grid_buffer);
        assert_eq!(
            (kernels.local, kernels.tiles, kernels.gu_outputs),
            (local, 1, (local / 64) * nr)
        );
        let candidate = record_frame(be, &fixtures, arm, kernels, &query);
        if variant == 0 {
            let skipped_gu = record_sparse_skipped_gu(be, &fixtures, kernels);
            reset_candidate_state(&fixtures, &resident, &layout, &ring);
            skipped_gu.replay().unwrap();
            assert_skipped_gu_rejected(&fixtures, &oracle);
            println!(
                "OWNED_SPARSE_NEGATIVE_CONTROL profile={} rows={rows} miss={miss:?} local={local} nr={nr} reset_all_scratch=true poison_selected_M_weights=true skipped_M_GU=true rejected_every_stage_row=true H_outputs_byteexact=true untimed=true",
                p.name
            );
        }
        let (base_us, candidate_us) =
            measure_pair(be, &baseline, &candidate, &query, fixtures.len());
        // The last timed arm may be baseline. Discard its scratch and copied weights
        // before a standalone candidate replay; no reset/readback enters measure_pair.
        reset_candidate_state(&fixtures, &resident, &layout, &ring);
        candidate.replay().unwrap();
        assert_fresh_candidate_scratch(&fixtures);
        verify_final(&fixtures, &oracle, &resident, &mut layout, &ring);
        let speedups: Vec<_> = base_us
            .iter()
            .zip(&candidate_us)
            .map(|(a, b)| a / b)
            .collect();
        println!(
            "OWNED_SPARSE_PIPELINE profile={} rows={rows} miss={miss:?} local={local} nr={nr} grid_buffer={grid_buffer} arm={arm:?} copy_local={} baseline_copy_local={} copy_groups={COPY_GROUPS} baseline_copy_groups={COPY_GROUPS} copy_invocations_per_dispatch={} copy_width_equal=true stages={} jobs={} copied_bytes_per_frame={} shared_resident_H_bank=true baseline=CuPartial reused_fixture_source_oracle=true warm_replays_per_arm=2 pairs=6 order=AB_BA timestamps=outer_only baseline_us_per_stage={base_us:?} candidate_us_per_stage={candidate_us:?} baseline_median_us={} candidate_median_us={} median_paired_speedup={} oracle=CuPartial_full_frame exact_stage_outputs=true exact_final_weights_guards=true correctness_untimed_fresh_reset=true reset_all_scratch=true poison_selected_M_weights=true candidate_only_replay=true H_projections_rewritten=true skipped_GU_negative_control=true",
            p.name,
            copy.local,
            copy.local,
            copy.local * COPY_GROUPS as usize,
            ring.stages,
            ring.ranges.len(),
            ring.stages * ring.payload_per_stage,
            median(&base_us),
            median(&candidate_us),
            median(&speedups)
        );
        drop(candidate);
    }
    drop(baseline); // Recordings drain before any shared buffer/source lease is released.
}

#[test]
#[ignore = "parent serial GPU; rows1 Down/three roles: 128NR2 and 256NR2/4/8; Gate stays NR2"]
fn gpu_owned_sparse_pipeline_native_rows1() {
    let be = fused_backend();
    for p in native_profiles() {
        for miss in [Miss::Gate, Miss::Down, Miss::ThreeRoles] {
            sparse_case(&be, p, 1, miss);
        }
    }
}

#[test]
#[ignore = "parent serial GPU; local128/256 NR2; same source/fixtures; three roles only, rows2"]
fn gpu_owned_sparse_pipeline_native_rows2_three_roles() {
    let be = fused_backend();
    for p in native_profiles() {
        sparse_case(&be, p, 2, Miss::ThreeRoles);
    }
}

#[test]
#[ignore = "parent serial GPU; matched native-width GU and copy; Down and three-role misses"]
fn gpu_owned_sparse_pipeline_local64_rows1() {
    let be = fused_backend();
    for p in native_profiles() {
        let selected: &[(usize, usize, bool, Arm)] = if p.gate_up == DType::Q5K {
            &[
                (64, 1, false, Arm::PipelinePartialHit),
                (64, 2, false, Arm::PipelinePartialHit),
            ]
        } else {
            &[
                (64, 2, false, Arm::PipelinePartialHit),
                (64, 2, true, Arm::PipelinePartialHit),
            ]
        };
        for miss in [Miss::Down, Miss::ThreeRoles] {
            sparse_case_variants(&be, p, 1, miss, selected);
        }
    }
}

#[test]
#[ignore = "parent serial GPU; matched GU width; two-row three-role misses"]
fn gpu_owned_sparse_pipeline_local64_rows2() {
    let be = fused_backend();
    for p in native_profiles() {
        let selected: &[(usize, usize, bool, Arm)] = if p.gate_up == DType::Q5K {
            &[
                (64, 1, false, Arm::PipelinePartialHit),
                (64, 2, false, Arm::PipelinePartialHit),
            ]
        } else {
            &[
                (64, 2, false, Arm::PipelinePartialHit),
                (64, 2, true, Arm::PipelinePartialHit),
            ]
        };
        sparse_case_variants(&be, p, 2, Miss::ThreeRoles, selected);
    }
}

fn phase2_entry_controls(rows: usize) {
    let be = fused_backend();
    let selected = [
        (256, 2, false, Arm::PipelinePartialHit),
        (256, 2, false, Arm::PipelinePartialHitNoEntry),
        (256, 2, false, Arm::PipelineNativeGu),
        (256, 2, false, Arm::PipelineNativeGuNoEntry),
    ];
    for p in native_profiles() {
        sparse_case_variants(&be, p, rows, Miss::ThreeRoles, &selected);
    }
}

#[test]
#[ignore = "parent serial GPU; same fixtures, kernels and copy width; isolate phase2 entry and native GU"]
fn gpu_owned_sparse_phase2_entry_controls_rows1() {
    phase2_entry_controls(1);
}

#[test]
#[ignore = "parent serial GPU; phase2 entry and native GU, two rows"]
fn gpu_owned_sparse_phase2_entry_controls_rows2() {
    phase2_entry_controls(2);
}

#[test]
#[ignore = "parent serial GPU; phase2 entry and native GU, four rows"]
fn gpu_owned_sparse_phase2_entry_controls_rows4() {
    phase2_entry_controls(4);
}

#[test]
fn sparse_pipeline_focused_variant_controls() {
    assert_eq!(copy_local(1024, 1024), 1024);
    assert_eq!(copy_local(1024, 512), 256);
    assert_eq!(copy_local(512, 1024), 256);
    assert_eq!(COPY_GROUPS, 8);
    assert_eq!(variants(1, Miss::Gate), &[(128, 2), (256, 2)]);
    for miss in [Miss::Down, Miss::ThreeRoles] {
        assert_eq!(variants(1, miss), &[(128, 2), (256, 2), (256, 4), (256, 8)]);
    }
    assert_eq!(variants(2, Miss::ThreeRoles), &[(128, 2), (256, 2)]);
    for p in native_profiles() {
        for miss in [Miss::Gate, Miss::Down, Miss::ThreeRoles] {
            require_builtin_variants(p, 1, miss);
        }
    }
}

#[test]
fn sparse_pipeline_fresh_reset_preserves_guards_and_poison_is_nan() {
    assert!(f32::from_bits(SCRATCH_NAN).is_nan());
    for words in [0, 1, 16, 4096] {
        let mut bytes = vec![POISON; words * 4 + GUARD];
        for word in bytes[..words * 4].chunks_exact_mut(4) {
            word.copy_from_slice(&0.125f32.to_le_bytes());
        }
        reset_scratch_bytes(&mut bytes);
        assert!(bytes[..words * 4]
            .chunks_exact(4)
            .all(|w| w == SCRATCH_NAN.to_le_bytes()));
        assert!(bytes[words * 4..].iter().all(|&b| b == POISON));
        let once = bytes.clone();
        reset_scratch_bytes(&mut bytes);
        assert_eq!(bytes, once);
    }
}

#[test]
#[should_panic]
fn sparse_pipeline_fresh_reset_rejects_corrupt_guard() {
    let mut bytes = vec![POISON; 4 + GUARD];
    bytes[4] ^= 1;
    reset_scratch_bytes(&mut bytes);
}

#[test]
#[ignore = "parent serial GPU; ordinary unfiltered payload only, no Tree64/fused/role-phase factory"]
fn gpu_owned_sparse_default_copy_remains_unfiltered() {
    let be = fused_backend();
    let limits = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    }
    .limits;
    let wide_supported = limits.max_compute_work_group_invocations >= 1024
        && limits.max_compute_work_group_size[0] >= 1024;
    let len = 2 * GUARD + 3 * 64;
    let source_bytes: Vec<u8> = (0..len)
        .map(|i| (i as u8).wrapping_mul(73).wrapping_add(19))
        .collect();
    let host = AlignedHostBuffer::new(len).unwrap();
    unsafe { std::ptr::copy_nonoverlapping(source_bytes.as_ptr(), host.as_ptr(), len) };
    let imported = be
        .try_import_host_shard(
            be.shared
                .external_memory_host
                .as_ref()
                .expect("ImportedHost required"),
            Arc::clone(&host),
            0,
            host.allocated_len(),
        )
        .unwrap()
        .buffer;
    assert!(matches!(
        &as_vk_buf(imported.as_ref()).unwrap().backing,
        Backing::ImportedHost { .. }
    ));
    let target = mapped(&be, len);
    let journal = mapped(&be, LAYERS * RESIDENT_JOURNAL_BYTES);
    let jobs = mapped(&be, LAYERS * CAPACITY * 32);
    for wide in [false, true] {
        if wide && !wide_supported {
            continue;
        }
        // Deliberately use the DEFAULT build and its FOUR-byte PC. No phase parameter,
        // external shader, Tree64 GU or COPY_ROLE_FILTER variant participates in this test.
        let kernel = be.kernel(
            if wide {
                "gpu_resident_copy_payload1024"
            } else {
                "gpu_resident_copy_payload"
            },
            crate::gemm::gpu_resident_copy_payload_spv(wide),
            2,
            4,
        );
        for miss in [Miss::Gate, Miss::Down, Miss::ThreeRoles] {
            for (status, cpu_first) in [(4u32, false), (5, false), (5, true)] {
                let mut expected = vec![POISON; len];
                write(target.as_ref(), &expected);
                let mut words = vec![0u32; LAYERS * CAPACITY * 8];
                for (j, &role) in miss.roles().iter().enumerate() {
                    let start = GUARD + role * 64;
                    let bytes = (role + 1) * 16;
                    let src = if cpu_first && j == 0 {
                        0
                    } else {
                        expected[start..start + bytes]
                            .copy_from_slice(&source_bytes[start..start + bytes]);
                        imported.device_addr().unwrap() + start as u64
                    };
                    let dst = target.device_addr().unwrap() + start as u64;
                    let at = (ORDINAL * CAPACITY + j) * 8;
                    words[at..at + 8].copy_from_slice(&[
                        src as u32,
                        (src >> 32) as u32,
                        dst as u32,
                        (dst >> 32) as u32,
                        bytes as u32,
                        (role * 16) as u32,
                        j as u32,
                        (role * 16) as u32,
                    ]);
                }
                write(jobs.as_ref(), bytemuck::cast_slice(&words));
                let mut journal_words = vec![0u32; LAYERS * RESIDENT_JOURNAL_WORDS];
                let base = ORDINAL * RESIDENT_JOURNAL_WORDS;
                journal_words[base + RESIDENT_JOURNAL_STATUS_WORD] = status;
                journal_words[base + RESIDENT_JOURNAL_JOB_COUNT_WORD] = miss.roles().len() as u32;
                write(journal.as_ref(), bytemuck::cast_slice(&journal_words));
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                rec.dispatch(
                    kernel,
                    &[
                        Recorder::vkb(journal.as_ref()),
                        Recorder::vkb(jobs.as_ref()),
                    ],
                    0,
                    bytemuck::bytes_of(&(ORDINAL as u32)),
                    8,
                );
                rec.global_barrier();
                rec.publish_host_reads();
                rec.finish().unwrap();
                assert_eq!(
                    read(target.as_ref()),
                    expected,
                    "default copy must include all nonzero-source roles: wide={wide} miss={miss:?} status={status} cpu_first={cpu_first}"
                );
                assert_eq!(
                    read(jobs.as_ref()),
                    bytemuck::cast_slice::<u32, u8>(&words),
                    "immutable jobs"
                );
                assert_eq!(
                    read(journal.as_ref()),
                    bytemuck::cast_slice::<u32, u8>(&journal_words),
                    "immutable journal"
                );
            }
        }
    }
    assert_eq!(
        unsafe { std::slice::from_raw_parts(host.as_ptr(), len) },
        source_bytes.as_slice(),
        "default copy never modifies imported source"
    );
}
