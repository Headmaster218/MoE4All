//! Private child of recorder_owned_sparse_pipeline_tests.rs; external SPIR-V only.
//! Host proofs apply to this immutable synthetic ABI, NOT arbitrary owner metadata.
use super::*;

const TASK_WORDS: usize = 8 + 2 * CAPACITY;
const W0: usize = 7;

fn trial_copy_groups(local: usize) -> u32 {
    if local == 1024 {
        return 8;
    }
    let groups = std::env::var("MOE4ALL_COPY_GU_GROUPS")
        .map(|value| value.parse::<u32>().expect("explicit copy workgroup count"))
        .unwrap_or(8);
    assert!(matches!(groups, 8 | 32));
    groups
}

fn role_copy_at_width(rec: &Recorder<'_>, f: &FusedFixture, copy: &SparseCopyKernels, phase: u32) {
    rec.dispatch(
        copy.roles,
        &[
            Recorder::vkb(f.inner.journal.as_ref()),
            Recorder::vkb(f.inner.jobs.as_ref()),
        ],
        0,
        bytemuck::cast_slice(&[
            ORDINAL as u32,
            phase,
            0,
            f.inner.p.used as u32,
            f.inner.p.used as u32,
        ]),
        trial_copy_groups(copy.local),
    );
}

#[derive(Clone, Copy, Debug)]
enum MixedArm {
    Separate256,
    Mixed256,
    Classified256,
    Separate1024,
}

struct MixedKernels {
    mixed: ComputeKernel,
    gu: ComputeKernel,
    classify: ComputeKernel,
    early: ComputeKernel,
    classified: ComputeKernel,
}

fn mixed_backend() -> VulkanBackend {
    assert_eq!(std::env::var("MOE4ALL_TEST_DEVICE").unwrap(), "Vulkan1");
    let be = fused_backend();
    let props = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    };
    let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy();
    println!("COPY_GU_V2_DEVICE actual={name} selected=Vulkan1");
    assert!(
        name.contains("RX 7900 XTX"),
        "wrong physical device: {name}"
    );
    assert!(props.limits.max_compute_work_group_invocations >= 1024);
    assert!(props.limits.max_compute_work_group_size[0] >= 1024);
    be
}

impl MixedKernels {
    fn new(be: &VulkanBackend, p: Profile) -> Self {
        let root = PathBuf::from(
            std::env::var_os("MOE4ALL_FUSION_V2_DIR")
                .expect("explicit external copy/GU SPIR-V directory required"),
        );
        let load = |variant: &str| {
            let file = format!("copy-gu-v2-{}-{variant}.spv", format_name(p));
            let bytes = std::fs::read(root.join(&file)).expect(&file);
            assert_eq!(bytes.len() % 4, 0);
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .collect();
            assert_eq!(words.first(), Some(&0x0723_0203));
            // Include the external root in the process-lifetime cache key.
            let name = Box::leak(format!("mixed_v2_{}_{file}", root.display()).into_boxed_str());
            be.kernel_sg(name, &words, 7, 64, 32)
        };
        Self {
            mixed: load("mixed"),
            gu: load("gu-control"),
            classify: load("classify"),
            early: load("early"),
            classified: load("classified"),
        }
    }
}

fn copy_at_width(be: &VulkanBackend, local: usize) -> SparseCopyKernels {
    assert!(matches!(local, 256 | 1024));
    let wide = local == 1024;
    SparseCopyKernels {
        full: be.kernel(
            if wide {
                "mixed_v2_full1024"
            } else {
                "mixed_v2_full256"
            },
            crate::gemm::gpu_resident_copy_payload_spv(wide),
            2,
            4,
        ),
        roles: be.kernel(
            if wide {
                "mixed_v2_roles1024"
            } else {
                "mixed_v2_roles256"
            },
            crate::gemm::gpu_owned_role_copy_spv(wide),
            2,
            20,
        ),
        local,
    }
}

fn tasks_buffer(be: &VulkanBackend) -> Box<dyn Buffer> {
    let buffer = be.alloc_mapped_predicate(TASK_WORDS * 4 + GUARD).unwrap();
    let mut bytes = vec![POISON; buffer.len_bytes()];
    bytes[..TASK_WORDS * 4].fill(0);
    write(buffer.as_ref(), &bytes);
    buffer
}

fn span(buffer: &dyn Buffer) -> std::result::Result<Range<u64>, String> {
    let base = buffer.device_addr().ok_or("BDA missing")?;
    let end = base
        .checked_add(buffer.len_bytes() as u64)
        .ok_or("BDA overflow")?;
    if base == 0 || end <= base {
        return Err("invalid BDA span".into());
    }
    Ok(base..end)
}

fn intersects(a: &Range<u64>, b: &Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}

// Prepared once, outside GPU timing. All backing allocations and ImportedHost leases
// outlive each recorded CB and replay(). No CPU source split/preclassification occurs.
// This proof deliberately rejects bad aliases instead of repeating a full-bank scan
// in hundreds of GU WGs. Mutation by a real owner producer is OUTSIDE this module ABI.
fn prove_stage(
    f: &FusedFixture,
    tasks: &dyn Buffer,
    resident: &Resident,
    layout: &WeightLayout,
    ring: &SparseRing,
    stage: usize,
) -> std::result::Result<(), String> {
    let i = &f.inner;
    let p = i.p;
    if !(1..=8).contains(&i.rows)
        || !(1..=16).contains(&p.used)
        || p.ne == 0
        || p.ne > 65536
        || !p.ne.is_multiple_of(256)
        || p.nff == 0
        || p.nff > 1024
        || !p.nff.is_multiple_of(32)
        || i.job_count > CAPACITY
        || stage >= ring.stages
        || tasks.len_bytes() < TASK_WORDS * 4 + GUARD
        || i.activation.len_bytes() < i.rows * f.physical_used() * p.nff * 4 + GUARD
        || i.input.len_bytes() < i.rows * p.ne * 4
        || i.ids.len_bytes() < i.rows * (p.used + 2) * 4
    {
        return Err("geometry/count".into());
    }
    let allocations = [
        i.weights.as_ref(),
        i.tape.as_ref(),
        i.ids.as_ref(),
        i.input.as_ref(),
        i.gate.as_ref(),
        i.up.as_ref(),
        i.activation.as_ref(),
        i.output.as_ref(),
        i.jobs.as_ref(),
        i.journal.as_ref(),
        tasks,
        ring.imported.as_ref(),
    ];
    let spans: Vec<_> = allocations
        .iter()
        .map(|b| span(*b))
        .collect::<std::result::Result<_, _>>()?;
    for (n, a) in spans.iter().enumerate() {
        if spans[..n].iter().any(|b| intersects(a, b)) {
            return Err("allocation alias".into());
        }
    }
    if let Some(shared) = &f.shared {
        for matrix in &shared.matrices {
            let a = span(matrix.as_ref())?;
            if spans.iter().any(|b| intersects(&a, b)) {
                return Err("shared alias".into());
            }
        }
    }
    if read(i.tape.as_ref()) != resident.tape_bytes
        || read(i.jobs.as_ref()) != bytemuck::cast_slice::<u32, u8>(&i.job_words)
    {
        return Err("immutable metadata mismatch".into());
    }
    let journal = read(i.journal.as_ref());
    let at = (ORDINAL * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_STATUS_WORD) * 4;
    let status = u32::from_le_bytes(journal[at..at + 4].try_into().unwrap());
    let count_at = (ORDINAL * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_JOB_COUNT_WORD) * 4;
    let count = u32::from_le_bytes(journal[count_at..count_at + 4].try_into().unwrap());
    if count as usize != i.job_count {
        return Err("immutable job count mismatch".into());
    }
    let base = i.weights.device_addr().unwrap();
    let mut banks = Vec::new();
    let tape = read(i.tape.as_ref());
    for role in 0..3 {
        for expert in 0..p.used {
            let r = &layout.ranges[role][expert];
            let address = u64::from_le_bytes(
                tape[(role * p.used + expert) * 8..(role * p.used + expert + 1) * 8]
                    .try_into()
                    .unwrap(),
            );
            if address != base + r.start as u64
                || r.end > i.weights.len_bytes()
                || (r.start | r.len()) & 15 != 0
            {
                return Err("bank lease/range".into());
            }
            let a = address..address + r.len() as u64;
            if banks.iter().any(|b| intersects(&a, b)) {
                return Err("bank alias".into());
            }
            banks.push(a);
        }
    }
    let mut seen = Vec::new();
    for j in 0..i.job_count {
        let w = &i.job_words[(ORDINAL * CAPACITY + j) * 8..][..8];
        let block = w[5] as usize;
        if block >= 3 * p.used || seen.contains(&block) {
            return Err("block/duplicate".into());
        }
        seen.push(block);
        let role = block / p.used;
        let expert = block % p.used;
        let target = &layout.ranges[role][expert];
        let src = w[0] as u64 | ((w[1] as u64) << 32);
        let dst = w[2] as u64 | ((w[3] as u64) << 32);
        if expert != MISS_EXPERT
            || dst != base + target.start as u64
            || w[4] as usize != target.len()
        {
            return Err("job target bank".into());
        }
        if src != 0 {
            let (_, r) = ring
                .ranges
                .iter()
                .find(|(r, _)| *r == role)
                .ok_or("source role")?;
            let expected =
                ring.imported.device_addr().unwrap() + (stage * ring.span + r.start) as u64;
            if src != expected || r.len() != target.len() {
                return Err("source lease/range".into());
            }
        } else if status != 5 {
            return Err("CPU-only job requires status5".into());
        }
    }
    Ok(())
}

fn dispatch_v2(
    rec: &Recorder<'_>,
    f: &FusedFixture,
    tasks: &dyn Buffer,
    kernel: ComputeKernel,
    groups: u32,
    layout: &WeightLayout,
) {
    let i = &f.inner;
    let p = i.p;
    let push = [
        p.ne as u32,
        p.nff as u32,
        i.rows as u32,
        p.used as u32,
        p.used as u32,
        f.physical_used() as u32,
        0,
        p.used as u32,
        (2 * p.used) as u32,
        ORDINAL as u32,
        trial_copy_groups(256),
        0,
        p.used as u32,
        (2 * p.used) as u32,
        layout.ranges[0][0].len() as u32,
        layout.ranges[2][0].len() as u32,
    ];
    rec.dispatch(
        kernel,
        &[
            Recorder::vkb(i.journal.as_ref()),
            Recorder::vkb(i.jobs.as_ref()),
            Recorder::vkb(i.input.as_ref()),
            Recorder::vkb(i.ids.as_ref()),
            Recorder::vkb(i.tape.as_ref()),
            Recorder::vkb(i.activation.as_ref()),
            Recorder::vkb(tasks),
        ],
        2,
        bytemuck::cast_slice(&push),
        groups,
    );
}

fn phase1(
    rec: &Recorder<'_>,
    f: &FusedFixture,
    tasks: &dyn Buffer,
    arm: MixedArm,
    k: &MixedKernels,
    copy: &SparseCopyKernels,
    layout: &WeightLayout,
) {
    if matches!(arm, MixedArm::Classified256) {
        dispatch_v2(rec, f, tasks, k.classify, 1, layout);
        rec.global_barrier(); // Included in OUTER timing, never free host classification.
    }
    let independent = rec.independent_dispatch_scope();
    if matches!(arm, MixedArm::Classified256) {
        dispatch_v2(rec, f, tasks, k.early, trial_copy_groups(256), layout);
    } else {
        role_copy_at_width(rec, f, copy, 1);
    }
    f.projections(rec, 1, 0);
    drop(independent);
}

fn phase2(
    rec: &Recorder<'_>,
    f: &FusedFixture,
    tasks: &dyn Buffer,
    arm: MixedArm,
    k: &MixedKernels,
    copy: &SparseCopyKernels,
    layout: &WeightLayout,
) {
    let gu_groups = (f.inner.rows * f.inner.p.used * f.inner.p.nff.div_ceil(8)) as u32;
    let independent = rec.independent_dispatch_scope();
    match arm {
        MixedArm::Separate256 | MixedArm::Separate1024 => {
            role_copy_at_width(rec, f, copy, 2);
            dispatch_v2(rec, f, tasks, k.gu, gu_groups, layout);
        }
        MixedArm::Mixed256 => dispatch_v2(
            rec,
            f,
            tasks,
            k.mixed,
            trial_copy_groups(256) + gu_groups,
            layout,
        ),
        MixedArm::Classified256 => dispatch_v2(
            rec,
            f,
            tasks,
            k.classified,
            trial_copy_groups(256) + gu_groups,
            layout,
        ),
    }
    drop(independent); // Unconditional global join, also when W0=0/status5.
    rec.begin_immutable_conditional_at(tasks, W0);
    f.down(rec, 2, 0);
    rec.end_conditional();
}

fn record_mixed(
    be: &VulkanBackend,
    fixtures: &[FusedFixture],
    tasks: &[Box<dyn Buffer>],
    arm: MixedArm,
    k: &MixedKernels,
    copy: &SparseCopyKernels,
    resident: &Resident,
    layout: &WeightLayout,
    ring: &SparseRing,
    query: &Timestamps<'_>,
) -> RecordedCmd {
    assert_eq!(fixtures.len(), tasks.len());
    for (stage, (f, t)) in fixtures.iter().zip(tasks).enumerate() {
        prove_stage(f, t.as_ref(), resident, layout, ring, stage).unwrap();
    }
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
    for (f, t) in fixtures.iter().zip(tasks) {
        phase1(&rec, f, t.as_ref(), arm, k, copy, layout);
        f.activation(&rec);
        f.down(&rec, 1, 0);
        phase2(&rec, f, t.as_ref(), arm, k, copy, layout);
        rec.global_barrier(); // Shared resident miss slots cannot be reused across stages early.
    }
    query.end(&rec);
    rec.publish_host_reads();
    rec.finish_record().unwrap()
}

fn small_ring(be: &VulkanBackend, p: Profile, layout: &WeightLayout) -> SparseRing {
    let mut span = GUARD;
    let ranges: Vec<_> = (0..3)
        .map(|role| {
            let start = span;
            span += layout.ranges[role][MISS_EXPERT].len() + GUARD;
            (role, start..span - GUARD)
        })
        .collect();
    let stages = 3;
    let host = AlignedHostBuffer::new(span * stages).unwrap();
    let bytes = unsafe { std::slice::from_raw_parts_mut(host.as_ptr(), span * stages) };
    bytes.fill(POISON);
    for stage in 0..stages {
        for (role, range) in &ranges {
            let dtype = if *role == 2 { p.down } else { p.gate_up };
            let weight =
                infr_testkit::synth_weight(dtype, p.ne * p.nff, SparseRing::seed(stage, *role));
            bytes[stage * span + range.start..stage * span + range.end].copy_from_slice(&weight);
        }
    }
    let imported = be
        .try_import_host_shard(
            be.shared.external_memory_host.as_ref().unwrap(),
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
    SparseRing {
        host,
        imported,
        payload_per_stage: ranges.iter().map(|(_, r)| r.len()).sum(),
        ranges,
        span,
        stages,
    }
}

fn reset_one(f: &FusedFixture, resident: &Resident, layout: &WeightLayout) {
    for b in [
        &f.inner.gate,
        &f.inner.up,
        &f.inner.activation,
        &f.inner.output,
    ] {
        let mut bytes = read(b.as_ref());
        reset_scratch_bytes(&mut bytes);
        write(b.as_ref(), &bytes);
    }
    let mut weights = layout.bytes.clone();
    for role in 0..3 {
        weights[layout.ranges[role][MISS_EXPERT].clone()].fill(POISON);
    }
    write(resident.weights.as_ref(), &weights);
}

fn assert_tasks(tasks: &dyn Buffer, ready: bool) {
    let bytes = read(tasks);
    assert_eq!(
        u32::from_le_bytes(bytes[W0 * 4..W0 * 4 + 4].try_into().unwrap()),
        u32::from(ready)
    );
    assert!(bytes[TASK_WORDS * 4..].iter().all(|&b| b == POISON));
}

fn expected_payload(
    f: &FusedFixture,
    layout: &WeightLayout,
    ring: &SparseRing,
    stage: usize,
    initial: &[u8],
    enabled: bool,
) -> Vec<u8> {
    let mut expected = initial.to_vec();
    if enabled {
        for j in 0..f.inner.job_count {
            let w = &f.inner.job_words[(ORDINAL * CAPACITY + j) * 8..][..8];
            if w[0] == 0 && w[1] == 0 {
                continue;
            }
            let role = w[5] as usize / f.inner.p.used;
            let (_, range) = ring.ranges.iter().find(|(r, _)| *r == role).unwrap();
            expected[layout.ranges[role][MISS_EXPERT].clone()]
                .copy_from_slice(ring.bytes(stage, range));
        }
    }
    expected
}

#[test]
#[ignore = "external SPIR-V; explicit Vulkan1 RX 7900 XTX lease required"]
fn gpu_owned_copy_gu_v2_exact_rows_1_2_4_8_shared() {
    let be = mixed_backend();
    let copy = copy_at_width(&be, 256);
    let query = Timestamps::new(&be);
    for mut p in native_profiles() {
        p.ne = 256;
        p.nff = 256;
        let layout = WeightLayout::new(p);
        let ring = small_ring(&be, p, &layout);
        let k = MixedKernels::new(&be, p);
        for rows in [1, 2, 4, 8] {
            for shared_slot in [false, true] {
                let resident = Resident::new(&be, p, rows, &layout);
                let mut f = stage_fixture(&be, p, rows, 0, &layout, &ring, &resident);
                if shared_slot {
                    f.shared = Some(SharedWeights::new(&be, p));
                    let n = rows * (p.used + 1);
                    f.inner.gate = scratch(&be, n * p.nff * 4);
                    f.inner.up = scratch(&be, n * p.nff * 4);
                    f.inner.activation = scratch(&be, n * p.nff * 4);
                    f.inner.output = scratch(&be, n * p.ne * 4);
                    f.shared_masks();
                }
                let tasks = tasks_buffer(&be);
                prove_stage(&f, tasks.as_ref(), &resident, &layout, &ring, 0).unwrap();
                let ids_before = read(f.inner.ids.as_ref());
                let shared_before: Vec<_> = f
                    .shared
                    .as_ref()
                    .map(|s| s.matrices.iter().map(|m| read(m.as_ref())).collect())
                    .unwrap_or_default();
                // Fully resident NATIVE G/U -> SiLU -> Down oracle, not another fused GU.
                let weights = expected_payload(&f, &layout, &ring, 0, &layout.bytes, true);
                write(resident.weights.as_ref(), &weights);
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                let all = (1u32 << f.physical_used()) - 1;
                f.projections(&rec, 0, all);
                f.activation(&rec);
                f.down(&rec, 0, all);
                rec.publish_host_reads();
                rec.finish().unwrap();
                let act = read(f.inner.activation.as_ref());
                let output = read(f.inner.output.as_ref());
                assert!(output[..output.len() - GUARD]
                    .chunks_exact(4)
                    .all(|w| f32::from_le_bytes(w.try_into().unwrap()).is_finite()));
                for arm in [
                    MixedArm::Separate256,
                    MixedArm::Mixed256,
                    MixedArm::Classified256,
                ] {
                    reset_one(&f, &resident, &layout);
                    let fixtures = std::slice::from_ref(&f);
                    let t = vec![tasks_buffer(&be)];
                    let cmd = record_mixed(
                        &be, fixtures, &t, arm, &k, &copy, &resident, &layout, &ring, &query,
                    );
                    cmd.replay().unwrap();
                    assert_eq!(
                        read(f.inner.activation.as_ref()),
                        act,
                        "native ACT {p:?} rows={rows} {arm:?} shared={shared_slot}"
                    );
                    assert_eq!(read(f.inner.output.as_ref()), output, "native Y {arm:?}");
                    assert_eq!(read(resident.weights.as_ref()), weights);
                    assert_eq!(read(f.inner.ids.as_ref()), ids_before);
                    assert_eq!(read(f.inner.tape.as_ref()), resident.tape_bytes);
                    assert_eq!(read(f.inner.input.as_ref()), resident.input_bytes);
                    assert_eq!(
                        read(f.inner.jobs.as_ref()),
                        bytemuck::cast_slice::<u32, u8>(&f.inner.job_words)
                    );
                    for (matrix, expected) in f
                        .shared
                        .as_ref()
                        .into_iter()
                        .flat_map(|s| &s.matrices)
                        .zip(&shared_before)
                    {
                        assert_eq!(read(matrix.as_ref()), *expected);
                    }
                    assert_tasks(t[0].as_ref(), true);
                    assert_guards(&f);
                    drop(cmd);
                    drop(t);
                }
            }
        }
    }
}

#[test]
#[ignore = "external SPIR-V; explicit Vulkan1 RX 7900 XTX lease required"]
fn gpu_owned_copy_gu_v2_protocol_status5_zero_masks_alias() {
    let be = mixed_backend();
    let copy = copy_at_width(&be, 256);
    let mut p = native_profiles()[0];
    p.ne = 256;
    p.nff = 256;
    let layout = WeightLayout::new(p);
    let ring = small_ring(&be, p, &layout);
    let k = MixedKernels::new(&be, p);
    for rows in [1, 2, 4, 8] {
        // No synthetic CPU-ready predicate: status5 keeps W0=0 throughout.
        for (status, count, mask_zero) in [
            (5, 3, false),
            (0, 3, false),
            (4, 0, false),
            (1, 0, true),
            (4, 3, true),
        ] {
            let resident = Resident::new(&be, p, rows, &layout);
            let mut f = stage_fixture(&be, p, rows, 0, &layout, &ring, &resident);
            f.inner.job_count = count;
            if status == 5 {
                let at = ORDINAL * CAPACITY * 8;
                f.inner.job_words[at..at + 2].fill(0); // Gate CPU-only; Up AND Down still mapped.
                write(
                    f.inner.jobs.as_ref(),
                    bytemuck::cast_slice(&f.inner.job_words),
                );
                assert!(f.inner.job_words[at + 8] != 0 || f.inner.job_words[at + 9] != 0);
                assert!(f.inner.job_words[at + 16] != 0 || f.inner.job_words[at + 17] != 0);
            }
            if mask_zero {
                if status == 1 {
                    f.inner.hits.fill((1 << p.used) - 1);
                    f.inner.write_masks();
                } else {
                    let mut ids = f.inner.routed.clone();
                    ids.extend_from_slice(&f.inner.hits);
                    ids.extend(std::iter::repeat_n(0u32, rows));
                    write(f.inner.ids.as_ref(), bytemuck::cast_slice(&ids));
                }
            }
            f.inner.set_status(status, count);
            let ids = read(f.inner.ids.as_ref());
            let journal = read(f.inner.journal.as_ref());
            for arm in [
                MixedArm::Separate256,
                MixedArm::Mixed256,
                MixedArm::Classified256,
            ] {
                reset_one(&f, &resident, &layout);
                if status == 1 {
                    write(resident.weights.as_ref(), &layout.bytes);
                }
                let initial = read(resident.weights.as_ref());
                let tasks = tasks_buffer(&be);
                // Stale ready must not survive SKIP/zero-job/status5 dispatches.
                let mut task_bytes = read(tasks.as_ref());
                task_bytes[W0 * 4..W0 * 4 + 4].copy_from_slice(&1u32.to_le_bytes());
                write(tasks.as_ref(), &task_bytes);
                prove_stage(&f, tasks.as_ref(), &resident, &layout, &ring, 0).unwrap();
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                phase1(&rec, &f, tasks.as_ref(), arm, &k, &copy, &layout);
                // ACT intentionally left at exact NaN poison: isolate W0 and M mask,
                // rather than manufacturing an all-M status5 numerical oracle.
                phase2(&rec, &f, tasks.as_ref(), arm, &k, &copy, &layout);
                rec.publish_host_reads();
                rec.finish().unwrap();
                let expected = expected_payload(
                    &f,
                    &layout,
                    &ring,
                    0,
                    &initial,
                    (status == 4 || status == 5) && count != 0,
                );
                assert_eq!(
                    read(resident.weights.as_ref()),
                    expected,
                    "mapped payload {arm:?} status={status}"
                );
                assert!(
                    read(f.inner.activation.as_ref())[..rows * p.used * p.nff * 4]
                        .chunks_exact(4)
                        .all(|w| w == SCRATCH_NAN.to_le_bytes()),
                    "W0=0 OR mask=0 must retain exact ACT poison"
                );
                assert!(read(f.inner.output.as_ref())[..rows * p.used * p.ne * 4]
                    .chunks_exact(4)
                    .all(|w| w == SCRATCH_NAN.to_le_bytes()));
                assert_tasks(tasks.as_ref(), status == 4 && count != 0);
                assert_guards(&f);
                assert_eq!(read(f.inner.ids.as_ref()), ids);
                assert_eq!(
                    read(f.inner.journal.as_ref()),
                    journal,
                    "copy does not publish READY"
                );
                assert_eq!(
                    read(f.inner.jobs.as_ref()),
                    bytemuck::cast_slice::<u32, u8>(&f.inner.job_words)
                );
            }
        }
    }
    // Bad aliases fail host admission BEFORE any CB is created. No costly GPU bank
    // validation is added to either timing arm just to support this negative test.
    let resident = Resident::new(&be, p, 1, &layout);
    let mut f = stage_fixture(&be, p, 1, 0, &layout, &ring, &resident);
    let tasks = tasks_buffer(&be);
    prove_stage(&f, tasks.as_ref(), &resident, &layout, &ring, 0).unwrap();
    let old_act = std::mem::replace(&mut f.inner.activation, shared(&resident.input));
    assert!(prove_stage(&f, tasks.as_ref(), &resident, &layout, &ring, 0).is_err());
    f.inner.activation = old_act;
    let down = (ORDINAL * CAPACITY + 2) * 8;
    let gate = ORDINAL * CAPACITY * 8;
    f.inner.job_words[down + 2] = f.inner.job_words[gate + 2];
    f.inner.job_words[down + 3] = f.inner.job_words[gate + 3];
    write(
        f.inner.jobs.as_ref(),
        bytemuck::cast_slice(&f.inner.job_words),
    );
    assert!(prove_stage(&f, tasks.as_ref(), &resident, &layout, &ring, 0).is_err());
    println!(
        "COPY_GU_V2_PROTOCOL W0_status5_zero=true mapped_up_down_copied=true CPU_gate_poison=true READY_not_published=true aliases_host_rejected=true actual_adapter_qualification=false"
    );
}

#[test]
#[ignore = "external SPIR-V; geometry rejection must clear a prior ready predicate"]
fn gpu_owned_copy_gu_v2_rejection_clears_stale_w0() {
    let be = mixed_backend();
    let mut p = native_profiles()[0];
    p.ne = 256;
    p.nff = 256;
    let layout = WeightLayout::new(p);
    let ring = small_ring(&be, p, &layout);
    let resident = Resident::new(&be, p, 1, &layout);
    let mut fixture = stage_fixture(&be, p, 1, 0, &layout, &ring, &resident);
    let kernels = MixedKernels::new(&be, p);
    let tasks = tasks_buffer(&be);
    let initial_activation = read(fixture.inner.activation.as_ref());
    let initial_weights = read(resident.weights.as_ref());
    for fault in 0..4 {
        fixture.inner.p = p;
        fixture.inner.rows = 1;
        match fault {
            0 => fixture.inner.p.ne = 65792,
            1 => fixture.inner.rows = 0,
            2 => fixture.inner.p.nff = 1056,
            _ => fixture.inner.p.ne = 257,
        }
        assert!(prove_stage(&fixture, tasks.as_ref(), &resident, &layout, &ring, 0).is_err());
        for kernel in [
            kernels.mixed,
            kernels.gu,
            kernels.classify,
            kernels.early,
            kernels.classified,
        ] {
            let mut bytes = read(tasks.as_ref());
            bytes[W0 * 4..W0 * 4 + 4].copy_from_slice(&1u32.to_le_bytes());
            write(tasks.as_ref(), &bytes);
            let rec = be.recorder().unwrap();
            rec.seed_barrier();
            // Deliberately bypass the host rejection solely to verify the GPU rejection path.
            dispatch_v2(&rec, &fixture, tasks.as_ref(), kernel, 1, &layout);
            rec.publish_host_reads();
            rec.finish().unwrap();
            bytes[W0 * 4..W0 * 4 + 4].copy_from_slice(&0u32.to_le_bytes());
            assert_eq!(read(tasks.as_ref()), bytes, "fault={fault}");
            assert_eq!(read(fixture.inner.activation.as_ref()), initial_activation);
            assert_eq!(read(resident.weights.as_ref()), initial_weights);
        }
    }
}

fn mixed_frame_probe(rows: usize) {
    let be = mixed_backend();
    let query = Timestamps::new(&be);
    let narrow = copy_at_width(&be, 256);
    let wide = copy_at_width(&be, 1024);
    for p in native_profiles() {
        let mut layout = WeightLayout::new(p);
        let ring = SparseRing::new(&be, p, Miss::ThreeRoles, &layout);
        let resident = Resident::new(&be, p, rows, &layout);
        let fixtures: Vec<_> = (0..ring.stages)
            .map(|stage| stage_fixture(&be, p, rows, stage, &layout, &ring, &resident))
            .collect();
        let tasks: Vec<_> = (0..ring.stages).map(|_| tasks_buffer(&be)).collect();
        let k = MixedKernels::new(&be, p);
        let baseline = record_sparse_baseline(&be, &fixtures, &wide, &query);
        reset_candidate_state(&fixtures, &resident, &layout, &ring);
        baseline.replay().unwrap();
        let oracle: Vec<_> = fixtures
            .iter()
            .map(|f| read(f.inner.output.as_ref()))
            .collect();
        verify_final(&fixtures, &oracle, &resident, &mut layout, &ring);
        let separate = record_mixed(
            &be,
            &fixtures,
            &tasks,
            MixedArm::Separate256,
            &k,
            &narrow,
            &resident,
            &layout,
            &ring,
            &query,
        );
        // Deliberately omitted M-GU must fail this exact data/scratch oracle first.
        let mut old = Kernels::tree64_gu(&be, p, 256, 2);
        old.roles = narrow.roles;
        let skipped = record_sparse_skipped_gu(&be, &fixtures, old);
        reset_candidate_state(&fixtures, &resident, &layout, &ring);
        skipped.replay().unwrap();
        assert_skipped_gu_rejected(&fixtures, &oracle);
        drop(skipped);
        for arm in [
            MixedArm::Mixed256,
            MixedArm::Classified256,
            MixedArm::Separate1024,
        ] {
            let copy = if matches!(arm, MixedArm::Separate1024) {
                &wide
            } else {
                &narrow
            };
            let cmd = record_mixed(
                &be, &fixtures, &tasks, arm, &k, copy, &resident, &layout, &ring, &query,
            );
            // Width=256 launch-fusion comparison is distinct from the original
            // CuPartial1024 baseline. BOTH query comparisons include the entire FFN.
            for (label, control) in [("separate256", &separate), ("CuPartial1024", &baseline)] {
                let (base_us, candidate_us) = measure_pair(&be, control, &cmd, &query, ring.stages);
                reset_candidate_state(&fixtures, &resident, &layout, &ring);
                cmd.replay().unwrap();
                assert_fresh_candidate_scratch(&fixtures);
                verify_final(&fixtures, &oracle, &resident, &mut layout, &ring);
                for t in &tasks {
                    assert_tasks(t.as_ref(), true);
                }
                let ratios: Vec<_> = base_us
                    .iter()
                    .zip(&candidate_us)
                    .map(|(a, b)| a / b)
                    .collect();
                println!(
                    "COPY_GU_V2 profile={} rows={rows} used={} hits={} miss_experts=1 roles=3 control={label} candidate={arm:?} copy_local={} copy_groups={} GU_local=256 GU_NR=2 outer_queries=2 pairs={FRAME_PAIRS} copied_frame={} classify_charged={} baseline_us={base_us:?} candidate_us={candidate_us:?} median_ratio={} exact=true fresh_negative_control=true full_bank_scan_in_GU=false",
                    p.name,
                    p.used,
                    p.used - 1,
                    copy.local,
                    trial_copy_groups(copy.local),
                    ring.payload_per_stage * ring.stages,
                    matches!(arm, MixedArm::Classified256),
                    median(&ratios)
                );
            }
        }
    }
}

#[test]
#[ignore = "external SPIR-V; serialized native-geometry paired frame probe"]
fn gpu_owned_copy_gu_v2_paired_rows1() {
    mixed_frame_probe(1);
}

#[test]
#[ignore = "external SPIR-V; serialized native-geometry paired frame probe"]
fn gpu_owned_copy_gu_v2_paired_rows2() {
    mixed_frame_probe(2);
}

#[test]
#[ignore = "external SPIR-V; serialized native-geometry paired frame probe"]
fn gpu_owned_copy_gu_v2_paired_rows4() {
    mixed_frame_probe(4);
}
