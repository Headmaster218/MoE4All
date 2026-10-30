// Test-only serial copy/publish prototype. No production hook or allocation policy change.
fn copy_count_predicate_word(ordinal: u32, journal_bytes: usize) -> Option<usize> {
    if !journal_bytes.is_multiple_of(RESIDENT_JOURNAL_BYTES)
        || ordinal as usize >= journal_bytes / RESIDENT_JOURNAL_BYTES
    {
        return None;
    }
    (ordinal as usize)
        .checked_mul(RESIDENT_JOURNAL_WORDS)?
        .checked_add(RESIDENT_JOURNAL_JOB_COUNT_WORD)
}

#[test]
fn gpu_owned_copy_count_predicate_layout_bounds() {
    assert_eq!(
        copy_count_predicate_word(0, RESIDENT_JOURNAL_BYTES),
        Some(RESIDENT_JOURNAL_JOB_COUNT_WORD)
    );
    assert_eq!(copy_count_predicate_word(1, RESIDENT_JOURNAL_BYTES), None);
    assert_eq!(
        copy_count_predicate_word(0, RESIDENT_JOURNAL_BYTES - 4),
        None
    );
    assert_eq!(
        copy_count_predicate_word(u32::MAX, RESIDENT_JOURNAL_BYTES),
        None
    );
    assert_eq!(copy_count_predicate_word(0, 0), None);
    let last = copy_count_predicate_word(127, 128 * RESIDENT_JOURNAL_BYTES).unwrap();
    assert_eq!(last * 4 % 4, 0);
    assert!(last * 4 + 4 <= 128 * RESIDENT_JOURNAL_BYTES);
}

struct CountPredicateFixture(CopyPublishFixture);

// Test-only replacement for the two production imports. This is a GLOBAL dependency,
// so it can consume the tracker's complete outstanding compute/transfer hazard set.
fn begin_joint_count_import(rec: &Recorder<'_>, journal: &dyn Buffer, word: usize) {
    assert!(!rec.conditional_active.get() && rec.resident_indirect.borrow().is_none());
    assert!(!rec.no_barrier && !rec.full_barrier && !rec.suppress.get());
    assert!(rec.independent_dispatch.borrow().is_none());
    assert!((word + 1) * 4 <= journal.len_bytes());
    let mut source_stage = vk::PipelineStageFlags::COMPUTE_SHADER;
    let mut source_access = vk::AccessFlags::SHADER_WRITE;
    if rec.dirty_transfer.get() {
        source_stage |= vk::PipelineStageFlags::TRANSFER;
        source_access |= vk::AccessFlags::TRANSFER_WRITE;
    }
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(source_access)
        .dst_access_mask(
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT,
        );
    let buffer = Recorder::vk_of(journal);
    unsafe {
        rec.be.shared.device.cmd_pipeline_barrier(
            rec.cmd,
            source_stage,
            vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
    rec.dirty_writes.borrow_mut().clear();
    rec.dirty_reads.borrow_mut().clear();
    rec.dirty_transfer.set(false);
    *rec.barriers.borrow_mut() += 1;
    rec.conditional_active.set(true);
    rec.conditional_mutable.set(false);
    rec.immutable_condition
        .set(Some((buffer.buffer, buffer.sub_offset, word)));
    unsafe {
        (rec.be
            .shared
            .conditional_rendering
            .as_ref()
            .unwrap()
            .fp()
            .cmd_begin_conditional_rendering_ext)(
            rec.cmd,
            &vk::ConditionalRenderingBeginInfoEXT::default()
                .buffer(buffer.buffer)
                .offset((buffer.sub_offset + word * 4) as u64),
        );
    }
}

struct CountPrepareInputs {
    // Initial directory/control snapshots, not the already admitted constructor state.
    resets: Vec<(usize, Box<dyn Buffer>)>,
    push: [u32; 19],
    geometry: [u32; 12],
    single_row: bool,
}

impl CountPrepareInputs {
    fn new(
        be: &VulkanBackend,
        fixture: &CountPredicateFixture,
        rows: usize,
        roles: [u32; 3],
    ) -> Self {
        let nodes = &fixture.0.fixture.nodes;
        let slot_base = nodes[1][2] as usize;
        let address_base = nodes[1][3] as usize;
        let mut overlay = vec![[0, 0, 0, NONE]; fixture.0.fixture.buffers[6].len_bytes() / 16];
        for slot in 0..nodes[0][2] as usize {
            let state = nodes[slot_base + slot];
            if state[3] == 0 {
                continue;
            }
            assert_eq!(state[3], 1);
            let address = nodes[address_base + slot];
            overlay[state[2] as usize] = [address[0], address[1], 1, slot as u32];
        }
        let mut control = vec![0; RESIDENT_CONTROL_BYTES / 4];
        control[0] = 1;
        control[1] = NONE;
        let snapshots: [(usize, &[u8]); 3] = [
            (2, bytemuck::cast_slice(&control)),
            (5, bytemuck::cast_slice(nodes)),
            (6, bytemuck::cast_slice(&overlay)),
        ];
        let resets = snapshots
            .into_iter()
            .map(|(binding, bytes)| {
                let buffer = be
                    .alloc(bytes.len(), infr_core::backend::BufferUsage::Staging)
                    .unwrap();
                be.upload(buffer.as_ref(), bytes).unwrap();
                (binding, buffer)
            })
            .collect();
        let mut geometry = [0u32; 12];
        for (role, base) in roles.into_iter().enumerate() {
            let pool = base / 4;
            assert!(pool < 2);
            geometry[role * 4..role * 4 + 4].copy_from_slice(&[
                pool,
                fixture.0.stride as u32,
                pool * 3,
                3,
            ]);
        }
        let push = gpu_copy_prepare_push(
            [rows as u32 * 3, 4, 0, 4, 8, fixture.0.ordinal, 3, 0],
            true,
            [roles[0], roles[1], roles[2], 0, 0, 0, 0, 0, 0],
            8,
        );
        Self {
            resets,
            push,
            geometry,
            single_row: rows == 1,
        }
    }

    fn record(&self, rec: &Recorder<'_>, fixture: &CountPredicateFixture) {
        // Previous COUNT is immutable only until this reuse. In particular, ordinary
        // compute barriers do not cover the old conditional-rendering predicate read.
        let reuse = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE);
        unsafe {
            rec.be.shared.device.cmd_pipeline_barrier(
                rec.cmd,
                vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[reuse],
                &[],
                &[],
            );
        }
        for (binding, seed) in &self.resets {
            rec.copy(
                seed.as_ref(),
                0,
                fixture.0.fixture.buffers[*binding].as_ref(),
                0,
                seed.len_bytes(),
            );
        }
        let bindings: Vec<_> = fixture
            .0
            .fixture
            .buffers
            .iter()
            .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
            .collect();
        if self.single_row {
            let kernel = rec.be.kernel(
                "gpu_owned_prepare_fused",
                crate::gemm::gpu_owned_prepare_fused_spv(false),
                8,
                GPU_OWNED_FAST_PUSH_BYTES,
            );
            rec.dispatch(
                kernel,
                &bindings,
                8,
                bytemuck::cast_slice(&gpu_owned_fast_push(&self.push, &self.geometry)),
                1,
            );
        } else {
            let kernel = rec.be.kernel(
                "gpu_owned_prepare",
                crate::gemm::gpu_owned_prepare_spv(),
                8,
                GPU_COPY_PREPARE_PUSH_BYTES,
            );
            rec.dispatch(kernel, &bindings, 8, bytemuck::cast_slice(&self.push), 1);
        }
        // Do NOT import or clear tracking here. Baseline's dispatch emits the original RAW;
        // candidate's joint import must handle these real GPU writes without a second RAW.
        assert!(rec
            .dirty_writes
            .borrow()
            .contains(&Recorder::vk_of(fixture.0.fixture.buffers[3].as_ref()).buffer));
    }
}

impl CountPredicateFixture {
    fn new(
        be: &VulkanBackend,
        rows: usize,
        status: u32,
        all_cpu: bool,
        roles: [u32; 3],
        stride: usize,
        host_source: bool,
    ) -> Self {
        assert!(
            be.shared.conditional_rendering.is_some(),
            "native conditional rendering is required by this isolated probe"
        );
        let mut inner =
            CopyPublishFixture::new(be, rows, status, all_cpu, roles, stride, host_source);
        let words = read_words(be, inner.fixture.buffers[3].as_ref());
        // The ordinary mapped journal does NOT carry CONDITIONAL_RENDERING usage. Allocate
        // this test journal through the existing certified predicate constructor instead.
        let journal = be
            .alloc_mapped_predicate(inner.fixture.buffers[3].len_bytes())
            .unwrap();
        upload_words(be, journal.as_ref(), &words);
        inner.fixture.buffers[3] = journal;
        Self(inner)
    }

    fn record(&self, rec: &Recorder<'_>, conditional: bool) {
        assert!(!rec.conditional_active.get() && rec.resident_indirect.borrow().is_none());
        let inner = &self.0;
        let journal = inner.fixture.buffers[3].as_ref();
        // Unsupported/out-of-bounds predicate layouts keep the original shader rejection.
        // Never begin a conditional region with an out-of-bounds word offset.
        let word = conditional
            .then(|| copy_count_predicate_word(inner.ordinal, journal.len_bytes()))
            .flatten();
        if let Some(word) = word {
            begin_joint_count_import(rec, journal, word);
        }
        let kernel = rec.be.kernel(
            "gpu_owned_copy_publish",
            crate::gemm::gpu_owned_copy_publish_spv(false),
            4,
            4,
        );
        let bindings =
            [3, 7, 5, 6].map(|index| Recorder::vkb_exact(inner.fixture.buffers[index].as_ref()));
        // Match production's read-only descriptor tracking: BDA/READY writes are protected
        // by the explicit global dependency below, not a new synthetic tracking policy.
        rec.dispatch(kernel, &bindings, 0, bytemuck::bytes_of(&inner.ordinal), 8);
        if word.is_some() {
            rec.end_conditional();
        }
        // This barrier is retained for BOTH arms, including COUNT=0. It joins all workgroups
        // and exposes BDA bytes plus READY to subsequent expert/owner consumers together.
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            rec.be.shared.device.cmd_pipeline_barrier(
                rec.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    fn run(&self, be: &VulkanBackend, conditional: bool) -> Vec<Vec<u32>> {
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        self.record(&rec, conditional);
        rec.copy_strided(
            self.0.fixture.destination.as_ref(),
            self.0.observer.as_ref(),
            1,
            self.0.fixture.destination.len_bytes() / 4,
            0,
            0,
            0,
            0,
        );
        rec.publish_host_reads();
        rec.finish().unwrap();
        let result = self.0.snapshot(be);
        assert_eq!(
            read_words(be, self.0.observer.as_ref()),
            result[8],
            "post-barrier GPU consumer observed incomplete bytes"
        );
        result
    }

    fn differential(&self, be: &VulkanBackend, seed: &[Vec<u32>]) -> Vec<Vec<u32>> {
        self.0.restore(be, seed);
        let unconditional = self.run(be, false);
        self.0.restore(be, seed);
        let conditional = self.run(be, true);
        assert_eq!(
            unconditional, conditional,
            "COUNT predicate changed payload or metadata semantics"
        );
        conditional
    }

    fn differential_prepared(
        &self,
        be: &VulkanBackend,
        inputs: &CountPrepareInputs,
        seed: &[Vec<u32>],
    ) -> Vec<Vec<u32>> {
        let mut results = Vec::new();
        for conditional in [false, true] {
            self.0.restore(be, seed);
            let rec = be.recorder().unwrap();
            rec.owned_copy_barrier(true, false);
            inputs.record(&rec, self);
            self.record(&rec, conditional);
            rec.copy_strided(
                self.0.fixture.destination.as_ref(),
                self.0.observer.as_ref(),
                1,
                self.0.fixture.destination.len_bytes() / 4,
                0,
                0,
                0,
                0,
            );
            rec.publish_host_reads();
            rec.finish().unwrap();
            let state = self.0.snapshot(be);
            assert_eq!(read_words(be, self.0.observer.as_ref()), state[8]);
            self.0.check_expected(seed, &state);
            results.push(state);
        }
        assert_eq!(
            results[0], results[1],
            "joint import differed after a real GPU journal producer"
        );
        results.pop().unwrap()
    }
}

#[test]
#[ignore = "central GPU runner only; native COUNT predicate exact hits, mapped, CPU-only, mixed and aliases"]
fn gpu_owned_copy_count_predicate_matches_unconditional() {
    let be = backend();
    for (rows, status, all_cpu, roles) in [
        (1, 1, false, [0, 0, 4]),
        (1, 4, false, [0, 0, 4]),
        (2, 4, false, [0, 4, 4]),
        (2, 5, false, [0, 4, 4]),
        (infr_core::backend::MAX_DECODE_ROWS, 5, true, [0, 0, 4]),
    ] {
        let fixture = CountPredicateFixture::new(&be, rows, status, all_cpu, roles, 8192, false);
        let seed = fixture.0.snapshot(&be);
        let result = fixture.differential(&be, &seed);
        fixture.0.check_expected(&seed, &result);
        let inputs = CountPrepareInputs::new(&be, &fixture, rows, roles);
        assert_eq!(fixture.differential_prepared(&be, &inputs, &seed), result);
        if status == 1 || all_cpu {
            assert_eq!(
                result, seed,
                "all-hit/CPU-only serial copy+mode0 must remain externally read-only"
            );
        }
    }
}

#[test]
#[ignore = "central GPU runner only; native COUNT predicate retains malformed and rejection semantics"]
fn gpu_owned_copy_count_predicate_keeps_malformed_semantics() {
    let be = backend();
    let mut fixture = CountPredicateFixture::new(&be, 1, 4, false, [0, 0, 4], 8192, false);
    let original = fixture.0.snapshot(&be);
    let job = fixture.0.ordinal as usize * RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
    let count_word =
        copy_count_predicate_word(fixture.0.ordinal, fixture.0.fixture.buffers[3].len_bytes())
            .unwrap();
    let status_word =
        fixture.0.ordinal as usize * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_STATUS_WORD;
    for fault in [
        "count_zero",
        "count_overflow",
        "status_error",
        "source_alignment",
        "source_wrap",
        "destination_zero",
        "bytes_zero",
        "block_bounds",
        "slot_bounds",
        "reverse_owner",
        "canonical_address",
        "overlay_state",
        "duplicate_block",
        "duplicate_slot",
        "header",
    ] {
        let mut seed = original.clone();
        let block = seed[7][job + 5] as usize;
        let slot = seed[7][job + 6] as usize;
        let slot_base = seed[5][6] as usize;
        let address_base = seed[5][7] as usize;
        match fault {
            "count_zero" => seed[3][count_word] = 0,
            "count_overflow" => seed[3][count_word] = (RESIDENT_JOURNAL_ID_WORDS * 3 + 1) as u32,
            "status_error" => seed[3][status_word] = 3,
            "source_alignment" => seed[7][job] |= 4,
            "source_wrap" => {
                seed[7][job] = 0xffff_fff0;
                seed[7][job + 1] = NONE;
            }
            "destination_zero" => seed[7][job + 2..job + 4].fill(0),
            "bytes_zero" => seed[7][job + 4] = 0,
            "block_bounds" => seed[7][job + 5] = 8,
            "slot_bounds" => seed[7][job + 6] = 6,
            "reverse_owner" => seed[5][(slot_base + slot) * 4 + 2] = NONE,
            "canonical_address" => seed[5][(address_base + slot) * 4] ^= 16,
            "overlay_state" => seed[6][block * 4 + 2] = 0,
            "duplicate_block" => seed[7][job + 8 + 5] = block as u32,
            "duplicate_slot" => seed[7][job + 8 + 6] = slot as u32,
            "header" => seed[5][0] = 99,
            _ => unreachable!(),
        }
        // Invalid BDAs above are rejected by the unchanged payload checks before dereference.
        let result = fixture.differential(&be, &seed);
        assert_eq!(
            result[5], seed[5],
            "failed owner batch published READY: {fault}"
        );
        assert_eq!(
            result[6], seed[6],
            "failed overlay batch published READY: {fault}"
        );
    }
    // status4/source0 intentionally has different payload and publication rejection:
    // old payload skips the whole batch, mode0 can still publish valid mapped siblings.
    let mut seed = original.clone();
    seed[7][job..job + 2].fill(0);
    let result = fixture.differential(&be, &seed);
    assert_eq!(result[8], seed[8]);
    assert_eq!(result[6][seed[7][job + 5] as usize * 4 + 2], 2);
    assert_eq!(result[6][seed[7][job + 8 + 5] as usize * 4 + 2], 1);
    // No predicate address is formed for an invalid ordinal; ordinary shader rejection wins.
    fixture.0.ordinal = 2;
    assert_eq!(fixture.differential(&be, &original), original);
}

struct CountPredicateQuery {
    device: ash::Device,
    pool: vk::QueryPool,
}

impl Drop for CountPredicateQuery {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.pool, None);
        }
    }
}

#[test]
#[ignore = "central GPU runner only; bounded paired native COUNT predicate GPU/host cost, production groups8"]
fn gpu_owned_copy_count_predicate_paired_latency_probe() {
    let be = backend();
    assert!(be.shared.submit_timestamp_valid_bits > 0);
    assert!(
        be.shared.host_overflow_type.is_some(),
        "host-RAM BDA source required; do not silently relabel VRAM"
    );
    let device = &be.shared.device;
    let query = CountPredicateQuery {
        device: device.clone(),
        pool: unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .unwrap(),
    };
    let period = be.shared.submit_timestamp_period_ns as f64;
    let started = std::time::Instant::now();
    const REPEATS: usize = 64;
    const ROUNDS: usize = 6;
    const WARMUP_ROUNDS: usize = 2;
    // One device/source/size/groups setting only. Six paired rounds, first two warmup.
    // No Cartesian sweep; the parent runner supplies its normal process deadline as well.
    for status in [1, 4, 5] {
        let fixture =
            CountPredicateFixture::new(&be, 1, status, false, [0, 0, 4], 1024 * 1024, true);
        let seed = fixture.0.snapshot(&be);
        let inputs = CountPrepareInputs::new(&be, &fixture, 1, [0, 0, 4]);
        let mut gpu = [Vec::new(), Vec::new()];
        let mut host = [Vec::new(), Vec::new()];
        for round in 0..ROUNDS {
            for arm in 0..2 {
                assert!(
                    started.elapsed().as_secs() < 90,
                    "bounded COUNT probe exceeded 90s between submits"
                );
                let variant = (round + arm) % 2;
                fixture.0.restore(&be, &seed);
                let rec = be.recorder().unwrap();
                rec.owned_copy_barrier(true, false);
                unsafe {
                    device.cmd_reset_query_pool(rec.cmd, query.pool, 0, 2);
                    device.cmd_write_timestamp(
                        rec.cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        query.pool,
                        0,
                    );
                }
                let recording = std::time::Instant::now();
                for _ in 0..REPEATS {
                    // Reset only owner/control state, then execute a real GPU prepare. Every
                    // repetition is the SAME miss/hit case, not a warmed-up all-hit successor.
                    inputs.record(&rec, &fixture);
                    let barriers = *rec.barriers.borrow();
                    fixture.record(&rec, variant == 1);
                    // Both arms import exactly one real GPU-producer dependency. The baseline
                    // dispatch emits its normal compute RAW; candidate uses the joint import.
                    assert_eq!(*rec.barriers.borrow() - barriers, 1);
                }
                let record_us = recording.elapsed().as_secs_f64() * 1e6 / REPEATS as f64;
                unsafe {
                    device.cmd_write_timestamp(
                        rec.cmd,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        query.pool,
                        1,
                    );
                }
                rec.publish_host_reads();
                rec.finish().unwrap();
                let mut ticks = [0u64; 2];
                unsafe {
                    device.get_query_pool_results(
                        query.pool,
                        0,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64,
                    )
                }
                .unwrap();
                let mut delta = ticks[1].wrapping_sub(ticks[0]);
                if be.shared.submit_timestamp_valid_bits < 64 {
                    delta &= (1u64 << be.shared.submit_timestamp_valid_bits) - 1;
                }
                if round >= WARMUP_ROUNDS {
                    gpu[variant].push(delta as f64 * period / 1000.0 / REPEATS as f64);
                    host[variant].push(record_us);
                }
                // Last iteration must reproduce the original GPU admission and publication,
                // including job count, masks, FIFO, clock, READY and all copied bytes.
                fixture.0.check_expected(&seed, &fixture.0.snapshot(&be));
            }
        }
        for arm in 0..2 {
            gpu[arm].sort_by(f64::total_cmp);
            host[arm].sort_by(f64::total_cmp);
            assert_eq!(gpu[arm].len(), ROUNDS - WARMUP_ROUNDS);
            eprintln!("COPY_COUNT_PREDICATE_V2 scope=gpu-reset+prepare+import+payload+ready-barrier source=host-RAM expert_bytes=1048576 status={status} groups=8 local=256 repeats={REPEATS} rounds={ROUNDS} warmup_rounds={WARMUP_ROUNDS} arm={} n={} gpu_p50_us={:.3} gpu_mean_us={:.3} host_record_p50_us={:.3}",
                if arm == 0 { "compute-RAW+unconditional" } else { "joint-compute-conditional+conditional" }, gpu[arm].len(),
                (gpu[arm][1] + gpu[arm][2]) * 0.5, gpu[arm].iter().sum::<f64>() / gpu[arm].len() as f64,
                (host[arm][1] + host[arm][2]) * 0.5);
        }
    }
}
