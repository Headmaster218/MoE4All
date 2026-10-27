//! Module-only prototype tests, included as a child of recorder_gpu_owner_performance.
//! Synthetic BDAs are metadata only; neither arm records payload or publish dispatches.
use super::*;

#[path = "recorder_gpu_owner_topk_overlap_tests.rs"]
mod copy_overlap;

#[path = "recorder_gpu_owner_topk_compaction_replay_tests.rs"]
mod compaction_replay;

#[path = "recorder_gpu_owner_topk_victim_prefix_tests.rs"]
mod victim_prefix;

const ORIGINAL_OWNER_SHA: &str = "1E12D03E96DAC3D27A7F6951C345F8BE57BF301CE719736DE68703F3160BB592";
const ORIGINAL_ROUTER_SHA: &str =
    "444AC5EB384F1226E0D02C9F3117719205DF12C9133827D0641E4F1E67296852";
const WEIGHT_CANARY: u32 = 0x6a12_3456;
const TAIL_CANARY: u32 = 0xa53c_91e7;
const SCALE: f32 = 1.375;

fn previous_predicate_word(ordinal: usize) -> usize {
    assert!(ordinal < RESIDENT_MAX_LAYERS);
    ordinal
        .checked_sub(1)
        .map_or(RESIDENT_SEED_WORD, |prior| 5 + prior * 2)
}

fn routing_push(owner: &[u32; 31], gating: u32, predicate: usize) -> [u32; 31] {
    let mut push = *owner;
    push[13] = SCALE.to_bits();
    push[14] = gating;
    push[15] = predicate as u32;
    push
}

#[test]
fn gpu_owned_topk_push_preserves_owner_geometry_and_predicate_identity() {
    assert_eq!(RESIDENT_MAX_LAYERS, 128, "frozen prototype control ABI");
    assert_eq!(RESIDENT_SEED_WORD, 261, "frozen prototype seed offset");
    assert_eq!(GPU_OWNED_FAST_PUSH_BYTES, 124);
    let base = std::array::from_fn(|i| i as u32 + 0x1000);
    assert_eq!(previous_predicate_word(0), RESIDENT_SEED_WORD);
    assert_eq!(previous_predicate_word(1), 5);
    assert_eq!(previous_predicate_word(127), 257);
    for ordinal in [0, 1, 127] {
        for gating in [0, 1] {
            let predicate = previous_predicate_word(ordinal);
            let push = routing_push(&base, gating, predicate);
            assert_eq!(&push[13..16], &[SCALE.to_bits(), gating, predicate as u32]);
            for i in 0..31 {
                if !(13..16).contains(&i) {
                    assert_eq!(push[i], base[i], "owner push word {i}");
                }
            }
        }
    }
}

#[test]
fn gpu_owned_topk_header_overlap_predicate_read_write_disjoint() {
    for ordinal in 0..RESIDENT_MAX_LAYERS {
        let input = previous_predicate_word(ordinal);
        let output = 5 + ordinal * 2;
        assert!(input < RESIDENT_CONTROL_BYTES / 4);
        assert_ne!(input, output);
        assert_ne!(input, output + 1);
        assert_ne!(RESIDENT_SEED_WORD, output);
        assert_ne!(RESIDENT_SEED_WORD, output + 1);
    }
}

fn external_words(variable: &str) -> Vec<u32> {
    let path = std::env::var_os(variable).unwrap_or_else(|| panic!("set {variable}"));
    eprintln!(
        "GPU_OWNER_TOPK_SHADER env={variable} path={}",
        std::path::Path::new(&path).display()
    );
    ash::util::read_spv(&mut std::io::Cursor::new(std::fs::read(path).unwrap())).unwrap()
}

struct TopkKernels {
    router: ComputeKernel,
    owner: ComputeKernel,
    fused: ComputeKernel,
}

impl TopkKernels {
    fn new(be: &VulkanBackend) -> Self {
        let required = vk::SubgroupFeatureFlags::BASIC
            | vk::SubgroupFeatureFlags::ARITHMETIC
            | vk::SubgroupFeatureFlags::BALLOT;
        assert!(be.shared.compute_subgroup_operations.contains(required));
        assert!(
            be.shared.conditional_rendering.is_some(),
            "this isolated differential probe needs the original conditional-rendering predicate"
        );
        eprintln!("GPU_OWNER_TOPK_SOURCE owner_sha256={ORIGINAL_OWNER_SHA} router_sha256={ORIGINAL_ROUTER_SHA} subgroup=Fixed32 metadata_only=true");
        Self {
            router: be.kernel_sg(
                "owner_topk_original_wave32_external",
                &external_words("MOE4ALL_OWNER_TOPK_REFERENCE_ROUTER_SPV"),
                5,
                36,
                32,
            ),
            owner: be.kernel_sg(
                "owner_topk_original_owner_external",
                &external_words("MOE4ALL_OWNER_TOPK_REFERENCE_OWNER_SPV"),
                8,
                GPU_OWNED_FAST_PUSH_BYTES,
                32,
            ),
            fused: be.kernel_sg(
                "owner_topk_fused_external",
                &external_words("MOE4ALL_OWNER_TOPK_FUSED_SPV"),
                10,
                GPU_OWNED_FAST_PUSH_BYTES,
                32,
            ),
        }
    }

    fn previous_fused(&self, be: &VulkanBackend) -> Self {
        let words = external_words("MOE4ALL_OWNER_TOPK_REFERENCE_FUSED_SPV");
        assert_ne!(
            words,
            external_words("MOE4ALL_OWNER_TOPK_FUSED_SPV"),
            "overlap oracle must be the frozen pre-overlap fused shader, not the candidate"
        );
        Self {
            router: self.router,
            owner: self.owner,
            fused: be.kernel_sg(
                "owner_topk_previous_fused_external",
                &words,
                10,
                GPU_OWNED_FAST_PUSH_BYTES,
                32,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Logits {
    ForcedRandom,
    Random,
    Ties,
    Nonfinite,
    AllNan,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    used: usize,
    gating: u32,
    workload: Workload,
    logits: Logits,
    ordinal: usize,
    predicate: bool,
    run: bool,
    alias: bool,
}

impl Case {
    fn normal(name: &'static str, used: usize, workload: Workload) -> Self {
        Self {
            name,
            used,
            gating: u32::from(used == 10),
            workload,
            logits: Logits::ForcedRandom,
            ordinal: 0,
            predicate: true,
            run: true,
            alias: false,
        }
    }

    fn logits(self) -> Vec<u32> {
        let mut seed = 0x9e37_79b9u32;
        let mut values: Vec<_> = (0..EXPERTS)
            .map(|i| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let random = (seed >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0;
                match self.logits {
                    Logits::ForcedRandom if i >= self.used => -20.0 - i as f32 / 1024.0,
                    Logits::Ties => 0.25,
                    Logits::AllNan => f32::from_bits(0x7fc0_0123),
                    _ => random,
                }
            })
            .collect();
        if matches!(self.logits, Logits::Nonfinite) {
            for (i, value) in [
                (0, f32::INFINITY),
                (1, f32::NEG_INFINITY),
                (2, f32::from_bits(0x7fc0_0123)),
                (3, -1.0e31),
                (31, f32::INFINITY),
                (511, f32::NEG_INFINITY),
            ] {
                values[i] = value;
            }
        }
        let mut bits: Vec<_> = values.into_iter().map(f32::to_bits).collect();
        bits.extend([TAIL_CANARY; 4]);
        bits
    }
}

struct TopkProbe {
    fixture: PoolProbe,
    logits: Box<dyn Buffer>,
    weights: Box<dyn Buffer>,
    dummy: Box<dyn Buffer>,
    logits_bits: Vec<u32>,
    weights_bits: Vec<u32>,
    expected_weights: Vec<u32>,
    owner_push: [u32; 31],
    fused_push: [u32; 31],
    router_push: [u32; 9],
    predicate: usize,
}

impl TopkProbe {
    fn new(be: &VulkanBackend, kernels: &TopkKernels, case: Case, slots: usize) -> Self {
        Self::new_with_experts(be, kernels, case, slots, EXPERTS)
    }

    fn new_with_experts(
        be: &VulkanBackend,
        kernels: &TopkKernels,
        case: Case,
        slots: usize,
        experts: usize,
    ) -> Self {
        assert!([256, 512].contains(&experts) && experts <= EXPERTS);
        assert!(case.used <= experts);
        assert!(experts == EXPERTS || matches!(case.logits, Logits::ForcedRandom));
        // Reuse the parent's canonical seed, then make this probe independent of its row layout.
        // A 20-ID seed guarantees forced top-8/10/16 hits without touching the global used env.
        assert_eq!(
            probe_used(),
            10,
            "leave MOE4ALL_OWNER_POOL_USED unset (or 10)"
        );
        let mut fixture = PoolProbe::new(be, slots, 20, case.workload);
        fixture.repeats = 1;
        fixture.push[0] = case.used as u32;
        // This is an active bank span, not the physical directory/Tape stride. Retain
        // all canonical 512-stride banks and pools so the independent reference stays valid.
        // Narrowing this raw shader span is not a CPU 256-expert registration contract.
        fixture.push[1] = experts as u32;
        fixture.push[5] = case.ordinal as u32;
        fixture.push[6] = case.used as u32;
        if case.alias {
            fixture.push[10] = fixture.push[9];
        }
        fixture.initial[0] = (0..case.used as u32).collect();
        fixture.initial[0].extend([TAIL_CANARY; 6]);
        fixture.initial[2][0] = u32::from(case.run);
        let predicate = previous_predicate_word(case.ordinal);
        fixture.initial[2][predicate] = u32::from(case.predicate);
        if case.ordinal > 0 {
            fixture.initial[2][RESIDENT_SEED_WORD] = u32::from(!case.predicate);
        }
        fixture.initial[3] = vec![TAIL_CANARY; (case.ordinal + 2) * RESIDENT_JOURNAL_WORDS];
        fixture.initial[7] =
            vec![TAIL_CANARY; (case.ordinal + 2) * RESIDENT_JOURNAL_ID_WORDS * 3 * 8];
        for i in [0, 3, 7] {
            fixture.buffers[i] = be
                .alloc_mapped_arena_bda(fixture.initial[i].len() * 4)
                .unwrap()
                .0;
        }
        fixture.buffers[5] = be
            .alloc_mapped_arena_bda((fixture.heap_base + 2 + slots * 4) * 16)
            .unwrap()
            .0;
        let mut logits_bits = case.logits();
        logits_bits.truncate(experts);
        logits_bits.extend([TAIL_CANARY; 4]);
        let weights_bits = vec![WEIGHT_CANARY; case.used + 4];
        let logits = be.alloc_mapped_arena_bda(logits_bits.len() * 4).unwrap().0;
        let weights = be.alloc_mapped_arena_bda(weights_bits.len() * 4).unwrap().0;
        let dummy = be.alloc_mapped_arena_bda(4).unwrap().0;
        upload(be, dummy.as_ref(), &[0]);
        let geometry = [
            0,
            SLOT_BYTES[0],
            0,
            slots as u32,
            0,
            SLOT_BYTES[0],
            0,
            slots as u32,
            1,
            SLOT_BYTES[1],
            slots as u32,
            slots as u32,
        ];
        let owner_push = gpu_owned_fast_push(&fixture.push, &geometry);
        let fused_push = routing_push(&owner_push, case.gating, predicate);
        let router_push = [
            experts as u32,
            case.used as u32,
            SCALE.to_bits(),
            case.gating,
            1,
            0,
            0,
            0,
            0,
        ];
        let mut probe = Self {
            fixture,
            logits,
            weights,
            dummy,
            logits_bits,
            weights_bits,
            expected_weights: Vec::new(),
            owner_push,
            fused_push,
            router_push,
            predicate,
        };
        probe.reset(be);
        // Independent routing oracle: the frozen original shader under the actual old predicate.
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        rec.begin_immutable_conditional_at(probe.fixture.buffers[2].as_ref(), predicate);
        probe.record_router(&rec, kernels);
        rec.end_conditional();
        rec.owned_copy_barrier(false, true);
        rec.finish().unwrap();
        let selected = read(be, probe.fixture.buffers[0].as_ref());
        probe.expected_weights = read(be, probe.weights.as_ref());
        let mut initial = probe.fixture.initial.clone();
        initial[0][..case.used].copy_from_slice(&selected[..case.used]);
        let mut expected;
        if case.run {
            let roles = std::array::from_fn(|r| probe.fixture.push[9 + r] as usize);
            let misses;
            (expected, misses) = reference(
                &initial,
                &selected[..case.used],
                roles,
                slots,
                probe.fixture.address_base,
            );
            probe.fixture.misses = misses;
            // reference() uses the parent probe's used10 mask layout. Recompute only row masks.
            let mask = (1u32 << case.used) - 1;
            let mut hits = 0u32;
            for (i, &id) in selected[..case.used].iter().enumerate() {
                if roles
                    .iter()
                    .all(|&base| initial[6][(base + id as usize) * 4 + 2] == 1)
                {
                    hits |= 1 << i;
                }
            }
            expected[0][case.used] = mask;
            expected[0][case.used + 1] = 0;
            expected[3][RESIDENT_JOURNAL_MASK_WORD] = hits;
            expected[2] = initial[2].clone();
            expected[2][2] = case.ordinal as u32 + 1;
            expected[2][4] = 1;
            expected[2][5 + case.ordinal * 2..7 + case.ordinal * 2].copy_from_slice(&[1, 1]);
            if case.ordinal != 0 {
                let journal = expected[3][..RESIDENT_JOURNAL_WORDS].to_vec();
                expected[3] = initial[3].clone();
                let base = case.ordinal * RESIDENT_JOURNAL_WORDS;
                expected[3][base..base + RESIDENT_JOURNAL_WORDS].copy_from_slice(&journal);
                let stride = RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
                let jobs = expected[7][..stride].to_vec();
                expected[7] = initial[7].clone();
                expected[7][case.ordinal * stride..(case.ordinal + 1) * stride]
                    .copy_from_slice(&jobs);
            }
        } else {
            expected = initial;
            let base = case.ordinal * RESIDENT_JOURNAL_WORDS;
            expected[3][base + RESIDENT_JOURNAL_JOB_COUNT_WORD
                ..base + RESIDENT_JOURNAL_JOB_COUNT_WORD + 3]
                .fill(0);
            expected[3][base + RESIDENT_JOURNAL_STATUS_WORD] = 0;
            expected[2][5 + case.ordinal * 2..7 + case.ordinal * 2].fill(0);
            probe.fixture.misses = 0;
        }
        probe.fixture.expected = expected;
        if case.run && case.predicate && matches!(case.logits, Logits::ForcedRandom) {
            let mut selected_ids = selected[..case.used].to_vec();
            selected_ids.sort_unstable();
            assert_eq!(selected_ids, (0..case.used as u32).collect::<Vec<_>>());
            match case.workload {
                Workload::AllHit => assert_eq!(probe.fixture.misses, 0),
                Workload::OneExpertMiss => {
                    let banks = probe.fixture.push[9..12]
                        .iter()
                        .collect::<std::collections::HashSet<_>>();
                    assert_eq!(probe.fixture.misses, banks.len());
                }
                _ => {}
            }
        }
        if !case.predicate {
            assert_eq!(
                probe.expected_weights, probe.weights_bits,
                "false predicate preserved weights"
            );
            assert_eq!(
                &selected[..case.used],
                &probe.fixture.initial[0][..case.used]
            );
        } else if case.ordinal > 0 {
            assert_ne!(
                probe.expected_weights, probe.weights_bits,
                "true prior predicate must route despite false seed"
            );
        }
        probe
    }

    fn reset(&self, be: &VulkanBackend) {
        self.fixture.reset(be, false, true);
        upload(be, self.logits.as_ref(), &self.logits_bits);
        upload(be, self.weights.as_ref(), &self.weights_bits);
    }

    fn record_router(&self, rec: &Recorder<'_>, kernels: &TopkKernels) {
        rec.dispatch(
            kernels.router,
            &[
                Recorder::vkb_exact(self.logits.as_ref()),
                Recorder::vkb_exact(self.dummy.as_ref()),
                Recorder::vkb_exact(self.dummy.as_ref()),
                Recorder::vkb_exact(self.fixture.buffers[0].as_ref()),
                Recorder::vkb_exact(self.weights.as_ref()),
            ],
            2,
            bytemuck::cast_slice(&self.router_push),
            1,
        );
    }

    fn raw_fused(
        &self,
        be: &VulkanBackend,
        kernel: ComputeKernel,
        mutate: impl FnOnce(&mut [Vec<u32>; 8], &mut [u32; 31]),
    ) -> [Vec<u32>; 10] {
        self.reset(be);
        // Mutate AFTER FIFO reset synthesizes the live V4 queue. Do not let reset silently
        // repair the malformed header/queue whose exact rejection is being compared.
        let mut state = std::array::from_fn(|i| read(be, self.fixture.buffers[i].as_ref()));
        let mut push = self.fused_push;
        mutate(&mut state, &mut push);
        for (buffer, words) in self.fixture.buffers.iter().zip(&state) {
            upload(be, buffer.as_ref(), words);
        }
        let mut bindings: Vec<_> = self
            .fixture
            .buffers
            .iter()
            .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
            .collect();
        bindings.extend([
            Recorder::vkb_exact(self.logits.as_ref()),
            Recorder::vkb_exact(self.weights.as_ref()),
        ]);
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        rec.dispatch(kernel, &bindings, 10, bytemuck::cast_slice(&push), 1);
        rec.owned_copy_barrier(false, true);
        rec.finish().unwrap();
        std::array::from_fn(|i| match i {
            8 => read(be, self.weights.as_ref()),
            9 => read(be, self.logits.as_ref()),
            _ => read(be, self.fixture.buffers[i].as_ref()),
        })
    }

    fn run(
        &self,
        be: &VulkanBackend,
        kernels: &TopkKernels,
        fused: bool,
        query: Option<&Queries<'_>>,
    ) -> ([Vec<u32>; 8], Option<u64>) {
        self.reset(be);
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        if let Some(query) = query {
            unsafe {
                be.shared
                    .device
                    .cmd_reset_query_pool(rec.cmd, query.pool, 0, 2);
            }
        }
        // Both arms pay the same conditional setup outside timestamps. Only the original
        // TopK executes inside it; owner initialization is unconditional in both arms.
        rec.begin_immutable_conditional_at(self.fixture.buffers[2].as_ref(), self.predicate);
        if let Some(query) = query {
            unsafe {
                be.shared.device.cmd_write_timestamp(
                    rec.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query.pool,
                    0,
                );
            }
        }
        if !fused {
            self.record_router(&rec, kernels);
        }
        rec.end_conditional();
        let mut bindings: Vec<_> = self
            .fixture
            .buffers
            .iter()
            .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
            .collect();
        if fused {
            bindings.extend([
                Recorder::vkb_exact(self.logits.as_ref()),
                Recorder::vkb_exact(self.weights.as_ref()),
            ]);
            rec.dispatch(
                kernels.fused,
                &bindings,
                10,
                bytemuck::cast_slice(&self.fused_push),
                1,
            );
        } else {
            rec.dispatch(
                kernels.owner,
                &bindings,
                8,
                bytemuck::cast_slice(&self.owner_push),
                1,
            );
        }
        if let Some(query) = query {
            unsafe {
                be.shared.device.cmd_write_timestamp(
                    rec.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    query.pool,
                    1,
                );
            }
        }
        rec.owned_copy_barrier(false, true);
        rec.finish().unwrap();
        self.fixture.assert_result(be, false, true, false, false);
        assert_eq!(
            read(be, self.weights.as_ref()),
            self.expected_weights,
            "all weight bits/tail"
        );
        assert_eq!(
            read(be, self.logits.as_ref()),
            self.logits_bits,
            "logits/tail unchanged"
        );
        let snapshot = std::array::from_fn(|i| read(be, self.fixture.buffers[i].as_ref()));
        let ticks = query.map(|query| {
            let mut values = [0u64; 2];
            unsafe {
                be.shared
                    .device
                    .get_query_pool_results(
                        query.pool,
                        0,
                        &mut values,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .unwrap();
            }
            let mut delta = values[1].wrapping_sub(values[0]);
            let bits = be.shared.submit_timestamp_valid_bits;
            if bits < 64 {
                delta &= (1u64 << bits) - 1;
            }
            delta
        });
        (snapshot, ticks)
    }
}

fn backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicit idle test device"));
    VulkanBackend::new_with(Arc::new(cfg)).unwrap()
}

#[test]
#[ignore = "parent serial GPU only; external TopK+owner module, exact bits/all eight buffers, no payload"]
fn gpu_owned_topk_fused_module_exact() {
    let be = backend();
    let kernels = TopkKernels::new(&be);
    let mut cases = Vec::new();
    for used in [8, 10, 16] {
        cases.push(Case::normal("forced_random_hit", used, Workload::AllHit));
        cases.push(Case::normal(
            "forced_random_miss",
            used,
            Workload::OneExpertMiss,
        ));
    }
    let mut ties = Case::normal("ties", 10, Workload::AllHit);
    ties.logits = Logits::Ties;
    cases.push(ties);
    let mut finite_alias = Case::normal("finite_bank_alias_hit", 10, Workload::AllHit);
    finite_alias.alias = true;
    cases.push(finite_alias);
    let mut random = Case::normal("random_all_experts", 16, Workload::OneExpertStale);
    random.logits = Logits::Random;
    cases.push(random);
    let mut nonfinite = Case::normal("nan_inf", 10, Workload::OneExpertStale);
    nonfinite.logits = Logits::Nonfinite;
    cases.push(nonfinite);
    let mut alias = Case::normal(
        "all_nan_duplicate_ids_bank_alias",
        8,
        Workload::OneExpertStale,
    );
    alias.logits = Logits::AllNan;
    alias.alias = true;
    cases.push(alias);
    for ordinal in [0, 1] {
        let mut skipped = Case::normal("predicate_false_run_true", 10, Workload::OneExpertMiss);
        skipped.ordinal = ordinal;
        skipped.predicate = false;
        cases.push(skipped);
    }
    let mut prior_true = Case::normal("prior_true_seed_false", 10, Workload::OneExpertMiss);
    prior_true.ordinal = 1;
    cases.push(prior_true);
    let mut stopped = Case::normal("predicate_true_run_false", 16, Workload::AllHit);
    stopped.run = false;
    cases.push(stopped);
    for slots in [40, 4096] {
        for (index, case) in cases.iter().copied().enumerate() {
            let probe = TopkProbe::new(&be, &kernels, case, slots);
            let first = index % 2 == 0;
            let (a, _) = probe.run(&be, &kernels, first, None);
            let (b, _) = probe.run(&be, &kernels, !first, None);
            assert_eq!(
                a, b,
                "raw eight-buffer equality case={case:?} slots={slots}"
            );
            eprintln!("GPU_OWNER_TOPK_EXACT case={} used={} gating={} ordinal={} predicate={} run={} slots={slots} misses={} all8_raw=pass weights_bits=pass cpu_lru=pass", case.name, case.used, case.gating, case.ordinal, case.predicate, case.run, probe.fixture.misses);
        }
    }
}

#[test]
#[ignore = "parent serial idle GPU only; paired rotating TopK+owner, six rounds, hot/one-miss only"]
fn gpu_owned_topk_fused_module_latency_probe() {
    let be = backend();
    let kernels = TopkKernels::new(&be);
    assert!(be.shared.submit_timestamp_valid_bits > 0);
    let queries = Queries {
        be: &be,
        pool: unsafe {
            be.shared
                .device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
                .unwrap()
        },
    };
    let period = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
            .limits
            .timestamp_period
    };
    for workload in [Workload::AllHit, Workload::OneExpertMiss] {
        let case = Case::normal("paired", 10, workload);
        let probe = TopkProbe::new(&be, &kernels, case, 4096);
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..6 {
            let mut snapshots = Vec::new();
            for offset in 0..2 {
                let arm = (round + offset) % 2;
                let (snapshot, ticks) = probe.run(&be, &kernels, arm == 1, Some(&queries));
                snapshots.push(snapshot);
                if round >= 2 {
                    samples[arm].push(ticks.unwrap() as f64 * period as f64 / 1000.0);
                }
            }
            assert_eq!(
                snapshots[0],
                snapshots[1],
                "round={round} workload={}",
                workload.name()
            );
        }
        for (name, mut values) in ["separate_original", "fused_external"]
            .into_iter()
            .zip(samples)
        {
            values.sort_by(f64::total_cmp);
            eprintln!("GPU_OWNER_TOPK_PAIRED variant={name} workload={} slots=4096 used=10 rounds=6 warmup=2 samples={} p50_us={:.3} mean_us={:.3} all8_raw=pass weights_bits=pass cpu_lru=pass", workload.name(), values.len(), values[values.len()/2], values.iter().sum::<f64>()/values.len() as f64);
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; overlapped TopK/header vs frozen fused and independent float/LRU oracle"]
fn gpu_owned_topk_header_overlap_module_exact() {
    let be = backend();
    let kernels = TopkKernels::new(&be);
    let previous = kernels.previous_fused(&be);
    let mut cases = Vec::new();
    for used in [8, 10, 16] {
        for gating in [0, 1] {
            for workload in [Workload::AllHit, Workload::OneExpertMiss] {
                let mut case = Case::normal("overlap_valid", used, workload);
                case.gating = gating;
                cases.push(case);
            }
        }
    }
    for logits in [Logits::Ties, Logits::Nonfinite, Logits::AllNan] {
        let mut case = Case::normal("overlap_float_bits_alias", 10, Workload::OneExpertStale);
        case.logits = logits;
        case.alias = true;
        cases.push(case);
    }
    for ordinal in [0, 1, 127] {
        for predicate in [false, true] {
            for run in [false, true] {
                let mut case = Case::normal("overlap_predicate_run", 10, Workload::AllHit);
                case.ordinal = ordinal;
                case.predicate = predicate;
                case.run = run;
                cases.push(case);
            }
        }
    }
    // One bounded geometry; wider pool timing is a separate paired test, not a matrix.
    for case in cases {
        let probe = TopkProbe::new(&be, &kernels, case, 40);
        let (independent, _) = probe.run(&be, &kernels, false, None);
        let (old_fused, _) = probe.run(&be, &previous, true, None);
        let (overlapped, _) = probe.run(&be, &kernels, true, None);
        assert_eq!(independent, old_fused, "frozen oracle case={case:?}");
        assert_eq!(old_fused, overlapped, "overlap state case={case:?}");
        for word in [probe.predicate, RESIDENT_SEED_WORD] {
            assert_eq!(
                overlapped[2][word], probe.fixture.initial[2][word],
                "input predicate must survive ordinal={} word={word}",
                case.ordinal
            );
        }
        // run() independently checks every float bit/tail and the CPU exact-LRU reference.
        eprintln!("GPU_OWNER_TOPK_HEADER_EXACT case={case:?} all8_raw=pass float_bits=pass exact_lru=pass");
    }
}

#[test]
#[ignore = "parent serial GPU only; previous fused rejection oracle, count/alias/sticky error and CPU misses"]
fn gpu_owned_topk_header_overlap_rejection_exact() {
    let be = backend();
    let kernels = TopkKernels::new(&be);
    let previous = kernels.previous_fused(&be);
    let mut case = Case::normal("overlap_rejection", 10, Workload::AllHit);
    case.ordinal = 1;
    let probe = TopkProbe::new(&be, &kernels, case, 40);
    for fault in [
        "count_zero",
        "count_mismatch",
        "count_large",
        "used_zero",
        "gating",
        "wrong_predicate",
        "ordinal_large",
        "sticky_invalid",
        "header",
        "queue_pointer",
        "queue_tail",
        "canonical",
        "reverse_owner",
        "partial_bank_alias",
        "tape_alias_different_bank",
        "recovery",
    ] {
        let mutate = |state: &mut [Vec<u32>; 8], push: &mut [u32; 31]| {
            // A nonzero current pair proves it is reset, without confusing it with the
            // prior input pair or seed. Journal padding is already a nonzero canary.
            state[2][5 + case.ordinal * 2..7 + case.ordinal * 2].fill(TAIL_CANARY);
            match fault {
                "count_zero" => push[0] = 0,
                "count_mismatch" => push[0] += 1,
                "count_large" => push[0] = RESIDENT_JOURNAL_ID_WORDS as u32 + 1,
                "used_zero" => push[6] = 0,
                "gating" => push[14] = 2,
                "wrong_predicate" => push[15] = (5 + case.ordinal * 2) as u32,
                "ordinal_large" => push[5] = RESIDENT_MAX_LAYERS as u32,
                "sticky_invalid" => state[2][3] = 1,
                "header" => state[5][0] = 99,
                "queue_pointer" => state[5][POOL_BASE * 4] = u32::MAX,
                "queue_tail" => {
                    let queue = state[5][POOL_BASE * 4] as usize * 4;
                    state[5][queue + 1] = state[5][queue + 2] + 1;
                }
                "canonical" => {
                    // ForcedRandom routes ID0 within the registered bank, not global block0.
                    let block = probe.owner_push[9] as usize;
                    let entry = &state[6][block * 4..block * 4 + 4];
                    assert_eq!(entry[2], 1, "corrupt an actual READY routed block");
                    let slot = entry[3] as usize;
                    assert!(slot < state[5][2] as usize);
                    let address = state[5][7] as usize;
                    assert!((address + slot) * 4 + 4 <= state[5].len());
                    state[5][(address + slot) * 4] ^= 16;
                }
                "reverse_owner" => {
                    let block = probe.owner_push[9] as usize;
                    let entry = &state[6][block * 4..block * 4 + 4];
                    assert_eq!(entry[2], 1, "corrupt an actual READY routed block");
                    let slot = entry[3] as usize;
                    assert!(slot < state[5][2] as usize);
                    let slots = state[5][6] as usize;
                    assert!((slots + slot) * 4 + 4 <= state[5].len());
                    assert_eq!(state[5][(slots + slot) * 4 + 2], block as u32);
                    state[5][(slots + slot) * 4 + 2] = u32::MAX;
                }
                "partial_bank_alias" => push[10] = push[9] + 1,
                "tape_alias_different_bank" => push[3] = push[2],
                "recovery" => push[12] = 1,
                _ => unreachable!(),
            }
        };
        for repeat in 0..4 {
            let old = probe.raw_fused(&be, previous.fused, mutate);
            let new = probe.raw_fused(&be, kernels.fused, mutate);
            assert_eq!(
                old, new,
                "fault={fault} repeat={repeat} all8+weights+logits"
            );
            assert_eq!(new[9], probe.logits_bits);
            assert_eq!(
                new[2][probe.predicate],
                probe.fixture.initial[2][probe.predicate]
            );
            assert_eq!(
                new[2][RESIDENT_SEED_WORD],
                probe.fixture.initial[2][RESIDENT_SEED_WORD]
            );
            if matches!(
                fault,
                "count_zero"
                    | "count_mismatch"
                    | "count_large"
                    | "used_zero"
                    | "gating"
                    | "wrong_predicate"
                    | "ordinal_large"
                    | "recovery"
            ) {
                assert_eq!(new[8], probe.weights_bits, "must not route: {fault}");
            }
        }
    }
    let cpu = TopkProbe::new(
        &be,
        &kernels,
        Case::normal("overlap_cpu_miss", 10, Workload::OneExpertMiss),
        40,
    );
    let cpu_source = |state: &mut [Vec<u32>; 8], _: &mut [u32; 31]| {
        let block = cpu.owner_push[9] as usize;
        assert_eq!(state[6][block * 4 + 2], 0, "ID0 must really miss");
        let source = &mut state[4][block * 4..block * 4 + 4];
        assert!(source[0] != 0 || source[1] != 0);
        assert!(source[2] > 0, "retain the registered source size/class");
        source[..2].fill(0); // This registered missing block is CPU-only, not a bad BDA.
    };
    let old = cpu.raw_fused(&be, previous.fused, cpu_source);
    let new = cpu.raw_fused(&be, kernels.fused, cpu_source);
    assert_eq!(old, new, "CPU-serviced miss metadata/float bits");
    assert_eq!(new[3][RESIDENT_JOURNAL_STATUS_WORD], 5);

    // Keep ID0 missing so the new partial continuation must supplement the cached hot
    // proof. ID1 is a real READY hit: testing stamps on an all-hit batch would never
    // exercise the cold-only stamp/source contract that this continuation must preserve.
    for fault in [
        "ready_stamp_zero",
        "ready_stamp_future",
        "source_alignment",
        "source_wrap",
        "source_pool_invalid",
        "source_pool_changed",
    ] {
        let mutate = |state: &mut [Vec<u32>; 8], _: &mut [u32; 31]| {
            let base = cpu.owner_push[9] as usize;
            assert_eq!(state[6][base * 4 + 2], 0, "ID0 must force a partial miss");
            let block = base + 1;
            let entry = &state[6][block * 4..block * 4 + 4];
            assert_eq!(entry[2], 1, "ID1 must be an actual READY hit");
            let slot = entry[3] as usize;
            assert!(slot < state[5][2] as usize);
            let slot_word = (state[5][6] as usize + slot) * 4;
            assert!(slot_word + 4 <= state[5].len());
            assert_eq!(state[5][slot_word + 2], block as u32);
            assert_eq!(state[5][slot_word + 3], 1);
            match fault {
                "ready_stamp_zero" => state[5][slot_word..slot_word + 2].fill(0),
                "ready_stamp_future" => {
                    let future = stamp(&state[5][10..12]).checked_add(1).unwrap();
                    state[5][slot_word..slot_word + 2].copy_from_slice(&pair(future));
                }
                "source_alignment" => state[4][block * 4] |= 4,
                "source_wrap" => {
                    let source = &mut state[4][base * 4..base * 4 + 4];
                    assert!(source[2] > 16);
                    source[..2].copy_from_slice(&[0xffff_fff0, u32::MAX]);
                }
                "source_pool_invalid" => state[4][base * 4 + 3] = state[5][1],
                "source_pool_changed" => {
                    assert_eq!(state[4][block * 4 + 3], 0);
                    state[4][block * 4 + 3] = 1;
                }
                _ => unreachable!(),
            }
        };
        let old = cpu.raw_fused(&be, previous.fused, mutate);
        let new = cpu.raw_fused(&be, kernels.fused, mutate);
        assert_eq!(old, new, "partial supplement fault={fault} all8+float bits");
        assert_eq!(new[2][0], 0, "stop on {fault}");
        assert_eq!(new[2][3], 1, "retain invalid on {fault}");
        assert_eq!(new[3][RESIDENT_JOURNAL_STATUS_WORD], 3);
        assert_eq!(
            new[8], cpu.expected_weights,
            "authorized TopK still ran: {fault}"
        );
        assert_eq!(new[9], cpu.logits_bits);
        assert_eq!(new[2][cpu.predicate], cpu.fixture.initial[2][cpu.predicate]);
    }
}

fn paired_probe_shape(used: Option<&str>, experts: Option<&str>) -> (usize, usize) {
    let used = used.map_or(10, |value| {
        value
            .parse()
            .expect("MOE4ALL_OWNER_TOPK_PAIRED_USED must be 8, 10 or 16")
    });
    let experts = experts.map_or(EXPERTS, |value| {
        value
            .parse()
            .expect("MOE4ALL_OWNER_TOPK_PAIRED_EXPERTS must be 256 or 512")
    });
    assert!(
        [8, 10, 16].contains(&used),
        "paired used must be 8, 10 or 16"
    );
    assert!(
        [256, 512].contains(&experts),
        "paired experts must be 256 or 512"
    );
    (used, experts)
}

#[test]
fn gpu_owned_topk_paired_shape_keeps_forced_routes_and_bank_stride() {
    assert_eq!(paired_probe_shape(None, None), (10, EXPERTS));
    for used in [8, 10, 16] {
        let case = Case::normal("paired_shape", used, Workload::AllHit);
        let original = case.logits();
        for experts in [256, 512] {
            assert_eq!(
                paired_probe_shape(Some(&used.to_string()), Some(&experts.to_string())),
                (used, experts)
            );
            let mut logits = original.clone();
            logits.truncate(experts);
            logits.extend([TAIL_CANARY; 4]);
            assert_eq!(&logits[..experts], &original[..experts]);
            assert_eq!(&logits[experts..], &[TAIL_CANARY; 4]);
            let selected_min = logits[..used]
                .iter()
                .copied()
                .map(f32::from_bits)
                .fold(f32::INFINITY, f32::min);
            assert!(logits[used..experts]
                .iter()
                .all(|&bits| f32::from_bits(bits) < selected_min));
            // Active ranges can be narrower than the original physical banks; source
            // class and CPU reference still use the unchanged canonical stride.
            let bases = [0, 1, 2].map(|role| ((LAYERS - 1) * 3 + role) * EXPERTS);
            for (role, base) in bases.into_iter().enumerate() {
                assert!(base + experts <= BLOCKS);
                for id in 0..experts {
                    assert_eq!(pool_of(base + id), usize::from(role == 2));
                }
            }
        }
    }
    for (used, experts) in [("7", "512"), ("17", "512"), ("10", "128"), ("bad", "512")] {
        assert!(
            std::panic::catch_unwind(|| paired_probe_shape(Some(used), Some(experts))).is_err()
        );
    }
}

#[test]
#[ignore = "parent serial idle GPU only; previous fused vs overlapped header, bounded paired latency"]
fn gpu_owned_topk_header_overlap_paired_latency_probe() {
    let used_env = std::env::var("MOE4ALL_OWNER_TOPK_PAIRED_USED").ok();
    let experts_env = std::env::var("MOE4ALL_OWNER_TOPK_PAIRED_EXPERTS").ok();
    let (used, experts) = paired_probe_shape(used_env.as_deref(), experts_env.as_deref());
    let be = backend();
    let kernels = TopkKernels::new(&be);
    let previous = kernels.previous_fused(&be);
    assert!(be.shared.submit_timestamp_valid_bits > 0);
    let queries = Queries {
        be: &be,
        pool: unsafe {
            be.shared
                .device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
                .unwrap()
        },
    };
    let period = be.shared.submit_timestamp_period_ns as f64;
    for workload in [Workload::AllHit, Workload::OneExpertMiss] {
        let case = Case::normal("overlap_paired", used, workload);
        let probe = TopkProbe::new_with_experts(&be, &kernels, case, 4096, experts);
        assert!(matches!(case.logits, Logits::ForcedRandom));
        assert_eq!(probe.router_push[0], experts as u32);
        assert_eq!(probe.owner_push[1], experts as u32);
        assert_eq!(probe.fused_push[1], experts as u32);
        let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
        for round in 0..6 {
            let mut states = Vec::new();
            for offset in 0..2 {
                let arm = (round + offset) % 2;
                let selected = if arm == 0 { &previous } else { &kernels };
                let (state, ticks) = probe.run(&be, selected, true, Some(&queries));
                states.push(state);
                if round >= 2 {
                    samples[arm].push(ticks.unwrap() as f64 * period / 1000.0);
                }
            }
            assert_eq!(
                states[0], states[1],
                "paired round={round} workload={workload:?} used={used} experts={experts}"
            );
        }
        for (name, mut values) in ["previous_fused", "overlapped_header"]
            .into_iter()
            .zip(samples)
        {
            values.sort_by(f64::total_cmp);
            eprintln!("GPU_OWNER_TOPK_HEADER_PAIRED variant={name} workload={} slots=4096 used={used} experts={experts} bank_stride={EXPERTS} gating={} rounds=6 warmup=2 n={} p50_us={:.3} mean_us={:.3} all8_raw=pass float_bits=pass exact_lru=pass",
                workload.name(), case.gating, values.len(), (values[1] + values[2]) * 0.5,
                values.iter().sum::<f64>() / values.len() as f64);
        }
    }
}
