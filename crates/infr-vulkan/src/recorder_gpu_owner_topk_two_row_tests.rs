//! External WG128 experiment. The oracle is the actual multirow router plus v69 WG64 owner.
//! Synthetic BDAs never reach payload dispatches; the recovery test replaces them with real BDAs.
use super::*;

type Raw = Vec<Vec<u32>>;
const CANARY: u32 = 0xa53c_91e7;
const WEIGHT_CANARY: u32 = 0x6a12_3456;
const SCALE: f32 = 1.375;

fn predicate_word(ordinal: usize) -> usize {
    ordinal
        .checked_sub(1)
        .map_or(RESIDENT_SEED_WORD, |n| 5 + n * 2)
}

fn external(variable: &str) -> Vec<u32> {
    let path = std::env::var_os(variable).unwrap_or_else(|| panic!("set {variable}"));
    eprintln!(
        "TWO_ROW_TOPK_SPV env={variable} path={}",
        std::path::Path::new(&path).display()
    );
    ash::util::read_spv(&mut std::io::Cursor::new(std::fs::read(path).unwrap())).unwrap()
}

struct Kernels {
    router: ComputeKernel,
    owner: [ComputeKernel; 2],
    candidate: [ComputeKernel; 2],
    fused_reference: Option<[ComputeKernel; 2]>,
}

impl Kernels {
    fn new(be: &VulkanBackend) -> Self {
        assert!(be.shared.compute_subgroup_operations.contains(
            vk::SubgroupFeatureFlags::BASIC
                | vk::SubgroupFeatureFlags::ARITHMETIC
                | vk::SubgroupFeatureFlags::BALLOT
        ));
        assert!(be.shared.conditional_rendering.is_some());
        let router = be.kernel_sg(
            "two_row_topk_actual_multirow",
            crate::gemm::moe_topk_sg_spv(),
            5,
            36,
            32,
        );
        let owner = [false, true].map(|overlap| {
            be.kernel(
                if overlap {
                    "two_row_topk_reference_owner_overlap"
                } else {
                    "two_row_topk_reference_owner"
                },
                crate::gemm::gpu_owned_prepare_fused_spv(overlap),
                if overlap { 9 } else { 8 },
                GPU_OWNED_FAST_PUSH_BYTES,
            )
        });
        let candidate = [
            (
                "two_row_topk_external_serial",
                "MOE4ALL_OWNER_TOPK_TWO_ROW_SERIAL_SPV",
                10,
            ),
            (
                "two_row_topk_external_overlap",
                "MOE4ALL_OWNER_TOPK_TWO_ROW_OVERLAP_SPV",
                11,
            ),
        ]
        .map(|(name, variable, bindings)| {
            be.kernel_sg(
                name,
                &external(variable),
                bindings,
                GPU_OWNED_FAST_PUSH_BYTES,
                32,
            )
        });
        let reference_serial = "MOE4ALL_OWNER_TOPK_TWO_ROW_REFERENCE_SERIAL_SPV";
        let reference_overlap = "MOE4ALL_OWNER_TOPK_TWO_ROW_REFERENCE_OVERLAP_SPV";
        let fused_reference = if std::env::var_os(reference_serial).is_some()
            || std::env::var_os(reference_overlap).is_some()
        {
            Some(
                [
                    ("two_row_topk_fused_reference_serial", reference_serial, 10),
                    (
                        "two_row_topk_fused_reference_overlap",
                        reference_overlap,
                        11,
                    ),
                ]
                .map(|(name, variable, bindings)| {
                    be.kernel_sg(
                        name,
                        &external(variable),
                        bindings,
                        GPU_OWNED_FAST_PUSH_BYTES,
                        32,
                    )
                }),
            )
        } else {
            None
        };
        eprintln!("TWO_ROW_TOPK_REFERENCE router=moe_topk_sg router_workgroups=2 owner=general_fused_wg64 router_spv_hash={:016x} serial_owner_spv_hash={:016x} overlap_owner_spv_hash={:016x} candidate_serial_spv_hash={:016x} candidate_overlap_spv_hash={:016x} hash_kind=internal_spv_hash_not_sha256 production_switch=false", router.spv_hash, owner[0].spv_hash, owner[1].spv_hash, candidate[0].spv_hash, candidate[1].spv_hash);
        Self {
            router,
            owner,
            candidate,
            fused_reference,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Distinct,
    Duplicate,
    Saturated,
    Nonfinite,
    AllNan,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    pattern: Pattern,
    gating: u32,
    workload: Workload,
    overlap: bool,
    predicate: bool,
    run: bool,
    ordinal: usize,
    bank_alias: u32,
}

impl Case {
    fn new(name: &'static str, pattern: Pattern, workload: Workload, overlap: bool) -> Self {
        Self {
            name,
            pattern,
            gating: 1,
            workload,
            overlap,
            predicate: true,
            run: true,
            ordinal: 0,
            bank_alias: 0,
        }
    }
}

struct Probe {
    pool: PoolProbe,
    case: Case,
    logits: Box<dyn Buffer>,
    weights: Box<dyn Buffer>,
    miss: Box<dyn Buffer>,
    logits_bits: Vec<u32>,
    weights_bits: Vec<u32>,
    expected_weights: Vec<u32>,
    selected: Vec<u32>,
    push: [u32; 31],
    router_push: [u32; 9],
}

impl Probe {
    fn new(be: &VulkanBackend, kernels: &Kernels, case: Case, slots: usize) -> Self {
        let used = probe_used();
        assert!(
            matches!(used, 8 | 10),
            "separate used8/10 subprocesses; no global env mutation"
        );
        let count = used * 2;
        let mut pool = PoolProbe::new(be, slots, count, case.workload);
        pool.repeats = 1;
        pool.push[5] = case.ordinal as u32;
        if case.bank_alias != 0 {
            pool.push[10] = pool.push[9] + case.bank_alias - 1;
        }
        pool.initial[0] = (0..count as u32).chain([CANARY; 8]).collect();
        pool.initial[2][0] = u32::from(case.run);
        pool.initial[2][predicate_word(case.ordinal)] = u32::from(case.predicate);
        if case.ordinal != 0 {
            pool.initial[2][RESIDENT_SEED_WORD] = u32::from(!case.predicate);
        }
        pool.initial[3] = vec![CANARY; (case.ordinal + 2) * RESIDENT_JOURNAL_WORDS];
        pool.initial[7] = vec![CANARY; (case.ordinal + 2) * RESIDENT_JOURNAL_ID_WORDS * 3 * 8];
        for binding in [0, 3, 7] {
            pool.buffers[binding] = be
                .alloc_mapped_arena_bda(pool.initial[binding].len() * 4)
                .unwrap()
                .0;
        }
        pool.buffers[5] = be
            .alloc_mapped_arena_bda((pool.heap_base + 2 + slots * 4) * 16)
            .unwrap()
            .0;
        let mut logits_bits = Vec::new();
        for row in 0..2 {
            let start = if matches!(case.pattern, Pattern::Duplicate) {
                0
            } else {
                row * used
            };
            for expert in 0..EXPERTS {
                let mut value = if (start..start + used).contains(&expert) {
                    2.0 - (expert - start) as f32 / 32.0
                } else {
                    -12.0 - expert as f32 / 1024.0
                };
                match case.pattern {
                    Pattern::Saturated => {
                        value = if expert == 0 {
                            20.0
                        } else if expert == 1 {
                            30.0
                        } else if expert < used {
                            2.0 - expert as f32 / 32.0
                        } else {
                            -12.0
                        };
                    }
                    Pattern::Nonfinite => {
                        value = match expert {
                            0 | 31 => f32::INFINITY,
                            1 => f32::NEG_INFINITY,
                            2 => f32::from_bits(0x7fc0_0123 + row as u32),
                            3 => -1.0e31,
                            _ => value,
                        };
                    }
                    Pattern::AllNan => value = f32::from_bits(0x7fc0_0123 + row as u32),
                    _ => (),
                }
                logits_bits.push(value.to_bits());
            }
        }
        logits_bits.extend([CANARY; 4]);
        let weights_bits = vec![WEIGHT_CANARY; count + 4];
        let logits = be.alloc_mapped_arena_bda(logits_bits.len() * 4).unwrap().0;
        let weights = be.alloc_mapped_arena_bda(weights_bits.len() * 4).unwrap().0;
        let miss = be
            .alloc_mapped_arena_bda((case.ordinal + 2) * 16)
            .unwrap()
            .0;
        let n = slots as u32;
        let geometry = [
            0,
            SLOT_BYTES[0],
            0,
            n,
            0,
            SLOT_BYTES[0],
            0,
            n,
            1,
            SLOT_BYTES[1],
            n,
            n,
        ];
        let mut push = gpu_owned_fast_push(&pool.push, &geometry);
        push[13..16].copy_from_slice(&[
            SCALE.to_bits(),
            case.gating,
            predicate_word(case.ordinal) as u32,
        ]);
        let router_push = [
            EXPERTS as u32,
            used as u32,
            SCALE.to_bits(),
            case.gating,
            1,
            0,
            0,
            0,
            0,
        ];
        let mut probe = Self {
            pool,
            case,
            logits,
            weights,
            miss,
            logits_bits,
            weights_bits,
            expected_weights: Vec::new(),
            selected: Vec::new(),
            push,
            router_push,
        };
        probe.reset(be);
        // Capture only the actual generic router, never the wave32 single-row substitute.
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        rec.begin_immutable_conditional_at(
            probe.pool.buffers[2].as_ref(),
            predicate_word(case.ordinal),
        );
        probe.record_router(&rec, kernels);
        rec.end_conditional();
        rec.owned_copy_barrier(false, true);
        rec.finish().unwrap();
        probe.selected = read(be, probe.pool.buffers[0].as_ref())[..count].to_vec();
        probe.expected_weights = read(be, probe.weights.as_ref());
        if !case.predicate {
            assert_eq!(probe.expected_weights, probe.weights_bits);
        }
        if case.predicate && matches!(case.pattern, Pattern::Saturated) {
            assert_eq!(
                &probe.selected[..2],
                &[0, 1],
                "generic sigmoid saturation tie, NOT raw-logit ranking"
            );
            assert_eq!(&probe.selected[used..used + 2], &[0, 1]);
        }
        probe.prepare_reference();
        if std::env::var("MOE4ALL_OWNER_ORACLE_LEGACY_TOUCH").as_deref() == Ok("1") {
            super::legacy_touch_tests::legacy_stamps_for(&mut probe.pool, &probe.selected);
        }
        probe
    }

    fn prepare_reference(&mut self) {
        let count = self.push[0] as usize;
        let mut initial = self.pool.initial.clone();
        initial[0][..count].copy_from_slice(&self.selected);
        let mut expected;
        if self.case.run {
            let roles = [self.push[9], self.push[10], self.push[11]].map(|x| x as usize);
            (expected, self.pool.misses) = reference(
                &initial,
                &self.selected,
                roles,
                self.pool.slots_per_pool,
                self.pool.address_base,
            );
            expected[0][count + 2..count + 4].fill(0);
            if self.case.overlap {
                for row in 0..2 {
                    let hits = expected[3][RESIDENT_JOURNAL_MASK_WORD + row];
                    expected[0][count + row] = hits;
                    expected[0][count + 2 + row] = ((1 << self.push[6]) - 1) ^ hits;
                }
            }
            expected[2] = initial[2].clone();
            expected[2][2] = self.case.ordinal as u32 + 1;
            expected[2][4] = 1;
            expected[2][5 + self.case.ordinal * 2..7 + self.case.ordinal * 2]
                .copy_from_slice(&[1, 1]);
            if self.case.ordinal != 0 {
                let journal = expected[3][..RESIDENT_JOURNAL_WORDS].to_vec();
                expected[3] = initial[3].clone();
                let base = self.case.ordinal * RESIDENT_JOURNAL_WORDS;
                expected[3][base..base + RESIDENT_JOURNAL_WORDS].copy_from_slice(&journal);
                let stride = RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
                let jobs = expected[7][..stride].to_vec();
                expected[7] = initial[7].clone();
                expected[7][self.case.ordinal * stride..(self.case.ordinal + 1) * stride]
                    .copy_from_slice(&jobs);
            }
        } else {
            expected = initial;
            let base = self.case.ordinal * RESIDENT_JOURNAL_WORDS;
            expected[3][base + RESIDENT_JOURNAL_JOB_COUNT_WORD
                ..base + RESIDENT_JOURNAL_JOB_COUNT_WORD + 3]
                .fill(0);
            expected[3][base + RESIDENT_JOURNAL_STATUS_WORD] = 0;
            expected[2][4] = 0;
            expected[2][5 + self.case.ordinal * 2..7 + self.case.ordinal * 2].fill(0);
            self.pool.misses = 0;
        }
        self.pool.expected = expected;
    }

    fn reset(&self, be: &VulkanBackend) {
        self.pool.reset(be, false, true);
        upload(be, self.logits.as_ref(), &self.logits_bits);
        upload(be, self.weights.as_ref(), &self.weights_bits);
        upload(
            be,
            self.miss.as_ref(),
            &vec![CANARY; (self.case.ordinal + 2) * 4],
        );
    }

    fn snapshot(&self, be: &VulkanBackend) -> Raw {
        let mut raw: Raw = self
            .pool
            .buffers
            .iter()
            .map(|b| read(be, b.as_ref()))
            .collect();
        raw.push(read(be, self.weights.as_ref()));
        raw.push(read(be, self.logits.as_ref()));
        if self.case.overlap {
            raw.push(read(be, self.miss.as_ref()));
        }
        raw
    }

    fn restore(&self, be: &VulkanBackend, raw: &Raw) {
        for (buffer, words) in self.pool.buffers.iter().zip(raw) {
            upload(be, buffer.as_ref(), words);
        }
        upload(be, self.weights.as_ref(), &raw[8]);
        upload(be, self.logits.as_ref(), &raw[9]);
        if self.case.overlap {
            upload(be, self.miss.as_ref(), &raw[10]);
        }
    }

    fn bindings(&self, fused: bool) -> Vec<vk::DescriptorBufferInfo> {
        let mut bindings: Vec<_> = self
            .pool
            .buffers
            .iter()
            .map(|b| Recorder::vkb_exact(b.as_ref()))
            .collect();
        if self.case.overlap {
            bindings.push(Recorder::vkb_exact(self.miss.as_ref()));
        }
        if fused {
            bindings.extend([
                Recorder::vkb_exact(self.logits.as_ref()),
                Recorder::vkb_exact(self.weights.as_ref()),
            ]);
        }
        bindings
    }

    fn record_router(&self, rec: &Recorder<'_>, kernels: &Kernels) {
        rec.dispatch(
            kernels.router,
            &[
                Recorder::vkb_exact(self.logits.as_ref()),
                Recorder::vkb_exact(self.logits.as_ref()),
                Recorder::vkb_exact(self.logits.as_ref()),
                Recorder::vkb_exact(self.pool.buffers[0].as_ref()),
                Recorder::vkb_exact(self.weights.as_ref()),
            ],
            2,
            bytemuck::cast_slice(&self.router_push),
            2,
        );
    }

    fn record(&self, rec: &Recorder<'_>, kernels: &Kernels, fused: bool, push: &[u32; 31]) {
        let reference = (!fused)
            .then_some(kernels.fused_reference.as_ref())
            .flatten();
        let fused_shader = fused || reference.is_some();
        // Keep the hardware predicate on routing only. Owner initialization is unconditional.
        rec.begin_immutable_conditional_at(
            self.pool.buffers[2].as_ref(),
            predicate_word(self.case.ordinal),
        );
        if !fused_shader && push[12] == 0 {
            self.record_router(rec, kernels);
        }
        rec.end_conditional();
        let mut owner_push = *push;
        if !fused_shader {
            owner_push[13..18].fill(0);
        }
        let bindings = self.bindings(fused_shader);
        rec.dispatch(
            if fused {
                kernels.candidate[self.case.overlap as usize]
            } else if let Some(reference) = reference {
                reference[self.case.overlap as usize]
            } else {
                kernels.owner[self.case.overlap as usize]
            },
            &bindings,
            bindings.len(),
            bytemuck::cast_slice(&owner_push),
            1,
        );
    }

    fn run(
        &self,
        be: &VulkanBackend,
        kernels: &Kernels,
        fused: bool,
        seed: &Raw,
        push: &[u32; 31],
        query: Option<&Queries<'_>>,
    ) -> (Raw, Option<f64>) {
        self.restore(be, seed);
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        if let Some(q) = query {
            unsafe {
                be.shared.device.cmd_reset_query_pool(rec.cmd, q.pool, 0, 2);
                be.shared.device.cmd_write_timestamp(
                    rec.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    q.pool,
                    0,
                );
            }
        }
        self.record(&rec, kernels, fused, push);
        if let Some(q) = query {
            unsafe {
                be.shared.device.cmd_write_timestamp(
                    rec.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    q.pool,
                    1,
                );
            }
        }
        rec.owned_copy_barrier(false, true);
        rec.finish().unwrap();
        let us = query.map(|q| {
            let mut ticks = [0u64; 2];
            unsafe {
                be.shared.device.get_query_pool_results(
                    q.pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            }
            .unwrap();
            let mut delta = ticks[1].wrapping_sub(ticks[0]);
            if be.shared.submit_timestamp_valid_bits < 64 {
                delta &= (1u64 << be.shared.submit_timestamp_valid_bits) - 1;
            }
            delta as f64 * be.shared.submit_timestamp_period_ns as f64 / 1000.0
        });
        (self.snapshot(be), us)
    }

    fn assert_oracle(&self, be: &VulkanBackend) {
        self.pool.assert_result(be, false, true, false, false);
        assert_eq!(
            read(be, self.weights.as_ref()),
            self.expected_weights,
            "all weight bits and canaries"
        );
        assert_eq!(
            read(be, self.logits.as_ref()),
            self.logits_bits,
            "all logits and canaries"
        );
        if self.case.overlap {
            let mut expected = vec![CANARY; (self.case.ordinal + 2) * 4];
            let i = self.case.ordinal * 4;
            expected[i..i + 4].copy_from_slice(&[
                self.pool.misses as u32,
                if self.pool.misses == 0 { 0 } else { 2 },
                1,
                1,
            ]);
            assert_eq!(read(be, self.miss.as_ref()), expected);
        }
    }
}

fn backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicit idle device"));
    VulkanBackend::new_with(Arc::new(cfg)).unwrap()
}

fn check(be: &VulkanBackend, kernels: &Kernels, case: Case) {
    let probe = Probe::new(be, kernels, case, 256);
    probe.reset(be);
    let seed = probe.snapshot(be);
    let (baseline, _) = probe.run(be, kernels, false, &seed, &probe.push, None);
    probe.assert_oracle(be);
    let (candidate, _) = probe.run(be, kernels, true, &seed, &probe.push, None);
    probe.assert_oracle(be);
    assert_eq!(
        candidate, baseline,
        "{} full raw, exact FP bits, physical FIFO unmasked",
        case.name
    );
    eprintln!("TWO_ROW_TOPK_EXACT case={} used={} overlap={} all_raw=true weights_bits=true exact_lru=true", case.name, probe_used(), case.overlap);
}

#[test]
#[ignore = "parent serial GPU only; actual multirow router, exact raw/fp/LRU/predicates and malformed inputs"]
fn gpu_owned_topk_two_row_exact() {
    let be = backend();
    let kernels = Kernels::new(&be);
    for overlap in [false, true] {
        for (pattern, workload) in [
            (Pattern::Distinct, Workload::AllHit),
            (Pattern::Duplicate, Workload::AllHit),
            (Pattern::Distinct, Workload::OneExpertMiss),
            (Pattern::Duplicate, Workload::OneExpertMiss),
        ] {
            check(
                &be,
                &kernels,
                Case::new("normal", pattern, workload, overlap),
            );
        }
    }
    for (pattern, gating, overlap) in [
        (Pattern::Saturated, 1, false),
        (Pattern::Nonfinite, 0, false),
        (Pattern::Nonfinite, 1, true),
        (Pattern::AllNan, 1, true),
    ] {
        let mut case = Case::new("fp_edges", pattern, Workload::AllHit, overlap);
        case.gating = gating;
        check(&be, &kernels, case);
    }
    for (predicate, run, overlap) in [(false, true, false), (true, false, true)] {
        let mut case = Case::new(
            "predicate_vs_run",
            Pattern::Distinct,
            Workload::OneExpertMiss,
            overlap,
        );
        case.ordinal = 1;
        case.predicate = predicate;
        case.run = run;
        check(&be, &kernels, case);
    }
    for alias in [1, 2] {
        let mut case = Case::new(
            "equal_or_partial_bank_alias",
            Pattern::Duplicate,
            Workload::OneExpertMiss,
            false,
        );
        case.bank_alias = alias;
        check(&be, &kernels, case);
    }
    for (fault, overlap) in [
        ("header", false),
        ("tape_alias", true),
        ("source_stride", false),
        ("preserved_bad_id", true),
    ] {
        let mut case = Case::new(fault, Pattern::Distinct, Workload::OneExpertMiss, overlap);
        if fault == "preserved_bad_id" {
            case.predicate = false;
        }
        let probe = Probe::new(&be, &kernels, case, 256);
        probe.reset(&be);
        let mut seed = probe.snapshot(&be);
        let mut push = probe.push;
        match fault {
            "header" => seed[5][0] = 2,
            "tape_alias" => push[3] = push[2],
            "source_stride" => {
                let b = push[10] as usize * 4;
                seed[4][b + 2..b + 4].copy_from_slice(&[SLOT_BYTES[1], 1]);
            }
            "preserved_bad_id" => seed[0][0] = EXPERTS as u32,
            _ => unreachable!(),
        }
        let (baseline, _) = probe.run(&be, &kernels, false, &seed, &push, None);
        let (candidate, _) = probe.run(&be, &kernels, true, &seed, &push, None);
        assert_eq!(candidate, baseline, "malformed={fault} all raw and FP bits");
        assert_eq!(candidate[3][RESIDENT_JOURNAL_STATUS_WORD], 3);
        assert_eq!(&candidate[2][..4], &[0, 0, 1, 1]);
        for binding in [1, 4, 5, 6, 7] {
            assert_eq!(
                candidate[binding], seed[binding],
                "reject before metadata commit {fault}/{binding}"
            );
        }
        assert_eq!(candidate[8], probe.expected_weights);
        assert_eq!(candidate[9], probe.logits_bits);
    }
}

fn node(words: &[u32], index: usize) -> [u32; 4] {
    words[index * 4..index * 4 + 4].try_into().unwrap()
}

fn put(words: &mut [u32], index: usize, value: [u32; 4]) {
    words[index * 4..index * 4 + 4].copy_from_slice(&value);
}

fn protected(probe: &Probe) -> HashSet<usize> {
    probe.push[9..12]
        .iter()
        .flat_map(|&base| {
            probe
                .selected
                .iter()
                .map(move |&id| base as usize + id as usize)
        })
        .collect()
}

fn fifo_seed(probe: &Probe, seed: &mut Raw, prefixes: [usize; 2], copying_head: bool) {
    let slots = probe.pool.slots_per_pool;
    let routed = protected(probe);
    let clock = stamp(&seed[5][10..12]);
    for pool in 0..2 {
        let mut order: Vec<_> = (pool * slots..(pool + 1) * slots).collect();
        order.sort_unstable_by_key(|&slot| {
            let n = node(&seed[5], SLOT_BASE + slot);
            (routed.contains(&(n[2] as usize)), stamp(&n), slot)
        });
        for (index, &slot) in order.iter().enumerate() {
            let mut n = node(&seed[5], SLOT_BASE + slot);
            n[..2].copy_from_slice(&pair(
                clock - (slots * 2) as u64 + (pool * slots + index + 1) as u64,
            ));
            if copying_head && pool == 0 && index == 0 {
                assert!(!routed.contains(&(n[2] as usize)));
                n[3] = 2;
                seed[6][n[2] as usize * 4 + 2] = 2;
            }
            put(&mut seed[5], SLOT_BASE + slot, n);
        }
        let header = seed[5][(POOL_BASE + pool * 2) * 4] as usize;
        let prefix = prefixes[pool];
        assert!(prefix + slots <= slots * 2);
        put(
            &mut seed[5],
            header,
            [0, (prefix + slots) as u32, (slots * 2) as u32, 0],
        );
        for index in 0..slots * 2 {
            let record = if index < prefix {
                let old = pair(index as u64 + 1);
                [old[0], old[1], order[0] as u32, 0]
            } else if index < prefix + slots {
                let slot = order[index - prefix];
                let n = node(&seed[5], SLOT_BASE + slot);
                [n[0], n[1], slot as u32, 0]
            } else {
                [CANARY; 4]
            };
            put(&mut seed[5], header + 1 + index, record);
        }
    }
}

fn assert_fifo(raw: &Raw, probe: &Probe) {
    let nodes = &raw[5];
    let slots = probe.pool.slots_per_pool;
    for pool in 0..2 {
        let header = nodes[(POOL_BASE + pool * 2) * 4] as usize;
        let q = node(nodes, header);
        assert!(q[0] <= q[1] && q[1] <= q[2]);
        assert_eq!(&q[2..], &[(slots * 2) as u32, 0]);
        let mut actual = Vec::new();
        for i in q[0]..q[1] {
            let record = node(nodes, header + 1 + i as usize);
            assert_eq!(record[3], 0);
            let slot = record[2] as usize;
            assert!((pool * slots..(pool + 1) * slots).contains(&slot));
            let live = node(nodes, SLOT_BASE + slot);
            if matches!(live[3], 1 | 2) && stamp(&live) == stamp(&record) {
                actual.push((stamp(&record), slot));
            }
        }
        let mut expected: Vec<_> = (pool * slots..(pool + 1) * slots)
            .filter(|&slot| matches!(nodes[(SLOT_BASE + slot) * 4 + 3], 1 | 2))
            .map(|slot| (stamp(&nodes[(SLOT_BASE + slot) * 4..]), slot))
            .collect();
        expected.sort_unstable();
        assert_eq!(
            actual, expected,
            "each READY/COPYING slot exactly once, stable FIFO LRU order"
        );
        assert_eq!(
            nodes[(POOL_BASE + pool * 2) * 4 + 1] as usize,
            expected.len()
        );
    }
}

fn assert_live_reference(raw: &Raw, seed: &Raw, probe: &Probe) {
    let mut input: [Vec<u32>; 8] = std::array::from_fn(|i| seed[i].clone());
    input[0][..probe.selected.len()].copy_from_slice(&probe.selected);
    let roles = [probe.push[9], probe.push[10], probe.push[11]].map(|x| x as usize);
    let (mut expected, _) = reference(
        &input,
        &probe.selected,
        roles,
        probe.pool.slots_per_pool,
        probe.pool.address_base,
    );
    expected[0][probe.selected.len() + 2..probe.selected.len() + 4].fill(0);
    for binding in [0, 1, 2, 3, 4, 6, 7] {
        assert_eq!(
            raw[binding], expected[binding],
            "independent live reference binding={binding}"
        );
    }
    assert_eq!(
        &raw[5][..probe.pool.heap_base * 4],
        &expected[5][..probe.pool.heap_base * 4],
        "exact owners/stamps/states/clock/stats/free counts/canonical addresses"
    );
    assert_eq!(raw[8], probe.expected_weights);
    assert_eq!(raw[9], probe.logits_bits);
}

#[test]
#[ignore = "parent serial GPU only; exact victim cutoff and stable WG64/WG128 physical compaction"]
fn gpu_owned_topk_two_row_fifo_boundaries_exact() {
    let be = backend();
    let kernels = Kernels::new(&be);
    for (cutoff, fault, copying) in [
        (63, "after", false),
        (64, "none", false),
        (64, "before", false),
        (127, "after", false),
        (128, "none", true),
        (128, "before", false),
    ] {
        let probe = Probe::new(
            &be,
            &kernels,
            Case::new(
                "victim_cutoff",
                Pattern::Duplicate,
                Workload::OneExpertMiss,
                false,
            ),
            256,
        );
        probe.reset(&be);
        let mut seed = probe.snapshot(&be);
        // Duplicate rows miss ID0 in three banks: exactly two pool0 victims and one pool1 victim.
        assert_eq!(probe.pool.misses, 3);
        let prefix = cutoff - if copying { 2 } else { 1 };
        fifo_seed(&probe, &mut seed, [prefix, cutoff], copying);
        let header = seed[5][POOL_BASE * 4] as usize;
        if fault != "none" {
            let at = if fault == "before" {
                cutoff - 1
            } else {
                cutoff + 1
            };
            seed[5][(header + 1 + at) * 4 + 3] = 1;
        }
        let (baseline, _) = probe.run(&be, &kernels, false, &seed, &probe.push, None);
        let (candidate, _) = probe.run(&be, &kernels, true, &seed, &probe.push, None);
        assert_eq!(
            candidate, baseline,
            "cutoff={cutoff} fault={fault} copying={copying} complete physical FIFO/raw/FP"
        );
        if fault == "before" {
            assert_eq!(candidate[3][RESIDENT_JOURNAL_STATUS_WORD], 3);
            for binding in [1, 4, 5, 6, 7] {
                assert_eq!(candidate[binding], seed[binding]);
            }
        } else {
            assert_eq!(candidate[3][RESIDENT_JOURNAL_STATUS_WORD], 4);
            assert_eq!(
                candidate[5][header * 4] as usize,
                prefix,
                "first matching READY/COPYING head"
            );
            assert_live_reference(&candidate, &seed, &probe);
            // An intentionally malformed record beyond the serial cutoff is allowed to
            // remain uninspected. Do not pretend this adversarial seed is a legal full log.
            if fault == "none" {
                assert_fifo(&candidate, &probe);
            }
        }
        eprintln!("TWO_ROW_TOPK_FIFO cutoff={cutoff} fault={fault} copying_head={copying} raw_physical_unmasked=true");
    }
    let probe = Probe::new(
        &be,
        &kernels,
        Case::new(
            "forced_compaction",
            Pattern::Distinct,
            Workload::AllHit,
            false,
        ),
        256,
    );
    for tail in [511usize, 512] {
        probe.reset(&be);
        let mut seed = probe.snapshot(&be);
        fifo_seed(&probe, &mut seed, [tail - 256; 2], false);
        let (baseline, _) = probe.run(&be, &kernels, false, &seed, &probe.push, None);
        let (candidate, _) = probe.run(&be, &kernels, true, &seed, &probe.push, None);
        assert_eq!(
            candidate, baseline,
            "WG64/WG128 complete stable compaction and raw FP bits tail={tail}"
        );
        assert_live_reference(&candidate, &seed, &probe);
        assert_fifo(&candidate, &probe);
        for pool in 0..2 {
            let header = seed[5][(POOL_BASE + pool * 2) * 4] as usize;
            assert_eq!(seed[5][header * 4 + 1] as usize, tail);
            assert_eq!(candidate[5][header * 4], 0);
            assert!(
                (candidate[5][header * 4 + 1] as usize) < tail,
                "real compaction coverage, including a partial final WG128 chunk"
            );
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; real BDA CPU-only copy, guarded recovery, changed logits must not reroute"]
fn gpu_owned_topk_two_row_cpu_recovery_exact() {
    let be = backend();
    let kernels = Kernels::new(&be);
    for overlap in [false, true] {
        let mut probe = Probe::new(
            &be,
            &kernels,
            Case::new(
                "cpu_recovery",
                Pattern::Duplicate,
                Workload::OneExpertMiss,
                overlap,
            ),
            128,
        );
        let (source, src) = be.alloc_mapped_arena_bda(SLOT_BYTES[1] as usize).unwrap();
        let bytes = probe.pool.slots_per_pool * (SLOT_BYTES[0] + SLOT_BYTES[1]) as usize;
        let (destination, dst) = be.alloc_mapped_arena_bda(bytes).unwrap();
        let pattern: Vec<u32> = (0..SLOT_BYTES[1] / 4).map(|i| 0x3f00_0000 ^ i).collect();
        upload(&be, source.as_ref(), &pattern);
        for block in 0..BLOCKS {
            probe.pool.initial[4][block * 4..block * 4 + 2].copy_from_slice(&pair(src));
        }
        let cpu_block = probe.push[9] as usize;
        probe.pool.initial[4][cpu_block * 4..cpu_block * 4 + 2].fill(0);
        for slot in 0..probe.pool.slots_per_pool * 2 {
            let pool = slot / probe.pool.slots_per_pool;
            let offset = if pool == 0 {
                slot * SLOT_BYTES[0] as usize
            } else {
                probe.pool.slots_per_pool * SLOT_BYTES[0] as usize
                    + (slot - probe.pool.slots_per_pool) * SLOT_BYTES[1] as usize
            };
            let address = pair(dst + offset as u64);
            let at = (probe.pool.address_base + slot) * 4;
            probe.pool.initial[5][at..at + 2].copy_from_slice(&address);
            let block = probe.pool.initial[5][(SLOT_BASE + slot) * 4 + 2] as usize;
            probe.pool.initial[6][block * 4..block * 4 + 2].copy_from_slice(&address);
        }
        for bad_generation in [false, true] {
            let mut baseline = None;
            for fused in [false, true] {
                probe.reset(&be);
                upload(&be, destination.as_ref(), &vec![0xa5a5_a5a5; bytes / 4]);
                let seed = probe.snapshot(&be);
                let (prepared, _) = probe.run(&be, &kernels, fused, &seed, &probe.push, None);
                assert_eq!(prepared[3][RESIDENT_JOURNAL_STATUS_WORD], 5);
                assert_eq!(prepared[3][RESIDENT_JOURNAL_JOB_COUNT_WORD], 3);
                assert_eq!(prepared[8], probe.expected_weights);
                let bindings = probe.bindings(false);
                let rec = be.recorder().unwrap();
                rec.owned_copy_barrier(false, false);
                let payload = be.kernel(
                    "gpu_owned_payload",
                    crate::gemm::gpu_resident_copy_payload_spv(false),
                    2,
                    4,
                );
                rec.dispatch(
                    payload,
                    &[bindings[3], bindings[7]],
                    2,
                    bytemuck::bytes_of(&0u32),
                    2,
                );
                rec.owned_copy_barrier(false, false);
                let publish = be.kernel(
                    "gpu_owned_publish",
                    crate::gemm::gpu_owned_publish_spv(),
                    4,
                    8,
                );
                let pub_bindings = [bindings[3], bindings[7], bindings[5], bindings[6]];
                rec.dispatch(
                    publish,
                    &pub_bindings,
                    2,
                    bytemuck::cast_slice(&[0u32, 0]),
                    1,
                );
                rec.owned_copy_barrier(false, true);
                rec.finish().unwrap();
                let stopped = probe.snapshot(&be);
                assert_fifo(&stopped, &probe);
                let mut expected_bits = vec![0xa5a5_a5a5; bytes / 4];
                let mut cpu_count = 0;
                for job in stopped[7][..24].chunks_exact(8) {
                    let offset = stamp(&job[2..4]).checked_sub(dst).unwrap() as usize;
                    let len = job[4] as usize;
                    assert!(offset <= bytes && len <= bytes - offset);
                    if stamp(job) == 0 {
                        cpu_count += 1;
                        let ptr = as_vk_buf(destination.as_ref())
                            .unwrap()
                            .mapped_ptr()
                            .unwrap();
                        crate::copy_to_mapped(
                            &bytemuck::cast_slice::<u32, u8>(&pattern)[..len],
                            unsafe { ptr.add(offset) },
                        );
                    }
                    expected_bits[offset / 4..(offset + len) / 4]
                        .copy_from_slice(&pattern[..len / 4]);
                }
                assert_eq!(cpu_count, 1);
                let mut changed_logits = probe.logits_bits.clone();
                changed_logits[..EXPERTS * 2].fill((-10.0f32).to_bits());
                changed_logits[EXPERTS - 1] = 10.0f32.to_bits();
                changed_logits[EXPERTS * 2 - 1] = 10.0f32.to_bits();
                upload(&be, probe.logits.as_ref(), &changed_logits);
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                rec.gpu_owned_recovery_seed(
                    probe.pool.buffers[2].as_ref(),
                    probe.pool.buffers[3].as_ref(),
                    probe.pool.buffers[5].as_ref(),
                    [0, 1, if bad_generation { 42 } else { 41 }, 7],
                );
                rec.begin_immutable_conditional_at(
                    probe.pool.buffers[2].as_ref(),
                    RESIDENT_SEED_WORD,
                );
                rec.owned_copy_barrier(true, false);
                rec.dispatch(
                    publish,
                    &pub_bindings,
                    2,
                    bytemuck::cast_slice(&[0u32, 1]),
                    1,
                );
                rec.owned_copy_barrier(false, false);
                rec.end_conditional();
                let mut push = probe.push;
                push[12] = 1;
                probe.record(&rec, &kernels, fused, &push);
                rec.owned_copy_barrier(false, true);
                rec.finish().unwrap();
                let actual = probe.snapshot(&be);
                assert_eq!(
                    actual[8], probe.expected_weights,
                    "recovery must not reroute after logits change"
                );
                assert_eq!(actual[9], changed_logits);
                assert_eq!(
                    actual[5][10..12],
                    stopped[5][10..12],
                    "no second LRU clock touch"
                );
                assert_eq!(
                    stamp(&actual[5][10..12]),
                    stamp(&seed[5][10..12]) + probe.push[0] as u64 * 3
                );
                if bad_generation {
                    assert_eq!(actual[2][0], 0);
                    assert_eq!(actual[2][3], 1);
                    assert_eq!(actual[2][RESIDENT_SEED_WORD], 0);
                    assert_eq!(actual[3][RESIDENT_JOURNAL_STATUS_WORD], 0);
                    for binding in [0, 1, 4, 5, 6, 7] {
                        assert_eq!(actual[binding], stopped[binding]);
                    }
                } else {
                    assert_eq!(&actual[2][..4], &[1, NONE, 1, 0]);
                    assert_eq!(actual[3][RESIDENT_JOURNAL_STATUS_WORD], 1);
                    for &block in &protected(&probe) {
                        assert_eq!(
                            actual[6][block * 4 + 2],
                            1,
                            "all requested slots published READY"
                        );
                    }
                }
                assert_fifo(&actual, &probe);
                let bits = read(&be, destination.as_ref());
                assert_eq!(
                    bits, expected_bits,
                    "all payload float bits and untouched bytes"
                );
                let result = (prepared, stopped, actual, bits);
                if let Some(expected) = &baseline {
                    assert_eq!(
                        &result, expected,
                        "all raw recovery overlap={overlap} bad_generation={bad_generation}"
                    );
                } else {
                    baseline = Some(result);
                }
            }
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; exact/recovery/boundary tests first, full router+owner paired timestamps"]
fn gpu_owned_topk_two_row_paired_latency_probe() {
    let be = backend();
    let kernels = Kernels::new(&be);
    let reference_name = if kernels.fused_reference.is_some() {
        "external_fused_wg128"
    } else {
        "actual_moe_topk_sg_2wg_plus_v69_owner_wg64"
    };
    assert!(be.shared.submit_timestamp_valid_bits > 0);
    let queries = Queries {
        be: &be,
        pool: unsafe {
            be.shared.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .unwrap(),
    };
    for overlap in [false, true] {
        for (pattern, workload) in [
            (Pattern::Distinct, Workload::AllHit),
            (Pattern::Duplicate, Workload::AllHit),
            (Pattern::Distinct, Workload::OneExpertMiss),
            (Pattern::Duplicate, Workload::OneExpertMiss),
        ] {
            let case = Case::new("paired", pattern, workload, overlap);
            let probe = Probe::new(&be, &kernels, case, 4096);
            probe.reset(&be);
            let seed = probe.snapshot(&be);
            // No timestamped sample until both arms pass full raw/weights/CPU LRU for this seed.
            let (expected, _) = probe.run(&be, &kernels, false, &seed, &probe.push, None);
            probe.assert_oracle(&be);
            let (actual, _) = probe.run(&be, &kernels, true, &seed, &probe.push, None);
            probe.assert_oracle(&be);
            assert_eq!(actual, expected);
            let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
            for round in 0..8 {
                for variant in [round % 2, 1 - round % 2] {
                    let (raw, us) = probe.run(
                        &be,
                        &kernels,
                        variant == 1,
                        &seed,
                        &probe.push,
                        Some(&queries),
                    );
                    assert_eq!(raw, expected, "every timed sample full raw and float bits");
                    if round >= 2 {
                        samples[variant].push(us.unwrap());
                    }
                }
            }
            let mean = |a: &[f64]| a.iter().sum::<f64>() / a.len() as f64;
            let paired: Vec<_> = samples[0]
                .iter()
                .zip(&samples[1])
                .map(|(a, b)| b - a)
                .collect();
            eprintln!("TWO_ROW_TOPK_PAIRED used={} rows=2 experts=512 slots_per_pool=4096 pattern={pattern:?} workload={workload:?} overlap={overlap} pairs={} reference={reference_name} candidate=external_wg128 full_router_owner=true reference_total_mean_us={:.6} candidate_total_mean_us={:.6} paired_delta_mean_us={:.6} reset_readback_outside_timestamps=true raw_fp_lru=pass module_only=true", probe_used(), paired.len(), mean(&samples[0]), mean(&samples[1]), mean(&paired));
        }
    }
}

#[test]
#[ignore = "central serial GPU only; external two-row legacy touch-order shader"]
fn gpu_owned_topk_two_row_legacy_touch_oracle() {
    let be = backend();
    let kernels = Kernels::new(&be);
    for overlap in [false, true] {
        for workload in [
            Workload::AllHit,
            Workload::OneExpertMiss,
            Workload::MixedFresh,
            Workload::MixedStale,
        ] {
            for (pattern, alias) in [
                (Pattern::Distinct, 0),
                (Pattern::Duplicate, 0),
                (Pattern::Distinct, 1),
                (Pattern::Distinct, 2),
            ] {
                let mut case = Case::new("legacy_touch", pattern, workload, overlap);
                case.bank_alias = alias;
                let mut probe = Probe::new(&be, &kernels, case, 256);
                super::legacy_touch_tests::legacy_stamps_for(&mut probe.pool, &probe.selected);
                probe.reset(&be);
                let seed = probe.snapshot(&be);
                probe.run(&be, &kernels, true, &seed, &probe.push, None);
                probe.assert_oracle(&be);
                eprintln!("TWO_ROW_LEGACY_TOUCH used={} overlap={overlap} workload={workload:?} pattern={pattern:?} alias={alias} exact_stamps_slots_jobs_weights=true fifo_live_order=exact", probe_used());
            }
        }
    }
}
