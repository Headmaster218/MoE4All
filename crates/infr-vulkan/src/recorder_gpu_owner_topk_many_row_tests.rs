//! Experimental metadata-only ManyRow qualification. Fake BDAs never reach payloads.
//! Oracle: actual moe_topk_sg (one WG/row), then the unchanged general WG64 owner.
use super::*;

#[path = "recorder_gpu_owner_recovery_candidate_tests.rs"]
mod recovery_candidate_tests;

type Raw = Vec<Vec<u32>>;
const GUARD: u32 = 0xa53c_91e7;
const SCALE: f32 = 1.375;

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Distinct,
    Duplicate,
    Saturated,
    Nonfinite,
    AllNan,
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    rows: usize,
    used: usize,
    experts: usize,
    gating: u32,
    pattern: Pattern,
    mixed: bool,
    overlap: bool,
    ordinal: usize,
    predicate: bool,
    run: bool,
    alias: u32,
}

impl Shape {
    fn new(rows: usize, used: usize) -> Self {
        Self {
            rows,
            used,
            experts: EXPERTS,
            gating: 1,
            pattern: Pattern::Distinct,
            mixed: false,
            overlap: false,
            ordinal: 0,
            predicate: true,
            run: true,
            alias: 0,
        }
    }
    fn count(self) -> usize {
        self.rows * self.used
    }
    fn predicate_word(self) -> usize {
        self.ordinal
            .checked_sub(1)
            .map_or(RESIDENT_SEED_WORD, |n| 5 + n * 2)
    }
}

struct Kernels {
    router: ComputeKernel,
    owner: [ComputeKernel; 2],
    many: [ComputeKernel; 2],
}

fn backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicit idle GPU required"));
    VulkanBackend::new_with(Arc::new(cfg)).unwrap()
}

impl Kernels {
    fn new(be: &VulkanBackend) -> Self {
        let properties = unsafe {
            be.shared
                .instance
                .get_physical_device_properties(be.shared.physical_device)
        };
        assert!(properties.limits.max_compute_work_group_invocations >= 256);
        assert!(properties.limits.max_compute_work_group_size[0] >= 256);
        assert!(properties.limits.max_compute_shared_memory_size >= 20 * 1024);
        assert!(be.shared.compute_subgroup_operations.contains(
            vk::SubgroupFeatureFlags::BASIC
                | vk::SubgroupFeatureFlags::ARITHMETIC
                | vk::SubgroupFeatureFlags::BALLOT
        ));
        assert!(be.shared.conditional_rendering.is_some());
        let router = be.kernel_sg(
            "many_row_actual_router",
            crate::gemm::moe_topk_sg_spv(),
            5,
            36,
            32,
        );
        let owner = [false, true].map(|overlap| {
            be.kernel(
                if overlap {
                    "many_row_reference_owner_overlap"
                } else {
                    "many_row_reference_owner"
                },
                crate::gemm::gpu_owned_prepare_fused_spv(overlap),
                if overlap { 9 } else { 8 },
                GPU_OWNED_FAST_PUSH_BYTES,
            )
        });
        let many = [false, true].map(|overlap| {
            be.kernel_sg(
                if overlap {
                    "gpu_owned_prepare_topk_fused_many_row_overlap"
                } else {
                    "gpu_owned_prepare_topk_fused_many_row"
                },
                crate::gemm::gpu_owned_prepare_topk_fused_many_row_spv(overlap),
                if overlap { 11 } else { 10 },
                GPU_OWNED_FAST_PUSH_BYTES,
                32,
            )
        });
        eprintln!(
            "MANY_ROW_TOPK kernels=embedded_many_row wg=256 subgroup=32 production_default=true reference=actual_router_plus_general_owner_wg64"
        );
        Self {
            router,
            owner,
            many,
        }
    }
}

const V94_MANY_ROW_DIR: &str = r"D:\AIinfr\infr\local-work\bin\infr\20261025-130100-gpu-owned-decode-c741\many-row-v94-reference";

fn external_many_words(dir: &std::path::Path) -> [Vec<u32>; 2] {
    [false, true].map(|overlap| {
        let file = if overlap {
            "gpu_owned_prepare_topk_fused_many_row_overlap.spv"
        } else {
            "gpu_owned_prepare_topk_fused_many_row.spv"
        };
        let path = dir.join(file);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("explicit ManyRow SPV {}: {e}", path.display()));
        // Diagnostic identity only; the frozen directory also retains its SHA256 manifest.
        let fingerprint = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
        });
        eprintln!(
            "MANY_ROW_SPV path={} bytes={} fnv1a64={fingerprint:016x}",
            path.display(),
            bytes.len()
        );
        ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap()
    })
}

fn comparison_kernels(be: &VulkanBackend) -> (Kernels, Kernels) {
    let mut candidate = Kernels::new(be);
    let reference_words = external_many_words(std::path::Path::new(V94_MANY_ROW_DIR));
    // Optional attribution pairs use the same two filenames, not a build.rs matrix.
    let external = std::env::var_os("MOE4ALL_MANY_ROW_CANDIDATE_SPV_DIR");
    let candidate_words = external.as_ref().map_or_else(
        || {
            [false, true].map(|overlap| {
                crate::gemm::gpu_owned_prepare_topk_fused_many_row_spv(overlap).to_vec()
            })
        },
        |dir| external_many_words(std::path::Path::new(dir)),
    );
    for overlap in 0..2 {
        assert_ne!(
            candidate_words[overlap], reference_words[overlap],
            "candidate must not silently alias frozen v94 ManyRow"
        );
    }
    if external.is_some() {
        candidate.many = [0, 1].map(|overlap| {
            be.kernel_sg(
                [
                    "many_row_external_candidate",
                    "many_row_external_candidate_overlap",
                ][overlap],
                &candidate_words[overlap],
                10 + overlap,
                GPU_OWNED_FAST_PUSH_BYTES,
                32,
            )
        });
    }
    let reference = Kernels {
        router: candidate.router,
        owner: candidate.owner,
        many: [0, 1].map(|overlap| {
            be.kernel_sg(
                ["many_row_v94_reference", "many_row_v94_reference_overlap"][overlap],
                &reference_words[overlap],
                10 + overlap,
                GPU_OWNED_FAST_PUSH_BYTES,
                32,
            )
        }),
    };
    eprintln!(
        "MANY_ROW_COMPARISON reference=frozen_v94_many_row_wg256 reference_dir={V94_MANY_ROW_DIR} candidate={} subgroup=32",
        if external.is_some() {
            "explicit_external"
        } else {
            "embedded_partial_and_two_pool_sg"
        }
    );
    (reference, candidate)
}

struct Probe {
    shape: Shape,
    buffers: Vec<Box<dyn Buffer>>,
    seed: Raw,
    push: [u32; 31],
    address_base: usize,
    slots: usize,
}

impl Probe {
    fn new(be: &VulkanBackend, shape: Shape, slots: usize) -> Self {
        assert!((3..=8).contains(&shape.rows) && (1..=16).contains(&shape.used));
        let count = shape.count();
        let total = 2 * slots;
        let address_base = SLOT_BASE + total;
        let heap_base = address_base + total;
        let roles = [0, 1, 2].map(|r| ((LAYERS - 1) * 3 + r) * EXPERTS);
        let clock = (1u64 << 32) - 10;
        let mut nodes = vec![[0u32; 4]; heap_base];
        nodes[0] = [4, 2, total as u32, 41];
        nodes[1] = [7, POOL_BASE as u32, SLOT_BASE as u32, address_base as u32];
        nodes[2] = [17, 23, pair(clock)[0], pair(clock)[1]];
        let mut sources = vec![[0; 4]; BLOCKS];
        let mut overlay = vec![[0, 0, 0, NONE]; BLOCKS];
        for (block, s) in sources.iter_mut().enumerate() {
            let a = pair(0x20_0000_0000 + block as u64 * 16384);
            *s = [
                a[0],
                a[1],
                SLOT_BYTES[pool_of(block)],
                pool_of(block) as u32,
            ];
        }
        let requested: HashSet<_> = roles
            .iter()
            .flat_map(|base| (0..count).map(move |id| base + id))
            .collect();
        for pool in 0..2 {
            let start = pool * slots;
            let mut residents: Vec<_> = requested
                .iter()
                .copied()
                .filter(|&block| {
                    pool_of(block) == pool && (!shape.mixed || block % EXPERTS % 2 == 0)
                })
                .collect();
            residents.sort_unstable();
            assert!(residents.len() <= slots);
            residents.extend(
                (0..BLOCKS)
                    .filter(|block| pool_of(*block) == pool && !requested.contains(block))
                    .take(slots - residents.len()),
            );
            assert_eq!(residents.len(), slots);
            let queue = nodes.len();
            nodes[POOL_BASE + 2 * pool] = [queue as u32, slots as u32, NONE, slots as u32];
            nodes[POOL_BASE + 2 * pool + 1] = [11, 7, 5, 0x1234];
            nodes.push([0, slots as u32, (2 * slots) as u32, 0]);
            for (local, block) in residents.into_iter().enumerate() {
                let slot = start + local;
                let live = pair(clock - total as u64 + slot as u64 + 1);
                let a = pair(0x10_0000_0000 + slot as u64 * 16384);
                nodes[SLOT_BASE + slot] = [live[0], live[1], block as u32, 1];
                nodes[address_base + slot] = [a[0], a[1], pool as u32, local as u32];
                overlay[block] = [a[0], a[1], 1, slot as u32];
                nodes.push([live[0], live[1], slot as u32, 0]);
            }
            nodes.resize(queue + 1 + 2 * slots, [0; 4]);
        }
        let mut seed = vec![Vec::new(); 11];
        seed[0] = (0..count as u32)
            .chain(std::iter::repeat_n(GUARD, 2 * shape.rows + 4))
            .collect();
        seed[1] = vec![GUARD; 3 * EXPERTS * 2 + 4];
        seed[2] = vec![0; RESIDENT_CONTROL_BYTES / 4];
        seed[2][0] = u32::from(shape.run);
        seed[2][1] = NONE;
        seed[2][shape.predicate_word()] = u32::from(shape.predicate);
        if shape.ordinal != 0 {
            seed[2][RESIDENT_SEED_WORD] = u32::from(!shape.predicate);
        }
        seed[3] = vec![GUARD; (shape.ordinal + 2) * RESIDENT_JOURNAL_WORDS];
        seed[4] = bytemuck::cast_slice(&sources).to_vec();
        seed[5] = bytemuck::cast_slice(&nodes).to_vec();
        seed[6] = bytemuck::cast_slice(&overlay).to_vec();
        seed[7] = vec![GUARD; (shape.ordinal + 2) * RESIDENT_JOURNAL_ID_WORDS * 3 * 8];
        seed[8] = vec![GUARD; (shape.ordinal + 2) * 4];
        for row in 0..shape.rows {
            for expert in 0..shape.experts {
                let start = if matches!(shape.pattern, Pattern::Duplicate) {
                    0
                } else {
                    row * shape.used
                };
                let mut value = if (start..start + shape.used).contains(&expert) {
                    2.0 - (expert - start) as f32 / 32.0
                } else {
                    -12.0 - expert as f32 / 1024.0
                };
                match shape.pattern {
                    Pattern::Saturated => {
                        value = if expert == 0 {
                            20.0
                        } else if expert == 1 {
                            30.0
                        } else if expert < shape.used {
                            2.0
                        } else {
                            -12.0
                        }
                    }
                    Pattern::Nonfinite => {
                        value = match expert {
                            0 | 31 => f32::INFINITY,
                            1 => f32::NEG_INFINITY,
                            2 => f32::from_bits(0x7fc0_0123 + row as u32),
                            3 => -1.0e31,
                            _ => value,
                        }
                    }
                    Pattern::AllNan => value = f32::from_bits(0x7fc0_0123 + row as u32),
                    _ => (),
                }
                seed[9].push(value.to_bits());
            }
        }
        seed[9].extend([GUARD; 4]);
        seed[10] = vec![GUARD; count + 4];
        let buffers = seed
            .iter()
            .map(|words| be.alloc_mapped_arena_bda(words.len() * 4).unwrap().0)
            .collect();
        let n = slots as u32;
        let base = gpu_copy_prepare_push(
            [
                count as u32,
                shape.experts as u32,
                0,
                EXPERTS as u32,
                (2 * EXPERTS) as u32,
                shape.ordinal as u32,
                shape.used as u32,
                1 << shape.used,
            ],
            true,
            [
                roles[0] as u32,
                if shape.alias == 0 {
                    roles[1] as u32
                } else {
                    roles[0] as u32 + shape.alias - 1
                },
                roles[2] as u32,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            8,
        );
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
        let mut push = gpu_owned_fast_push(&base, &geometry);
        push[13..16].copy_from_slice(&[
            SCALE.to_bits(),
            shape.gating,
            shape.predicate_word() as u32,
        ]);
        Self {
            shape,
            buffers,
            seed,
            push,
            address_base,
            slots,
        }
    }

    fn bindings(&self, fused: bool) -> Vec<vk::DescriptorBufferInfo> {
        let mut out: Vec<_> = self.buffers[..8]
            .iter()
            .map(|b| Recorder::vkb_exact(b.as_ref()))
            .collect();
        if self.shape.overlap {
            out.push(Recorder::vkb_exact(self.buffers[8].as_ref()));
        }
        if fused {
            out.extend([
                Recorder::vkb_exact(self.buffers[9].as_ref()),
                Recorder::vkb_exact(self.buffers[10].as_ref()),
            ]);
        }
        out
    }

    fn run(
        &self,
        be: &VulkanBackend,
        kernels: &Kernels,
        fused: bool,
        queries: Option<&Queries<'_>>,
    ) -> (Raw, Option<f64>) {
        for (b, words) in self.buffers.iter().zip(&self.seed) {
            upload(be, b.as_ref(), words);
        }
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        if let Some(q) = queries {
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
        if !fused {
            rec.begin_immutable_conditional_at(
                self.buffers[2].as_ref(),
                self.shape.predicate_word(),
            );
            let router = [
                self.shape.experts as u32,
                self.shape.used as u32,
                SCALE.to_bits(),
                self.shape.gating,
                1,
                0,
                0,
                0,
                0,
            ];
            rec.dispatch(
                kernels.router,
                &[
                    Recorder::vkb_exact(self.buffers[9].as_ref()),
                    Recorder::vkb_exact(self.buffers[9].as_ref()),
                    Recorder::vkb_exact(self.buffers[9].as_ref()),
                    Recorder::vkb_exact(self.buffers[0].as_ref()),
                    Recorder::vkb_exact(self.buffers[10].as_ref()),
                ],
                2,
                bytemuck::cast_slice(&router),
                self.shape.rows as u32,
            );
            rec.end_conditional();
        }
        let mut push = self.push;
        if !fused {
            push[13..18].fill(0);
        }
        let bindings = self.bindings(fused);
        rec.dispatch(
            if fused {
                kernels.many[self.shape.overlap as usize]
            } else {
                kernels.owner[self.shape.overlap as usize]
            },
            &bindings,
            bindings.len(),
            bytemuck::cast_slice(&push),
            1,
        );
        if let Some(q) = queries {
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
        let us = queries.map(|q| {
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
        (
            self.buffers.iter().map(|b| read(be, b.as_ref())).collect(),
            us,
        )
    }

    fn check_guards_and_valid(&self, raw: &Raw) {
        let s = self.shape;
        let count = s.count();
        assert_eq!(raw[9], self.seed[9], "all logits including canaries");
        assert_eq!(&raw[10][count..], &[GUARD; 4]);
        assert_eq!(&raw[0][count + s.rows * 2..], &[GUARD; 4]);
        assert_eq!(&raw[1][3 * EXPERTS * 2..], &[GUARD; 4]);
        assert_eq!(raw[4], self.seed[4], "source directory immutable");
        if !s.predicate {
            assert_eq!(&raw[0][..count], &self.seed[0][..count]);
            assert_eq!(raw[10], self.seed[10]);
        }
        let base = s.ordinal * RESIDENT_JOURNAL_WORDS;
        assert_eq!(&raw[3][..base], &self.seed[3][..base]);
        assert_eq!(
            &raw[3][base + RESIDENT_JOURNAL_WORDS..],
            &self.seed[3][base + RESIDENT_JOURNAL_WORDS..]
        );
        let jb = s.ordinal * RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
        let jobs = raw[3][base + RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
        assert!(jobs <= 3 * count && jobs <= 384);
        assert_eq!(&raw[7][..jb], &self.seed[7][..jb]);
        assert_eq!(&raw[7][jb + jobs * 8..], &self.seed[7][jb + jobs * 8..]);
        if !s.run {
            assert_eq!(raw[3][base + RESIDENT_JOURNAL_STATUS_WORD], 0);
            return;
        }
        assert!(matches!(raw[3][base + RESIDENT_JOURNAL_STATUS_WORD], 1 | 4));
        if !s.mixed && s.alias == 0 && matches!(s.pattern, Pattern::Distinct | Pattern::Duplicate) {
            assert_eq!(raw[3][base + RESIDENT_JOURNAL_STATUS_WORD], 1);
            assert_eq!(jobs, 0, "genuine all hit, not a hit-only compute budget");
        }
        let before = stamp(&self.seed[5][10..12]);
        assert_eq!(stamp(&raw[5][10..12]), before + (3 * count) as u64);
        for row in 0..s.rows {
            let mut hits = 0;
            for rank in 0..s.used {
                let id = raw[0][row * s.used + rank] as usize;
                let hit = (0..3)
                    .all(|role| self.seed[6][(self.push[9 + role] as usize + id) * 4 + 2] == 1);
                hits |= u32::from(hit) << rank;
                for role in 0..3 {
                    let block = self.push[9 + role] as usize + id;
                    let slot = raw[6][block * 4 + 3] as usize;
                    assert!(slot < 2 * self.slots);
                    assert_eq!(raw[5][(SLOT_BASE + slot) * 4 + 2], block as u32);
                    let tape = (role * EXPERTS + id) * 2;
                    let address = (self.address_base + slot) * 4;
                    assert_eq!(&raw[1][tape..tape + 2], &raw[5][address..address + 2]);
                }
            }
            assert_eq!(raw[3][base + RESIDENT_JOURNAL_MASK_WORD + row], hits);
            assert_eq!(
                raw[0][count + row],
                if s.overlap {
                    hits | (1 << s.used)
                } else {
                    ((1 << s.used) - 1) | (1 << s.used)
                }
            );
            assert_eq!(
                raw[0][count + s.rows + row],
                if s.overlap {
                    ((1 << s.used) - 1) ^ hits
                } else {
                    0
                }
            );
        }
    }
}

fn check(be: &VulkanBackend, kernels: &Kernels, shape: Shape) {
    let probe = Probe::new(be, shape, 256);
    let (expected, _) = probe.run(be, kernels, false, None);
    let (actual, _) = probe.run(be, kernels, true, None);
    assert_eq!(
        actual, expected,
        "full raw including exact weight bits, physical FIFO and guards: {shape:?}"
    );
    probe.check_guards_and_valid(&actual);
}

fn compare_probe(
    be: &VulkanBackend,
    reference: &Kernels,
    candidate: &Kernels,
    probe: &Probe,
) -> Raw {
    let (oracle, _) = probe.run(be, candidate, false, None);
    let (old, _) = probe.run(be, reference, true, None);
    let (new, _) = probe.run(be, candidate, true, None);
    assert_eq!(
        old, oracle,
        "frozen v94 vs independent oracle: {:?}",
        probe.shape
    );
    assert_eq!(
        new, old,
        "candidate vs frozen v94, every raw word: {:?}",
        probe.shape
    );
    new
}

#[test]
#[ignore = "central serial GPU only; bounded partial/two-pool candidate vs frozen v94 and independent oracle"]
fn gpu_owned_topk_many_row_partial_fifo_exact_3_to_8() {
    let be = backend();
    let (reference, candidate) = comparison_kernels(&be);
    for rows in 3..=8 {
        for used in [8, 10] {
            for overlap in [false, true] {
                for pattern in [Pattern::Distinct, Pattern::Duplicate] {
                    let p = Probe::new(
                        &be,
                        Shape {
                            mixed: true,
                            overlap,
                            pattern,
                            ..Shape::new(rows, used)
                        },
                        256,
                    );
                    assert!(3 * p.shape.count() <= 256);
                    let actual = compare_probe(&be, &reference, &candidate, &p);
                    p.check_guards_and_valid(&actual);
                }
            }
        }
    }
    // Counts 85/86 have no rectangular 3..8-row, used<=16 representation.
    // The nearest valid sides are 84 (252 role requests) and 88 (264).
    for (rows, used) in [(6, 14), (8, 11), (8, 16)] {
        for overlap in [false, true] {
            let p = Probe::new(
                &be,
                Shape {
                    mixed: true,
                    overlap,
                    ..Shape::new(rows, used)
                },
                256,
            );
            assert_eq!(3 * p.shape.count() <= 256, rows == 6);
            let actual = compare_probe(&be, &reference, &candidate, &p);
            p.check_guards_and_valid(&actual);
        }
    }
    for alias in [1, 2] {
        let p = Probe::new(
            &be,
            Shape {
                mixed: true,
                overlap: true,
                alias,
                ..Shape::new(4, 10)
            },
            256,
        );
        let actual = compare_probe(&be, &reference, &candidate, &p);
        p.check_guards_and_valid(&actual);
    }
}

#[test]
#[ignore = "central serial GPU only; bounded partial rejection, CPU-only jobs, duplicate/stale FIFO and free pops"]
fn gpu_owned_topk_many_row_partial_rejection_fifo_3_to_8() {
    let be = backend();
    let (reference, candidate) = comparison_kernels(&be);
    for rows in 3..=8 {
        for fault in 0..6 {
            let mut p = Probe::new(
                &be,
                Shape {
                    mixed: true,
                    overlap: true,
                    ..Shape::new(rows, 10)
                },
                256,
            );
            let gate = p.push[9] as usize;
            let expected_status = match fault {
                0 => {
                    p.seed[4][(gate + 1) * 4 + 2] += 1;
                    3
                }
                1 => {
                    let slot = p.seed[6][gate * 4 + 3] as usize;
                    let future = pair(stamp(&p.seed[5][10..12]) + 1);
                    p.seed[5][(SLOT_BASE + slot) * 4..(SLOT_BASE + slot) * 4 + 2]
                        .copy_from_slice(&future);
                    3
                }
                2 => {
                    p.seed[4][(gate + 1) * 4..(gate + 1) * 4 + 2].fill(0);
                    5
                }
                3 => {
                    p.push[3] = p.push[2] + 1; // Partial tape alias must decline hot conversion.
                    3
                }
                4 => {
                    let header = p.seed[5][POOL_BASE * 4] as usize;
                    p.seed[5][(header + 1) * 4 + 3] = 1; // Malformed before FIFO cutoff.
                    3
                }
                _ => {
                    let header = p.seed[5][POOL_BASE * 4] as usize;
                    // Pool0 has count live requested hits followed by eligible unrelated owners.
                    let at = (header + 1 + p.shape.count()) * 4;
                    let entry: [u32; 4] = p.seed[5][at..at + 4].try_into().unwrap();
                    p.seed[5][at + 4..at + 8].copy_from_slice(&entry);
                    3
                }
            };
            let actual = compare_probe(&be, &reference, &candidate, &p);
            assert_eq!(
                actual[3][RESIDENT_JOURNAL_STATUS_WORD], expected_status,
                "rows={rows} fault={fault}"
            );
            if expected_status == 3 {
                for binding in [1, 4, 5, 6, 7] {
                    assert_eq!(
                        actual[binding], p.seed[binding],
                        "reject before ownership mutation: rows={rows} fault={fault}"
                    );
                }
            }
        }
        let mut p = Probe::new(
            &be,
            Shape {
                mixed: true,
                overlap: true,
                ..Shape::new(rows, 10)
            },
            256,
        );
        for pool in 0..2 {
            let list = (POOL_BASE + 2 * pool) * 4;
            let header = p.seed[5][list] as usize;
            let hits = if pool == 0 {
                p.shape.count()
            } else {
                p.shape.count() / 2
            };
            let mut free = [0u32; 2];
            for (i, slot) in free.iter_mut().enumerate() {
                *slot = p.seed[5][(header + 1 + hits + i) * 4 + 2];
                let block = p.seed[5][(SLOT_BASE + *slot as usize) * 4 + 2] as usize;
                p.seed[6][block * 4..block * 4 + 4].copy_from_slice(&[0, 0, 0, NONE]);
            }
            p.seed[5][list + 1] -= 2;
            p.seed[5][list + 2] = free[0];
            for (i, slot) in free.into_iter().enumerate() {
                let at = (SLOT_BASE + slot as usize) * 4;
                p.seed[5][at..at + 4].copy_from_slice(&[
                    0,
                    if i == 0 { free[1] } else { NONE },
                    NONE,
                    0,
                ]);
            }
            let start = (header + 1) * 4;
            let tail = p.seed[5][header * 4 + 1] as usize;
            p.seed[5].copy_within(start..start + tail * 4, start + 4);
            p.seed[5][start..start + 4].copy_from_slice(&[1, 0, (pool * 256) as u32, 0]);
            p.seed[5][header * 4 + 1] += 1;
        }
        let actual = compare_probe(&be, &reference, &candidate, &p);
        p.check_guards_and_valid(&actual);
    }
}

#[test]
#[ignore = "central serial GPU only; ManyRow actual-router plus WG64-owner oracle"]
fn gpu_owned_topk_many_row_exact_3_to_8() {
    let be = backend();
    let kernels = Kernels::new(&be);
    for rows in 3..=8 {
        for used in [1, 8, 10, 16] {
            check(
                &be,
                &kernels,
                Shape {
                    gating: (rows % 2) as u32,
                    overlap: rows % 2 == 0,
                    ..Shape::new(rows, used)
                },
            );
        }
        check(
            &be,
            &kernels,
            Shape {
                pattern: Pattern::Duplicate,
                mixed: true,
                overlap: true,
                ..Shape::new(rows, 10)
            },
        );
        check(
            &be,
            &kernels,
            Shape {
                mixed: true,
                ..Shape::new(rows, 16)
            },
        );
    }
    for pattern in [Pattern::Saturated, Pattern::Nonfinite, Pattern::AllNan] {
        for gating in [0, 1] {
            for overlap in [false, true] {
                check(
                    &be,
                    &kernels,
                    Shape {
                        pattern,
                        gating,
                        overlap,
                        ..Shape::new(4, 10)
                    },
                );
            }
        }
    }
    for ordinal in [0, 1, 127] {
        for (predicate, run) in [(false, true), (true, false), (false, false)] {
            check(
                &be,
                &kernels,
                Shape {
                    ordinal,
                    predicate,
                    run,
                    overlap: true,
                    ..Shape::new(3, 8)
                },
            );
        }
    }
    for experts in [1, 17, 31, 32, 33, 127, 128, 129, 511] {
        check(
            &be,
            &kernels,
            Shape {
                experts,
                overlap: true,
                ..Shape::new(3, 16)
            },
        );
    }
    // Equal and partially overlapping banks retain exact cold classification.
    for alias in [1, 2] {
        check(
            &be,
            &kernels,
            Shape {
                alias,
                mixed: true,
                overlap: true,
                ..Shape::new(8, 16)
            },
        );
    }
}

#[test]
#[ignore = "central serial GPU only; malformed, CPU-only, stale FIFO and compaction parity"]
fn gpu_owned_topk_many_row_rejection_and_fifo() {
    let be = backend();
    let kernels = Kernels::new(&be);
    for overlap in [false, true] {
        for fault in 0..6 {
            let mut p = Probe::new(
                &be,
                Shape {
                    mixed: true,
                    overlap,
                    ..Shape::new(4, 16)
                },
                256,
            );
            let gate = p.push[9] as usize;
            let expected_status = match fault {
                0 => {
                    p.seed[5][0] = 99;
                    3
                }
                1 => {
                    p.push[3] = p.push[2] + 1;
                    3
                }
                2 => {
                    p.seed[4][gate * 4 + 2] += 1;
                    3
                }
                3 => {
                    let slot = p.seed[6][gate * 4 + 3] as usize;
                    p.seed[6][gate * 4 + 2] = 2;
                    p.seed[5][(SLOT_BASE + slot) * 4 + 3] = 2;
                    3
                }
                4 => {
                    // One CPU-only source among mapped siblings. No payload submitted.
                    p.seed[4][(gate + 1) * 4..(gate + 1) * 4 + 2].fill(0);
                    5
                }
                _ => {
                    p.shape.predicate = false;
                    p.seed[2][p.shape.predicate_word()] = 0;
                    p.seed[0][63] = EXPERTS as u32;
                    3
                }
            };
            let (expected, _) = p.run(&be, &kernels, false, None);
            let (actual, _) = p.run(&be, &kernels, true, None);
            assert_eq!(
                actual, expected,
                "fault={fault} overlap={overlap}, full raw"
            );
            assert_eq!(actual[3][RESIDENT_JOURNAL_STATUS_WORD], expected_status);
            if expected_status == 3 {
                for binding in [1, 4, 5, 6, 7] {
                    assert_eq!(
                        actual[binding], p.seed[binding],
                        "reject before directory/Tape/jobs mutation"
                    );
                }
            }
        }
        // Stale log followed by every live record. count128 all-hit append forces
        // compaction; mixed misses additionally exercise WG256 victim selection.
        for mixed in [false, true] {
            let mut p = Probe::new(
                &be,
                Shape {
                    mixed,
                    overlap,
                    ..Shape::new(8, 16)
                },
                256,
            );
            for pool in 0..2 {
                let header = p.seed[5][(POOL_BASE + 2 * pool) * 4] as usize;
                let start = (header + 1) * 4;
                let tail = p.seed[5][header * 4 + 1] as usize;
                p.seed[5].copy_within(start..start + tail * 4, start + 4);
                p.seed[5][start..start + 4].copy_from_slice(&[1, 0, (pool * 256) as u32, 0]);
                p.seed[5][header * 4 + 1] += 1;
            }
            let (expected, _) = p.run(&be, &kernels, false, None);
            let (actual, _) = p.run(&be, &kernels, true, None);
            assert_eq!(
                actual, expected,
                "stale compaction mixed={mixed} overlap={overlap}"
            );
            p.check_guards_and_valid(&actual);
        }
    }
}

#[test]
#[ignore = "central serial GPU only; exact first/last/unique for every prefix length 1..128"]
fn gpu_owned_topk_many_row_prefix_1_to_128() {
    let be = backend();
    let _capabilities = Kernels::new(&be);
    let kernel = be.kernel_sg(
        "many_row_prefix_probe",
        crate::gemm::gpu_owned_topk_many_row_classify_probe_spv(),
        10,
        GPU_OWNED_FAST_PUSH_BYTES,
        32,
    );
    let lengths = [132, 1, 1, 4, 4, 4, 4, 129 * 4, 1, 1];
    let buffers: Vec<_> = lengths
        .iter()
        .map(|n| be.alloc_mapped_arena_bda(n * 4).unwrap().0)
        .collect();
    for count in 1..=128 {
        for pattern in 0..3 {
            let mut seed: Raw = lengths.iter().map(|n| vec![GUARD; *n]).collect();
            let ids: Vec<_> = (0..count)
                .map(|i| match pattern {
                    0 => i as u32,
                    1 => (i % 7) as u32,
                    _ => ((i % 32).min(31 - i % 32) + (i / 64) * 16) as u32,
                })
                .collect();
            seed[0][..count].copy_from_slice(&ids);
            for (b, words) in buffers.iter().zip(&seed) {
                upload(&be, b.as_ref(), words);
            }
            let mut push = [0u32; 31];
            push[0] = count as u32;
            let bindings: Vec<_> = buffers
                .iter()
                .map(|b| Recorder::vkb_exact(b.as_ref()))
                .collect();
            let rec = be.recorder().unwrap();
            rec.owned_copy_barrier(true, false);
            rec.dispatch(
                kernel,
                &bindings,
                bindings.len(),
                bytemuck::cast_slice(&push),
                1,
            );
            rec.owned_copy_barrier(false, true);
            rec.finish().unwrap();
            let mut expected = seed;
            expected[3][0] = ids.iter().copied().collect::<HashSet<_>>().len() as u32;
            for i in 0..count {
                let first = ids.iter().position(|id| *id == ids[i]).unwrap();
                let last = ids.iter().rposition(|id| *id == ids[i]).unwrap();
                expected[7][4 * i..4 * i + 4].copy_from_slice(&[
                    first as u32,
                    last as u32,
                    u32::from(first == i) | (u32::from(last == i) << 1),
                    ids[i],
                ]);
            }
            let actual: Raw = buffers.iter().map(|b| read(&be, b.as_ref())).collect();
            assert_eq!(
                actual, expected,
                "count={count} pattern={pattern}, all prefix indices and guards"
            );
        }
    }
}

#[test]
#[ignore = "central serial GPU only; at most four rows, outer-only paired router+owner timestamps"]
fn gpu_owned_topk_many_row_paired_latency_probe() {
    let be = backend();
    let kernels = Kernels::new(&be);
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
    for rows in [3, 4] {
        for used in [8, 10] {
            for mixed in [false, true] {
                let shape = Shape {
                    mixed,
                    overlap: true,
                    ..Shape::new(rows, used)
                };
                let probe = Probe::new(&be, shape, 4096);
                let (expected, _) = probe.run(&be, &kernels, false, None);
                let (actual, _) = probe.run(&be, &kernels, true, None);
                assert_eq!(actual, expected);
                probe.check_guards_and_valid(&actual);
                let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                for round in 0..8 {
                    for variant in [round % 2, 1 - round % 2] {
                        let (raw, us) = probe.run(&be, &kernels, variant == 1, Some(&queries));
                        assert_eq!(raw, expected, "every timed sample full raw");
                        samples[variant].push(us.unwrap());
                    }
                }
                let mean = |a: &[f64]| a.iter().sum::<f64>() / a.len() as f64;
                let paired: Vec<_> = samples[0]
                    .iter()
                    .zip(&samples[1])
                    .map(|(a, b)| b - a)
                    .collect();
                eprintln!(
                    "MANY_ROW_TOPK_PAIRED rows={rows} used={used} experts=512 mixed={mixed} overlap=true slots_per_pool=4096 pairs=8 reference=actual_router_plus_general_owner_wg64 candidate=many_row_wg256 baseline_us={:.6} candidate_us={:.6} paired_delta_us={:.6} outer_only=true reset_readback_outside_timestamps=true exact_raw_fp_fifo=pass module_only=true",
                    mean(&samples[0]),
                    mean(&samples[1]),
                    mean(&paired)
                );
            }
        }
    }
}

#[test]
#[ignore = "central serial GPU only; frozen v94 ManyRow vs bounded candidate, eight outer timestamp pairs"]
fn gpu_owned_topk_many_row_v94_paired_latency_probe() {
    let be = backend();
    let (reference, candidate) = comparison_kernels(&be);
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
    for rows in [3, 4] {
        for used in [8, 10] {
            for mixed in [false, true] {
                let probe = Probe::new(
                    &be,
                    Shape {
                        mixed,
                        overlap: true,
                        ..Shape::new(rows, used)
                    },
                    4096,
                );
                let expected = compare_probe(&be, &reference, &candidate, &probe);
                probe.check_guards_and_valid(&expected);
                let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
                for round in 0..8 {
                    for variant in [round % 2, 1 - round % 2] {
                        let kernels = if variant == 0 { &reference } else { &candidate };
                        let (raw, us) = probe.run(&be, kernels, true, Some(&queries));
                        assert_eq!(
                            raw, expected,
                            "every fresh timed replay: rows={rows} used={used} mixed={mixed} variant={variant}"
                        );
                        samples[variant].push(us.unwrap());
                    }
                }
                let mean = |a: &[f64]| a.iter().sum::<f64>() / a.len() as f64;
                let paired: Vec<_> = samples[0]
                    .iter()
                    .zip(&samples[1])
                    .map(|(old, new)| new - old)
                    .collect();
                eprintln!(
                    "MANY_ROW_V94_PAIRED rows={rows} used={used} experts=512 mixed={mixed} overlap=true slots_per_pool=4096 pairs=8 reference=frozen_v94_many_row_wg256 candidate={} baseline_us={:.6} candidate_us={:.6} paired_delta_us={:.6} outer_only=true reset_readback_outside_timestamps=true exact_raw_fp_fifo=pass module_only=true",
                    if std::env::var_os("MOE4ALL_MANY_ROW_CANDIDATE_SPV_DIR").is_some() {
                        "explicit_external"
                    } else {
                        "embedded_partial_and_two_pool_sg"
                    },
                    mean(&samples[0]),
                    mean(&samples[1]),
                    mean(&paired)
                );
            }
        }
    }
}

#[test]
#[ignore = "central serial GPU only; same-kernel 4x10 true-route vs preserved-ID hot-owner attribution"]
fn gpu_owned_topk_many_row_hot_route_attribution_4x10() {
    let be = backend();
    let kernels = Kernels::new(&be);
    assert!(be.shared.submit_timestamp_valid_bits > 0);
    let shape = Shape {
        overlap: true,
        ..Shape::new(4, 10)
    };
    let mut probes = [
        Probe::new(&be, shape, 4096),
        Probe::new(
            &be,
            Shape {
                predicate: false,
                ..shape
            },
            4096,
        ),
    ];
    let count = shape.count();
    let input_word = shape.predicate_word();
    assert_eq!(count, 40);
    assert_eq!(input_word, RESIDENT_SEED_WORD);
    let (oracle, _) = probes[0].run(&be, &kernels, false, None);
    probes[0].check_guards_and_valid(&oracle);

    // Only routing differs. Both arms start with identical oracle IDs/weight bits;
    // every other input (including initial owner/log state) remains identical.
    for probe in &mut probes {
        probe.seed[0][..count].copy_from_slice(&oracle[0][..count]);
        probe.seed[10].clone_from(&oracle[10]);
    }
    let mut owner_seed = probes[1].seed.clone();
    assert_eq!(owner_seed[2][input_word], 0);
    assert_eq!(probes[0].seed[2][input_word], 1);
    owner_seed[2][input_word] = 1;
    assert_eq!(probes[0].seed, owner_seed);
    let mut expected = [oracle.clone(), oracle];
    assert_eq!(expected[0][2][input_word], 1);
    expected[1][2][input_word] = 0;
    // Seed has room for all touches: this probe cannot silently time compaction.
    for pool in 0..2 {
        let queue = probes[0].seed[5][(POOL_BASE + 2 * pool) * 4] as usize;
        assert_eq!(
            &probes[0].seed[5][queue * 4..queue * 4 + 4],
            &[0, 4096, 8192, 0]
        );
        assert_eq!(
            &expected[0][5][queue * 4..queue * 4 + 4],
            &[0, 4096 + [80u32, 40][pool], 8192, 0]
        );
    }
    let check = |variant: usize, raw: &Raw| {
        let probe = &probes[variant];
        assert_eq!(
            raw[2][input_word], probe.seed[2][input_word],
            "immutable input routing predicate must remain unchanged"
        );
        // Exact full raw comparison: the expected owner-only snapshot differs
        // solely in that one input bit. No output, weight or guard is masked.
        assert_eq!(
            raw, &expected[variant],
            "variant={variant}: exact raw/FP/FIFO/guards"
        );
        probe.check_guards_and_valid(raw);
    };
    // Warm BOTH paths; run() restores every seed outside the timestamp bracket.
    for _ in 0..2 {
        for variant in 0..2 {
            let (raw, _) = probes[variant].run(&be, &kernels, true, None);
            check(variant, &raw);
        }
    }
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
    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::new());
    let mut paired = Vec::new();
    for round in 0..8 {
        let mut us = [0.0; 2];
        for variant in [round % 2, 1 - round % 2] {
            let (raw, time) = probes[variant].run(&be, &kernels, true, Some(&queries));
            check(variant, &raw);
            us[variant] = time.unwrap();
            assert!(us[variant].is_finite() && us[variant] >= 0.0);
            samples[variant].push(us[variant]);
        }
        let delta = us[1] - us[0];
        paired.push(delta);
        eprintln!(
            "MANY_ROW_HOT_ROUTE_PAIR pair={round} order={} true_route_us={:.6} owner_only_us={:.6} owner_minus_route_us={delta:.6} exact_raw_fp_fifo_guards=pass",
            if round % 2 == 0 { "AB" } else { "BA" }, us[0], us[1],
        );
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let min = paired.iter().copied().fold(f64::INFINITY, f64::min);
    let max = paired.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(samples[0].len(), 8);
    assert_eq!(samples[1].len(), 8);
    eprintln!(
        "MANY_ROW_HOT_ROUTE_ATTRIBUTION rows=4 used=10 experts=512 gating=1 mixed=false overlap=true slots_per_pool=4096 kernel=embedded_many_row_wg256 subgroup=32 dispatches_per_sample=1 warmups_per_arm=2 pairs=8 timed_samples=16 true_route_mean_us={:.6} owner_only_mean_us={:.6} owner_minus_route_mean_us={:.6} paired_min_us={min:.6} paired_max_us={max:.6} outer_only=true reset_readback_outside_timestamps=true fixed_overhead_subtracted=false empty_bracket_measured=false immutable_input_unchanged=true exact_raw_fp_fifo_guards=pass module_only=true",
        mean(&samples[0]), mean(&samples[1]), mean(&paired),
    );
}
