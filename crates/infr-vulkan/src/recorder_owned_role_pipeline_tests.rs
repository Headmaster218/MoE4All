// Included in gpu_owned_tests to reuse its private FIFO/copy-publish fixture unchanged.
mod role_pipeline_tests {
    use super::*;
    use crate::recorder::owned_role_pipeline::{self as pipeline, contract, Geometry, Settings};
    use infr_core::DType;

    const GUARD: usize = 256;
    const POISON: u8 = 0xa5;
    const NE: usize = 256;
    const NFF: usize = 256;
    const USED: usize = 3;
    const EXPERTS: usize = 4;

    #[test]
    fn owned_role_pipeline_cpu_no_entry_fixed_suffix() {
        assert!(contract::phase2_suffix_matches(10, 7, 4, (14, 10)));
        assert!(contract::phase2_suffix_matches(10, 7, 2, (12, 9)));
        for actual in [(15, 10), (14, 11), (13, 10), (14, 9)] {
            assert!(
                !contract::phase2_suffix_matches(10, 7, 4, actual),
                "extra helper, transfer or missing join must decline: {actual:?}"
            );
        }
        assert!(!contract::phase2_suffix_matches(usize::MAX, 7, 4, (3, 10)));
        assert!(!contract::phase2_suffix_matches(10, usize::MAX, 4, (14, 2)));
        assert!(!contract::phase2_suffix_matches(10, 7, 3, (13, 9)));
    }

    fn geometry(rows: usize, shared: bool) -> Geometry {
        Geometry {
            ne: NE,
            nff: NFF,
            rows,
            used: USED,
            experts: EXPERTS,
            physical_used: USED + usize::from(shared),
            // Deliberately NOT the canonical block bases; Up and Gate alias only there.
            tape_bases: [0, 4, 8],
            block_bases: [0, 0, 4],
        }
    }

    #[test]
    fn owned_role_pipeline_cpu_scoped_controls_only() {
        assert!(contract::resolved_settings(None).is_none());
        assert!(contract::resolved_settings(Some(None)).is_none());
        let selected =
            contract::resolved_settings(Some(Some(pipeline::DEFAULT_SETTINGS))).unwrap();
        assert_eq!((selected.local, selected.nr), (256, 2));
        let selected =
            contract::resolved_settings(Some(Some(Settings { local: 128, nr: 2 }))).unwrap();
        assert_eq!(
            (selected.local, selected.nr),
            (128, 2),
            "explicit test scope selects its variant"
        );
        assert!(contract::resolved_settings(None).is_none(), "no global enable");
    }

    #[test]
    fn owned_role_pipeline_cpu_default_nr2_cache_labels() {
        let settings = pipeline::DEFAULT_SETTINGS;
        assert_eq!((settings.local, settings.nr), (256, 2));
        for dtype in [DType::Q4K, DType::Q5K, DType::Q6K, DType::Iq2S, DType::Iq3S] {
            let selected = contract::kernel_name(dtype, settings).unwrap();
            assert!(selected.ends_with("_256_nr2"));
            assert_ne!(
                Some(selected),
                contract::kernel_name(dtype, Settings { local: 256, nr: 1 })
            );
            assert!(contract::kernel_name(dtype, Settings { local: 128, nr: 2 })
                .unwrap()
                .ends_with("_128_nr2"));
        }
    }

    #[test]
    fn owned_role_pipeline_cpu_preserves_native_sg_selection() {
        let mut gemv = infr_core::config::Config::default().kernels.vulkan.gemv;
        gemv.no_id_sg = false;
        gemv.no_sg = false;
        gemv.sg_minout = 2048;
        gemv.sg_maxout = 8192;
        gemv.sg_nr = 2;
        assert!(contract::native_tree_compatible(
            DType::Q5K,
            2048,
            512,
            &gemv
        ));
        assert!(contract::native_tree_compatible(
            DType::Iq2S,
            2560,
            640,
            &gemv
        ));
        assert!(!contract::native_tree_compatible(
            DType::Q5K,
            512,
            2048,
            &gemv
        ));
        assert!(!contract::native_tree_compatible(
            DType::Iq3S,
            512,
            2048,
            &gemv
        ));
        gemv.sg_minout = 512;
        assert!(!contract::native_tree_compatible(
            DType::Q5K,
            2048,
            512,
            &gemv
        ));
        gemv.no_id_sg = true;
        assert!(contract::native_tree_compatible(
            DType::Q5K,
            2048,
            512,
            &gemv
        ));
    }

    #[test]
    fn owned_role_pipeline_cpu_policy_geometry_and_ranges() {
        assert!(!contract::policy_enabled(None));
        assert!(!contract::policy_enabled(Some(false)));
        assert!(contract::policy_enabled(Some(true)));
        assert!(
            contract::route_eligible(true, true, true, false, false, true),
            "prepared captures static coordinates"
        );
        assert!(contract::route_eligible(
            true, false, true, true, true, true
        ));
        assert!(!contract::route_eligible(
            true, false, true, false, false, true
        ));
        assert!(
            !contract::route_eligible(false, true, true, true, true, true),
            "MTP retains old route"
        );
        assert!(!contract::route_eligible(
            true, true, false, true, true, true
        ));
        assert!(!contract::route_eligible(
            true, true, true, true, true, false
        ));
        for rows in [1, 2, 4, 8] {
            for shared in [false, true] {
                assert!(contract::geometry_valid(geometry(rows, shared)));
            }
        }
        for rows in [0, 9, usize::MAX] {
            assert!(!contract::geometry_valid(geometry(rows, false)));
        }
        let g = geometry(2, true);
        for invalid in [
            Geometry {
                block_bases: [0, 0, 0],
                ..g
            },
            Geometry {
                block_bases: [0, 0, 3],
                ..g
            },
            Geometry {
                block_bases: [u32::MAX, 0, 4],
                ..g
            },
            Geometry {
                tape_bases: [u32::MAX, 4, 8],
                ..g
            },
            Geometry {
                physical_used: USED + 2,
                ..g
            },
            Geometry { ne: 255, ..g },
            Geometry { nff: 2048, ..g },
        ] {
            assert!(!contract::geometry_valid(invalid), "{invalid:?}");
        }
        let jobs = RESIDENT_JOURNAL_ID_WORDS * 3 * 32;
        for ordinal in [0, 1, RESIDENT_MAX_LAYERS - 1] {
            let layers = ordinal + 1;
            assert!(contract::sidecars_fit(
                ordinal as u32,
                layers * RESIDENT_JOURNAL_BYTES,
                layers * jobs,
                layers * 16
            ));
            assert!(!contract::sidecars_fit(
                ordinal as u32,
                layers * RESIDENT_JOURNAL_BYTES,
                layers * jobs,
                layers * 16 - 1
            ));
            assert!(!contract::sidecars_fit(
                ordinal as u32,
                layers * RESIDENT_JOURNAL_BYTES,
                layers * jobs - 1,
                layers * 16
            ));
            assert!(!contract::sidecars_fit(
                ordinal as u32,
                layers * RESIDENT_JOURNAL_BYTES - 1,
                layers * jobs,
                layers * 16
            ));
        }
        assert!(!contract::sidecars_fit(
            u32::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX
        ));
    }

    fn poison(be: &VulkanBackend, bytes: usize) -> Box<dyn Buffer> {
        let buffer = be.alloc_mapped_arena_bda(bytes).unwrap().0;
        be.upload(buffer.as_ref(), &vec![POISON; bytes]).unwrap();
        buffer
    }

    fn bytes(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<u8> {
        let mut result = vec![0; buffer.len_bytes()];
        be.download(buffer, &mut result).unwrap();
        result
    }

    struct Ffn {
        input: Box<dyn Buffer>,
        gate: Box<dyn Buffer>,
        up: Box<dyn Buffer>,
        activation: Box<dyn Buffer>,
        output: Box<dyn Buffer>,
        shared: Option<[Box<dyn Buffer>; 3]>,
        g: Geometry,
        dtype: DType,
        down: DType,
    }

    impl Ffn {
        fn new(be: &VulkanBackend, rows: usize, shared: bool, dtype: DType, down: DType) -> Self {
            let g = geometry(rows, shared);
            let scratch = || poison(be, rows * g.physical_used * NFF * 4 + GUARD);
            let input = poison(be, rows * NE * 4 + GUARD);
            let values: Vec<f32> = (0..rows * NE)
                .map(|i| ((i % 31) as f32 - 15.0) / 64.0)
                .collect();
            be.upload(input.as_ref(), bytemuck::cast_slice(&values))
                .unwrap();
            let shared = shared.then(|| {
                std::array::from_fn(|role| {
                    let weight =
                        infr_testkit::synth_weight(DType::Q8_0, NE * NFF, 301 + role as u32);
                    let buffer = poison(be, weight.len() + GUARD);
                    be.upload(buffer.as_ref(), &weight).unwrap();
                    buffer
                })
            });
            Self {
                input,
                gate: scratch(),
                up: scratch(),
                activation: scratch(),
                output: scratch(),
                shared,
                g,
                dtype,
                down,
            }
        }

        fn reset(&self, be: &VulkanBackend) {
            for buffer in [&self.gate, &self.up, &self.activation, &self.output] {
                be.upload(buffer.as_ref(), &vec![POISON; buffer.len_bytes()])
                    .unwrap();
            }
        }

        fn projection(&self, rec: &Recorder<'_>, f: &CopyPublishFixture, role: usize, bank: u32) {
            let output = if role == 0 { &self.gate } else { &self.up };
            if let Some(shared) = &self.shared {
                rec.linear_native_id_multi_paged_shared(
                    self.dtype,
                    0,
                    0,
                    f.fixture.buffers[1].as_ref(),
                    f.fixture.buffers[0].as_ref(),
                    USED,
                    self.g.tape_bases[role] as usize,
                    shared[role].as_ref(),
                    self.input.as_ref(),
                    false,
                    output.as_ref(),
                    NE,
                    NFF,
                    self.g.rows,
                    0,
                    bank,
                );
            } else {
                rec.linear_native_id_multi_paged(
                    self.dtype,
                    0,
                    0,
                    f.fixture.buffers[1].as_ref(),
                    f.fixture.buffers[0].as_ref(),
                    USED,
                    self.g.tape_bases[role] as usize,
                    self.input.as_ref(),
                    false,
                    output.as_ref(),
                    NE,
                    NFF,
                    self.g.rows,
                    0,
                    bank,
                );
            }
        }

        fn activate(&self, rec: &Recorder<'_>, f: &CopyPublishFixture, misses: bool) {
            rec.gpu_resident_silu_bank(
                self.gate.as_ref(),
                self.up.as_ref(),
                f.fixture.buffers[0].as_ref(),
                self.activation.as_ref(),
                [
                    self.g.rows as u32,
                    USED as u32,
                    self.g.physical_used as u32,
                    NFF as u32,
                ],
                misses,
            );
        }

        fn down(&self, rec: &Recorder<'_>, f: &CopyPublishFixture, bank: u32) {
            if let Some(shared) = &self.shared {
                rec.linear_native_id_multi_paged_shared(
                    self.down,
                    0,
                    0,
                    f.fixture.buffers[1].as_ref(),
                    f.fixture.buffers[0].as_ref(),
                    USED,
                    self.g.tape_bases[2] as usize,
                    shared[2].as_ref(),
                    self.activation.as_ref(),
                    true,
                    self.output.as_ref(),
                    NFF,
                    NE,
                    self.g.rows,
                    0,
                    bank,
                );
            } else {
                rec.linear_native_id_multi_paged(
                    self.down,
                    0,
                    0,
                    f.fixture.buffers[1].as_ref(),
                    f.fixture.buffers[0].as_ref(),
                    USED,
                    self.g.tape_bases[2] as usize,
                    self.activation.as_ref(),
                    true,
                    self.output.as_ref(),
                    NFF,
                    NE,
                    self.g.rows,
                    0,
                    bank,
                );
            }
        }

        fn guards(&self, be: &VulkanBackend) {
            for buffer in [
                &self.input,
                &self.gate,
                &self.up,
                &self.activation,
                &self.output,
            ] {
                let actual = bytes(be, buffer.as_ref());
                assert!(actual[actual.len() - GUARD..]
                    .iter()
                    .all(|&byte| byte == POISON));
            }
        }
    }

    fn reset_admission(be: &VulkanBackend, f: &mut CopyPublishFixture, ffn: &Ffn, status: u32) {
        let mut source = vec![POISON; 8 * f.stride];
        for block in 0..8 {
            let dtype = if block < 4 { ffn.dtype } else { ffn.down };
            let weight = infr_testkit::synth_weight(dtype, NE * NFF, 19 + block as u32);
            source[block * f.stride..block * f.stride + weight.len()].copy_from_slice(&weight);
        }
        f.fixture.source_words = source
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect();
        be.upload(f.fixture.source.as_ref(), &source).unwrap();
        let mut payload = vec![POISON; 6 * f.stride];
        let mut overlay = vec![[0, 0, 0, NONE]; 8];
        for slot in 0..6 {
            let node = f.fixture.nodes[SLOT_BASE + slot];
            let address = f.fixture.nodes[ADDRESS_BASE + slot];
            if node[3] == 1 {
                let block = node[2] as usize;
                overlay[block] = [address[0], address[1], 1, slot as u32];
                payload[slot * f.stride..(slot + 1) * f.stride]
                    .copy_from_slice(&source[block * f.stride..(block + 1) * f.stride]);
            }
        }
        be.upload(f.fixture.destination.as_ref(), &payload).unwrap();
        be.upload(
            f.fixture.buffers[5].as_ref(),
            bytemuck::cast_slice(&f.fixture.nodes),
        )
        .unwrap();
        be.upload(
            f.fixture.buffers[6].as_ref(),
            bytemuck::cast_slice(&overlay),
        )
        .unwrap();
        let routed = if status == 1 { [1, 1, 1] } else { [0, 1, 2] };
        let mut ids: Vec<u32> = (0..ffn.g.rows).flat_map(|_| routed).collect();
        ids.resize(ffn.g.rows * (USED + 2), 0);
        upload_words(be, f.fixture.buffers[0].as_ref(), &ids);
        let mut control = vec![0; f.fixture.buffers[2].len_bytes() / 4];
        control[0] = u32::from(status != 0);
        upload_words(be, f.fixture.buffers[2].as_ref(), &control);
        for index in [3, 7] {
            upload_words(
                be,
                f.fixture.buffers[index].as_ref(),
                &vec![0; f.fixture.buffers[index].len_bytes() / 4],
            );
        }
    }

    fn prepare(rec: &Recorder<'_>, f: &CopyPublishFixture, ffn: &Ffn, predicate: &dyn Buffer) {
        prepare_groups(rec, f, ffn, predicate, 2);
    }

    fn prepare_groups(
        rec: &Recorder<'_>,
        f: &CopyPublishFixture,
        ffn: &Ffn,
        predicate: &dyn Buffer,
        groups: u32,
    ) {
        let mut bindings: Vec<_> = f
            .fixture
            .buffers
            .iter()
            .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
            .collect();
        bindings.push(Recorder::vkb_exact(predicate));
        let push = gpu_copy_prepare_push(
            [
                (ffn.g.rows * USED) as u32,
                EXPERTS as u32,
                0,
                4,
                8,
                f.ordinal,
                USED as u32,
                if ffn.shared.is_some() { 1 << USED } else { 0 },
            ],
            true,
            [0, 0, 4, 0, 0, 0, 0, 0, 0],
            groups,
        );
        let kernel = rec.be.kernel(
            "gpu_owned_prepare_overlap",
            crate::gemm::gpu_owned_prepare_overlap_spv(),
            9,
            GPU_COPY_PREPARE_PUSH_BYTES,
        );
        rec.dispatch(kernel, &bindings, 9, bytemuck::cast_slice(&push), 1);
    }

    fn compare(actual: &[u8], expected: &[u8]) -> f64 {
        assert_eq!(actual.len(), expected.len());
        let mut maxerr = 0.0f64;
        for (a, b) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
            if a == b {
                continue;
            }
            assert_ne!(b, &[POISON; 4], "masked slots must stay exactly poisoned");
            let a = f32::from_le_bytes(a.try_into().unwrap()) as f64;
            let b = f32::from_le_bytes(b.try_into().unwrap()) as f64;
            let error = (a - b).abs();
            assert!(
                a.is_finite() && b.is_finite() && error <= 2e-6 + 1e-3 * b.abs(),
                "native FFN oracle: actual={a} expected={b} error={error}"
            );
            maxerr = maxerr.max(error);
        }
        maxerr
    }

    #[test]
    #[ignore = "parent serialized GPU; real FIFO prepare, split copy, EXT pause/resume, native FFN oracle"]
    fn gpu_owned_role_pipeline_status_rows_shared_contract() {
        let be = backend();
        let phase2_choices = pipeline::test_selection_scope(&be);
        let settings = pipeline::DEFAULT_SETTINGS;
        let _scope = pipeline::test_variant_scope(&be, Some(settings));
        for (dtype, down) in [(DType::Q5K, DType::Q5K), (DType::Iq2S, DType::Iq4Nl)] {
            let weight = infr_testkit::synth_weight(dtype, NE * NFF, 7);
            let down_weight = infr_testkit::synth_weight(down, NE * NFF, 7);
            let stride = weight.len().max(down_weight.len()) + GUARD;
            for rows in [1, 2, 4, 8] {
                for shared in [false, true] {
                    for status in [0, 1, 4, 5] {
                        let mut f = CopyPublishFixture::new(
                            &be,
                            rows,
                            if status == 0 { 4 } else { status },
                            false,
                            [0, 0, 4],
                            stride,
                            true,
                        );
                        let ffn = Ffn::new(&be, rows, shared, dtype, down);
                        reset_admission(&be, &mut f, &ffn, status);
                        let initial = f.snapshot(&be);
                        let predicate = be
                            .alloc_mapped_predicate((f.ordinal as usize + 1) * 16)
                            .unwrap();
                        let stale = vec![99; predicate.len_bytes() / 4];
                        let before_publish = poison(&be, f.fixture.buffers[5].len_bytes());
                        let before_phase2 = poison(&be, f.fixture.buffers[5].len_bytes());
                        let shared_before = ffn.shared.as_ref().map(|weights| {
                            weights.each_ref().map(|weight| bytes(&be, weight.as_ref()))
                        });
                        let mut oracle = Vec::new();
                        let mut oracle_activation = Vec::new();
                        let mut oracle_state = Vec::new();
                        for candidate in [false, true] {
                            f.restore(&be, &initial);
                            ffn.reset(&be);
                            upload_words(&be, predicate.as_ref(), &stale);
                            let rec = be.recorder().unwrap();
                            let dispatch = contract::fixture_dispatch(
                                &rec,
                                f.fixture.buffers[3].as_ref(),
                                f.fixture.buffers[7].as_ref(),
                                predicate.as_ref(),
                                f.fixture.buffers[0].as_ref(),
                                f.fixture.buffers[1].as_ref(),
                                ffn.input.as_ref(),
                                ffn.activation.as_ref(),
                                ffn.g,
                                f.ordinal,
                                dtype,
                                false,
                                2,
                                settings,
                            )
                            .expect("qualified EXT device");
                            rec.seed_barrier();
                            prepare(&rec, &f, &ffn, predicate.as_ref());
                            if !candidate {
                                f.record(&rec, false, false, 2);
                            }
                            rec.begin_immutable_conditional_at(
                                f.fixture.buffers[2].as_ref(),
                                6 + f.ordinal as usize * 2,
                            );
                            if candidate {
                                let independent = rec.independent_dispatch_scope();
                                contract::phase1(&dispatch, &rec);
                                for role in [0, 1] {
                                    ffn.projection(&rec, &f, role, 1);
                                }
                                drop(independent);
                            } else {
                                for role in [0, 1] {
                                    ffn.projection(&rec, &f, role, 1);
                                }
                            }
                            ffn.activate(&rec, &f, false);
                            ffn.down(&rec, &f, 1);
                            rec.end_conditional();
                            if candidate {
                                rec.copy(
                                    f.fixture.buffers[5].as_ref(),
                                    0,
                                    before_phase2.as_ref(),
                                    0,
                                    f.fixture.buffers[5].len_bytes(),
                                );
                                let before_choices = phase2_choices.phase2_counts();
                                contract::phase2_finish(
                                    &dispatch,
                                    &rec,
                                    f.fixture.buffers[5].as_ref(),
                                    f.fixture.buffers[6].as_ref(),
                                    before_publish.as_ref(),
                                );
                                assert_eq!(
                                    phase2_choices.phase2_counts(),
                                    [before_choices[0] + 1, 0],
                                    "mid-phase transfer snapshots have no pure-compute Plan proof"
                                );
                            } else {
                                rec.begin_immutable_conditional_at(
                                    predicate.as_ref(),
                                    f.ordinal as usize * 4,
                                );
                                for role in [0, 1] {
                                    ffn.projection(&rec, &f, role, 2);
                                }
                                ffn.activate(&rec, &f, true);
                            }
                            ffn.down(&rec, &f, 2);
                            rec.end_conditional();
                            assert!(rec.immutable_condition.get().is_none());
                            rec.resident_stage_handoff_barrier();
                            rec.publish_host_reads();
                            rec.finish().unwrap();
                            ffn.guards(&be);
                            let state = f.snapshot(&be);
                            let actual = bytes(&be, ffn.output.as_ref());
                            let activation = bytes(&be, ffn.activation.as_ref());
                            let pred = read_words(&be, predicate.as_ref());
                            let journal = &state[3];
                            let base = f.ordinal as usize * RESIDENT_JOURNAL_WORDS;
                            assert_eq!(journal[base + RESIDENT_JOURNAL_STATUS_WORD], status);
                            let count = journal[base + RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
                            assert_eq!(
                                pred[f.ordinal as usize * 4],
                                if status == 4 { count as u32 } else { 0 }
                            );
                            assert!(pred[..f.ordinal as usize * 4].iter().all(|&v| v == 99));
                            assert_eq!(state[9], initial[9], "source and source guards exact");
                            let destination = bytes(&be, f.fixture.destination.as_ref());
                            for slot in 0..6 {
                                let used = if slot < 3 {
                                    weight.len()
                                } else {
                                    down_weight.len()
                                };
                                assert!(
                                    destination[slot * stride + used..(slot + 1) * stride]
                                        .iter()
                                        .all(|&byte| byte == POISON),
                                    "copy guard slot={slot}"
                                );
                            }
                            assert_fifo(&state[5]);
                            if !candidate {
                                if status != 0 {
                                    assert!(
                                        actual[..actual.len() - GUARD].chunks_exact(4).any(
                                            |word| {
                                                let value =
                                                    f32::from_le_bytes(word.try_into().unwrap());
                                                value.is_finite() && value.abs() > 1e-7
                                            }
                                        ),
                                        "nonzero native FFN oracle"
                                    );
                                }
                                oracle = actual;
                                oracle_activation = activation;
                                oracle_state = state;
                                continue;
                            }
                            let maxerr = compare(&actual, &oracle);
                            let acterr = compare(&activation, &oracle_activation);
                            assert_eq!(
                                state, oracle_state,
                                "payload guards, journals, jobs, owner FIFO exact"
                            );
                            for snapshot in [before_phase2.as_ref(), before_publish.as_ref()] {
                                let nodes = read_words(&be, snapshot);
                                for j in 0..count {
                                    let job =
                                        f.ordinal as usize * RESIDENT_JOURNAL_ID_WORDS * 3 * 8
                                            + j * 8;
                                    let slot = state[7][job + 6] as usize;
                                    assert_eq!(
                                        nodes[(SLOT_BASE + slot) * 4 + 3],
                                        2,
                                        "no early READY"
                                    );
                                }
                            }
                            if let (Some(weights), Some(expected)) = (&ffn.shared, &shared_before) {
                                for (weight, reference) in weights.iter().zip(expected) {
                                    assert_eq!(
                                        bytes(&be, weight.as_ref()),
                                        *reference,
                                        "shared exact"
                                    );
                                }
                            }
                            println!("OWNED_ROLE_PIPELINE dtype={dtype:?} down={down:?} rows={rows} shared={shared} status={status} jobs={count} local={} nr={} maxerr={maxerr} act_maxerr={acterr} exact_fifo_weights_guards=true early_ready=false",
                                settings.local, settings.nr);
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "parent serialized GPU; Down-only and CPU-GU phase boundaries, no repeated copy or early READY"]
    fn gpu_owned_role_pipeline_down_early_phase_boundary() {
        let be = backend();
        let weight = infr_testkit::synth_weight(DType::Q5K, NE * NFF, 7);
        let stride = weight.len() + GUARD;
        for cpu_gu in [false, true] {
            for wide in [false, true] {
                let status = if cpu_gu { 5 } else { 4 };
                let mut f = CopyPublishFixture::new(&be, 2, status, false, [0, 0, 4], stride, true);
                let ffn = Ffn::new(&be, 2, true, DType::Q5K, DType::Q5K);
                if !cpu_gu {
                    // Preserve the FIFO entry/stamp, but seed resident Gate/Up expert 0.
                    f.fixture.nodes[SLOT_BASE][2] = 0;
                }
                reset_admission(&be, &mut f, &ffn, status);
                f.fixture.reset_inputs(&be, &[0, 1, 0, 0, 1, 0]);
                if cpu_gu {
                    let mut sources = read_words(&be, f.fixture.buffers[4].as_ref());
                    sources[..2].fill(0);
                    upload_words(&be, f.fixture.buffers[4].as_ref(), &sources);
                }
                let initial = f.snapshot(&be);
                let predicate = be.alloc_mapped_predicate(32).unwrap();
                upload_words(&be, predicate.as_ref(), &[99; 8]);
                let before_publish = poison(&be, f.fixture.buffers[5].len_bytes());
                let shared_before = ffn
                    .shared
                    .as_ref()
                    .unwrap()
                    .each_ref()
                    .map(|b| bytes(&be, b.as_ref()));
                let rec = be.recorder().unwrap();
                let dispatch = contract::fixture_dispatch(
                    &rec,
                    f.fixture.buffers[3].as_ref(),
                    f.fixture.buffers[7].as_ref(),
                    predicate.as_ref(),
                    f.fixture.buffers[0].as_ref(),
                    f.fixture.buffers[1].as_ref(),
                    ffn.input.as_ref(),
                    ffn.activation.as_ref(),
                    ffn.g,
                    f.ordinal,
                    ffn.dtype,
                    wide,
                    8,
                    pipeline::DEFAULT_SETTINGS,
                )
                .expect("qualified EXT device and copy width");
                rec.seed_barrier();
                prepare_groups(&rec, &f, &ffn, predicate.as_ref(), 8);
                rec.begin_immutable_conditional_at(
                    f.fixture.buffers[2].as_ref(),
                    6 + f.ordinal as usize * 2,
                );
                let independent = rec.independent_dispatch_scope();
                contract::phase1(&dispatch, &rec);
                for role in [0, 1] {
                    ffn.projection(&rec, &f, role, 1);
                }
                drop(independent);
                ffn.activate(&rec, &f, false);
                ffn.down(&rec, &f, 1);
                rec.end_conditional();
                rec.resident_stage_handoff_barrier();
                rec.publish_host_reads();
                rec.finish().unwrap();

                // Deliberately drain at this boundary: this is a copy/protocol test,
                // not a continuous-frame timing or prepared-replay lifetime test.
                let phase1 = f.snapshot(&be);
                let journal = f.ordinal as usize * RESIDENT_JOURNAL_WORDS;
                assert_eq!(phase1[3][journal + RESIDENT_JOURNAL_STATUS_WORD], status);
                let count = phase1[3][journal + RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
                assert_eq!(count, if cpu_gu { 2 } else { 1 });
                let start = f.ordinal as usize * RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
                let jobs: Vec<_> = phase1[7][start..start + count * 8]
                    .chunks_exact(8)
                    .map(|job| <[u32; 8]>::try_from(job).unwrap())
                    .collect();
                let mut blocks: Vec<_> = jobs.iter().map(|job| job[5]).collect();
                blocks.sort_unstable();
                assert_eq!(blocks, if cpu_gu { vec![0, 4] } else { vec![4] });
                let mut expected_payload = initial[8].clone();
                for job in &jobs {
                    let mapped = job[0] != 0 || job[1] != 0;
                    assert_eq!(mapped, job[5] == 4, "only Down is mapped");
                    assert_eq!(job[4] as usize, stride);
                    let slot = job[6] as usize;
                    assert_eq!(
                        phase1[5][(SLOT_BASE + slot) * 4 + 3],
                        2,
                        "phase1: no early READY"
                    );
                    assert_eq!(
                        phase1[6][job[5] as usize * 4 + 2],
                        2,
                        "phase1: overlay still COPYING"
                    );
                    if mapped {
                        expected_payload[slot * stride / 4..(slot + 1) * stride / 4]
                            .copy_from_slice(&initial[9][4 * stride / 4..5 * stride / 4]);
                    }
                }
                assert_eq!(
                    phase1[8], expected_payload,
                    "phase1 copied Down, skipped CPU GU, guards exact"
                );
                assert_eq!(phase1[9], initial[9], "phase1 source exact");
                let pred = read_words(&be, predicate.as_ref());
                assert_eq!(&pred[..4], &[99; 4], "other ordinal untouched");
                assert_eq!(&pred[4..], &[if cpu_gu { 0 } else { 1 }, 8, 1, 1]);
                assert_eq!(
                    &phase1[0][2 * USED..2 * USED + 2],
                    &[10, 10],
                    "GPU H masks include shared"
                );
                assert_eq!(
                    &phase1[0][2 * USED + 2..],
                    &[5, 5],
                    "GPU M masks exclude shared; W0 alone suppresses status5 GU"
                );
                let phase1_activation = bytes(&be, ffn.activation.as_ref());
                assert_fifo(&phase1[5]);

                // Change bytes, not addresses/jobs: a second Down copy can no longer
                // hide behind idempotent final payload equality. Source guards stay exact.
                let mut changed_source = initial[9].clone();
                for word in &mut changed_source[4 * stride / 4..(4 * stride + weight.len()) / 4] {
                    *word ^= 0x5a5a_5a5a;
                }
                assert_ne!(changed_source, initial[9]);
                upload_words(&be, f.fixture.source.as_ref(), &changed_source);
                let rec = be.recorder().unwrap();
                rec.seed_barrier();
                contract::phase2_finish(
                    &dispatch,
                    &rec,
                    f.fixture.buffers[5].as_ref(),
                    f.fixture.buffers[6].as_ref(),
                    before_publish.as_ref(),
                );
                rec.end_conditional();
                rec.resident_stage_handoff_barrier();
                rec.publish_host_reads();
                rec.finish().unwrap();
                let phase2 = f.snapshot(&be);
                assert_eq!(
                    phase2[8], expected_payload,
                    "phase2 must NOT copy Down again"
                );
                assert_eq!(phase2[9], changed_source, "phase2 does not write source");
                assert_eq!(
                    read_words(&be, before_publish.as_ref()),
                    phase1[5],
                    "phase2 join before the only READY publish"
                );
                let mut expected_nodes = phase1[5].clone();
                let mut expected_overlay = phase1[6].clone();
                for job in &jobs {
                    if job[0] != 0 || job[1] != 0 {
                        expected_nodes[(SLOT_BASE + job[6] as usize) * 4 + 3] = 1;
                        expected_overlay[job[5] as usize * 4 + 2] = 1;
                    }
                }
                assert_eq!(phase2[5], expected_nodes, "only mapped Down becomes READY");
                assert_eq!(phase2[6], expected_overlay);
                for binding in [0, 1, 2, 3, 4, 7] {
                    assert_eq!(
                        phase2[binding], phase1[binding],
                        "immutable protocol binding={binding}"
                    );
                }
                assert_eq!(
                    read_words(&be, predicate.as_ref()),
                    pred,
                    "W0/indirect args immutable"
                );
                if cpu_gu {
                    assert_eq!(
                        bytes(&be, ffn.activation.as_ref()),
                        phase1_activation,
                        "status5 must not execute GU even though M masks are nonzero"
                    );
                }
                for (buffer, reference) in ffn.shared.as_ref().unwrap().iter().zip(&shared_before) {
                    assert_eq!(bytes(&be, buffer.as_ref()), *reference, "shared exact");
                }
                ffn.guards(&be);
                assert_fifo(&phase2[5]);
                println!("OWNED_ROLE_DOWN_EARLY status={status} rows=2 shared=true copy_local={} groups=8 jobs={count} phase1_down=true phase2_recopy=false early_ready=false cpu_gu_pending={cpu_gu} exact_payload_guards_fifo=true",
                    if wide { 1024 } else { 256 });
            }
        }
    }

    #[test]
    #[ignore = "parent serialized GPU; EXT candidate rejects indirect emulation before recording"]
    fn gpu_owned_role_pipeline_rejects_indirect_and_alias_before_recording() {
        let be = backend();
        let f = CopyPublishFixture::new(&be, 1, 4, false, [0, 0, 4], 65536, true);
        let ffn = Ffn::new(&be, 1, false, DType::Q5K, DType::Q5K);
        let predicate = be.alloc_mapped_predicate(32).unwrap();
        let rec = be.recorder().unwrap();
        let settings = pipeline::DEFAULT_SETTINGS;
        let attempt = |input: &dyn Buffer, activation: &dyn Buffer| {
            contract::fixture_dispatch(
                &rec,
                f.fixture.buffers[3].as_ref(),
                f.fixture.buffers[7].as_ref(),
                predicate.as_ref(),
                f.fixture.buffers[0].as_ref(),
                f.fixture.buffers[1].as_ref(),
                input,
                activation,
                ffn.g,
                f.ordinal,
                ffn.dtype,
                false,
                2,
                settings,
            )
            .is_some()
        };
        let aliased = ffn.activation.as_ref();
        assert!(
            aliased.len_bytes() >= ffn.g.rows * ffn.g.ne * 4,
            "alias input capacity valid"
        );
        assert!(
            aliased.len_bytes() >= ffn.g.rows * ffn.g.physical_used * ffn.g.nff * 4,
            "alias activation capacity valid"
        );
        let before = rec.dispatches();
        assert!(
            attempt(ffn.input.as_ref(), aliased),
            "non-alias control must qualify"
        );
        assert!(
            !attempt(aliased, aliased),
            "reject capacity-valid scratch alias"
        );
        assert_eq!(rec.dispatches(), before);
        rec.use_resident_indirect(Arc::new(ResidentIndirect::new(&be).unwrap()));
        assert!(!attempt(ffn.input.as_ref(), ffn.activation.as_ref()));
        assert_eq!(
            rec.dispatches(),
            before,
            "fallback must not partially record"
        );
        rec.finish().unwrap();
    }
}
