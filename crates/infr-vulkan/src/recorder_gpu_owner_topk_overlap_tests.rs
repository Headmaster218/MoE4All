//! Raw differential: independent original router + unchanged overlap owner vs 11-binding fusion.
//! The inherited fixture has synthetic BDAs: never record payload work in this module.
use super::*;

fn kernels(be: &VulkanBackend) -> TopkKernels {
    TopkKernels {
        router: be.kernel_sg(
            "topk_copy_overlap_original_router",
            crate::gemm::moe_topk_sigmoid_wave32_spv(),
            5,
            36,
            32,
        ),
        owner: be.kernel_sg(
            "topk_copy_overlap_original_owner",
            crate::gemm::gpu_owned_prepare_fused_spv(true),
            9,
            GPU_OWNED_FAST_PUSH_BYTES,
            32,
        ),
        fused: be.kernel_sg(
            "gpu_owned_prepare_topk_fused_overlap",
            crate::gemm::gpu_owned_prepare_topk_fused_overlap_spv(),
            11,
            GPU_OWNED_FAST_PUSH_BYTES,
            32,
        ),
    }
}

fn raw(
    be: &VulkanBackend,
    kernels: &TopkKernels,
    probe: &TopkProbe,
    fused: bool,
    mutate: impl FnOnce(&mut [Vec<u32>; 9], &mut [u32; 31]),
) -> [Vec<u32>; 11] {
    probe.reset(be);
    let ordinal = probe.fixture.push[5] as usize;
    let args = be.alloc_mapped_predicate((ordinal + 2) * 16).unwrap();
    upload(be, args.as_ref(), &vec![99u32; (ordinal + 2) * 4]);
    let mut state: [Vec<u32>; 9] = std::array::from_fn(|i| {
        read(
            be,
            if i == 8 {
                args.as_ref()
            } else {
                probe.fixture.buffers[i].as_ref()
            },
        )
    });
    let mut push = probe.fused_push;
    push[7] = 1 << probe.fixture.push[6];
    mutate(&mut state, &mut push);
    for (i, words) in state.iter().enumerate() {
        upload(
            be,
            if i == 8 {
                args.as_ref()
            } else {
                probe.fixture.buffers[i].as_ref()
            },
            words,
        );
    }
    let mut bindings: Vec<_> = probe
        .fixture
        .buffers
        .iter()
        .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
        .collect();
    bindings.push(Recorder::vkb_exact(args.as_ref()));
    let rec = be.recorder().unwrap();
    rec.owned_copy_barrier(true, false);
    if fused {
        bindings.extend([
            Recorder::vkb_exact(probe.logits.as_ref()),
            Recorder::vkb_exact(probe.weights.as_ref()),
        ]);
        rec.dispatch(kernels.fused, &bindings, 11, bytemuck::cast_slice(&push), 1);
    } else {
        rec.begin_immutable_conditional_at(probe.fixture.buffers[2].as_ref(), probe.predicate);
        probe.record_router(&rec, kernels);
        rec.end_conditional();
        rec.dispatch(kernels.owner, &bindings, 9, bytemuck::cast_slice(&push), 1);
    }
    rec.owned_copy_barrier(false, true);
    rec.finish().unwrap();
    std::array::from_fn(|i| match i {
        8 => read(be, args.as_ref()),
        9 => read(be, probe.weights.as_ref()),
        10 => read(be, probe.logits.as_ref()),
        _ => read(be, probe.fixture.buffers[i].as_ref()),
    })
}

fn compare(
    be: &VulkanBackend,
    kernels: &TopkKernels,
    probe: &TopkProbe,
    mutate: impl Fn(&mut [Vec<u32>; 9], &mut [u32; 31]),
) -> [Vec<u32>; 11] {
    let old = raw(be, kernels, probe, false, &mutate);
    let new = raw(be, kernels, probe, true, &mutate);
    assert_eq!(old, new, "all9 raw buffers + float bits/logits must match");
    let ordinal = probe.fixture.push[5] as usize;
    for other in 0..ordinal + 2 {
        if other != ordinal {
            assert_eq!(&new[8][other * 4..other * 4 + 4], &[99; 4]);
        }
    }
    new
}

#[test]
#[ignore = "central serial GPU runner; raw TopK/copy-overlap differential, no BDA payload"]
fn gpu_owned_topk_copy_overlap_raw_exact() {
    let be = backend();
    let kernels = kernels(&be);
    for used in [8, 10, 16] {
        let experts = if used == 8 { 256 } else { 512 };
        for workload in [Workload::AllHit, Workload::OneExpertMiss] {
            let probe = TopkProbe::new_with_experts(
                &be,
                &kernels,
                Case::normal("copy_overlap", used, workload),
                128,
                experts,
            );
            for cpu in [false, true] {
                let result = compare(&be, &kernels, &probe, |state, _| {
                    if cpu {
                        let block = probe.owner_push[11] as usize;
                        state[4][block * 4..block * 4 + 2].fill(0);
                    }
                });
                let status = result[3][RESIDENT_JOURNAL_STATUS_WORD];
                let jobs = result[3][RESIDENT_JOURNAL_JOB_COUNT_WORD];
                let wanted = if matches!(workload, Workload::AllHit) {
                    1
                } else if cpu {
                    5
                } else {
                    4
                };
                assert_eq!(status, wanted);
                assert_eq!(
                    &result[8][..4],
                    &[
                        if status == 4 { jobs } else { 0 },
                        if jobs != 0 { 2 } else { 0 },
                        1,
                        1
                    ]
                );
                let hit_mask = result[3][RESIDENT_JOURNAL_MASK_WORD];
                assert_eq!(result[0][used], hit_mask | (1 << used));
                assert_eq!(result[0][used + 1], ((1 << used) - 1) ^ hit_mask);
                assert_eq!(result[9], probe.expected_weights);
                assert_eq!(result[10], probe.logits_bits);
                if !cpu {
                    // Reuse the independent CPU exact-LRU/physical-owner oracle after
                    // adapting only serial's masks to overlap's disjoint H/M contract.
                    let mut expected = probe.fixture.expected.clone();
                    expected[0][used] = hit_mask | (1 << used);
                    expected[0][used + 1] = ((1 << used) - 1) ^ hit_mask;
                    for i in [1, 4, 6, 7] {
                        assert_eq!(result[i], expected[i], "CPU oracle buffer {i}");
                    }
                    for slot in 0..probe.fixture.slots_per_pool * 2 {
                        let base = (SLOT_BASE + slot) * 4;
                        assert_eq!(&result[5][base..base + 4], &expected[5][base..base + 4]);
                    }
                }
            }
        }
    }
    for kind in 0..9 {
        let mut case = Case::normal("copy_overlap_edge", 10, Workload::OneExpertMiss);
        match kind {
            0 => case.alias = true,
            1 => case.ordinal = RESIDENT_MAX_LAYERS - 1,
            2 => case.logits = Logits::AllNan,
            3 => case.logits = Logits::Nonfinite,
            4 => case.logits = Logits::Ties,
            5 => case.predicate = false,
            6 => case.run = false,
            7 => {}
            8 => case.predicate = false,
            _ => unreachable!(),
        }
        let probe = TopkProbe::new(&be, &kernels, case, 128);
        let result = compare(&be, &kernels, &probe, |state, _| {
            if kind == 7 {
                for r in 0..3 {
                    let block = probe.owner_push[9 + r] as usize;
                    state[4][block * 4..block * 4 + 2].fill(0);
                }
            }
            if kind == 8 {
                state[0][..case.used].fill(1);
            }
        });
        if kind == 7 {
            assert_eq!(result[3][RESIDENT_JOURNAL_STATUS_WORD], 5);
            assert_eq!(&result[8][..4], &[0, 0, 1, 1]);
        }
    }
}

#[test]
#[ignore = "central serial GPU only; external single-row legacy touch-order shaders"]
fn gpu_owned_topk_single_legacy_touch_oracle() {
    let be = backend();
    for overlap in [false, true] {
        let variable = if overlap {
            "MOE4ALL_OWNER_LEGACY_TOPK_SINGLE_OVERLAP_SPV"
        } else {
            "MOE4ALL_OWNER_LEGACY_TOPK_SINGLE_SERIAL_SPV"
        };
        let mut kernels = kernels(&be);
        kernels.fused = be.kernel_sg(
            if overlap {
                "legacy_single_overlap"
            } else {
                "legacy_single_serial"
            },
            &external_words(variable),
            if overlap { 11 } else { 10 },
            GPU_OWNED_FAST_PUSH_BYTES,
            32,
        );
        for (used, experts) in [(8, 256), (10, 512)] {
            let patterns = if experts == 256 {
                vec![Logits::ForcedRandom]
            } else {
                vec![
                    Logits::ForcedRandom,
                    Logits::Random,
                    Logits::Ties,
                    Logits::Nonfinite,
                    Logits::AllNan,
                ]
            };
            for logits in patterns {
                for workload in [
                    Workload::AllHit,
                    Workload::OneExpertMiss,
                    Workload::MixedStale,
                ] {
                    for alias in [false, true] {
                        let mut case = Case::normal("legacy_single", used, workload);
                        case.logits = logits;
                        case.alias = alias;
                        let mut probe =
                            TopkProbe::new_with_experts(&be, &kernels, case, 128, experts);
                        let selected = probe.fixture.expected[0][..used].to_vec();
                        super::super::legacy_touch_tests::legacy_stamps_for(
                            &mut probe.fixture,
                            &selected,
                        );
                        if overlap {
                            let mask = probe.fixture.expected[3][RESIDENT_JOURNAL_MASK_WORD];
                            probe.fixture.expected[0][used] = mask | (1 << used);
                            probe.fixture.expected[0][used + 1] = ((1 << used) - 1) ^ mask;
                            let result = raw(&be, &kernels, &probe, true, |_, _| {});
                            probe.fixture.assert_result(&be, false, true, false, false);
                            assert_eq!(result[9], probe.expected_weights);
                            assert_eq!(result[10], probe.logits_bits);
                        } else {
                            probe.run(&be, &kernels, true, None);
                        }
                        eprintln!("SINGLE_ROW_LEGACY_TOUCH used={used} experts={experts} overlap={overlap} logits={logits:?} workload={workload:?} alias={alias} slots_stamps_jobs_weights=exact fifo_live_order=exact");
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "central serial GPU runner; overlap metadata faults and invalid route inputs"]
fn gpu_owned_topk_copy_overlap_raw_rejection_exact() {
    let be = backend();
    let kernels = kernels(&be);
    let probe = TopkProbe::new(
        &be,
        &kernels,
        Case::normal("overlap_fault", 10, Workload::OneExpertMiss),
        128,
    );
    for fault in 0..6 {
        let result = compare(&be, &kernels, &probe, |state, _| {
            let missing = probe.owner_push[9] as usize;
            let ready = missing + 1;
            let slot = state[6][ready * 4 + 3] as usize;
            assert_eq!(state[6][ready * 4 + 2], 1);
            assert!(slot < state[5][2] as usize);
            match fault {
                0 => state[4][missing * 4] |= 1,
                1 => state[4][missing * 4 + 3] = u32::MAX,
                2 => state[5][(SLOT_BASE + slot) * 4..(SLOT_BASE + slot) * 4 + 2].fill(0),
                3 => state[5][(SLOT_BASE + slot) * 4 + 2] = u32::MAX,
                4 => state[6][ready * 4] ^= 16,
                5 => state[5][0] = 0,
                _ => unreachable!(),
            }
        });
        assert_eq!(result[3][RESIDENT_JOURNAL_STATUS_WORD], 3);
        assert_eq!(&result[8][..4], &[0, 0, 1, 1]);
        assert_eq!(result[2][3], 1);
    }
    for fault in 0..3 {
        let result = raw(&be, &kernels, &probe, true, |_, push| match fault {
            0 => push[0] += 1,
            1 => push[15] += 1,
            2 => push[6] = 0,
            _ => unreachable!(),
        });
        assert_eq!(result[3][RESIDENT_JOURNAL_STATUS_WORD], 3);
        assert_eq!(
            result[9], probe.weights_bits,
            "invalid route must not evaluate TopK"
        );
        assert_eq!(&result[8][..4], &[0, 0, 1, 1]);
    }
}
