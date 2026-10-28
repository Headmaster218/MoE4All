//! GU-only LOCAL64 qualification against the actual selected native GU path.
use super::*;

fn native_profiles() -> [Profile; 5] {
    [
        Profile {
            name: "q4k_2048_512",
            ne: 2048,
            nff: 512,
            used: 8,
            gate_up: DType::Q4K,
            down: DType::Q4K,
        },
        Profile {
            name: "q5k_2048_512",
            ne: 2048,
            nff: 512,
            used: 8,
            gate_up: DType::Q5K,
            down: DType::Q5K,
        },
        Profile {
            name: "q6k_2048_512",
            ne: 2048,
            nff: 512,
            used: 8,
            gate_up: DType::Q6K,
            down: DType::Q6K,
        },
        Profile {
            name: "iq2s_iq4nl_2560_640",
            ne: 2560,
            nff: 640,
            used: 10,
            gate_up: DType::Iq2S,
            down: DType::Iq4Nl,
        },
        Profile {
            name: "iq3s_iq4nl_2560_640",
            ne: 2560,
            nff: 640,
            used: 10,
            gate_up: DType::Iq3S,
            down: DType::Iq4Nl,
        },
    ]
}

fn native_grid(be: &VulkanBackend, p: Profile) -> &dyn Buffer {
    let grid = match p.gate_up {
        DType::Iq2S => &be.iq2s_grid,
        DType::Iq3S => &be.iq3s_grid,
        _ => panic!("only IQ2S/IQ3S have native buffer grids"),
    };
    grid.as_ref()
        .expect("existing backend native grid")
        .as_ref()
}

fn download(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<u8> {
    let mut bytes = vec![0; buffer.len_bytes()];
    be.download(buffer, &mut bytes).unwrap();
    bytes
}

fn exact_native_gu(grid_buffer: bool) {
    let be = fused_backend();
    for p in native_profiles() {
        if grid_buffer && !matches!(p.gate_up, DType::Iq2S | DType::Iq3S) {
            continue;
        }
        let source = Sources::new(&be, p, 0);
        assert_eq!(source.banks, 3, "correctness-only source banks");
        let variants = [1, 2].map(|nr| {
            let kernels = if grid_buffer {
                Kernels::tree64_gu64_grid(&be, p, nr)
            } else {
                Kernels::tree64_gu(&be, p, 64, nr)
            };
            assert_eq!(kernels.gu_outputs, nr);
            assert_eq!(kernels.grid_buffer, grid_buffer);
            (kernels, nr)
        });
        for rows in [1, 2, 4, 8] {
            let bank = (rows - 1) % source.banks;
            let mut fixture = FusedFixture::new(&be, &source, p, rows, false);
            fixture.inner.prepare(&source, bank, true);
            let rec = be.recorder().unwrap();
            rec.seed_barrier();
            fixture.projections(&rec, 0, (1u32 << p.used) - 1);
            fixture.activation(&rec);
            rec.publish_host_reads();
            rec.finish().unwrap();
            let expected = read(fixture.inner.activation.as_ref());
            let native_kernels: Vec<_> = be
                .built_kernel_names()
                .into_iter()
                .filter(|name| name.starts_with("native_idm_") && name.contains(format_name(p)))
                .collect();
            assert!(!native_kernels.is_empty(), "native GU oracle must record");
            assert!(
                native_kernels.iter().all(|name| !name.contains("_sg")),
                "selected native SG reduction is not Tree64: {native_kernels:?}"
            );
            assert!(
                expected[..expected.len() - GUARD]
                    .chunks_exact(4)
                    .all(|word| f32::from_le_bytes(word.try_into().unwrap()).is_finite())
            );
            assert!(
                expected[expected.len() - GUARD..]
                    .iter()
                    .all(|&b| b == POISON)
            );
            println!(
                "TREE64_LOCAL64_ORACLE profile={} rows={rows} native_kernels={native_kernels:?} native_grid_policy={} native_grid_nr={} native_lds_nr={} candidate_grid_buffer={grid_buffer} artifact=builtin_sdk_spv",
                p.name,
                be.use_paged_grid_buffer(p.ne, p.nff, rows),
                be.paged_grid_buffer_nr(),
                be.cfg().kernels.vulkan.gemv.id_grid_nr
            );
            for (kernels, nr) in variants {
                fixture.inner.prepare(&source, bank, true);
                let f = &fixture.inner;
                let width = p.nff * 4;
                // Resident weights do not make the masks all-hit. Reset every M ACT
                // to NaN after prepare so skipped GU cannot reuse a prior oracle.
                let mut seed = read(f.activation.as_ref());
                let mut hits = 0;
                let mut misses = 0;
                for assignment in 0..rows * p.used {
                    let hit = f.hits[assignment / p.used] & (1 << (assignment % p.used)) != 0;
                    if hit {
                        hits += 1;
                    } else {
                        misses += 1;
                        let range = assignment * width..(assignment + 1) * width;
                        for word in seed[range].chunks_exact_mut(4) {
                            word.copy_from_slice(&f32::NAN.to_le_bytes());
                        }
                    }
                }
                assert!(hits > 0 && misses > 0);
                write(f.activation.as_ref(), &seed);
                assert_eq!(read(f.activation.as_ref()), seed, "fresh ACT seed");
                let immutable: [(&str, &dyn Buffer); 11] = [
                    ("input", f.input.as_ref()),
                    ("ids_masks", f.ids.as_ref()),
                    ("tape", f.tape.as_ref()),
                    ("gate", f.gate.as_ref()),
                    ("up", f.up.as_ref()),
                    ("down_output", f.output.as_ref()),
                    ("journal", f.journal.as_ref()),
                    ("jobs", f.jobs.as_ref()),
                    ("commands", f.commands.as_ref()),
                    ("commands_late", f.commands_late.as_ref()),
                    ("dummy", f.dummy.as_ref()),
                ];
                let before: Vec<_> = immutable.iter().map(|(_, b)| read(*b)).collect();
                let grid_before = grid_buffer.then(|| download(&be, native_grid(&be, p)));
                sample_fused(&be, &fixture, Arm::FusedGu, kernels);
                let actual = read(f.activation.as_ref());
                for assignment in 0..rows * p.used {
                    let range = assignment * width..(assignment + 1) * width;
                    let hit = f.hits[assignment / p.used] & (1 << (assignment % p.used)) != 0;
                    if hit {
                        assert_eq!(actual[range.clone()], seed[range], "H ACT untouched");
                    } else {
                        assert_ne!(
                            seed[range.clone()],
                            expected[range.clone()],
                            "no-op must fail"
                        );
                        if let Some(byte) = actual[range.clone()]
                            .iter()
                            .zip(&expected[range.clone()])
                            .position(|(a, b)| a != b)
                        {
                            let at = range.start + byte / 4 * 4;
                            let got = u32::from_le_bytes(actual[at..at + 4].try_into().unwrap());
                            let want = u32::from_le_bytes(expected[at..at + 4].try_into().unwrap());
                            panic!(
                                "M ACT not byte-exact: {p:?} rows={rows} local=64 nr={nr} grid_buffer={grid_buffer} assignment={assignment} element={} actual_bits={got:08x} expected_bits={want:08x} native_kernels={native_kernels:?}",
                                byte / 4
                            );
                        }
                    }
                }
                assert_eq!(
                    actual[actual.len() - GUARD..],
                    seed[seed.len() - GUARD..],
                    "ACT guard"
                );
                assert_eq!(
                    read(f.weights.as_ref()),
                    f.expected_weights,
                    "weights and payload guards"
                );
                for ((name, buffer), initial) in immutable.iter().zip(&before) {
                    assert_eq!(read(*buffer), *initial, "GU-only changed {name}");
                }
                if let Some(initial) = grid_before {
                    assert_eq!(
                        download(&be, native_grid(&be, p)),
                        initial,
                        "native grid immutable"
                    );
                }
                println!(
                    "TREE64_LOCAL64_EXACT profile={} ne={} nff={} rows={rows} used={} nr={nr} grid_buffer={grid_buffer} outputs_per_wg={} M_ACT_byteexact=true M_ACT_fresh_nan=true H_ACT_untouched=true guards_weights_exact=true source_banks=3 copy_fullhit_down_pipelines=false",
                    p.name, p.ne, p.nff, p.used, kernels.gu_outputs
                );
            }
        }
    }
}

#[test]
fn owned_tree64_local64_factory_matrix() {
    for dtype in [DType::Q4K, DType::Q5K, DType::Q6K, DType::Iq2S, DType::Iq3S] {
        for nr in [1, 2] {
            assert!(crate::gemm::gpu_owned_miss_gu_tree64_spv(dtype, 64, nr).is_some());
            assert_eq!(
                crate::gemm::gpu_owned_miss_gu_tree64_grid_buffer_spv(dtype, 64, nr).is_some(),
                matches!(dtype, DType::Iq2S | DType::Iq3S)
            );
        }
        assert!(crate::gemm::gpu_owned_miss_gu_tree64_spv(dtype, 64, 4).is_none());
        assert!(crate::gemm::gpu_owned_miss_gu_tree64_grid_buffer_spv(dtype, 64, 8).is_none());
        assert!(crate::gemm::gpu_owned_miss_gu_tree64_grid_buffer_spv(dtype, 128, 2).is_none());
    }
    assert!(crate::gemm::gpu_owned_miss_gu_tree64_spv(DType::F32, 64, 1).is_none());
}

#[test]
#[ignore = "parent serial GPU; builtin LOCAL64 LDS NR1/2, all five formats, rows1/2/4/8"]
fn gpu_owned_tree64_gu64_lds_native_exact_rows_1_2_4_8() {
    exact_native_gu(false);
}

#[test]
#[ignore = "parent serial GPU; existing native IQ2/IQ3 buffers, builtin LOCAL64 NR1/2, rows1/2/4/8"]
fn gpu_owned_tree64_gu64_grid_native_exact_rows_1_2_4_8() {
    exact_native_gu(true);
}
