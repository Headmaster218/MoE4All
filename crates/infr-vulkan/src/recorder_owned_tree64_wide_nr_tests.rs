//! Child of recorder_owned_fused_tests; parent registers the module separately.
//! Built-in NR4/NR8 GU only. No copy, FullHit or Down pipeline is needed.
use super::*;

fn gu_kernels(be: &VulkanBackend, p: Profile, local: usize, nr: usize) -> Kernels {
    assert!(matches!(local, 128 | 256) && matches!(nr, 4 | 8));
    let spv = crate::gemm::gpu_owned_miss_gu_tree64_spv(p.gate_up, local as u32, nr as u32)
        .expect("registered built-in wide-NR Tree64 GU");
    let name = Box::leak(
        format!(
            "fused_probe_builtin_tree64_{}_{}_nr{}",
            format_name(p),
            local,
            nr
        )
        .into_boxed_str(),
    );
    let gu = be.kernel_sg(name, spv, 4, 40, 32);
    Kernels {
        hit: gu,
        gu,
        roles: gu, // Unused placeholders: only Arm::FusedGu may use this test bundle.
        local,
        tiles: 1,
        gu_outputs: (local / 64) * nr,
        grid_buffer: false,
    }
}

fn native_profiles() -> [Profile; 3] {
    [
        Profile {
            name: "q5k_2048_512",
            ne: 2048,
            nff: 512,
            used: 8,
            gate_up: DType::Q5K,
            down: DType::Q5K,
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

fn exact_native_gu(nr: usize) {
    let be = fused_backend();
    for p in native_profiles() {
        let source = Sources::new(&be, p, 0);
        assert_eq!(
            source.banks, 3,
            "correctness-only sources, not the performance ring"
        );
        let variants = [128, 256].map(|local| gu_kernels(&be, p, local, nr));
        for rows in [1, 2] {
            let bank = rows - 1;
            let mut fixture = FusedFixture::new(&be, &source, p, rows, false);
            fixture.inner.prepare(&source, bank, true);
            let rec = be.recorder().unwrap();
            rec.seed_barrier();
            fixture.projections(&rec, 0, (1u32 << p.used) - 1);
            fixture.activation(&rec);
            rec.publish_host_reads();
            rec.finish().unwrap();
            let expected = read(fixture.inner.activation.as_ref());
            let names: Vec<_> = be
                .built_kernel_names()
                .into_iter()
                .filter(|name| name.starts_with("native_idm_") && name.contains(format_name(p)))
                .collect();
            println!(
                "TREE64_WIDE_NR_ORACLE profile={} rows={rows} native_kernels={names:?} grid_buffer_policy={} grid_buffer_nr={} id_grid_nr={} candidate_nr={nr} artifact=builtin_sdk_spv",
                p.name,
                be.use_paged_grid_buffer(p.ne, p.nff, rows),
                be.paged_grid_buffer_nr(),
                be.cfg().kernels.vulkan.gemv.id_grid_nr
            );
            assert!(
                !names.is_empty(),
                "ordinary native GU oracle must record a kernel"
            );
            assert!(
                names.iter().all(|name| !name.contains("_sg")),
                "non-Tree64 native SG path selected for {p:?}, rows={rows}: {names:?}; no tolerance or oracle substitution permitted"
            );
            assert!(
                expected[expected.len() - GUARD..]
                    .iter()
                    .all(|&v| v == POISON)
            );
            assert!(
                expected[..expected.len() - GUARD]
                    .chunks_exact(4)
                    .all(|word| f32::from_le_bytes(word.try_into().unwrap()).is_finite())
            );
            for kernels in variants {
                fixture.inner.prepare(&source, bank, true);
                let f = &fixture.inner;
                let unchanged: [(&str, &dyn Buffer); 11] = [
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
                let before: Vec<_> = unchanged.iter().map(|(_, b)| read(*b)).collect();
                sample_fused(&be, &fixture, Arm::FusedGu, kernels);
                let actual = read(f.activation.as_ref());
                let width = p.nff * 4;
                let mut hits = 0;
                let mut misses = 0;
                for assignment in 0..rows * p.used {
                    let range = assignment * width..(assignment + 1) * width;
                    let hit = f.hits[assignment / p.used] & (1 << (assignment % p.used)) != 0;
                    if hit {
                        hits += 1;
                        assert!(
                            actual[range]
                                .chunks_exact(4)
                                .all(|word| word == 0.125f32.to_le_bytes()),
                            "H assignment changed: {p:?} rows={rows} local={} nr={nr} assignment={assignment}",
                            kernels.local
                        );
                    } else {
                        misses += 1;
                        if let Some(byte) = actual[range.clone()]
                            .iter()
                            .zip(&expected[range.clone()])
                            .position(|(a, b)| a != b)
                        {
                            let at = range.start + byte / 4 * 4;
                            let got = u32::from_le_bytes(actual[at..at + 4].try_into().unwrap());
                            let want = u32::from_le_bytes(expected[at..at + 4].try_into().unwrap());
                            panic!(
                                "M ACT not byte-exact: {p:?} rows={rows} local={} nr={nr} assignment={assignment} element={} actual_bits={got:08x} expected_bits={want:08x} native_kernels={names:?}",
                                kernels.local,
                                byte / 4
                            );
                        }
                    }
                }
                assert!(hits > 0 && misses > 0);
                assert!(
                    actual[actual.len() - GUARD..].iter().all(|&v| v == POISON),
                    "ACT guard"
                );
                assert_eq!(
                    read(f.weights.as_ref()),
                    f.expected_weights,
                    "weights and payload guards"
                );
                for ((name, buffer), seed) in unchanged.iter().zip(&before) {
                    assert_eq!(read(*buffer), *seed, "GU-only changed {name}");
                }
                println!(
                    "TREE64_WIDE_NR_EXACT profile={} ne={} nff={} rows={rows} used={} local={} nr={nr} outputs_per_wg={} source_banks=3 M_ACT_byteexact=true H_untouched_0_125=true guards_weights_exact=true copy_fullhit_down_pipelines=false",
                    p.name, p.ne, p.nff, p.used, kernels.local, kernels.gu_outputs
                );
            }
        }
    }
}

#[test]
#[ignore = "parent serial GPU only; native Q5/IQ2/IQ3 GU NR4, rows1/2, local128/256"]
fn gpu_owned_tree64_gu_nr4_native_exact_rows_1_2() {
    exact_native_gu(4);
}

#[test]
#[ignore = "parent serial GPU only; native Q5/IQ2/IQ3 GU NR8, rows1/2, local128/256"]
fn gpu_owned_tree64_gu_nr8_native_exact_rows_1_2() {
    exact_native_gu(8);
}
