// Isolated staged shared IQ2S NR8 versus the actual staged shared NR1 kernel.
// Included only by pair_verify_contract.rs; no production policy or build hooks.
// Parent supplies iq2s_nr8_shared.spv from native_gemv_id_multi.comp with
// -DPAGED -DUSE_GRID -DFMT_IQ2S -DSHARED_Q8 -DNR=8 and production grid includes.
#[cfg(test)]
mod shared_nr8 {
    use super::*;

    const OUTPUT_SLOTS: usize = USED + 1;
    const SHARED_BIT: u32 = 1 << USED;
    const SHARED_PREFIX: usize = 256;

    fn shared_routes(pattern: Routes, hit_quarters: u32) -> Vec<u32> {
        let mut ids = routing(pattern, Masks::ResidencyById { hit_quarters });
        check_residency_by_id(&ids, hit_quarters);
        for row in 0..ROWS {
            ids[ASSIGNMENTS + row] |= SHARED_BIT;
            assert_eq!(ids[ASSIGNMENTS + ROWS + row] & SHARED_BIT, 0);
        }
        ids
    }

    fn shared_active(ids: &[u32], phase: u32, assignment: usize) -> bool {
        let row = assignment / OUTPUT_SLOTS;
        let slot = assignment % OUTPUT_SLOTS;
        let mask = if phase == 0 {
            ALL | SHARED_BIT
        } else {
            ids[ASSIGNMENTS + (phase as usize - 1) * ROWS + row]
        };
        mask & (1 << slot) != 0
    }

    #[test]
    fn shared_masks_preserve_stride11_and_disjoint_hit_miss_coverage() {
        assert_eq!(OUTPUT_SLOTS, 11);
        for (pattern, pairs) in [(Routes::Pairs6, 6), (Routes::Pairs10, 10)] {
            for quarters in [1, 3] {
                let ids = shared_routes(pattern, quarters);
                check_residency_by_id(&ids, quarters);
                assert_eq!(expected_meta(&ids, 1)[64], pairs);
                assert_eq!(ids.len(), ASSIGNMENTS + 2 * ROWS);
                for assignment in 0..ROWS * OUTPUT_SLOTS {
                    let hit = shared_active(&ids, 1, assignment);
                    let miss = shared_active(&ids, 2, assignment);
                    assert!(shared_active(&ids, 0, assignment));
                    assert_ne!(hit, miss);
                    let slot = assignment % OUTPUT_SLOTS;
                    if slot == USED {
                        assert!(hit);
                        assert!(!miss);
                    } else {
                        let routed = assignment / OUTPUT_SLOTS * USED + slot;
                        assert_eq!(hit, active(&ids, 1, routed));
                        assert_eq!(miss, active(&ids, 2, routed));
                    }
                }
            }
        }
    }

    struct SharedBank {
        routed: Bank,
        _shared: [Box<dyn Buffer>; 2],
        shared_addr: [u64; 2],
    }

    impl SharedBank {
        fn new(be: &VulkanBackend, bank: usize) -> Self {
            let routed = Bank::with_output_slots(be, DType::Iq2S, bank, OUTPUT_SLOTS);
            let (elements, bytes) = infr_core::decode_spec::block_layout(DType::Q8_0);
            assert_eq!((elements, bytes), (32, 34));
            let shared_bytes = K * N / elements * bytes;
            let shared: [Box<dyn Buffer>; 2] = std::array::from_fn(|role| {
                let mut data = vec![0xa5; SHARED_PREFIX + shared_bytes + 512];
                let mut seed = 0x61c9_234du32
                    .wrapping_add(bank as u32 * 0x1_0307)
                    .wrapping_add(role as u32 * 0x3_0109);
                for block in
                    data[SHARED_PREFIX..SHARED_PREFIX + shared_bytes].chunks_exact_mut(bytes)
                {
                    for byte in &mut block[2..] {
                        *byte = next(&mut seed) as u8;
                    }
                    let scale: u16 = if next(&mut seed) & 1 == 0 {
                        0x1400
                    } else {
                        0x9400
                    };
                    block[..2].copy_from_slice(&scale.to_le_bytes());
                }
                upload(be, &data, BufferUsage::Weights)
            });
            let shared_addr = std::array::from_fn(|role| {
                let base = shared[role].device_addr().expect("Q8 shared BDA owner");
                assert_ne!(base, 0);
                base + SHARED_PREFIX as u64
            });
            assert!(shared_addr.iter().all(|addr| *addr != 0 && *addr % 16 == 0));
            Self {
                routed,
                _shared: shared,
                shared_addr,
            }
        }
    }

    struct SharedKernels {
        baseline: ComputeKernel,
        candidate: ComputeKernel,
    }

    impl SharedKernels {
        fn new(be: &VulkanBackend, candidate: &[u32]) -> Self {
            let (name, baseline) = crate::gemm::native_idm_paged_shared_build_spv(DType::Iq2S)
                .expect("built-in shared IQ2S NR1 baseline");
            assert_eq!(name, "native_idm_iq2s_paged_shexp");
            Self {
                baseline: be.kernel(name, baseline, 5, 56),
                candidate: be.kernel("module_iq2s_nr8_shared", candidate, 5, 56),
            }
        }

        fn record(&self, r: &Recorder<'_>, bank: &SharedBank, candidate: bool, phases: &[u32]) {
            let nr = if candidate { 8 } else { 1 };
            let kernel = if candidate {
                self.candidate
            } else {
                self.baseline
            };
            for &phase in phases {
                assert!(phase <= 2);
                for role in 0..2 {
                    let address = bank.shared_addr[role];
                    let push = [
                        K as u32,
                        N as u32,
                        OUTPUT_SLOTS as u32,
                        LUT_BASE as u32,
                        0,
                        ROWS as u32,
                        0,
                        0,
                        0,
                        ALL | SHARED_BIT,
                        phase,
                        address as u32,
                        (address >> 32) as u32,
                        USED as u32,
                    ];
                    let b = &bank.routed;
                    let bindings = [
                        Recorder::vkb(b.lut[role].as_ref()),
                        Recorder::vkb(b.x.as_ref()),
                        Recorder::vkb(b.ids.as_ref()),
                        Recorder::vkb(b.lut[role].as_ref()),
                        b.out[usize::from(candidate)][role].descriptor(),
                    ];
                    r.dispatch3(
                        kernel,
                        &bindings,
                        1,
                        bytemuck::cast_slice(&push),
                        (ROWS * OUTPUT_SLOTS * N.div_ceil(nr)) as u32,
                        1,
                        1,
                    );
                }
            }
        }
    }

    fn check_shared(be: &VulkanBackend, banks: &[SharedBank], ids: &[u32], phases: &[u32]) {
        for bank in banks {
            for role in 0..2 {
                let baseline = bank.routed.out[0][role].read(be);
                let candidate = bank.routed.out[1][role].read(be);
                assert_eq!(baseline.len(), ROWS * OUTPUT_SLOTS * N);
                assert_eq!(candidate.len(), baseline.len());
                for assignment in 0..ROWS * OUTPUT_SLOTS {
                    let enabled = phases
                        .iter()
                        .any(|&phase| shared_active(ids, phase, assignment));
                    let range = assignment * N..(assignment + 1) * N;
                    for word in range.clone() {
                        assert_eq!(
                            candidate[word],
                            baseline[word],
                            "shared NR1/NR8 exactness: role={role} physical_slot={assignment} o={}",
                            word % N
                        );
                        if enabled {
                            assert!(
                                f32::from_bits(candidate[word]).is_finite(),
                                "active output unwritten/nonfinite"
                            );
                        } else {
                            assert_eq!(
                                candidate[word], POISON,
                                "inactive/shared miss slot was written"
                            );
                        }
                    }
                    if enabled {
                        assert!(
                            candidate[range].iter().any(|&v| f32::from_bits(v) != 0.0),
                            "vacuous active result"
                        );
                    }
                }
            }
        }
    }

    fn shared_contracts(
        be: &VulkanBackend,
        kernels: &SharedKernels,
        banks: &[SharedBank],
        ids: &[u32],
    ) {
        let mut full = Vec::new();
        for phases in [&[0u32][..], &[1][..], &[2][..], &[1, 2][..]] {
            for bank in banks {
                for leg in &bank.routed.out {
                    for out in leg {
                        out.reset(be);
                    }
                }
            }
            let r = be.recorder().unwrap();
            for bank in banks {
                kernels.record(&r, bank, false, phases);
                kernels.record(&r, bank, true, phases);
                r.global_barrier();
            }
            r.finish().unwrap();
            check_shared(be, banks, ids, phases);
            if phases == [0] {
                full = banks
                    .iter()
                    .flat_map(|bank| bank.routed.out[0].iter().map(|out| out.read(be)))
                    .collect::<Vec<_>>();
            } else if phases == [1, 2] {
                let output = banks
                    .iter()
                    .flat_map(|bank| bank.routed.out[1].iter().map(|out| out.read(be)))
                    .collect::<Vec<_>>();
                assert_eq!(
                    output, full,
                    "H+M must reconstruct full NR1, shared slot exactly once"
                );
            }
        }
    }

    fn shared_sweep(
        be: &VulkanBackend,
        kernels: &SharedKernels,
        banks: &[SharedBank],
        candidate: bool,
        phases: &[u32],
        timed: bool,
    ) -> (f64, f64) {
        let device = &be.shared.device;
        let timer = timed.then(|| Timer {
            be,
            pool: unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
            }
            .unwrap(),
        });
        let start = Instant::now();
        let r = be.recorder().unwrap();
        if let Some(timer) = &timer {
            unsafe {
                device.cmd_reset_query_pool(r.cmd, timer.pool, 0, 2);
                device.cmd_write_timestamp(
                    r.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    timer.pool,
                    0,
                );
            }
        }
        for bank in banks {
            kernels.record(&r, bank, candidate, phases);
            r.global_barrier();
        }
        if let Some(timer) = &timer {
            unsafe {
                device.cmd_write_timestamp(
                    r.cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    timer.pool,
                    1,
                );
            }
        }
        r.finish().unwrap();
        let wall = start.elapsed().as_secs_f64() * 1e6 / banks.len() as f64;
        let gpu = if let Some(timer) = &timer {
            let mut ticks = [0u64; 2];
            unsafe {
                device.get_query_pool_results(
                    timer.pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            }
            .unwrap();
            assert!(ticks[1] > ticks[0], "invalid timestamp interval");
            (ticks[1] - ticks[0]) as f64 * be.shared.submit_timestamp_period_ns as f64
                / 1000.0
                / banks.len() as f64
        } else {
            0.0
        };
        (gpu, wall)
    }

    fn shared_timing(
        be: &VulkanBackend,
        kernels: &SharedKernels,
        banks: &[SharedBank],
        ids: &[u32],
        pattern: Routes,
        quarters: u32,
        phases: &[u32],
    ) {
        let mut distinct = ids[..ASSIGNMENTS].to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let (qpb, bpb) = infr_core::decode_spec::block_layout(DType::Iq2S);
        let routed_bytes = banks.len() * distinct.len() * 2 * K * N / qpb * bpb;
        assert!(
            routed_bytes > 128 * 1024 * 1024,
            "visited routed weights, excluding padding/shared, must exceed 128 MiB"
        );
        let (q8pb, q8bpb) = infr_core::decode_spec::block_layout(DType::Q8_0);
        let shared_bytes = banks.len() * 2 * K * N / q8pb * q8bpb;
        let counts = expected_meta(ids, 1);
        let hit_assignments = (0..ASSIGNMENTS).filter(|&a| active(ids, 1, a)).count();
        let mut gpu = [Vec::new(), Vec::new()];
        let mut wall = [Vec::new(), Vec::new()];
        for round in 0..3 {
            let order = if round % 2 == 0 {
                [false, true, true, false]
            } else {
                [true, false, false, true]
            };
            for candidate in order {
                for bank in banks {
                    for out in &bank.routed.out[usize::from(candidate)] {
                        out.reset(be);
                    }
                }
                // Matched full-bank priming; no upload/readback in the measured sweep.
                shared_sweep(be, kernels, banks, candidate, phases, false);
                let (g, w) = shared_sweep(be, kernels, banks, candidate, phases, true);
                gpu[usize::from(candidate)].push(g);
                wall[usize::from(candidate)].push(w);
            }
            check_shared(be, banks, ids, phases);
        }
        eprintln!("SHARED_IQ2_NR8 routes={pattern:?} nominal_hit_pct={} routed_hit_assignments={hit_assignments}/{ASSIGNMENTS} pairs_full_hit_miss={:?} phases={phases:?} m={ROWS} k={K} n={N} experts={EXPERTS} routed_used={USED} physical_stride={OUTPUT_SLOTS} baseline=native_idm_iq2s_paged_shexp baseline_nr=1 candidate_nr=8 baseline_groups={} candidate_groups={} rotating={} visited_routed_bytes={routed_bytes} visited_shared_bytes={shared_bytes} residency_by_id=true order=ABBA_BAAB unit=GateUp_module_us baseline_gpu={:?} candidate_gpu={:?} baseline_wall={:?} candidate_wall={:?} gpu_ratio={} wall_ratio={} includes=Gate+Up+shared_Q8+barriers NOT_full_FFN=true NOT_pair_classifier=true",
            quarters * 25, &counts[64..67], ROWS * OUTPUT_SLOTS * N, ROWS * OUTPUT_SLOTS * N.div_ceil(8), banks.len(), gpu[0], gpu[1], wall[0], wall[1],
            median(&gpu[1]) / median(&gpu[0]), median(&wall[1]) / median(&wall[0]));
    }

    #[test]
    #[ignore = "parent-serialized shared IQ2S NR8 module; explicit MOE4ALL_TEST_DEVICE and iq2s_nr8_shared.spv required"]
    fn shared_iq2s_nr8_flash_verify4_numeric_and_module_cost() {
        let selector = std::env::var("MOE4ALL_TEST_DEVICE")
            .expect("select an idle GPU explicitly with VulkanN or its index");
        let selector = selector.trim();
        let index = selector
            .strip_prefix("Vulkan")
            .unwrap_or(selector)
            .parse::<usize>()
            .expect("MOE4ALL_TEST_DEVICE must be VulkanN or a bare device index");
        let root = std::env::var_os("MOE4ALL_PAIR_VERIFY_SHADER_DIR")
            .expect("parent must compile the shared candidate first");
        let root = Path::new(&root);
        assert!(root.is_absolute());
        let candidate = external_spv(root, "iq2s_nr8_shared.spv");
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev = Some(format!("Vulkan{index}"));
        let be = VulkanBackend::new_on_with(index, std::sync::Arc::new(cfg))
            .expect("explicit Vulkan device is required, no skip or fallback");
        let kernels = SharedKernels::new(&be, &candidate);
        let banks: Vec<_> = (0..BANKS).map(|bank| SharedBank::new(&be, bank)).collect();
        for pattern in [Routes::Pairs6, Routes::Pairs10] {
            for quarters in [1, 3] {
                let ids = shared_routes(pattern, quarters);
                for bank in &banks {
                    be.upload(bank.routed.ids.as_ref(), bytemuck::cast_slice(&ids))
                        .unwrap();
                }
                shared_contracts(&be, &kernels, &banks, &ids);
                if quarters == 1 {
                    shared_timing(&be, &kernels, &banks, &ids, pattern, quarters, &[0]);
                }
                shared_timing(&be, &kernels, &banks, &ids, pattern, quarters, &[1, 2]);
            }
        }
    }
}
