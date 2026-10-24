//! Ignored, parent-serialized contracts for the fused classifier and production pair kernels.
#[cfg(test)]
mod fixtures {
    use super::super::*;
    use infr_core::{backend::BufferUsage, Backend};

    const K: usize = 2560;
    const N: usize = 640;
    const USED: usize = 10;
    const EXPERTS: usize = 512;
    const CANARY: u32 = 0xc5f3_5555;

    fn device_selector(value: &str) -> String {
        let value = value.trim();
        assert!(
            !value.is_empty(),
            "explicit idle device selector is required"
        );
        if value.bytes().all(|byte| byte.is_ascii_digit()) {
            format!("Vulkan{value}")
        } else {
            value.to_owned()
        }
    }

    #[test]
    fn device_selector_accepts_standard_name_and_bare_index() {
        assert_eq!(device_selector("Vulkan1"), "Vulkan1");
        assert_eq!(device_selector("1"), "Vulkan1");
        assert_eq!(device_selector("  Vulkan1  "), "Vulkan1");
        assert_eq!(device_selector("0"), "Vulkan0");
    }

    fn routes(pairs: usize) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..40).collect();
        for slot in 0..pairs / 2 {
            ids[10 + slot] = ((slot * 3 + 1) % 10) as u32;
            ids[30 + slot] = (20 + (slot * 7 + 3) % 10) as u32;
        }
        ids.resize(48, 0);
        ids
    }

    fn resident(id: u32, quarters: u32) -> bool {
        ((id ^ (id >> 3)) & 3) < quarters
    }

    fn upload(backend: &VulkanBackend, words: &[u32]) -> Arc<dyn Buffer> {
        let buffer: Arc<dyn Buffer> = Arc::from(
            backend
                .alloc(words.len() * 4, BufferUsage::Readback)
                .unwrap(),
        );
        backend
            .upload(buffer.as_ref(), bytemuck::cast_slice(words))
            .unwrap();
        buffer
    }

    fn read(backend: &VulkanBackend, buffer: &dyn Buffer) -> Vec<u32> {
        let mut words = vec![0; buffer.len_bytes() / 4];
        backend
            .download(buffer, bytemuck::cast_slice_mut(&mut words))
            .unwrap();
        words
    }

    fn random(seed: &mut u32) -> u32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        *seed
    }

    fn weights(
        backend: &VulkanBackend,
        dtype: DType,
        experts: usize,
        mut seed: u32,
    ) -> Arc<dyn Buffer> {
        let (elements, bytes) = infr_core::decode_spec::block_layout(dtype);
        let mut data = vec![0; experts * K * N / elements * bytes];
        for block in data.chunks_exact_mut(bytes) {
            for byte in block.iter_mut() {
                *byte = random(&mut seed) as u8;
            }
            block[..2].copy_from_slice(&0x1400u16.to_le_bytes());
        }
        let buffer: Arc<dyn Buffer> =
            Arc::from(backend.alloc(data.len(), BufferUsage::Weights).unwrap());
        backend.upload(buffer.as_ref(), &data).unwrap();
        buffer
    }

    struct Event<'a>(&'a VulkanBackend, ash::vk::Event);
    impl Drop for Event<'_> {
        fn drop(&mut self) {
            unsafe {
                self.0.shared.device.destroy_event(self.1, None);
            }
        }
    }

    fn cohort(shared: bool) -> Cohort {
        Cohort {
            verify: true,
            independent_rows: false,
            rows: 4,
            used: USED,
            experts: EXPERTS,
            in_f: K,
            out_f: N,
            shared_q8: shared,
        }
    }

    fn expected_counts(ids: &[u32], quarters: u32) -> [u32; 3] {
        let mut counts = [0; 3];
        for a in 0..40 {
            let row = a / USED;
            if row & 1 != 0 {
                continue;
            }
            let n = ids[(row ^ 1) * USED..(row ^ 1) * USED + USED]
                .iter()
                .filter(|&&id| id == ids[a])
                .count();
            if n != 1 {
                continue;
            }
            counts[0] += 1;
            counts[if resident(ids[a], quarters) { 1 } else { 2 }] += 1;
        }
        counts
    }

    #[test]
    fn route_oracle_covers_actual_thresholds_and_coherent_phases() {
        assert!(
            infr_core::config::Config::default()
                .kernels
                .vulkan
                .verify_pair
        );
        for pairs in [0, 6, 10, 20] {
            let ids = routes(pairs);
            for quarters in [0, 1, 3, 4] {
                let counts = expected_counts(&ids, quarters);
                assert_eq!(counts[0], pairs as u32);
                assert_eq!(counts[1] + counts[2], counts[0]);
            }
        }
    }

    #[test]
    #[ignore = "parent-serialized GPU cache lifetime test; select MOE4ALL_TEST_DEVICE explicitly"]
    fn verify_pair_phase_cache_reuses_identity_and_stabilizes_resource_epoch() {
        use crate::adapter::{RuntimePhase, RuntimePhaseArena};
        use std::sync::atomic::Ordering;
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev = Some(device_selector(
            &std::env::var("MOE4ALL_TEST_DEVICE")
                .expect("explicit idle device selector is required"),
        ));
        cfg.kernels.vulkan.verify_pair = true;
        cfg.kernels.vulkan.gemv.id_grid_buffer = Some(false);
        let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        let pool = backend.init_unified_vram(8 * 1024 * 1024).unwrap();
        backend.with_unified_exclusive(|| {
            let c = cohort(false);
            let policy = RolePolicy::measured(DType::Iq3S, "native_idm_iq3s_paged", c);
            let plan = LayerPlan::new(c, [policy, policy]).unwrap();
            let key = |count| vec![Some(plan); count];
            let mut phase = RuntimePhaseArena::default();
            let layout = Vec::new();
            phase.begin_execute(RuntimePhase::Decode, &layout);
            let before = pool.stats().live_allocations;
            let mut identities = Vec::new();
            // Two actual split-graph-sized keys plus two bounded alternate layouts.
            for count in [1, 47, 2, 3] {
                let plans = key(count);
                let lease = phase
                    .verify_pairs
                    .acquire(&plans, false, || {
                        VerifyPairWindow::allocate(&backend, &plans)
                            .map(|window| window.map(Arc::new))
                    })
                    .unwrap()
                    .unwrap();
                identities.push((count, Arc::downgrade(&lease)));
            }
            assert_eq!(phase.verify_pairs.entries.len(), WindowCache::CAPACITY);
            let warm_allocations = pool.stats().live_allocations;
            let warm_generation = pool.generation();
            let prefix_epoch = backend.shared.resource_epoch.load(Ordering::Acquire);
            for _ in 0..8 {
                for (count, identity) in &identities {
                    phase.begin_execute(RuntimePhase::Decode, &layout);
                    let plans = key(*count);
                    let lease = phase
                        .verify_pairs
                        .acquire(&plans, true, || {
                            panic!("warm/frozen cache performed a unified allocation")
                        })
                        .unwrap()
                        .unwrap();
                    assert!(std::sync::Weak::ptr_eq(identity, &Arc::downgrade(&lease)));
                    let layer = lease.layer(0).unwrap();
                    let pointer = as_vk_buf(layer.buffer.as_ref())
                        .unwrap()
                        .mapped_ptr()
                        .unwrap();
                    let initial_config = unsafe {
                        std::slice::from_raw_parts(
                            pointer.cast::<u32>().add(CONFIG),
                            WORDS - CONFIG,
                        )
                        .to_vec()
                    };
                    let rec = backend.recorder().unwrap();
                    rec.retain_buffer(Arc::clone(&layer.buffer));
                    // Dynamic region is GPU-owned; a warm hit must preserve it, not host-reset it.
                    rec.zero(layer.buffer.as_ref(), CONFIG);
                    let pending = rec.finish_nowait().unwrap();
                    drop(layer);
                    drop(lease);
                    assert!(phase
                        .verify_pairs
                        .acquire(&plans, true, || panic!(
                            "queued reader created a replacement"
                        ))
                        .is_err());
                    pending.wait().unwrap();
                    let reused = phase
                        .verify_pairs
                        .acquire(&plans, true, || panic!("retired window failed to reuse"))
                        .unwrap()
                        .unwrap();
                    let layer = reused.layer(0).unwrap();
                    let pointer = as_vk_buf(layer.buffer.as_ref())
                        .unwrap()
                        .mapped_ptr()
                        .unwrap();
                    let words = unsafe { std::slice::from_raw_parts(pointer.cast::<u32>(), WORDS) };
                    assert!(words[..CONFIG].iter().all(|&word| word == 0));
                    assert_eq!(&words[CONFIG..], initial_config.as_slice());
                    drop(layer);
                    drop(reused);
                    assert_eq!(pool.stats().live_allocations, warm_allocations);
                    assert_eq!(pool.generation(), warm_generation);
                    assert_eq!(
                        backend.shared.resource_epoch.load(Ordering::Acquire),
                        prefix_epoch,
                        "warm sidecar reuse invalidated the epoch used by prepared prefixes"
                    );
                }
            }
            assert!(phase
                .verify_pairs
                .acquire(&key(4), true, || panic!("frozen cold key allocated"))
                .is_err());
            assert_eq!(
                backend.shared.resource_epoch.load(Ordering::Acquire),
                prefix_epoch
            );
            let cold = key(4);
            drop(
                phase
                    .verify_pairs
                    .acquire(&cold, false, || {
                        VerifyPairWindow::allocate(&backend, &cold)
                            .map(|window| window.map(Arc::new))
                    })
                    .unwrap(),
            );
            assert_eq!(phase.verify_pairs.entries.len(), WindowCache::CAPACITY);
            assert!(
                identities[0].1.upgrade().is_none(),
                "bounded cache did not evict the oldest key"
            );
            phase.begin_execute(RuntimePhase::Prefill, &layout);
            assert!(phase.verify_pairs.entries.is_empty());
            assert_eq!(pool.stats().live_allocations, before);
            let plans = key(1);
            drop(
                phase
                    .verify_pairs
                    .acquire(&plans, false, || {
                        VerifyPairWindow::allocate(&backend, &plans)
                            .map(|window| window.map(Arc::new))
                    })
                    .unwrap(),
            );
            phase.release_phase();
            assert!(phase.verify_pairs.entries.is_empty());
            assert_eq!(pool.stats().live_allocations, before);
        });
    }

    #[test]
    #[ignore = "parent-serialized GPU contract; select MOE4ALL_TEST_DEVICE explicitly"]
    fn verify_pair_fused_mixed_roles_shared_stride_and_resume() {
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev = Some(device_selector(
            &std::env::var("MOE4ALL_TEST_DEVICE")
                .expect("explicit idle device selector is required"),
        ));
        cfg.kernels.vulkan.verify_pair = true;
        cfg.kernels.vulkan.verify_shared_iq2_nr8 = true;
        cfg.kernels.vulkan.gemv.id_grid_buffer = Some(false);
        cfg.kernels.vulkan.gemv.id_grid_nr = 8;
        // No copy buffer participates: the legacy CPU-promotion Verify path is independent.
        cfg.kernels.vulkan.decode_gpu_resident_copy = Some(false);
        let backend = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        let event = Event(&backend, unsafe {
            backend
                .shared
                .device
                .create_event(&ash::vk::EventCreateInfo::default(), None)
                .unwrap()
        });
        let dtypes = [DType::Iq2S, DType::Iq3S];
        let routed = dtypes.map(|dtype| weights(&backend, dtype, 40, 0x1594_7821));
        let shared_weights = [
            weights(&backend, DType::Q8_0, 1, 0x1234_5678),
            weights(&backend, DType::Q8_0, 1, 0x9764_2318),
        ];
        let mut seed = 0x4311_783d;
        let x: Vec<f32> = (0..4 * K)
            .map(|_| (random(&mut seed) as i32 as f32 / i32::MAX as f32) * 0.75)
            .collect();
        let x = upload(&backend, bytemuck::cast_slice(&x));
        let mut full_tape = vec![0u64; EXPERTS * 3];
        for role in 0..2 {
            let (qpb, bpb) = infr_core::decode_spec::block_layout(dtypes[role]);
            let stride = K * N / qpb * bpb;
            let base = routed[role].device_addr().unwrap();
            for id in 0..40 {
                full_tape[role * EXPERTS + id] = base + (id * stride) as u64;
            }
        }
        full_tape[EXPERTS * 2..EXPERTS * 2 + 40].fill(1);
        let tape = upload(&backend, bytemuck::cast_slice(&full_tape));
        let ids_buffer = upload(&backend, &[0; 48]);
        let control = upload(&backend, &[0; crate::recorder::RESIDENT_CONTROL_BYTES / 4]);
        let journal = upload(&backend, &vec![0; crate::recorder::RESIDENT_JOURNAL_WORDS]);
        for shared in [false, true] {
            let cohort = cohort(shared);
            let policies = dtypes
                .map(|dtype| crate::recorder::verify_pair_baseline_policy(&backend, dtype, cohort));
            assert!(policies[1].is_some());
            assert_eq!(
                policies[0].is_none(),
                shared,
                "shared IQ2S NR8 baseline must not enroll an unqualified pair kernel"
            );
            let plan = LayerPlan::new(cohort, policies).unwrap();
            let layer = Layer {
                buffer: upload(&backend, &plan.initial_words()),
                cohort,
                roles: policies,
            };
            let physical = USED + usize::from(shared);
            let mut poison = vec![0x7fc0_1234; 4 * physical * N + 16];
            poison[4 * physical * N..].fill(CANARY);
            let baseline = [upload(&backend, &poison), upload(&backend, &poison)];
            let candidate = [upload(&backend, &poison), upload(&backend, &poison)];
            for pairs in [0, 6, 10, 20] {
                for quarters in [0, 1, 3, 4] {
                    let ids = routes(pairs);
                    backend
                        .upload(ids_buffer.as_ref(), bytemuck::cast_slice(&ids))
                        .unwrap();
                    let mut addresses = full_tape.clone();
                    for id in 0..40 {
                        if !resident(id as u32, quarters) {
                            addresses[2 * EXPERTS + id] = 0;
                        }
                    }
                    backend
                        .upload(tape.as_ref(), bytemuck::cast_slice(&addresses))
                        .unwrap();
                    let mut seed_control = vec![0; crate::recorder::RESIDENT_CONTROL_BYTES / 4];
                    seed_control[0] = 1;
                    seed_control[1] = u32::MAX;
                    backend
                        .upload(control.as_ref(), bytemuck::cast_slice(&seed_control))
                        .unwrap();
                    unsafe {
                        backend.shared.device.reset_event(event.1).unwrap();
                    }
                    let rec = backend.recorder().unwrap();
                    rec.gpu_resident_hit_check_pair(
                        ids_buffer.as_ref(),
                        tape.as_ref(),
                        control.as_ref(),
                        journal.as_ref(),
                        [40, 512, 0, 512, 1024, 0, 10, if shared { 0x400 } else { 0 }],
                        event.1,
                        &layer,
                    );
                    rec.finish_nowait().unwrap().wait().unwrap();
                    let meta = read(&backend, layer.buffer.as_ref());
                    let masks = read(&backend, ids_buffer.as_ref());
                    let counts = expected_counts(&ids, quarters);
                    assert_eq!(&meta[64..67], &counts, "pairs={pairs} quarters={quarters}");
                    assert_eq!(meta[VALID], 1);
                    let notice = read(&backend, journal.as_ref());
                    for row in 0..4 {
                        let hit = (0..USED).fold(0u32, |mask, slot| {
                            mask | (u32::from(resident(ids[row * USED + slot], quarters)) << slot)
                        });
                        assert_eq!(masks[40 + row], hit | if shared { 0x400 } else { 0 });
                        assert_eq!(masks[44 + row], 0x3ff ^ hit);
                        assert_eq!(
                            notice[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + row],
                            hit
                        );
                    }
                    for role in [Role::Gate, Role::Up] {
                        for phase in [Phase::Hits, Phase::Misses] {
                            let phase_index = phase as usize;
                            let phase_active =
                                (0..4).any(|row| masks[40 + (phase_index - 1) * 4 + row] != 0);
                            let enabled = policies[role as usize].is_some() && phase_active;
                            let paired = enabled
                                && counts[0] >= policies[role as usize].unwrap().threshold
                                && counts[phase_index] != 0;
                            for path in [Path::Baseline, Path::Pair] {
                                let word = command_byte_offset(role, phase, path) as usize / 4;
                                assert_eq!(
                                    meta[word] != 0,
                                    enabled && (paired == (path == Path::Pair))
                                );
                            }
                        }
                    }
                    // A CPU miss promotion changes LUT residency, but must not reclassify or reset
                    // this metadata. H and M still refer to the original frozen masks.
                    backend
                        .upload(tape.as_ref(), bytemuck::cast_slice(&full_tape))
                        .unwrap();
                    let rec = backend.recorder().unwrap();
                    for role in 0..2 {
                        backend
                            .upload(baseline[role].as_ref(), bytemuck::cast_slice(&poison))
                            .unwrap();
                        backend
                            .upload(candidate[role].as_ref(), bytemuck::cast_slice(&poison))
                            .unwrap();
                        let shared_weight = shared.then_some(shared_weights[role].as_ref());
                        let ordinary = |out: &dyn Buffer, bank, causal_verify| {
                            if let Some(weight) = shared_weight {
                                rec.linear_native_id_multi_paged_shared_for_verify(
                                    dtypes[role],
                                    0,
                                    0,
                                    tape.as_ref(),
                                    ids_buffer.as_ref(),
                                    USED,
                                    role * EXPERTS,
                                    weight,
                                    x.as_ref(),
                                    false,
                                    out,
                                    K,
                                    N,
                                    4,
                                    0x7ff,
                                    bank,
                                    causal_verify,
                                );
                            } else {
                                rec.linear_native_id_multi_paged(
                                    dtypes[role],
                                    0,
                                    0,
                                    tape.as_ref(),
                                    ids_buffer.as_ref(),
                                    USED,
                                    role * EXPERTS,
                                    x.as_ref(),
                                    false,
                                    out,
                                    K,
                                    N,
                                    4,
                                    0x3ff,
                                    bank,
                                );
                            }
                        };
                        ordinary(baseline[role].as_ref(), 0, false);
                        for phase in [1, 2] {
                            if !rec.linear_verify_pair(
                                &layer,
                                if role == 0 { Role::Gate } else { Role::Up },
                                dtypes[role],
                                0,
                                0,
                                tape.as_ref(),
                                ids_buffer.as_ref(),
                                role * EXPERTS,
                                shared_weight,
                                x.as_ref(),
                                candidate[role].as_ref(),
                                0,
                                phase,
                            ) {
                                ordinary(candidate[role].as_ref(), phase, true);
                            }
                        }
                    }
                    rec.finish_nowait().unwrap().wait().unwrap();
                    assert_eq!(
                        read(&backend, layer.buffer.as_ref()),
                        meta,
                        "CPU resume overwrote sidecar"
                    );
                    for role in 0..2 {
                        let expected = read(&backend, baseline[role].as_ref());
                        assert!(expected[..4 * physical * N]
                            .iter()
                            .all(|&v| f32::from_bits(v).is_finite()));
                        assert!(expected[4 * physical * N..].iter().all(|&v| v == CANARY));
                        assert_eq!(read(&backend, candidate[role].as_ref()), expected,
                            "bit mismatch shared={shared} role={role} pairs={pairs} quarters={quarters}");
                    }
                }
            }
            // Reuse populated metadata to prove skipped/invalid checks clear stale commands.
            for invalid_id in [false, true] {
                let mut ids = routes(20);
                if invalid_id {
                    ids[0] = EXPERTS as u32;
                }
                backend
                    .upload(ids_buffer.as_ref(), bytemuck::cast_slice(&ids))
                    .unwrap();
                let mut seed = vec![0; crate::recorder::RESIDENT_CONTROL_BYTES / 4];
                seed[0] = u32::from(invalid_id);
                backend
                    .upload(control.as_ref(), bytemuck::cast_slice(&seed))
                    .unwrap();
                unsafe {
                    backend.shared.device.reset_event(event.1).unwrap();
                }
                let rec = backend.recorder().unwrap();
                rec.gpu_resident_hit_check_pair(
                    ids_buffer.as_ref(),
                    tape.as_ref(),
                    control.as_ref(),
                    journal.as_ref(),
                    [40, 512, 0, 512, 1024, 0, 10, if shared { 0x400 } else { 0 }],
                    event.1,
                    &layer,
                );
                rec.finish_nowait().unwrap().wait().unwrap();
                let meta = read(&backend, layer.buffer.as_ref());
                assert_eq!(meta[VALID], 0);
                for word in (COMMANDS..CONFIG).step_by(3) {
                    assert_eq!(meta[word], 0);
                }
                let notice = read(&backend, journal.as_ref());
                assert_eq!(
                    notice[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD],
                    if invalid_id { 3 } else { 0 }
                );
            }
        }
    }
}
