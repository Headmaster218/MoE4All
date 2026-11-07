#[test]
#[ignore = "central GPU runner only; GPU-owned overlap masks, payload, and CPU recovery"]
fn gpu_owned_overlap_preserves_union_masks_aliases_and_cpu_recovery() {
    let be = backend();
    for rows in [1, 2, infr_core::backend::MAX_DECODE_ROWS] {
        for (routed, cpu_job, all_cpu) in [
            ([1, 1], false, false),
            ([1, 2], false, false),
            ([0, 2], false, false),
            ([1, 2], true, false),
            ([1, 2], true, true),
        ] {
            for roles in [[0u32, 0, 4], [0, 4, 4]] {
                for ordinal in [0usize, RESIDENT_MAX_LAYERS - 1] {
                    let count = rows * 2;
                    let mut fixture = Fixture::new(&be, count, cpu_job);
                    if all_cpu {
                        let mut sources = read_words(&be, fixture.buffers[4].as_ref());
                        for source in sources.chunks_exact_mut(4) {
                            source[..2].fill(0);
                        }
                        upload_words(&be, fixture.buffers[4].as_ref(), &sources);
                    }
                    for (binding, bytes) in [
                        (0, (count + 2 * rows) * 4),
                        (3, (ordinal + 1) * RESIDENT_JOURNAL_BYTES),
                        (7, (ordinal + 1) * RESIDENT_JOURNAL_ID_WORDS * 3 * 32),
                    ] {
                        fixture.buffers[binding] = be.alloc_mapped_arena_bda(bytes).unwrap().0;
                        upload_words(&be, fixture.buffers[binding].as_ref(), &vec![0; bytes / 4]);
                    }
                    let ids: Vec<u32> = (0..rows).flat_map(|_| routed).collect();
                    fixture.reset_inputs(&be, &ids);
                    let predicates = be.alloc_mapped_arena_bda((ordinal + 1) * 16).unwrap().0;
                    // Reused slots must be explicitly reset even for all-hit or skipped layers.
                    upload_words(&be, predicates.as_ref(), &vec![99; (ordinal + 1) * 4]);
                    let mut push = gpu_copy_prepare_push(
                        [count as u32, 4, 0, 4, 8, ordinal as u32, 2, 1 << 2],
                        true,
                        [roles[0], roles[1], roles[2], 0, 0, 0, 0, 0, 0],
                        2,
                    );
                    let prepare = |rec: &Recorder<'_>, push: &[u32; 19]| {
                        let mut bindings: Vec<_> = fixture
                            .buffers
                            .iter()
                            .map(|buffer| Recorder::vkb(buffer.as_ref()))
                            .collect();
                        bindings.push(Recorder::vkb(predicates.as_ref()));
                        let kernel = be.kernel(
                            "gpu_owned_prepare_reference_overlap",
                            crate::gemm::gpu_owned_prepare_reference_overlap_spv(),
                            9,
                            GPU_COPY_PREPARE_PUSH_BYTES,
                        );
                        rec.owned_copy_barrier(true, false);
                        rec.dispatch(kernel, &bindings, 9, bytemuck::cast_slice(push), 1);
                    };
                    let read = |binding: usize| read_words(&be, fixture.buffers[binding].as_ref());
                    let rec = be.recorder().unwrap();
                    prepare(&rec, &push);
                    rec.publish_host_reads();
                    rec.finish().unwrap();
                    let journal = read(3);
                    let base = ordinal * RESIDENT_JOURNAL_WORDS;
                    let status = journal[base + RESIDENT_JOURNAL_STATUS_WORD];
                    let jobs_count = journal[base + RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
                    let expected_hit = routed
                        .iter()
                        .enumerate()
                        .fold(0u32, |mask, (i, id)| mask | (u32::from(*id == 1) << i));
                    let expected_status = if expected_hit == 3 {
                        1
                    } else if cpu_job {
                        5
                    } else {
                        4
                    };
                    assert_eq!(status, expected_status);
                    let masks = read(0);
                    for row in 0..rows {
                        assert_eq!(masks[count + row], expected_hit | (1 << 2));
                        assert_eq!(masks[count + rows + row], 3 ^ expected_hit);
                        assert_eq!(
                            journal[base + RESIDENT_JOURNAL_MASK_WORD + row],
                            expected_hit
                        );
                    }
                    let args = read_words(&be, predicates.as_ref());
                    assert_eq!(
                        &args[ordinal * 4..ordinal * 4 + 4],
                        &[
                            if status == 4 { jobs_count as u32 } else { 0 },
                            if jobs_count != 0 && !all_cpu { 2 } else { 0 },
                            1,
                            1,
                        ]
                    );
                    let control = read(2);
                    assert_eq!(control[0], u32::from(status != 5));
                    assert_eq!(control[6 + ordinal * 2], 1, "CPU fallback may run hits");
                    assert_eq!(
                        control[5 + ordinal * 2],
                        u32::from(status != 5),
                        "CPU fallback cannot commit residual"
                    );
                    let jobs = read(7);
                    let job_base = ordinal * RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
                    let before_nodes = read(5);
                    let before_overlay = read(6);
                    if routed.contains(&1) {
                        for role in roles {
                            let block = (role + 1) as usize;
                            let entry = &before_overlay[block * 4..block * 4 + 4];
                            assert_eq!(entry[2], 1, "current hit must remain READY");
                            assert_eq!(
                                before_nodes[(SLOT_BASE + entry[3] as usize) * 4 + 2],
                                block as u32,
                                "admission cannot overwrite any role's current hit"
                            );
                        }
                    }
                    let mut unique_blocks = std::collections::HashSet::new();
                    let mut unique_slots = std::collections::HashSet::new();
                    for job in jobs[job_base..job_base + jobs_count * 8].chunks_exact(8) {
                        assert!(unique_blocks.insert(job[5]));
                        assert!(unique_slots.insert(job[6]));
                        assert_eq!(before_overlay[job[5] as usize * 4 + 2], 2);
                        assert_eq!(before_nodes[(SLOT_BASE + job[6] as usize) * 4 + 3], 2);
                        // Requested READY blocks from every row/role remain protected.
                        assert!(!roles.iter().any(|role| job[5] == role + 1));
                    }
                    let gate = be.alloc_mapped_arena_bda(rows * 3 * 4).unwrap().0;
                    let up = be.alloc_mapped_arena_bda(rows * 3 * 4).unwrap().0;
                    let activated = be.alloc_mapped_arena_bda(rows * 3 * 4).unwrap().0;
                    be.upload(gate.as_ref(), bytemuck::cast_slice(&vec![1f32; rows * 3]))
                        .unwrap();
                    be.upload(up.as_ref(), bytemuck::cast_slice(&vec![2f32; rows * 3]))
                        .unwrap();
                    be.upload(
                        activated.as_ref(),
                        bytemuck::cast_slice(&vec![-99f32; rows * 3]),
                    )
                    .unwrap();
                    let rec = be.recorder().unwrap();
                    let scope = rec.independent_dispatch_scope();
                    rec.gpu_resident_silu_bank(
                        gate.as_ref(),
                        up.as_ref(),
                        fixture.buffers[0].as_ref(),
                        activated.as_ref(),
                        [rows as u32, 2, 3, 1],
                        false,
                    );
                    let kernel = be.kernel(
                        "gpu_owned_overlap_test_payload",
                        crate::gemm::gpu_resident_copy_payload_spv(false),
                        2,
                        4,
                    );
                    let args_buffer = Recorder::vk_of(predicates.as_ref());
                    // A fresh recording consumes the GPU-produced indirect triple.
                    rec.dispatch_indirect(
                        kernel,
                        &[
                            Recorder::vkb(fixture.buffers[3].as_ref()),
                            Recorder::vkb(fixture.buffers[7].as_ref()),
                        ],
                        0,
                        bytemuck::bytes_of(&(ordinal as u32)),
                        args_buffer.buffer,
                        (args_buffer.sub_offset + (ordinal * 4 + 1) * 4) as u64,
                    );
                    drop(scope);
                    let publish = |rec: &Recorder<'_>, mode: u32| {
                        let kernel = be.kernel(
                            "gpu_owned_publish",
                            crate::gemm::gpu_owned_publish_spv(),
                            4,
                            8,
                        );
                        let bindings =
                            [3, 7, 5, 6].map(|i| Recorder::vkb(fixture.buffers[i].as_ref()));
                        rec.dispatch(
                            kernel,
                            &bindings,
                            2,
                            bytemuck::cast_slice(&[ordinal as u32, mode]),
                            1,
                        );
                        rec.owned_copy_barrier(false, false);
                    };
                    publish(&rec, 0);
                    if status == 4 {
                        rec.gpu_resident_silu_bank(
                            gate.as_ref(),
                            up.as_ref(),
                            fixture.buffers[0].as_ref(),
                            activated.as_ref(),
                            [rows as u32, 2, 3, 1],
                            true,
                        );
                    }
                    rec.publish_host_reads();
                    rec.finish().unwrap();
                    let mut values = vec![0f32; rows * 3];
                    be.download(activated.as_ref(), bytemuck::cast_slice_mut(&mut values))
                        .unwrap();
                    for row in 0..rows {
                        for slot in 0..3 {
                            let hit = slot == 2 || expected_hit & (1 << slot) != 0;
                            let expected = if hit || status == 4 {
                                2.0 / (1.0 + (-1f32).exp())
                            } else {
                                -99.0
                            };
                            assert!((values[row * 3 + slot] - expected).abs() < 1e-5);
                        }
                    }
                    let overlay = read(6);
                    let mut data = read_words(&be, fixture.destination.as_ref());
                    for job in jobs[job_base..job_base + jobs_count * 8].chunks_exact(8) {
                        let mapped = job[0] != 0 || job[1] != 0;
                        let block = job[5] as usize;
                        let slot = job[6] as usize;
                        assert_eq!(overlay[block * 4 + 2], if mapped { 1 } else { 2 });
                        if mapped {
                            assert_eq!(
                                &data[slot * 4..slot * 4 + 4],
                                &fixture.source_words[block * 4..block * 4 + 4]
                            );
                        } else {
                            data[slot * 4..slot * 4 + 4]
                                .copy_from_slice(&fixture.source_words[block * 4..block * 4 + 4]);
                        }
                    }
                    if status == 5 {
                        upload_words(&be, fixture.destination.as_ref(), &data);
                        let rec = be.recorder().unwrap();
                        rec.owned_copy_barrier(true, false);
                        publish(&rec, 1);
                        rec.publish_host_reads();
                        rec.finish().unwrap();
                        let mut control = vec![0; RESIDENT_CONTROL_BYTES / 4];
                        control[0] = 1;
                        control[1] = NONE;
                        upload_words(&be, fixture.buffers[2].as_ref(), &control);
                        push[12] = 1; // Recovery must not repeat timestamps or counters.
                        let rec = be.recorder().unwrap();
                        prepare(&rec, &push);
                        rec.publish_host_reads();
                        rec.finish().unwrap();
                        let recovered = read(5);
                        assert_eq!(&recovered[8..12], &before_nodes[8..12]);
                        let journal = read(3);
                        assert_eq!(journal[base + RESIDENT_JOURNAL_STATUS_WORD], 1);
                        assert_eq!(journal[base + RESIDENT_JOURNAL_JOB_COUNT_WORD], 0);
                        let recovered_args = read_words(&be, predicates.as_ref());
                        assert_eq!(&recovered_args[ordinal * 4..ordinal * 4 + 4], &[0, 0, 1, 1]);
                        for row in 0..rows {
                            assert_eq!(read(0)[count + row], 7);
                            assert_eq!(read(0)[count + rows + row], 0);
                        }
                    }
                    assert_heaps(&read(5));
                }
            }
        }
    }
}

#[test]
#[ignore = "central GPU runner only; expanded resident control seed ABI"]
fn gpu_owned_resident_seed_uses_the_final_control_word() {
    let be = backend();
    let fixture = Fixture::new(&be, 2, false);
    upload_words(
        &be,
        fixture.buffers[2].as_ref(),
        &vec![0xabcdef01; RESIDENT_CONTROL_BYTES / 4],
    );
    let rec = be.recorder().unwrap();
    rec.gpu_resident_check(
        fixture.buffers[0].as_ref(),
        fixture.buffers[1].as_ref(),
        fixture.buffers[2].as_ref(),
        fixture.buffers[3].as_ref(),
        [0; 6],
    );
    rec.publish_host_reads();
    rec.finish().unwrap();
    let control = read_words(&be, fixture.buffers[2].as_ref());
    assert_eq!(&control[..4], &[1, NONE, 0, 0]);
    assert_eq!(control[RESIDENT_SEED_WORD], 1);
    assert!(control[5..RESIDENT_SEED_WORD]
        .iter()
        .all(|word| *word == 0xabcdef01));
}

#[test]
#[ignore = "central GPU runner only; invalid and skipped overlap admission must not mutate ownership"]
fn gpu_owned_overlap_invalid_and_skip_clear_published_work_without_mutation() {
    let be = backend();
    for (ids, used, run, status) in [([4, 1], 2, 1, 3), ([1, 2], 0, 1, 3), ([1, 2], 2, 0, 0)] {
        let fixture = Fixture::new(&be, 2, false);
        fixture.reset_inputs(&be, &ids);
        let nodes = read_words(&be, fixture.buffers[5].as_ref());
        let overlay = read_words(&be, fixture.buffers[6].as_ref());
        let mut control = vec![0; RESIDENT_CONTROL_BYTES / 4];
        control[0] = run;
        upload_words(&be, fixture.buffers[2].as_ref(), &control);
        let predicates = be.alloc_mapped_arena_bda(16).unwrap().0;
        upload_words(&be, predicates.as_ref(), &[99; 4]);
        let push = gpu_copy_prepare_push(
            [2, 4, 0, 4, 8, 0, used, 4],
            true,
            [0, 0, 4, 0, 0, 0, 0, 0, 0],
            2,
        );
        let mut bindings: Vec<_> = fixture
            .buffers
            .iter()
            .map(|buffer| Recorder::vkb(buffer.as_ref()))
            .collect();
        bindings.push(Recorder::vkb(predicates.as_ref()));
        let rec = be.recorder().unwrap();
        let kernel = be.kernel(
            "gpu_owned_prepare_reference_overlap",
            crate::gemm::gpu_owned_prepare_reference_overlap_spv(),
            9,
            GPU_COPY_PREPARE_PUSH_BYTES,
        );
        rec.owned_copy_barrier(true, false);
        rec.dispatch(kernel, &bindings, 9, bytemuck::cast_slice(&push), 1);
        rec.publish_host_reads();
        rec.finish().unwrap();
        let journal = read_words(&be, fixture.buffers[3].as_ref());
        assert_eq!(journal[RESIDENT_JOURNAL_STATUS_WORD], status);
        assert_eq!(journal[RESIDENT_JOURNAL_JOB_COUNT_WORD], 0);
        assert_eq!(read_words(&be, predicates.as_ref()), [0, 0, 1, 1]);
        assert_eq!(read_words(&be, fixture.buffers[5].as_ref()), nodes);
        assert_eq!(read_words(&be, fixture.buffers[6].as_ref()), overlay);
    }
}

#[test]
#[ignore = "central GPU runner only; serial CPU-service recovery must preserve hit/shared outputs"]
fn gpu_owned_serial_cpu_recovery_masks_preserve_outputs_and_suballocation_bounds() {
    let be = backend();
    be.init_unified_vram(8 * 1024 * 1024).unwrap();
    let _prefix = be
        .alloc_unified(256, crate::unified::UnifiedVramClass::LlmRuntime)
        .unwrap();
    upload_words(&be, _prefix.as_ref(), &[0; 64]);
    for used in [1usize, 2, 16] {
        for rows in [1usize, 2, infr_core::backend::MAX_DECODE_ROWS] {
            let count = rows * used;
            let ids = be
                .alloc_unified(
                    (count + rows * 2) * 4,
                    crate::unified::UnifiedVramClass::LlmRuntime,
                )
                .unwrap();
            assert!(
                Recorder::vk_of(ids.as_ref()).sub_offset > 0,
                "exercise update-buffer offsets within the shared arena"
            );
            let all = (1u32 << used) - 1;
            let hits: Vec<u32> = (0..rows)
                .map(|row| match row % 3 {
                    0 => all & 0x5555,
                    1 => all,
                    _ => 0,
                })
                .collect();
            let mut initial_ids: Vec<u32> = (0..count as u32).collect();
            initial_ids.extend(hits.iter().map(|mask| mask | (1 << used)));
            initial_ids.extend(vec![0; rows]);
            upload_words(&be, ids.as_ref(), &initial_ids);
            let words = rows * (used + 1);
            let map = || be.alloc_mapped_arena_bda(words * 4).unwrap().0;
            let gate = map();
            let up = map();
            let new_gate = map();
            let new_up = map();
            let output = map();
            for (buffer, value) in [
                (gate.as_ref(), 1f32),
                (up.as_ref(), 2f32),
                (new_gate.as_ref(), 3f32),
                (new_up.as_ref(), 4f32),
                (output.as_ref(), -99f32),
            ] {
                be.upload(buffer, bytemuck::cast_slice(&vec![value; words]))
                    .unwrap();
            }
            let rec = be.recorder().unwrap();
            rec.owned_copy_barrier(true, false);
            rec.gpu_resident_silu_bank(
                gate.as_ref(),
                up.as_ref(),
                ids.as_ref(),
                output.as_ref(),
                [rows as u32, used as u32, (used + 1) as u32, 1],
                false,
            );
            // Give recovery different inputs: rerunning a hit or shared slot becomes observable.
            rec.copy(new_gate.as_ref(), 0, gate.as_ref(), 0, words * 4);
            rec.copy(new_up.as_ref(), 0, up.as_ref(), 0, words * 4);
            rec.gpu_owned_recovery_miss_masks(ids.as_ref(), count, used, &hits);
            rec.gpu_resident_silu_bank(
                gate.as_ref(),
                up.as_ref(),
                ids.as_ref(),
                output.as_ref(),
                [rows as u32, used as u32, (used + 1) as u32, 1],
                false,
            );
            rec.publish_host_reads();
            rec.finish().unwrap();
            let masks = read_words(&be, ids.as_ref());
            assert_eq!(&masks[..count], &initial_ids[..count]);
            assert_eq!(&masks[count + rows..count + rows * 2], &vec![0; rows]);
            let mut values = vec![0f32; words];
            be.download(output.as_ref(), bytemuck::cast_slice_mut(&mut values))
                .unwrap();
            for row in 0..rows {
                assert_eq!(masks[count + row], all ^ hits[row]);
                for slot in 0..=used {
                    let first_pass = slot == used || hits[row] & (1 << slot) != 0;
                    let expected = if first_pass {
                        2.0 / (1.0 + (-1f32).exp())
                    } else {
                        12.0 / (1.0 + (-3f32).exp())
                    };
                    assert!((values[row * (used + 1) + slot] - expected).abs() < 1e-5);
                }
            }
            assert_eq!(
                read_words(&be, _prefix.as_ref()),
                vec![0; 64],
                "the update must not target the buffer's absolute front"
            );
        }
    }
}

#[test]
#[ignore = "central GPU runner only; recovery mask input guards"]
fn gpu_owned_recovery_mask_guards_reject_invalid_inputs_before_recording() {
    let be = backend();
    let fixture = Fixture::new(&be, 2, false);
    let rec = be.recorder().unwrap();
    let before = rec.dispatches();
    for (count, used, masks) in [
        (0, 2, vec![0]),
        (2, 0, vec![0]),
        (2, 17, vec![0]),
        (3, 2, vec![0]),
        (2, 2, vec![]),
        (2, 2, vec![4]),
        (RESIDENT_JOURNAL_ID_WORDS + 1, 1, vec![0]),
        (4, 2, vec![0, 0]),
    ] {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rec.gpu_owned_recovery_miss_masks(fixture.buffers[0].as_ref(), count, used, &masks);
        }))
        .is_err());
        assert_eq!(rec.dispatches(), before);
    }
    rec.discard().unwrap();
}
