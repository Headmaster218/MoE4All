// Included in recorder::gpu_owned_tests; no model files or global debug switches.
#[test]
fn gpu_owned_recovery_seed_rejects_noncanonical_protocol_shapes() {
    let valid = [0, 1, 41, 7];
    assert!(owned_recovery_seed_shape_valid(
        RESIDENT_CONTROL_BYTES,
        RESIDENT_JOURNAL_BYTES,
        48,
        valid
    ));
    for control in [
        0,
        20,
        RESIDENT_CONTROL_BYTES - 4,
        RESIDENT_CONTROL_BYTES + 4,
        RESIDENT_CONTROL_BYTES + 1,
    ] {
        assert!(!owned_recovery_seed_shape_valid(
            control,
            RESIDENT_JOURNAL_BYTES,
            48,
            valid
        ));
    }
    for journal in [0, RESIDENT_JOURNAL_BYTES - 4, RESIDENT_JOURNAL_BYTES + 4] {
        assert!(!owned_recovery_seed_shape_valid(
            RESIDENT_CONTROL_BYTES,
            journal,
            48,
            valid
        ));
    }
    for candidates in [0, 32, 47, 49] {
        assert!(!owned_recovery_seed_shape_valid(
            RESIDENT_CONTROL_BYTES,
            RESIDENT_JOURNAL_BYTES,
            candidates,
            valid
        ));
    }
    for params in [
        [0, 0, 41, 7],
        [1, 1, 41, 7],
        [0, RESIDENT_MAX_LAYERS as u32 + 1, 41, 7],
        [1, 2, 41, 7],
    ] {
        assert!(!owned_recovery_seed_shape_valid(
            RESIDENT_CONTROL_BYTES,
            RESIDENT_JOURNAL_BYTES,
            48,
            params
        ));
    }
    assert!(owned_recovery_seed_shape_valid(
        RESIDENT_CONTROL_BYTES,
        RESIDENT_MAX_LAYERS * RESIDENT_JOURNAL_BYTES,
        48,
        [
            RESIDENT_MAX_LAYERS as u32 - 1,
            RESIDENT_MAX_LAYERS as u32,
            41,
            7
        ]
    ));
}

#[test]
#[ignore = "central GPU runner only; exact old-control/journal/current-generation guard and sticky failure"]
fn gpu_owned_recovery_seed_checks_all_old_epoch_fields_and_stays_invalid() {
    let be = backend();
    let control = be.alloc_mapped_arena_bda(RESIDENT_CONTROL_BYTES).unwrap().0;
    let journal = be
        .alloc_mapped_arena_bda(RESIDENT_MAX_LAYERS * RESIDENT_JOURNAL_BYTES)
        .unwrap()
        .0;
    let nodes = be.alloc_mapped_arena_bda(48).unwrap().0;
    for ordinal in [0usize, 17, RESIDENT_MAX_LAYERS - 1] {
        for fault in 0..16 {
            if ordinal != 0 && fault != 0 {
                continue;
            }
            let mut old = vec![0x5a5a_5a5a; RESIDENT_CONTROL_BYTES / 4];
            old[..5].copy_from_slice(&[0, ordinal as u32, ordinal as u32 + 1, 0, 1]);
            old[RESIDENT_SEED_WORD] = 1;
            let mut words = vec![0; RESIDENT_MAX_LAYERS * RESIDENT_JOURNAL_WORDS];
            let base = ordinal * RESIDENT_JOURNAL_WORDS;
            words[base + RESIDENT_JOURNAL_JOB_COUNT_WORD
                ..base + RESIDENT_JOURNAL_JOB_COUNT_WORD + 3]
                .copy_from_slice(&[1, 41, 7]);
            words[base + RESIDENT_JOURNAL_STATUS_WORD] = 5;
            let mut header = [4, 2, 6, 41, 7, 3, 7, 13, 0, 0, 2, 0];
            let mut params = [ordinal as u32, ordinal as u32 + 1, 41, 7];
            match fault {
                0 => (),
                1 => old[0] = 1,
                2 => old[1] += 1,
                3 => old[2] += 1,
                4 => old[3] = 1,
                5 => words[base + RESIDENT_JOURNAL_STATUS_WORD] = 4,
                6 => words[base + RESIDENT_JOURNAL_JOB_COUNT_WORD + 1] ^= 1,
                7 => words[base + RESIDENT_JOURNAL_JOB_COUNT_WORD + 2] ^= 1,
                8 => header[0] = 2,
                9 => header[3] ^= 1,
                10 => header[4] ^= 1,
                11 => words[base + RESIDENT_JOURNAL_JOB_COUNT_WORD] = 0,
                12 => {
                    words[base + RESIDENT_JOURNAL_JOB_COUNT_WORD] =
                        (RESIDENT_JOURNAL_ID_WORDS * 3 + 1) as u32
                }
                13 => params[2] ^= 1,
                14 => params[3] ^= 1,
                15 => {
                    params[0] = 1;
                    params[1] = 2;
                }
                _ => unreachable!(),
            }
            upload_words(&be, control.as_ref(), &old);
            upload_words(&be, journal.as_ref(), &words);
            upload_words(&be, nodes.as_ref(), &header);
            let rec = be.recorder().unwrap();
            rec.gpu_owned_recovery_seed(control.as_ref(), journal.as_ref(), nodes.as_ref(), params);
            // A repeated legitimate recovery cannot clear a previous failed guard. No
            // arbitrary fresh-forward seed is inserted between these guarded operations.
            if fault != 0 {
                rec.gpu_owned_recovery_seed(
                    control.as_ref(),
                    journal.as_ref(),
                    nodes.as_ref(),
                    params,
                );
            }
            rec.publish_host_reads();
            rec.finish().unwrap();
            let actual = read_words(&be, control.as_ref());
            let mut expected = old;
            expected[4] = 0;
            expected[RESIDENT_SEED_WORD] = u32::from(fault == 0);
            expected[0] = u32::from(fault == 0);
            if fault == 0 {
                expected[1] = NONE;
                expected[2] = 0;
            } else {
                expected[3] = 1;
            }
            assert_eq!(actual, expected, "ordinal={ordinal} fault={fault}");
            assert_eq!(read_words(&be, journal.as_ref()), words);
            assert_eq!(read_words(&be, nodes.as_ref()), header);
        }
    }
}

fn record_recovery_fixture_prepare(
    be: &VulkanBackend,
    rec: &Recorder<'_>,
    fixture: &Fixture,
    recovery: bool,
) {
    let bindings: Vec<_> = fixture
        .buffers
        .iter()
        .map(|b| Recorder::vkb_exact(b.as_ref()))
        .collect();
    let push = gpu_copy_prepare_push(
        [3, 4, 0, 4, 8, 0, 3, 0],
        true,
        [0, 0, 4, u32::from(recovery), 0, 0, 0, 0, 0],
        2,
    );
    let prepare = be.kernel(
        "gpu_owned_prepare",
        crate::gemm::gpu_owned_prepare_spv(),
        8,
        GPU_COPY_PREPARE_PUSH_BYTES,
    );
    rec.owned_copy_barrier(true, false);
    rec.dispatch(prepare, &bindings, 8, bytemuck::cast_slice(&push), 1);
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
    fixture.publish(rec, 0);
    rec.owned_copy_barrier(false, false);
}

#[test]
#[ignore = "central GPU runner only; guarded READY, sticky invalid, model predicate and submit-before-old-retire"]
fn gpu_owned_recovery_seed_guards_publication_and_queues_before_old_retirement() {
    let be = backend();
    for fault in 0..3 {
        let invalid = fault != 0;
        let fixture = Fixture::new_fifo(&be, 3, true);
        if fault == 1 {
            let mut control = read_words(&be, fixture.buffers[2].as_ref());
            control[3] = 1;
            upload_words(&be, fixture.buffers[2].as_ref(), &control);
        }
        let output = be.alloc_mapped_arena_bda(16).unwrap().0;
        be.upload(output.as_ref(), bytemuck::cast_slice(&[19.0f32; 4]))
            .unwrap();
        let old = be.recorder().unwrap();
        old.seed_barrier();
        record_recovery_fixture_prepare(&be, &old, &fixture, false);
        let old = old.finish_nowait().unwrap();
        // The known CPU-only job targets slot 3. No GPU payload reads or writes these
        // bytes; only recovery publishes them. Real producer prevalidation supplies this proof.
        let pointer = as_vk_buf(fixture.destination.as_ref())
            .unwrap()
            .mapped_ptr()
            .unwrap();
        crate::copy_to_mapped(
            bytemuck::cast_slice(&fixture.source_words[24..28]),
            unsafe { pointer.add(48) },
        );
        let rec = be.recorder().unwrap();
        rec.seed_barrier();
        rec.gpu_owned_recovery_seed(
            fixture.buffers[2].as_ref(),
            fixture.buffers[3].as_ref(),
            fixture.buffers[5].as_ref(),
            [0, 1, if fault == 2 { 42 } else { 41 }, 7],
        );
        rec.begin_immutable_conditional_at(fixture.buffers[2].as_ref(), RESIDENT_SEED_WORD);
        rec.owned_copy_barrier(true, false);
        fixture.publish(&rec, 1);
        rec.owned_copy_barrier(false, false);
        rec.end_conditional();
        // This is the actual recovery ordering: guarded mode1 -> owner prepare -> FFN
        // under the freshly written layer predicate, never an unconditional second seed.
        record_recovery_fixture_prepare(&be, &rec, &fixture, true);
        rec.begin_immutable_conditional_at(fixture.buffers[2].as_ref(), 5);
        rec.scale(output.as_ref(), 0.0, 4);
        rec.end_conditional();
        rec.publish_host_reads();
        let recovery = rec.finish_nowait().unwrap();
        // Both submissions are leased here. Retiring the old query/fence only happens
        // after recovery has reached the queue; FIFO plus the explicit WAR protects readers.
        old.wait().unwrap();
        recovery.wait().unwrap();
        let control = read_words(&be, fixture.buffers[2].as_ref());
        let nodes = read_words(&be, fixture.buffers[5].as_ref());
        let overlay = read_words(&be, fixture.buffers[6].as_ref());
        let words = read_words(&be, fixture.buffers[3].as_ref());
        let mut values = [0f32; 4];
        be.download(output.as_ref(), bytemuck::cast_slice_mut(&mut values))
            .unwrap();
        if invalid {
            assert_eq!(control[0], 0);
            assert_eq!(control[3], 1);
            assert_eq!(control[RESIDENT_SEED_WORD], 0);
            assert_eq!(words[RESIDENT_JOURNAL_STATUS_WORD], 0);
            assert_eq!(values, [19.0; 4]);
            if fault == 1 {
                // The failed old prepare never admitted any new destinations.
                assert_eq!(nodes, bytemuck::cast_slice::<[u32; 4], u32>(&fixture.nodes));
            } else {
                // An admitted CPU-only destination remains COPYING when a later guard
                // rejects Started's generation; neither mode1 nor the FFN may run.
                assert_eq!(nodes[(SLOT_BASE + 3) * 4 + 3], 2);
                assert_eq!(overlay[6 * 4 + 2], 2);
            }
            assert_ne!(overlay[6 * 4 + 2], 1);
        } else {
            assert_eq!(&control[..4], &[1, NONE, 1, 0]);
            assert_eq!(control[RESIDENT_SEED_WORD], 1);
            assert_eq!(words[RESIDENT_JOURNAL_STATUS_WORD], 1);
            assert_eq!(nodes[(SLOT_BASE + 3) * 4 + 3], 1);
            assert_eq!(overlay[6 * 4 + 2], 1);
            assert_eq!(values, [0.0; 4]);
            assert_eq!(
                &read_words(&be, fixture.destination.as_ref())[12..16],
                &fixture.source_words[24..28]
            );
        }
    }
}
