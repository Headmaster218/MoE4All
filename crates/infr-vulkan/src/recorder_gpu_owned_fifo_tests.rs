// Included in gpu_owned_tests; the independent slot/timestamp CPU reference is shared.
#[test]
#[ignore = "central GPU runner only; full owner-window journal/jobs/predicates/payload boundaries"]
fn gpu_owned_fifo_full_window_ordinals_keep_payload_and_lru_exact() {
    let be = backend();
    for ordinal in [16u32, 31, 47, 127] {
        let mut fixture = Fixture::new_fifo(&be, 3, false);
        let layers = ordinal as usize + 1;
        for (binding, bytes) in [
            (3, layers * RESIDENT_JOURNAL_BYTES),
            (7, layers * RESIDENT_JOURNAL_ID_WORDS * 3 * 8 * 4),
        ] {
            fixture.buffers[binding] = be.alloc_mapped_arena_bda(bytes).unwrap().0;
            be.upload(fixture.buffers[binding].as_ref(), &vec![0; bytes])
                .unwrap();
        }
        let bindings: Vec<_> = fixture
            .buffers
            .iter()
            .map(|buffer| Recorder::vkb(buffer.as_ref()))
            .collect();
        let base = gpu_copy_prepare_push(
            [3, 4, 0, 4, 8, ordinal, 3, 0],
            true,
            [0, 0, 4, 0, 0, 0, 0, 0, 0],
            2,
        );
        let geometry = [0, 16, 0, 3, 0, 16, 0, 3, 1, 16, 3, 3];
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        let fast = be.kernel(
            "gpu_owned_prepare_fast",
            crate::gemm::gpu_owned_prepare_fast_spv(false),
            8,
            GPU_OWNED_FAST_PUSH_BYTES,
        );
        rec.dispatch(
            fast,
            &bindings,
            8,
            bytemuck::cast_slice(&gpu_owned_fast_push(&base, &geometry)),
            1,
        );
        rec.owned_copy_barrier(false, false);
        let slow = be.kernel(
            "gpu_owned_prepare_slow",
            crate::gemm::gpu_owned_prepare_slow_spv(false),
            8,
            GPU_COPY_PREPARE_PUSH_BYTES,
        );
        rec.dispatch(slow, &bindings, 8, bytemuck::cast_slice(&base), 1);
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
            bytemuck::bytes_of(&ordinal),
            2,
        );
        rec.owned_copy_barrier(false, false);
        let publish = be.kernel(
            "gpu_owned_publish",
            crate::gemm::gpu_owned_publish_spv(),
            4,
            8,
        );
        rec.dispatch(
            publish,
            &[bindings[3], bindings[7], bindings[5], bindings[6]],
            4,
            bytemuck::cast_slice(&[ordinal, 0]),
            1,
        );
        rec.owned_copy_barrier(false, true);
        rec.finish().unwrap();
        let journal = read_words(&be, fixture.buffers[3].as_ref());
        let offset = ordinal as usize * RESIDENT_JOURNAL_WORDS;
        assert!(journal[..offset].iter().all(|&word| word == 0));
        assert_eq!(journal[offset + RESIDENT_JOURNAL_STATUS_WORD], 4);
        assert_eq!(journal[offset + RESIDENT_JOURNAL_JOB_COUNT_WORD], 4);
        let control = read_words(&be, fixture.buffers[2].as_ref());
        assert_eq!(&control[..5], &[1, NONE, ordinal + 1, 0, 1]);
        assert_eq!(
            &control[5 + ordinal as usize * 2..7 + ordinal as usize * 2],
            &[1, 1]
        );
        let jobs = read_words(&be, fixture.buffers[7].as_ref());
        let job_offset = ordinal as usize * RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
        assert!(jobs[..job_offset].iter().all(|&word| word == 0));
        let mut reference = CpuReference::new();
        assert_eq!(reference.request(&[0, 1, 2], [0, 0, 4]), 4);
        reference.assert_fixture(&be, &fixture);
    }
}

#[test]
fn gpu_owned_fast_push_preserves_protocol_and_fits_portable_push_limit() {
    let base = std::array::from_fn(|index| index as u32 + 100);
    let geometry = [0, 4096, 0, 1024, 0, 4096, 0, 1024, 1, 8192, 1024, 512];
    let push = gpu_owned_fast_push(&base, &geometry);
    assert_eq!(&push[..19], &base);
    assert_eq!(&push[19..], &geometry);
    assert_eq!(bytemuck::cast_slice::<u32, u8>(&push).len(), 124);
    assert_eq!(GPU_OWNED_FAST_PUSH_BYTES, 124);
    assert!(GPU_OWNED_FAST_PUSH_BYTES <= 128);
}

impl Fixture {
    fn new_fifo(be: &VulkanBackend, count: usize, cpu_job: bool) -> Self {
        let mut fixture = Self::new(be, count, cpu_job);
        let mut nodes = fixture.nodes[..HEAP_BASE].to_vec();
        nodes[0][0] = 4;
        for pool in 0..2 {
            let header = nodes.len();
            nodes[POOL_BASE + pool * 2][0] = header as u32;
            nodes.push([0, 2, 6, 0]);
            nodes.push([1, 0, (pool * 3) as u32, 0]);
            nodes.push([2, 0, (pool * 3 + 1) as u32, 0]);
            nodes.resize(header + 7, [0; 4]);
        }
        fixture.buffers[5] = be.alloc_mapped_arena_bda(nodes.len() * 16).unwrap().0;
        be.upload(fixture.buffers[5].as_ref(), bytemuck::cast_slice(&nodes))
            .unwrap();
        fixture.nodes = nodes;
        let mut payload = vec![0xa5a5_a5a5; 24];
        for (slot, block) in [(0, 3), (1, 1), (3, 7), (4, 5)] {
            payload[slot * 4..slot * 4 + 4]
                .copy_from_slice(&fixture.source_words[block * 4..block * 4 + 4]);
        }
        upload_words(be, fixture.destination.as_ref(), &payload);
        fixture
    }
}

fn assert_fifo(nodes: &[u32]) {
    assert_eq!(&nodes[..3], &[4, 2, 6]);
    assert_eq!(nodes.len(), (HEAP_BASE + 14) * 4);
    let clock = stamp(&nodes[10..12]);
    for pool in 0..2 {
        let header = &nodes[(POOL_BASE + pool * 2) * 4..];
        let base = HEAP_BASE + pool * 7;
        assert_eq!(header[0] as usize, base);
        let queue = &nodes[base * 4..base * 4 + 4];
        assert!(queue[0] <= queue[1] && queue[1] <= queue[2]);
        assert_eq!(&queue[2..], &[6, 0]);
        let residents = lru_slots(nodes, pool);
        assert_eq!(header[1] as usize, residents.len());
        let mut live = std::collections::HashSet::new();
        let mut previous = 0;
        for index in queue[0] as usize..queue[1] as usize {
            let entry = &nodes[(base + 1 + index) * 4..];
            let slot = entry[2] as usize;
            assert!((pool * 3..pool * 3 + 3).contains(&slot));
            assert_eq!(entry[3], 0);
            let value = stamp(entry);
            assert!(value > previous && value <= clock);
            previous = value;
            let actual = &nodes[(SLOT_BASE + slot) * 4..];
            assert!(value <= stamp(actual));
            if value == stamp(actual) {
                assert!(matches!(actual[3], 1 | 2));
                assert!(live.insert(slot));
            }
        }
        assert_eq!(live, residents.into_iter().collect());
    }
}

#[test]
#[ignore = "central GPU runner only; exact FIFO aliases, eviction, compaction, carry, payload"]
fn gpu_owned_fifo_matches_cpu_lru_across_compaction() {
    let be = backend();
    let fixture = Fixture::new_fifo(&be, 2, false);
    let mut reference = CpuReference::new();
    let mut data = vec![0xa5a5_a5a5; 24];
    for (slot, block) in reference.blocks.iter().enumerate() {
        if let Some(block) = *block {
            data[slot * 4..slot * 4 + 4]
                .copy_from_slice(&fixture.source_words[block * 4..block * 4 + 4]);
        }
    }
    upload_words(&be, fixture.destination.as_ref(), &data);
    reference.clock = u64::from(u32::MAX) - 2;
    let mut seed = fixture.nodes.clone();
    seed[2] = [0, 0, reference.clock as u32, (reference.clock >> 32) as u32];
    be.upload(fixture.buffers[5].as_ref(), bytemuck::cast_slice(&seed))
        .unwrap();
    let patterns = [[0, 0, 4], [4, 0, 4], [0, 4, 0], [4, 4, 0]];
    let mut rng = 0x61c8_8647u32;
    for iteration in 0..80 {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let count = 1 + (rng as usize & 1);
        let ids = [(rng >> 8) & 3, (rng >> 16) & 3];
        let roles = patterns[(rng >> 24) as usize & 3];
        fixture.reset_inputs(&be, &ids[..count]);
        let misses = reference.request(&ids[..count], roles);
        fixture.run_with_roles(&be, count as u32, roles);
        let journal = read_words(&be, fixture.buffers[3].as_ref());
        assert_eq!(
            journal[RESIDENT_JOURNAL_STATUS_WORD],
            if misses == 0 { 1 } else { 4 },
            "iteration {iteration}"
        );
        assert_eq!(journal[RESIDENT_JOURNAL_JOB_COUNT_WORD], misses as u32);
        reference.assert_fixture(&be, &fixture);
    }
}

#[test]
#[ignore = "central GPU runner only; FIFO mixed CPU/GPU publication"]
fn gpu_owned_fifo_mixed_cpu_publication_preserves_order() {
    let be = backend();
    let fixture = Fixture::new_fifo(&be, 3, true);
    fixture.run(&be, 3);
    let journal = read_words(&be, fixture.buffers[3].as_ref());
    assert_eq!(journal[RESIDENT_JOURNAL_STATUS_WORD], 5);
    assert_eq!(journal[RESIDENT_JOURNAL_JOB_COUNT_WORD], 4);
    assert_eq!(journal[RESIDENT_JOURNAL_MASK_WORD], 2);
    assert_eq!(&read_words(&be, fixture.buffers[0].as_ref())[3..5], &[2, 0]);
    assert_eq!(
        &read_words(&be, fixture.buffers[2].as_ref())[..7],
        &[0, 0, 1, 0, 1, 0, 1]
    );
    let nodes = read_words(&be, fixture.buffers[5].as_ref());
    assert_eq!(nodes[(SLOT_BASE + 3) * 4 + 3], 2);
    assert_fifo(&nodes);
    let pointer = as_vk_buf(fixture.destination.as_ref())
        .unwrap()
        .mapped_ptr()
        .unwrap();
    crate::copy_to_mapped(
        bytemuck::cast_slice(&fixture.source_words[24..28]),
        unsafe { pointer.add(3 * 16) },
    );
    let rec = be.recorder().unwrap();
    rec.owned_copy_barrier(true, false);
    fixture.publish(&rec, 1);
    rec.owned_copy_barrier(false, true);
    rec.finish().unwrap();
    let data = read_words(&be, fixture.destination.as_ref());
    assert_eq!(&data[12..16], &fixture.source_words[24..28]);
    let mut reference = CpuReference::new();
    reference.request(&[0, 1, 2], [0, 0, 4]);
    reference.assert_fixture(&be, &fixture);
}

#[test]
#[ignore = "central GPU runner only; FIFO preflight failure cannot mutate ownership"]
fn gpu_owned_fifo_invalid_queue_is_atomic() {
    let be = backend();
    for corruption in 0..5 {
        let fixture = Fixture::new_fifo(&be, 3, false);
        let mut nodes = fixture.nodes.clone();
        match corruption {
            0 => nodes[HEAP_BASE][1] = 7,
            1 => nodes[HEAP_BASE][0] = 3,
            2 => nodes[HEAP_BASE][2] = 5,
            3 => nodes[HEAP_BASE + 1][2] = 6,
            4 => nodes[HEAP_BASE + 1][3] = 1,
            _ => unreachable!(),
        }
        be.upload(fixture.buffers[5].as_ref(), bytemuck::cast_slice(&nodes))
            .unwrap();
        let before_overlay = read_words(&be, fixture.buffers[6].as_ref());
        let before_payload = read_words(&be, fixture.destination.as_ref());
        fixture.run(&be, 3);
        let journal = read_words(&be, fixture.buffers[3].as_ref());
        assert_eq!(
            journal[RESIDENT_JOURNAL_STATUS_WORD], 3,
            "corruption {corruption}"
        );
        assert_eq!(
            read_words(&be, fixture.buffers[5].as_ref()),
            bytemuck::cast_slice::<[u32; 4], u32>(&nodes)
        );
        assert_eq!(read_words(&be, fixture.buffers[6].as_ref()), before_overlay);
        assert_eq!(
            read_words(&be, fixture.destination.as_ref()),
            before_payload
        );
    }
}

#[test]
#[ignore = "central GPU runner only; FIFO compaction must retain in-flight entries"]
fn gpu_owned_fifo_copying_survives_hot_compaction_and_is_not_a_victim() {
    let be = backend();
    let fixture = Fixture::new_fifo(&be, 2, false);
    let mut nodes = fixture.nodes.clone();
    nodes[SLOT_BASE][3] = 2;
    let mut overlay = read_words(&be, fixture.buffers[6].as_ref());
    overlay[3 * 4 + 2] = 2;
    be.upload(fixture.buffers[5].as_ref(), bytemuck::cast_slice(&nodes))
        .unwrap();
    upload_words(&be, fixture.buffers[6].as_ref(), &overlay);
    // Repeated exact hits force many compactions without touching the pending block.
    for _ in 0..16 {
        fixture.reset_inputs(&be, &[1]);
        fixture.run_with_roles(&be, 1, [0, 0, 4]);
        let actual = read_words(&be, fixture.buffers[5].as_ref());
        assert_fifo(&actual);
        assert_eq!(&actual[SLOT_BASE * 4..SLOT_BASE * 4 + 4], &[1, 0, 3, 2]);
    }
    for id in [2, 0] {
        fixture.reset_inputs(&be, &[id]);
        fixture.run_with_roles(&be, 1, [0, 0, 4]);
        let actual = read_words(&be, fixture.buffers[5].as_ref());
        assert_fifo(&actual);
        assert_eq!(&actual[SLOT_BASE * 4..SLOT_BASE * 4 + 4], &[1, 0, 3, 2]);
    }
}
