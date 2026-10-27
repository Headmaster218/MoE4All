// Included only in recorder::gpu_owned_tests. The production recorder is not switched here.
struct CopyPublishFixture {
    fixture: Fixture,
    ordinal: u32,
    stride: usize,
    observer: Box<dyn Buffer>,
}

impl CopyPublishFixture {
    fn new(
        be: &VulkanBackend,
        rows: usize,
        status: u32,
        all_cpu: bool,
        roles: [u32; 3],
        stride: usize,
        host_source: bool,
    ) -> Self {
        assert!(stride >= 16 && stride % 16 == 0);
        assert!((1..=infr_core::backend::MAX_DECODE_ROWS).contains(&rows));
        let count = rows * 3;
        let ordinal = 1u32;
        let mut fixture = Fixture::new_fifo(be, count, status == 5);
        let source: Box<dyn Buffer> = if host_source {
            // Same non-device-local BDA allocation used for host KV; no new public API.
            Box::new(be.alloc_kv_host(8 * stride).unwrap())
        } else {
            be.alloc_mapped_arena_bda(8 * stride).unwrap().0
        };
        // These two dedicated constructors select exactly these memory types (no allocator
        // fallback). Check both the actual backing class and that constructor's physical flags.
        let source_vk = crate::as_vk_buf(source.as_ref()).unwrap();
        assert!(
            matches!(&source_vk.backing, crate::Backing::Vram { spilled, .. } if *spilled == host_source)
        );
        let properties = unsafe {
            be.shared
                .instance
                .get_physical_device_memory_properties(be.shared.physical_device)
        };
        let memory_type = if host_source {
            be.shared.host_overflow_type.unwrap()
        } else {
            let requirements = unsafe {
                be.shared
                    .device
                    .get_buffer_memory_requirements(source_vk.buffer)
            };
            be.find_memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap()
        };
        let flags = properties.memory_types[memory_type as usize].property_flags;
        let heap_flags = properties.memory_heaps
            [properties.memory_types[memory_type as usize].heap_index as usize]
            .flags;
        assert!(flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        ));
        assert_eq!(
            flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
            !host_source
        );
        if host_source {
            assert!(!heap_flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL));
        }
        eprintln!(
            "COPY_PUBLISH_SOURCE class={} flags={flags:?} heap_flags={heap_flags:?} uma={}",
            if host_source {
                "host-RAM"
            } else {
                "device-local"
            },
            be.caps().unified_memory
        );
        let source_address = source
            .device_addr()
            .expect("source requires a retained BDA");
        let (destination, destination_address) = be.alloc_mapped_arena_bda(6 * stride).unwrap();
        let source_words: Vec<u32> = (0..8 * stride / 4)
            .map(|word| (word as u32).wrapping_mul(0x9e37_79b9) ^ 0x12ab_34cd)
            .collect();
        upload_words(be, source.as_ref(), &source_words);
        let mut payload = vec![0xa5a5_a5a5; 6 * stride / 4];
        let mut overlay = vec![[0, 0, 0, NONE]; 8];
        for slot in 0..6 {
            let address = destination_address + (slot * stride) as u64;
            fixture.nodes[ADDRESS_BASE + slot] = [
                address as u32,
                (address >> 32) as u32,
                (slot / 3) as u32,
                (slot % 3) as u32,
            ];
            if slot % 3 < 2 {
                let block = slot / 3 * 4 + if slot % 3 == 0 { 3 } else { 1 };
                overlay[block] = [address as u32, (address >> 32) as u32, 1, slot as u32];
                payload[slot * stride / 4..(slot + 1) * stride / 4]
                    .copy_from_slice(&source_words[block * stride / 4..(block + 1) * stride / 4]);
            }
        }
        upload_words(be, destination.as_ref(), &payload);
        let sources: Vec<[u32; 4]> = (0..8)
            .map(|block| {
                let address = if status == 5 && (all_cpu || block == 6) {
                    0
                } else {
                    source_address + (block * stride) as u64
                };
                [
                    address as u32,
                    (address >> 32) as u32,
                    stride as u32,
                    (block / 4) as u32,
                ]
            })
            .collect();
        for (binding, bytes) in [
            (0, (count + rows * 2) * 4),
            (3, (ordinal as usize + 1) * RESIDENT_JOURNAL_BYTES),
            (
                7,
                (ordinal as usize + 1) * RESIDENT_JOURNAL_ID_WORDS * 3 * 32,
            ),
        ] {
            fixture.buffers[binding] = be.alloc_mapped_arena_bda(bytes).unwrap().0;
            upload_words(be, fixture.buffers[binding].as_ref(), &vec![0; bytes / 4]);
        }
        be.upload(fixture.buffers[4].as_ref(), bytemuck::cast_slice(&sources))
            .unwrap();
        be.upload(
            fixture.buffers[5].as_ref(),
            bytemuck::cast_slice(&fixture.nodes),
        )
        .unwrap();
        be.upload(fixture.buffers[6].as_ref(), bytemuck::cast_slice(&overlay))
            .unwrap();
        let routed = if status == 1 { [1, 1, 1] } else { [0, 1, 2] };
        let ids: Vec<u32> = (0..rows).flat_map(|_| routed).collect();
        fixture.reset_inputs(be, &ids);
        fixture.source = source;
        fixture.destination = destination;
        fixture.source_words = source_words;
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        let bindings: Vec<_> = fixture
            .buffers
            .iter()
            .map(|b| Recorder::vkb_exact(b.as_ref()))
            .collect();
        let push = gpu_copy_prepare_push(
            [count as u32, 4, 0, 4, 8, ordinal, 3, 0],
            true,
            [roles[0], roles[1], roles[2], 0, 0, 0, 0, 0, 0],
            2,
        );
        let kernel = be.kernel(
            "gpu_owned_prepare",
            crate::gemm::gpu_owned_prepare_spv(),
            8,
            GPU_COPY_PREPARE_PUSH_BYTES,
        );
        rec.dispatch(kernel, &bindings, 8, bytemuck::cast_slice(&push), 1);
        rec.publish_host_reads();
        rec.finish().unwrap();
        assert_eq!(
            read_words(be, fixture.buffers[3].as_ref())
                [ordinal as usize * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_STATUS_WORD],
            status
        );
        let observer = be.alloc_mapped_arena_bda(6 * stride).unwrap().0;
        Self {
            fixture,
            ordinal,
            stride,
            observer,
        }
    }

    fn snapshot(&self, be: &VulkanBackend) -> Vec<Vec<u32>> {
        self.fixture
            .buffers
            .iter()
            .map(|b| read_words(be, b.as_ref()))
            .chain([
                read_words(be, self.fixture.destination.as_ref()),
                read_words(be, self.fixture.source.as_ref()),
            ])
            .collect()
    }

    fn restore(&self, be: &VulkanBackend, snapshot: &[Vec<u32>]) {
        assert_eq!(snapshot.len(), 10);
        for (buffer, words) in self.fixture.buffers.iter().zip(snapshot) {
            upload_words(be, buffer.as_ref(), words);
        }
        upload_words(be, self.fixture.destination.as_ref(), &snapshot[8]);
        upload_words(be, self.fixture.source.as_ref(), &snapshot[9]);
    }

    fn record(&self, rec: &Recorder<'_>, fused: bool, wide: bool, groups: u32) {
        assert!(groups > 0);
        let b = &self.fixture.buffers;
        let bindings = [3, 7, 5, 6].map(|i| Recorder::vkb_exact(b[i].as_ref()));
        if fused {
            let kernel = rec.be.kernel(
                if wide {
                    "gpu_owned_copy_publish1024"
                } else {
                    "gpu_owned_copy_publish"
                },
                crate::gemm::gpu_owned_copy_publish_spv(wide),
                4,
                4,
            );
            rec.dispatch(
                kernel,
                &bindings,
                2,
                bytemuck::bytes_of(&self.ordinal),
                groups,
            );
        } else {
            let payload = rec.be.kernel(
                if wide {
                    "gpu_resident_copy_payload1024"
                } else {
                    "gpu_resident_copy_payload"
                },
                crate::gemm::gpu_resident_copy_payload_spv(wide),
                2,
                4,
            );
            rec.dispatch(
                payload,
                &bindings[..2],
                0,
                bytemuck::bytes_of(&self.ordinal),
                groups,
            );
            rec.owned_copy_barrier(false, false);
            let publish = rec.be.kernel(
                "gpu_owned_publish",
                crate::gemm::gpu_owned_publish_spv(),
                4,
                8,
            );
            rec.dispatch(
                publish,
                &bindings,
                2,
                bytemuck::cast_slice(&[self.ordinal, 0]),
                1,
            );
        }
        // Exact serial contract: the barrier, not WG0's early READY stores, admits consumers.
        rec.owned_copy_barrier(false, false);
    }

    fn run(&self, be: &VulkanBackend, fused: bool, wide: bool, groups: u32) -> Vec<Vec<u32>> {
        let rec = be.recorder().unwrap();
        rec.owned_copy_barrier(true, false);
        self.record(&rec, fused, wide, groups);
        rec.copy_strided(
            self.fixture.destination.as_ref(),
            self.observer.as_ref(),
            1,
            self.fixture.destination.len_bytes() / 4,
            0,
            0,
            0,
            0,
        );
        rec.publish_host_reads();
        rec.finish().unwrap();
        let snapshot = self.snapshot(be);
        assert_eq!(
            read_words(be, self.observer.as_ref()),
            snapshot[8],
            "immediate GPU consumer saw incomplete payload"
        );
        snapshot
    }

    fn differential(
        &self,
        be: &VulkanBackend,
        seed: &[Vec<u32>],
        wide: bool,
        groups: u32,
    ) -> Vec<Vec<u32>> {
        self.restore(be, seed);
        let old = self.run(be, false, wide, groups);
        self.restore(be, seed);
        let fused = self.run(be, true, wide, groups);
        for (binding, (left, right)) in old.iter().zip(&fused).enumerate() {
            assert_eq!(
                left, right,
                "serial copy/publish differs at buffer {binding}, wide={wide}, groups={groups}"
            );
        }
        fused
    }

    fn check_expected(&self, seed: &[Vec<u32>], result: &[Vec<u32>]) {
        let base = self.ordinal as usize * RESIDENT_JOURNAL_WORDS;
        let count = seed[3][base + RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
        let jobs = self.ordinal as usize * RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
        let mut payload = seed[8].clone();
        let mut nodes = seed[5].clone();
        let mut overlay = seed[6].clone();
        let slot_base = nodes[6] as usize;
        for job in seed[7][jobs..jobs + count * 8].chunks_exact(8) {
            if job[0] == 0 && job[1] == 0 {
                continue;
            }
            let block = job[5] as usize;
            let slot = job[6] as usize;
            payload[slot * self.stride / 4..(slot + 1) * self.stride / 4]
                .copy_from_slice(&seed[9][block * self.stride / 4..(block + 1) * self.stride / 4]);
            nodes[(slot_base + slot) * 4 + 3] = 1;
            overlay[block * 4 + 2] = 1;
        }
        assert_eq!(result[8], payload);
        assert_eq!(result[5], nodes);
        assert_eq!(result[6], overlay);
        for binding in [0, 1, 2, 3, 4, 7, 9] {
            assert_eq!(result[binding], seed[binding]);
        }
    }
}

#[test]
#[ignore = "central GPU runner only; serial fused multi-WG bytes, owner metadata, aliases and CPU recovery"]
fn gpu_owned_copy_publish_multi_workgroup_matches_serial() {
    let be = backend();
    for (rows, status, all_cpu, roles, groups) in [
        (1, 1, false, [0, 0, 4], 2),
        (1, 4, false, [0, 0, 4], 7),
        (2, 4, false, [0, 4, 4], 3),
        (infr_core::backend::MAX_DECODE_ROWS, 4, false, [0, 0, 4], 7),
        (1, 5, false, [0, 0, 4], 3),
        (2, 5, false, [0, 4, 4], 7),
        (infr_core::backend::MAX_DECODE_ROWS, 5, true, [0, 0, 4], 2),
    ] {
        let fixture = CopyPublishFixture::new(&be, rows, status, all_cpu, roles, 128 * 1024, false);
        let seed = fixture.snapshot(&be);
        for wide in [false, true] {
            let result = fixture.differential(&be, &seed, wide, groups);
            fixture.check_expected(&seed, &result);
            if status == 5 {
                // CPU supplies only zero-BDA jobs. Mode 1 remains a separate ordered operation.
                let start = fixture.ordinal as usize * RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
                let count = seed[3][fixture.ordinal as usize * RESIDENT_JOURNAL_WORDS
                    + RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
                let mut uploaded = result[8].clone();
                for job in seed[7][start..start + count * 8].chunks_exact(8) {
                    if job[0] != 0 || job[1] != 0 {
                        continue;
                    }
                    let (block, slot) = (job[5] as usize, job[6] as usize);
                    uploaded[slot * fixture.stride / 4..(slot + 1) * fixture.stride / 4]
                        .copy_from_slice(
                            &seed[9][block * fixture.stride / 4..(block + 1) * fixture.stride / 4],
                        );
                }
                upload_words(&be, fixture.fixture.destination.as_ref(), &uploaded);
                let rec = be.recorder().unwrap();
                rec.owned_copy_barrier(true, false);
                let b = &fixture.fixture.buffers;
                let bindings = [3, 7, 5, 6].map(|i| Recorder::vkb_exact(b[i].as_ref()));
                let publish = be.kernel(
                    "gpu_owned_publish",
                    crate::gemm::gpu_owned_publish_spv(),
                    4,
                    8,
                );
                rec.dispatch(
                    publish,
                    &bindings,
                    2,
                    bytemuck::cast_slice(&[fixture.ordinal, 1]),
                    1,
                );
                rec.owned_copy_barrier(false, false);
                rec.copy_strided(
                    fixture.fixture.destination.as_ref(),
                    fixture.observer.as_ref(),
                    1,
                    uploaded.len(),
                    0,
                    0,
                    0,
                    0,
                );
                rec.publish_host_reads();
                rec.finish().unwrap();
                assert_eq!(read_words(&be, fixture.observer.as_ref()), uploaded);
                let final_state = fixture.snapshot(&be);
                let slot_base = final_state[5][6] as usize;
                for job in seed[7][start..start + count * 8].chunks_exact(8) {
                    assert_eq!(final_state[5][(slot_base + job[6] as usize) * 4 + 3], 1);
                    assert_eq!(final_state[6][job[5] as usize * 4 + 2], 1);
                }
            }
        }
    }
}

#[test]
#[ignore = "central GPU runner only; invalid publication is atomic without suppressing old payload semantics"]
fn gpu_owned_copy_publish_invalid_batches_match_independent_rejection() {
    let be = backend();
    let fixture = CopyPublishFixture::new(&be, 1, 4, false, [0, 0, 4], 32 * 1024, false);
    let original = fixture.snapshot(&be);
    let job = fixture.ordinal as usize * RESIDENT_JOURNAL_ID_WORDS * 3 * 8;
    let count_word =
        fixture.ordinal as usize * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_JOB_COUNT_WORD;
    let status_word =
        fixture.ordinal as usize * RESIDENT_JOURNAL_WORDS + RESIDENT_JOURNAL_STATUS_WORD;
    // Corrupted BDA inputs below always fail payload validation before dereferencing them.
    for fault in [
        "source_alignment",
        "source_wrap",
        "destination_zero",
        "destination_alignment",
        "bytes_zero",
        "block_bounds",
        "slot_bounds",
        "reverse_owner",
        "canonical_address",
        "overlay_state",
        "duplicate_block",
        "duplicate_slot",
        "header",
        "count_overflow",
        "count_zero",
        "status_error",
    ] {
        let mut seed = original.clone();
        let (block, slot) = (seed[7][job + 5] as usize, seed[7][job + 6] as usize);
        let slot_base = seed[5][6] as usize;
        let address_base = seed[5][7] as usize;
        match fault {
            "source_alignment" => seed[7][job] |= 4,
            "source_wrap" => {
                seed[7][job] = 0xffff_fff0;
                seed[7][job + 1] = NONE;
            }
            "destination_zero" => seed[7][job + 2..job + 4].fill(0),
            "destination_alignment" => seed[7][job + 2] |= 4,
            "bytes_zero" => seed[7][job + 4] = 0,
            "block_bounds" => seed[7][job + 5] = 8,
            "slot_bounds" => seed[7][job + 6] = 6,
            "reverse_owner" => seed[5][(slot_base + slot) * 4 + 2] = NONE,
            "canonical_address" => seed[5][(address_base + slot) * 4] ^= 16,
            "overlay_state" => seed[6][block * 4 + 2] = 0,
            "duplicate_block" => seed[7][job + 8 + 5] = block as u32,
            "duplicate_slot" => seed[7][job + 8 + 6] = slot as u32,
            "header" => seed[5][0] = 99,
            "count_overflow" => seed[3][count_word] = (RESIDENT_JOURNAL_ID_WORDS * 3 + 1) as u32,
            "count_zero" => seed[3][count_word] = 0,
            "status_error" => seed[3][status_word] = 3,
            _ => unreachable!(),
        }
        for wide in [false, true] {
            let result = fixture.differential(&be, &seed, wide, 3);
            assert_eq!(
                result[5], seed[5],
                "failed batch published owner READY: {fault}"
            );
            assert_eq!(
                result[6], seed[6],
                "failed batch published overlay READY: {fault}"
            );
        }
    }
    // The existing two shaders do NOT share a common rejection predicate: zero source
    // in status4 aborts the whole copy, but mode0 publication skips only that job.
    // Preserve this exact negative-input behavior rather than claiming a stronger contract.
    let mut seed = original.clone();
    seed[7][job..job + 2].fill(0);
    let result = fixture.differential(&be, &seed, false, 7);
    assert_eq!(result[8], seed[8]);
    assert_eq!(result[6][seed[7][job + 5] as usize * 4 + 2], 2);
    let sibling = seed[7][job + 8 + 5] as usize;
    assert_eq!(result[6][sibling * 4 + 2], 1);
    // Out-of-range ordinals must reject before any journal/jobs read.
    let mut invalid_ordinal = fixture;
    invalid_ordinal.ordinal = 2;
    let result = invalid_ordinal.differential(&be, &original, false, 3);
    assert_eq!(result, original);
}

#[test]
#[ignore = "central GPU runner only; paired serial copy/publish GPU intervals, 0.5/1/2 MiB per expert"]
fn gpu_owned_copy_publish_paired_latency_probe() {
    let be = backend();
    assert!(be.shared.submit_timestamp_valid_bits > 0);
    let device = &be.shared.device;
    let query = unsafe {
        device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )
    }
    .unwrap();
    let properties = unsafe {
        be.shared
            .instance
            .get_physical_device_properties(be.shared.physical_device)
    };
    let period = properties.limits.timestamp_period as f64;
    let wide = std::env::var("MOE4ALL_COPY_PUBLISH_PERF_WIDE").as_deref() == Ok("1");
    let smoke = std::env::var("MOE4ALL_COPY_PUBLISH_PERF_SMOKE").as_deref() == Ok("1");
    let groups = std::env::var("MOE4ALL_COPY_PUBLISH_PERF_GROUPS")
        .ok()
        .map(|value| value.parse::<u32>().expect("positive GPU workgroup count"))
        .unwrap_or(2);
    assert!(groups > 0 && groups <= properties.limits.max_compute_work_group_count[0]);
    for host_source in [false, true] {
        if smoke && !host_source {
            continue;
        }
        if host_source && be.shared.host_overflow_type.is_none() {
            eprintln!("COPY_PUBLISH_PROBE host-RAM BDA unavailable; not substituting VRAM");
            continue;
        }
        for stride in [512 * 1024, 1024 * 1024, 2 * 1024 * 1024] {
            if smoke && stride != 1024 * 1024 {
                continue;
            }
            for status in [1, 4, 5] {
                if smoke && status == 5 {
                    continue;
                }
                let fixture =
                    CopyPublishFixture::new(&be, 1, status, false, [0, 0, 4], stride, host_source);
                let seed = fixture.snapshot(&be);
                let mut samples = [Vec::new(), Vec::new()];
                for round in 0..if smoke { 3 } else { 10 } {
                    for arm in 0..2 {
                        let variant = (round + arm) % 2;
                        fixture.restore(&be, &seed);
                        let rec = be.recorder().unwrap();
                        rec.owned_copy_barrier(true, false);
                        unsafe {
                            device.cmd_reset_query_pool(rec.cmd, query, 0, 2);
                            device.cmd_write_timestamp(
                                rec.cmd,
                                vk::PipelineStageFlags::TOP_OF_PIPE,
                                query,
                                0,
                            );
                        }
                        fixture.record(&rec, variant == 1, wide, groups);
                        unsafe {
                            device.cmd_write_timestamp(
                                rec.cmd,
                                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                                query,
                                1,
                            );
                        }
                        rec.finish().unwrap();
                        let mut ticks = [0u64; 2];
                        unsafe {
                            device.get_query_pool_results(
                                query,
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
                        if round >= if smoke { 1 } else { 2 } {
                            samples[variant].push(delta as f64 * period / 1000.0);
                        }
                        fixture.check_expected(&seed, &fixture.snapshot(&be));
                    }
                }
                for (variant, values) in samples.iter_mut().enumerate() {
                    values.sort_by(f64::total_cmp);
                    eprintln!("COPY_PUBLISH_PROBE source={} expert_bytes={stride} status={status} wide={wide} groups={groups} variant={} n={} p50_us={:.3} mean_us={:.3}",
                                if host_source { "host-RAM" } else { "device-local" }, if variant == 0 { "separate" } else { "fused" }, values.len(), values[values.len()/2], values.iter().sum::<f64>() / values.len() as f64);
                }
            }
        }
    }
    unsafe {
        device.destroy_query_pool(query, None);
    }
}
