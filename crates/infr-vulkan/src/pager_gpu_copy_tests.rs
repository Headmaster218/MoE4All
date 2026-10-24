#[cfg(test)]
mod fixtures {
    use super::super::*;

    fn read_words(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<u32> {
        let mut words = vec![0; buffer.len_bytes() / 4];
        be.download(buffer, bytemuck::cast_slice_mut(&mut words))
            .unwrap();
        words
    }

    #[test]
    #[ignore = "requires an explicitly selected GPU with imported host BDA"]
    fn serial_gpu_copy_contract() {
        serial_gpu_copy_contract_with_groups(0);
    }

    #[test]
    #[ignore = "requires an explicitly selected GPU with imported host BDA"]
    fn serial_gpu_copy_groups_2_4_8_contract() {
        for groups in [2, 4, 8] {
            serial_gpu_copy_contract_with_groups(groups);
        }
    }

    fn serial_gpu_copy_contract_with_groups(groups: usize) {
        const EXPERTS: usize = 8;
        const STRIDE: usize = 64;
        const SLOTS: usize = 20;
        let mut cfg = infr_core::config::Config::default();
        cfg.device.dev =
            Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select the test GPU explicitly"));
        cfg.kernels.vulkan.decode_gpu_resident_copy = Some(true);
        cfg.kernels.vulkan.decode_gpu_resident_copy_groups = groups;
        let be_ = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
        be_.init_unified_vram(16 * 1024 * 1024).unwrap();
        let host = Arc::new(AlignedHostBuffer::new(EXPERTS * 3 * STRIDE).unwrap());
        let expected: Vec<u32> = (0..EXPERTS * 3 * STRIDE / 4)
            .map(|i| (i as u32).wrapping_mul(2654435761) ^ 0x5a739d12)
            .collect();
        unsafe {
            std::ptr::copy_nonoverlapping(
                expected.as_ptr().cast::<u8>(),
                host.as_ptr(),
                expected.len() * 4,
            )
        };
        let ext = be_
            .shared
            .external_memory_host
            .as_ref()
            .expect("host import required");
        let imported = be_
            .try_import_host_shard(ext, Arc::clone(&host), 0, host.allocated_len())
            .unwrap();
        let address = as_vk_buf(imported.buffer.as_ref())
            .unwrap()
            .device_addr()
            .unwrap();
        assert_ne!(address, 0);
        let directory: Vec<u32> = (0..EXPERTS * 3)
            .flat_map(|id| {
                let a = address + (id * STRIDE) as u64;
                [a as u32, (a >> 32) as u32, STRIDE as u32, 0]
            })
            .collect();
        let copy = GpuCopyBuffers::new(
            &be_,
            GpuCopySources {
                words: directory.clone(),
                keepalive: vec![Arc::clone(&imported.buffer)],
                pools: 1,
            },
            2,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            copy.copy_groups,
            effective_copy_groups(groups, copy.wide_copy)
        );
        let (destination, destination_address) =
            be_.alloc_mapped_arena_bda(SLOTS * STRIDE).unwrap();
        let mapped = |bytes| be_.alloc_mapped_arena_bda(bytes).unwrap().0;
        let ids = mapped((infr_core::backend::MAX_DECODE_ROWS * 18) * 4);
        let tape = mapped(EXPERTS * 3 * 8);
        let control = mapped(crate::recorder::RESIDENT_CONTROL_BYTES);
        let journal = mapped(2 * crate::recorder::RESIDENT_JOURNAL_BYTES);
        let event = unsafe {
            be_.shared
                .device
                .create_event(&vk::EventCreateInfo::default(), None)
        }
        .unwrap();
        let status_word = crate::recorder::RESIDENT_JOURNAL_STATUS_WORD;
        let mask_word = crate::recorder::RESIDENT_JOURNAL_MASK_WORD;
        let all_candidates: Vec<u32> = (0..SLOTS)
            .flat_map(|slot| {
                let a = destination_address + (slot * STRIDE) as u64;
                [a as u32, (a >> 32) as u32, NOT_RESIDENT, 0]
            })
            .collect();
        let mut checked = 0;
        for rows in [1, 2, infr_core::backend::MAX_DECODE_ROWS] {
            for mode in [
                "all_hit",
                "all_miss",
                "mixed",
                "unmapped",
                "exhausted",
                "unaligned",
            ] {
                let mut sources = directory.clone();
                if mode == "unmapped" {
                    sources[EXPERTS * 4] = 0;
                    sources[EXPERTS * 4 + 1] = 0;
                }
                if mode == "unaligned" {
                    sources[EXPERTS * 4] += 4;
                }
                mapped_write(copy.sources.as_ref(), bytemuck::cast_slice(&sources)).unwrap();
                mapped_write(
                    copy.candidates.as_ref(),
                    bytemuck::cast_slice(&all_candidates),
                )
                .unwrap();
                let routed: Vec<u32> = (0..rows * 4).map(|i| (i % 4) as u32).collect();
                mapped_write(ids.as_ref(), bytemuck::cast_slice(&routed)).unwrap();
                let mut addresses = vec![0u64; EXPERTS * 3];
                for role in 0..3 {
                    for id in 0..4 {
                        if mode == "all_hit" || (mode == "mixed" && id == 0) {
                            addresses[role * EXPERTS + id] =
                                destination_address + ((16 + role) * STRIDE) as u64;
                        }
                    }
                }
                mapped_write(tape.as_ref(), bytemuck::cast_slice(&addresses)).unwrap();
                let count = if mode == "exhausted" { 2 } else { 16 };
                let rec = be_.recorder().unwrap();
                rec.zero(copy.overlay.as_ref(), copy.overlay.len_bytes() / 4);
                rec.zero(journal.as_ref(), journal.len_bytes() / 4);
                rec.zero(destination.as_ref(), destination.len_bytes() / 4);
                rec.gpu_resident_check(
                    ids.as_ref(),
                    tape.as_ref(),
                    control.as_ref(),
                    journal.as_ref(),
                    [0; 6],
                );
                rec.gpu_resident_copy_check(
                    ids.as_ref(),
                    tape.as_ref(),
                    control.as_ref(),
                    journal.as_ref(),
                    &copy,
                    [
                        (rows * 4) as u32,
                        EXPERTS as u32,
                        0,
                        EXPERTS as u32,
                        (EXPERTS * 2) as u32,
                        0,
                        4,
                        0,
                    ],
                    [
                        0,
                        EXPERTS as u32,
                        (EXPERTS * 2) as u32,
                        0,
                        0,
                        0,
                        count,
                        count,
                        count,
                    ],
                    event,
                );
                rec.publish_host_reads();
                rec.finish().unwrap();
                let result = read_words(&be_, journal.as_ref());
                let expected_status = match mode {
                    "all_hit" => 1,
                    "all_miss" | "mixed" => 4,
                    _ => 2,
                };
                assert_eq!(
                    result[status_word], expected_status,
                    "rows={rows} mode={mode}"
                );
                assert_eq!(&result[..routed.len()], routed.as_slice());
                for row in 0..rows {
                    assert_eq!(
                        result[mask_word + row],
                        if mode == "all_hit" {
                            15
                        } else if mode == "mixed" {
                            1
                        } else {
                            0
                        }
                    );
                }
                let candidates = read_words(&be_, copy.candidates.as_ref());
                let output = read_words(&be_, destination.as_ref());
                if expected_status == 4 {
                    let jobs = read_words(&be_, copy.jobs.as_ref());
                    let count = result[JOB_COUNT_WORD] as usize;
                    assert_eq!(
                        count,
                        if mode == "mixed" { 9 } else { 12 },
                        "deduplicate repeated routes"
                    );
                    let resolved = read_words(&be_, tape.as_ref());
                    for job in jobs[..count * JOB_WORDS].chunks_exact(JOB_WORDS) {
                        let block = job[5] as usize;
                        let dst = job[2] as u64 | ((job[3] as u64) << 32);
                        let offset = (dst - destination_address) as usize / 4;
                        assert_eq!(
                            &output[offset..offset + STRIDE / 4],
                            &expected[block * STRIDE / 4..(block + 1) * STRIDE / 4]
                        );
                        assert_eq!(
                            &resolved[job[7] as usize * 2..job[7] as usize * 2 + 2],
                            &job[2..4]
                        );
                        assert_eq!(candidates[job[6] as usize * 4 + 3], 1);
                    }
                    let masks = read_words(&be_, ids.as_ref());
                    assert!(masks[routed.len()..routed.len() + rows]
                        .iter()
                        .all(|&m| m == 15));
                    assert!(masks[routed.len() + rows..routed.len() + rows * 2]
                        .iter()
                        .all(|&m| m == 0));
                } else {
                    assert_eq!(result[JOB_COUNT_WORD], 0);
                    assert_eq!(
                        &candidates[..all_candidates.len()],
                        all_candidates.as_slice(),
                        "fallback must not mutate slot ownership"
                    );
                    assert!(output.iter().all(|&word| word == 0));
                }
                unsafe { be_.shared.device.reset_event(event) }.unwrap();
                checked += 1;
            }
        }
        // A later frozen LUT may still mention a replaced owner. The overlay must invalidate it.
        mapped_write(copy.sources.as_ref(), bytemuck::cast_slice(&directory)).unwrap();
        let mut candidates = all_candidates.clone();
        candidates[2] = 6;
        mapped_write(copy.candidates.as_ref(), bytemuck::cast_slice(&candidates)).unwrap();
        let mut addresses = vec![0u64; EXPERTS * 3];
        addresses[6] = destination_address;
        mapped_write(tape.as_ref(), bytemuck::cast_slice(&addresses)).unwrap();
        mapped_write(ids.as_ref(), bytemuck::cast_slice(&[0u32])).unwrap();
        let rec = be_.recorder().unwrap();
        rec.zero(copy.overlay.as_ref(), copy.overlay.len_bytes() / 4);
        rec.zero(journal.as_ref(), journal.len_bytes() / 4);
        rec.gpu_resident_check(
            ids.as_ref(),
            tape.as_ref(),
            control.as_ref(),
            journal.as_ref(),
            [0; 6],
        );
        rec.gpu_resident_copy_check(
            ids.as_ref(),
            tape.as_ref(),
            control.as_ref(),
            journal.as_ref(),
            &copy,
            [1, 8, 0, 8, 16, 0, 1, 0],
            [0, 8, 16, 0, 0, 0, 16, 16, 16],
            event,
        );
        rec.finish().unwrap();
        let overlay = read_words(&be_, copy.overlay.as_ref());
        assert_eq!(&overlay[6 * 4..6 * 4 + 2], &[NOT_RESIDENT; 2]);
        mapped_write(ids.as_ref(), bytemuck::cast_slice(&[6u32])).unwrap();
        unsafe { be_.shared.device.reset_event(event) }.unwrap();
        let rec = be_.recorder().unwrap();
        rec.gpu_resident_copy_check(
            ids.as_ref(),
            tape.as_ref(),
            control.as_ref(),
            journal.as_ref(),
            &copy,
            [1, 8, 0, 8, 16, 1, 1, 0],
            [0, 8, 16, 0, 0, 0, 16, 16, 16],
            event,
        );
        rec.finish().unwrap();
        let result = read_words(&be_, journal.as_ref());
        assert_eq!(
            result[crate::recorder::RESIDENT_JOURNAL_WORDS + status_word],
            4
        );
        assert_eq!(
            result[crate::recorder::RESIDENT_JOURNAL_WORDS + JOB_COUNT_WORD],
            3
        );
        let addresses = read_words(&be_, tape.as_ref());
        assert_ne!(
            addresses[6 * 2] as u64 | ((addresses[6 * 2 + 1] as u64) << 32),
            destination_address
        );
        checked += 1;
        unsafe { be_.shared.device.destroy_event(event, None) };
        println!("serial GPU copy contract: {checked} cases passed, including stale LUT eviction");
    }
}
