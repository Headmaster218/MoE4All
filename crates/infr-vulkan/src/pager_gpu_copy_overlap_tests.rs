use super::*;
use infr_core::DType;

const WIDTH: usize = 32;
const EXPERTS: usize = 4;
const USED: usize = 2;
const PHYSICAL: usize = USED + 1;
const STRIDE: usize = 4096;

fn words(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<u32> {
    let mut values = vec![0; buffer.len_bytes() / 4];
    be.download(buffer, bytemuck::cast_slice_mut(&mut values))
        .unwrap();
    values
}

#[test]
#[ignore = "requires MOE4ALL_TEST_DEVICE selecting an idle GPU with imported-host BDA"]
fn gpu_copy_overlap_preserves_masks_shared_stride_and_zero_job_dispatch() {
    overlap_copy_contract_with_groups(0);
}

#[test]
#[ignore = "requires MOE4ALL_TEST_DEVICE selecting an idle GPU with imported-host BDA"]
fn gpu_copy_overlap_groups_2_4_8_preserve_masks_and_shared_stride() {
    for groups in [2, 4, 8] {
        overlap_copy_contract_with_groups(groups);
    }
}

fn overlap_copy_contract_with_groups(groups: usize) {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("select the test GPU explicitly"));
    cfg.kernels.vulkan.decode_gpu_resident_copy = Some(true);
    cfg.kernels.vulkan.decode_gpu_resident_copy_groups = groups;
    cfg.kernels.vulkan.decode_gpu_resident_copy_overlap = true;
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    be.init_unified_vram(8 * 1024 * 1024).unwrap();
    let host = Arc::new(AlignedHostBuffer::new(3 * EXPERTS * STRIDE).unwrap());
    let mut source = vec![0u8; 3 * EXPERTS * STRIDE];
    for role in 0..3 {
        for expert in 0..EXPERTS {
            let scale = half::f16::from_f32((1 + role + expert) as f32 / 128.0)
                .to_bits()
                .to_le_bytes();
            for row in 0..WIDTH {
                let offset = (role * EXPERTS + expert) * STRIDE + row * 18;
                source[offset..offset + 2].copy_from_slice(&scale);
                // IQ4_NL code 8 is +1, giving a constant matrix with known row sum.
                source[offset + 2..offset + 18].fill(0x88);
            }
        }
    }
    unsafe {
        std::ptr::copy_nonoverlapping(source.as_ptr(), host.as_ptr(), source.len());
    }
    let imported = be
        .try_import_host_shard(
            be.shared
                .external_memory_host
                .as_ref()
                .expect("host import required"),
            Arc::clone(&host),
            0,
            host.allocated_len(),
        )
        .unwrap();
    let source_address = imported.buffer.device_addr().expect("host BDA required");
    let directory: Vec<u32> = (0..3 * EXPERTS)
        .flat_map(|block| {
            let address = source_address + (block * STRIDE) as u64;
            [address as u32, (address >> 32) as u32, STRIDE as u32, 0]
        })
        .collect();
    let copy = GpuCopyBuffers::new(
        &be,
        GpuCopySources {
            words: directory.clone(),
            keepalive: vec![Arc::clone(&imported.buffer)],
            pools: 1,
        },
        1,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        copy.copy_groups,
        effective_copy_groups(groups, copy.wide_copy)
    );
    let map = |bytes| be.alloc_mapped_arena_bda(bytes).unwrap().0;
    let (destination, destination_address) = be.alloc_mapped_arena_bda(24 * STRIDE).unwrap();
    let tape = map(3 * EXPERTS * 8);
    let control = be
        .alloc_mapped_predicate(crate::recorder::RESIDENT_CONTROL_BYTES)
        .unwrap();
    let journal = map(crate::recorder::RESIDENT_JOURNAL_BYTES);
    let event = unsafe {
        be.shared
            .device
            .create_event(&vk::EventCreateInfo::default(), None)
    }
    .unwrap();
    let candidates: Vec<u32> = (12..24)
        .flat_map(|slot| {
            let address = destination_address + (slot * STRIDE) as u64;
            [address as u32, (address >> 32) as u32, NOT_RESIDENT, 0]
        })
        .collect();
    let shared = be.alloc(WIDTH * 34, BufferUsage::Weights).unwrap();
    let mut shared_weights = vec![0u8; WIDTH * 34];
    for row in 0..WIDTH {
        shared_weights[row * 34..row * 34 + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        shared_weights[row * 34 + 2 + row] = 1;
    }
    be.upload(shared.as_ref(), &shared_weights).unwrap();

    for (rows, verify) in [(1, false), (2, false), (2, true), (3, true), (4, true)] {
        let ids = map(rows * (USED + 2) * 4);
        let input = map(rows * WIDTH * 4);
        let gate = map(rows * PHYSICAL * WIDTH * 4);
        let up = map(rows * PHYSICAL * WIDTH * 4);
        let activated = map(rows * PHYSICAL * WIDTH * 4);
        let output = map(rows * PHYSICAL * WIDTH * 4);
        let weights = map(rows * USED * 4);
        let shared_gate = map(rows * 4);
        let hidden = map(rows * WIDTH * 4);
        let wts: Vec<f32> = (0..rows).flat_map(|_| [0.25, 0.75]).collect();
        mapped_write(weights.as_ref(), bytemuck::cast_slice(&wts)).unwrap();
        mapped_write(
            shared_gate.as_ref(),
            bytemuck::cast_slice(&vec![0f32; rows]),
        )
        .unwrap();
        let routes = [[0, 1], [2, 0], [1, 3], [3, 2]];
        let routed: Vec<u32> = (0..rows).flat_map(|row| routes[row]).collect();
        let x: Vec<f32> = (0..rows)
            .flat_map(|row| vec![1.0 + row as f32 * 0.25; WIDTH])
            .collect();
        mapped_write(input.as_ref(), bytemuck::cast_slice(&x)).unwrap();
        for mode in [
            "all_miss",
            "all_hit",
            "mixed",
            "unmapped",
            "all_hit",
            "partial_gate",
            "partial_up",
            "partial_down",
            "mixed",
        ] {
            mapped_write(destination.as_ref(), &source).unwrap();
            mapped_write(copy.candidates.as_ref(), bytemuck::cast_slice(&candidates)).unwrap();
            mapped_write(ids.as_ref(), bytemuck::cast_slice(&routed)).unwrap();
            let mut addresses = vec![0u64; 3 * EXPERTS];
            for role in 0..3 {
                for expert in 0..EXPERTS {
                    let missing_role = match mode {
                        "partial_gate" => Some(0),
                        "partial_up" => Some(1),
                        "partial_down" => Some(2),
                        _ => None,
                    };
                    let present = mode == "all_hit"
                        || (mode != "all_miss"
                            && (expert % 2 == 0
                                || missing_role.is_some_and(|missing| missing != role)));
                    if present {
                        addresses[role * EXPERTS + expert] =
                            destination_address + ((role * EXPERTS + expert) * STRIDE) as u64;
                    }
                }
            }
            mapped_write(tape.as_ref(), bytemuck::cast_slice(&addresses)).unwrap();
            let mut sources = directory.clone();
            if mode == "unmapped" {
                sources[4..6].fill(0);
            }
            mapped_write(copy.sources.as_ref(), bytemuck::cast_slice(&sources)).unwrap();
            let rc = if verify {
                crate::recorder::Recorder::new_prepared(&be)
            } else {
                be.recorder()
            }
            .unwrap();
            assert!(rc.gpu_copy_overlap_supported(
                verify,
                rows,
                false,
                false,
                control.as_ref(),
                &copy
            ));
            assert!(!rc.gpu_copy_overlap_supported(
                verify,
                rows,
                false,
                true,
                control.as_ref(),
                &copy
            ));
            if verify {
                assert!(!rc.gpu_copy_overlap_supported(
                    true,
                    rows,
                    true,
                    false,
                    control.as_ref(),
                    &copy
                ));
            }
            rc.zero(copy.overlay.as_ref(), copy.overlay.len_bytes() / 4);
            rc.zero(output.as_ref(), output.len_bytes() / 4);
            rc.gpu_resident_check(
                ids.as_ref(),
                tape.as_ref(),
                control.as_ref(),
                journal.as_ref(),
                [0; 6],
            );
            rc.gpu_resident_copy_prepare_overlap(
                ids.as_ref(),
                tape.as_ref(),
                control.as_ref(),
                journal.as_ref(),
                &copy,
                [
                    (rows * USED) as u32,
                    EXPERTS as u32,
                    0,
                    EXPERTS as u32,
                    (2 * EXPERTS) as u32,
                    0,
                    USED as u32,
                    1 << USED,
                ],
                [0, EXPERTS as u32, (2 * EXPERTS) as u32, 0, 0, 0, 12, 12, 12],
                event,
            );
            rc.begin_immutable_conditional_at(control.as_ref(), 6);
            let scope = rc.independent_dispatch_scope();
            for (role, out) in [(0, gate.as_ref()), (1, up.as_ref())] {
                rc.linear_native_id_multi_paged_shared(
                    DType::Iq4Nl,
                    destination_address,
                    STRIDE as u32,
                    tape.as_ref(),
                    ids.as_ref(),
                    USED,
                    role * EXPERTS,
                    shared.as_ref(),
                    input.as_ref(),
                    false,
                    out,
                    WIDTH,
                    WIDTH,
                    rows,
                    0,
                    1,
                );
            }
            rc.gpu_resident_copy_payload_overlap(&copy, journal.as_ref(), 0);
            drop(scope);
            let finish_half = |misses: bool| {
                rc.gpu_resident_silu_bank(
                    gate.as_ref(),
                    up.as_ref(),
                    ids.as_ref(),
                    activated.as_ref(),
                    [rows as u32, USED as u32, PHYSICAL as u32, WIDTH as u32],
                    misses,
                );
                rc.linear_native_id_multi_paged_shared(
                    DType::Iq4Nl,
                    destination_address,
                    STRIDE as u32,
                    tape.as_ref(),
                    ids.as_ref(),
                    USED,
                    2 * EXPERTS,
                    shared.as_ref(),
                    activated.as_ref(),
                    true,
                    output.as_ref(),
                    WIDTH,
                    WIDTH,
                    rows,
                    0,
                    if misses { 2 } else { 1 },
                );
            };
            finish_half(false);
            rc.end_conditional();
            rc.begin_immutable_conditional_at(copy.miss_predicates.as_ref().unwrap().as_ref(), 0);
            for (role, out) in [(0, gate.as_ref()), (1, up.as_ref())] {
                // Shared binding remains present even though its bit is absent from M.
                rc.linear_native_id_multi_paged_shared(
                    DType::Iq4Nl,
                    destination_address,
                    STRIDE as u32,
                    tape.as_ref(),
                    ids.as_ref(),
                    USED,
                    role * EXPERTS,
                    shared.as_ref(),
                    input.as_ref(),
                    false,
                    out,
                    WIDTH,
                    WIDTH,
                    rows,
                    0,
                    2,
                );
            }
            finish_half(true);
            rc.end_conditional();
            if mode != "unmapped" {
                rc.moe_accumulate_shared(
                    output.as_ref(),
                    weights.as_ref(),
                    shared_gate.as_ref(),
                    hidden.as_ref(),
                    WIDTH,
                    USED,
                    rows,
                );
            }
            rc.publish_host_reads();
            if verify {
                rc.finish_prepared()
                    .unwrap()
                    .submit(&be)
                    .unwrap()
                    .wait()
                    .unwrap();
            } else {
                rc.finish().unwrap();
            }

            let masks = words(&be, ids.as_ref());
            let notice = words(&be, journal.as_ref());
            let args = words(&be, copy.miss_predicates.as_ref().unwrap().as_ref());
            let mut required = std::collections::HashSet::new();
            for &expert in &routed {
                for role in 0..3 {
                    if addresses[role * EXPERTS + expert as usize] == 0 {
                        required.insert(role * EXPERTS + expert as usize);
                    }
                }
            }
            let jobs = if mode == "unmapped" {
                0
            } else {
                required.len() as u32
            };
            assert_eq!(
                &args[..4],
                &[jobs, if jobs == 0 { 0 } else { copy.copy_groups }, 1, 1]
            );
            assert_eq!(
                notice[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD],
                jobs
            );
            assert_eq!(
                notice[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD],
                if mode == "all_hit" {
                    1
                } else if mode == "unmapped" {
                    2
                } else {
                    4
                }
            );
            let mut actual = vec![0f32; rows * PHYSICAL * WIDTH];
            be.download(output.as_ref(), bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            let mut combined = vec![0f32; rows * WIDTH];
            if mode != "unmapped" {
                be.download(hidden.as_ref(), bytemuck::cast_slice_mut(&mut combined))
                    .unwrap();
            }
            for row in 0..rows {
                let mut hits = 0;
                let mut expected_hidden = 0f32;
                for slot in 0..USED {
                    let expert = routed[row * USED + slot] as usize;
                    let hit = (0..3).all(|role| addresses[role * EXPERTS + expert] != 0);
                    if hit {
                        hits |= 1 << slot;
                    }
                    let value = x[row * WIDTH];
                    let g = WIDTH as f32 * (1 + expert) as f32 / 128.0 * value;
                    let u = WIDTH as f32 * (2 + expert) as f32 / 128.0 * value;
                    let expected = if mode == "unmapped" && !hit {
                        0.0
                    } else {
                        g / (1.0 + (-g).exp()) * u * WIDTH as f32 * (3 + expert) as f32 / 128.0
                    };
                    expected_hidden += wts[row * USED + slot] * expected;
                    for &v in &actual
                        [(row * PHYSICAL + slot) * WIDTH..(row * PHYSICAL + slot + 1) * WIDTH]
                    {
                        assert!(
                            (v - expected).abs() < 1e-5,
                            "{mode} row={row} slot={slot}: {v} != {expected}"
                        );
                    }
                }
                assert_eq!(masks[rows * USED + row], hits | (1 << USED));
                assert_eq!(masks[rows * USED + rows + row], ((1 << USED) - 1) ^ hits);
                assert_eq!(
                    notice[crate::recorder::RESIDENT_JOURNAL_MASK_WORD + row],
                    hits
                );
                let value = x[row * WIDTH];
                let expected = value / (1.0 + (-value).exp()) * value;
                expected_hidden += 0.5 * expected;
                for &v in &actual[(row * PHYSICAL + USED) * WIDTH..(row + 1) * PHYSICAL * WIDTH] {
                    assert!(
                        (v - expected).abs() < 1e-5,
                        "shared slot must be evaluated once: {v} != {expected}"
                    );
                }
                if mode != "unmapped" {
                    assert!(
                        combined[row * WIDTH..(row + 1) * WIDTH]
                            .iter()
                            .all(|value| (value - expected_hidden).abs() < 1e-5),
                        "{mode} verify={verify} row={row}: routed/shared accumulation differs"
                    );
                }
            }
            unsafe { be.shared.device.reset_event(event) }.unwrap();
        }
    }
    unsafe {
        be.shared.device.destroy_event(event, None);
    }
}
