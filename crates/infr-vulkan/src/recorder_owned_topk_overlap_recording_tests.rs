//! Real owner/session entrypoint coverage; adapter integration is deliberately separate.
use super::*;
use crate::pager::{
    ExpertSource, GpuCopyBuffers, MoeHostChunkSpec, MoePagerLayout, MoePoolSpec, Role,
};
use infr_core::backend::Backend;
use infr_core::blockio::{BlockDesc, BlockExtent, BlockIo};

const EXPERTS: usize = 16;
const BYTES: usize = 64 * 1024;
const ROLES: [usize; 3] = [501, 502, 503];

fn pattern(role: usize, expert: usize) -> u32 {
    0x1200_0000 | ((role as u32) << 16) | expert as u32
}

struct PatternIo;
impl BlockIo for PatternIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> infr_core::error::Result<()> {
        assert_eq!(desc.nbytes(), EXPERTS * BYTES);
        for expert in 0..EXPERTS {
            let bytes = pattern(desc.id as usize, expert).to_le_bytes();
            for word in dst[expert * BYTES..(expert + 1) * BYTES].chunks_exact_mut(4) {
                word.copy_from_slice(&bytes);
            }
        }
        Ok(())
    }
}

fn upload(be: &VulkanBackend, buffer: &dyn Buffer, words: &[u32]) {
    be.upload(buffer, bytemuck::cast_slice(words)).unwrap();
}
fn read(be: &VulkanBackend, buffer: &dyn Buffer) -> Vec<u32> {
    let mut words = vec![0; buffer.len_bytes() / 4];
    be.download(buffer, bytemuck::cast_slice_mut(&mut words))
        .unwrap();
    words
}

fn backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev =
        Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("central runner selects idle GPU"));
    cfg.paging.host_dma = true;
    cfg.paging.expert_prefetch = false;
    cfg.kernels.vulkan.decode_gpu_resident = true;
    cfg.kernels.vulkan.decode_gpu_resident_copy = Some(true);
    cfg.kernels.vulkan.decode_gpu_resident_copy_overlap = true;
    cfg.kernels.vulkan.decode_gpu_resident_copy_groups = 2;
    VulkanBackend::new_with(Arc::new(cfg)).unwrap()
}

fn run(used: usize, cpu: bool, fallback: bool, with_event: bool) {
    run_route(used, cpu, fallback, with_event, 1, 1, false);
}

fn run_route(
    used: usize,
    cpu: bool,
    fallback: bool,
    with_event: bool,
    rows: usize,
    gating: u32,
    distinct: bool,
) {
    let count = rows * used;
    let physical_count = rows * (used + 1);
    let be = backend();
    assert!(device_descriptors(&be, true));
    assert!(subgroup_capabilities(be.shared.compute_subgroup_operations));
    assert!(be.shared.conditional_rendering.is_some());
    be.init_moe_pager(MoePagerLayout {
        load_reserve_bytes: 0,
        n_blocks: EXPERTS,
        pools: vec![MoePoolSpec {
            slot_bytes: BYTES,
            n_slots: 128,
            min_enabled_slots: 48,
        }],
        host_tier: None,
        host_store_io: Some(Arc::new(PatternIo)),
        dynamic_state_reserve_bytes: 0,
        dynamic_state_max_allocation_bytes: 0,
        prefill_min_lane_bytes: 0,
        runtime_reserve_bytes: (64 * BYTES) as u64,
        host_chunks: vec![MoeHostChunkSpec {
            base_offset: 0,
            bytes: 3 * EXPERTS * BYTES,
        }],
        prefill_target_lanes: 1,
        prefill_cache_bytes: (48 * BYTES) as u64,
    })
    .unwrap();
    assert_eq!(be.unified_vram().unwrap().shard_sizes(), vec![128 * BYTES]);
    for (r, role) in [Role::Gate, Role::Up, Role::Down].into_iter().enumerate() {
        be.register_paged_expert(
            role,
            ROLES[r],
            ExpertSource {
                bank: Arc::new(vec![0u8; EXPERTS * BYTES]),
                stride_bytes: BYTES,
                layer_base: 0,
                host_offset: r * EXPERTS * BYTES,
                file: Some(BlockDesc {
                    id: r as u32,
                    extents: vec![BlockExtent {
                        offset: 0,
                        len: EXPERTS * BYTES,
                    }],
                }),
            },
            EXPERTS,
        )
        .unwrap();
    }
    let transfer_sources = {
        let mut guard = be.moe_pager().lock().unwrap();
        guard.as_mut().unwrap().take_transfer_sources()
    };
    // Transport import and unified claims must see the installed session, with no
    // pager mutex held while the backend allocates or retires Expert filler.
    let plan = be.build_session_transfer_plan(transfer_sources);
    be.moe_pager()
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .install_transfer_plan(Arc::new(plan));
    let hits: Vec<_> = (1..(used + usize::from(rows == 2)) as u32).collect();
    for role in ROLES {
        let push = {
            let mut guard = be.moe_pager().lock().unwrap();
            let session = guard.as_mut().unwrap();
            session.begin_role_batches(&[role]).unwrap();
            session
                .push_roles_cpu(&be, &[(role, &hits)], false)
                .unwrap()
        };
        push.complete_without_recorder(&be).unwrap();
    }
    let (windows, tape, mut sources, seed) = {
        let mut guard = be.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        let mut cursor = 0;
        let windows = ROLES.map(|role| {
            session
                .resident_lut_window(&mut cursor, role, EXPERTS)
                .unwrap()
        });
        (
            windows,
            session.retained_tape(),
            session.gpu_copy_sources().unwrap().unwrap(),
            crate::pager::owned_seed(session, 0).unwrap(),
        )
    };
    if cpu {
        sources.test_cpu_only_source(2 * EXPERTS);
    }
    // Keep the real session installed throughout metadata/workspace allocations.
    let copy = GpuCopyBuffers::new_owned(&be, sources, 1, seed)
        .unwrap()
        .unwrap();
    let extra = {
        let guard = be.moe_pager().lock().unwrap();
        copy.role_params(guard.as_ref().unwrap(), &ROLES).unwrap()
    };
    let mapped = |n| be.alloc_mapped_arena_bda(n).unwrap().0;
    let ids = mapped((count + 2 * rows) * 4);
    let control = be.alloc_mapped_predicate(RESIDENT_CONTROL_BYTES).unwrap();
    let journal = mapped(RESIDENT_JOURNAL_BYTES);
    let logits = mapped(rows * EXPERTS * 4);
    let weights = mapped(count * 4);
    let reference_ids = mapped(count * 4);
    let reference_weights = mapped(count * 4);
    let gate = mapped(physical_count * 4);
    let up = mapped(physical_count * 4);
    let activation = mapped(physical_count * 4);
    upload(&be, ids.as_ref(), &vec![0; count + 2 * rows]);
    upload(&be, control.as_ref(), &vec![0; RESIDENT_CONTROL_BYTES / 4]);
    upload(&be, journal.as_ref(), &vec![0; RESIDENT_JOURNAL_WORDS]);
    upload(
        &be,
        logits.as_ref(),
        &(0..rows * EXPERTS)
            .map(|at| {
                let row = at / EXPERTS;
                let expert = at % EXPERTS;
                let rank = (expert + EXPERTS - usize::from(distinct && row == 1)) % EXPERTS;
                // Two-row sigmoid saturation must use general selection ties,
                // not the single-row wrapper's raw-logit ordering.
                let shift = if rows == 2 && gating == 1 { 20.0 } else { 0.0 };
                (shift + (EXPERTS - rank) as f32).to_bits()
            })
            .collect::<Vec<_>>(),
    );
    upload(&be, weights.as_ref(), &vec![0xdead_beef; count]);
    upload(&be, gate.as_ref(), &vec![1f32.to_bits(); physical_count]);
    upload(&be, up.as_ref(), &vec![2f32.to_bits(); physical_count]);
    upload(
        &be,
        activation.as_ref(),
        &vec![(-99f32).to_bits(); physical_count],
    );
    // Execute the ordinary router as the bitwise reference before owner capture.
    let reference = be.recorder().unwrap();
    reference.moe_topk_with_tap(
        logits.as_ref(),
        reference_ids.as_ref(),
        reference_weights.as_ref(),
        logits.as_ref(),
        rows,
        EXPERTS,
        used,
        1.375,
        gating,
        true,
        false,
        0,
        0,
        logits.as_ref(),
        false,
        None,
    );
    reference.publish_host_reads();
    reference.finish().unwrap();
    let expected_ids = read(&be, reference_ids.as_ref());
    let expected_weights = read(&be, reference_weights.as_ref());
    {
        let mut guard = be.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        // All runtime claims are complete. Refresh any filler retirements before
        // publishing this owner; subsequent unexpected claims must respect ownership.
        copy.begin(&be, session).unwrap();
        session.gpu_decode_owner = Some(Arc::clone(&copy));
    }
    let _counting = if rows >= 3 {
        test_many_row_scope(&be)
    } else {
        test_scope(&be, true)
    };
    let event = if with_event {
        unsafe {
            be.shared
                .device
                .create_event(&vk::EventCreateInfo::default(), None)
                .unwrap()
        }
    } else {
        vk::Event::null()
    };
    let params = [
        count as u32,
        EXPERTS as u32,
        windows[0],
        windows[1],
        windows[2],
        0,
        used as u32,
        1 << used,
    ];
    let rec = be.recorder().unwrap();
    rec.gpu_resident_check(
        ids.as_ref(),
        tape.as_ref(),
        control.as_ref(),
        journal.as_ref(),
        [0; 6],
    );
    rec.begin_immutable_conditional_at(control.as_ref(), RESIDENT_SEED_WORD);
    let route_params = TopkParams {
        rows,
        experts: EXPERTS,
        used,
        scale: 1.375,
        gating,
        norm_w: true,
        has_bias: false,
        groups: 0,
        groups_used: 0,
        hash: false,
    };
    assert!(
        rec.defer_owned_topk(
            &copy,
            control.as_ref(),
            logits.as_ref(),
            weights.as_ref(),
            [extra[0], extra[1], extra[2]],
            0,
            TopkParams {
                rows: if rows == 1 {
                    2
                } else if rows == 2 {
                    4
                } else {
                    9
                },
                ..route_params
            },
            true,
            false
        )
        .is_none()
    );
    let mut route = rec
        .defer_owned_topk(
            &copy,
            control.as_ref(),
            logits.as_ref(),
            weights.as_ref(),
            [extra[0], extra[1], extra[2]],
            0,
            route_params,
            true,
            false,
        )
        .expect("certified ordinary route");
    assert_eq!(
        route.mode,
        if rows == 1 {
            OwnedTopkMode::Wave32
        } else if rows == 2 {
            OwnedTopkMode::TwoRowGeneral
        } else {
            OwnedTopkMode::ManyRowGeneral
        }
    );
    rec.end_conditional();
    let captured = route;
    if fallback {
        // Exercise the same fallback as a ten-binding-only device without lying
        // about bank geometry or adding a global production debug switch.
        route.overlap_descriptors = false;
    }
    let prepare_before = rec.dispatches();
    rec.gpu_resident_copy_prepare_overlap_topk(
        ids.as_ref(),
        tape.as_ref(),
        control.as_ref(),
        journal.as_ref(),
        &copy,
        params,
        extra,
        event,
        route,
    );
    if rows >= 3 && fallback {
        assert_eq!(
            rec.dispatches() - prepare_before,
            3,
            "ordinary router plus split owner"
        );
        let names = be.built_kernel_names();
        assert!(names.contains(&"gpu_owned_prepare_fast_split_overlap"));
        assert!(names.contains(&"gpu_owned_prepare_slow_subgroup_overlap"));
        assert!(
            !names.contains(&"gpu_owned_prepare_fused_overlap"),
            "ordinary many-row non-TopK admission must stay split"
        );
    }
    rec.begin_immutable_conditional_at(control.as_ref(), 6);
    let scope = rec.independent_dispatch_scope();
    rec.gpu_resident_silu_bank(
        gate.as_ref(),
        up.as_ref(),
        ids.as_ref(),
        activation.as_ref(),
        [rows as u32, used as u32, used as u32 + 1, 1],
        false,
    );
    rec.gpu_resident_copy_payload_overlap(&copy, journal.as_ref(), 0);
    drop(scope);
    rec.end_conditional();
    rec.gpu_resident_copy_finish_overlap(&copy, journal.as_ref(), 0);
    rec.begin_immutable_conditional_at(copy.miss_predicates.as_ref().unwrap().as_ref(), 0);
    rec.gpu_resident_silu_bank(
        gate.as_ref(),
        up.as_ref(),
        ids.as_ref(),
        activation.as_ref(),
        [rows as u32, used as u32, used as u32 + 1, 1],
        true,
    );
    rec.end_conditional();
    rec.publish_host_reads();
    rec.finish().unwrap();
    if with_event {
        assert!(unsafe { be.shared.device.get_event_status(event).unwrap() });
        unsafe { be.shared.device.destroy_event(event, None) };
    }
    let words: [u32; RESIDENT_JOURNAL_WORDS] = read(&be, journal.as_ref()).try_into().unwrap();
    assert_eq!(words[RESIDENT_JOURNAL_STATUS_WORD], if cpu { 5 } else { 4 });
    let job_count = words[RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
    assert_eq!(job_count, 3);
    let jobs = read(&be, copy.jobs.as_ref());
    let entries = read(&be, copy.overlay.as_ref());
    let args = read(&be, copy.miss_predicates.as_ref().unwrap().as_ref());
    assert_eq!(args[0], if cpu { 0 } else { 3 });
    assert!(
        args[1] > 0,
        "fixture requires a real mapped-source payload path"
    );
    for job in jobs[..job_count * 8].chunks_exact(8) {
        let mapped = job[0] != 0 || job[1] != 0;
        assert_eq!(entries[job[5] as usize * 4 + 2], if mapped { 1 } else { 2 });
        if mapped {
            let address = u64::from(job[2]) | (u64::from(job[3]) << 32);
            let mut payload = vec![0u32; BYTES / 4];
            {
                let guard = be.moe_pager().lock().unwrap();
                let arena = guard.as_ref().unwrap().arena(ROLES[0]).unwrap();
                let offset =
                    usize::try_from(address.checked_sub(arena.device_addr().unwrap()).unwrap())
                        .unwrap();
                assert!(
                    offset + BYTES <= arena.len_bytes(),
                    "small fixture expects one arena shard"
                );
                // download_range uses its raw staging allocator, not a unified claim.
                be.download_range(arena, offset, bytemuck::cast_slice_mut(&mut payload))
                    .unwrap();
            }
            assert!(
                payload
                    .iter()
                    .all(|&v| v == pattern(job[5] as usize / EXPERTS, 0))
            );
        }
    }
    let old_hits = words[RESIDENT_JOURNAL_MASK_WORD..RESIDENT_JOURNAL_MASK_WORD + rows].to_vec();
    let output = read(&be, activation.as_ref());
    let routed = read(&be, ids.as_ref());
    assert_eq!(&routed[..count], expected_ids.as_slice());
    assert_eq!(read(&be, weights.as_ref()), expected_weights);
    for row in 0..rows {
        let mask = routed[row * used..(row + 1) * used]
            .iter()
            .enumerate()
            .fold(0u32, |mask, (rank, &id)| {
                mask | (u32::from(id != 0) << rank)
            });
        assert_eq!(old_hits[row], mask);
    }
    if cpu {
        for (rank, &id) in routed[..count].iter().enumerate() {
            if id == 0 {
                let physical_rank = rank / used * (used + 1) + rank % used;
                assert_eq!(f32::from_bits(output[physical_rank]), -99.0);
            }
        }
        let clock = read(&be, copy.candidates.as_ref())[8..12].to_vec();
        let weights_before = read(&be, weights.as_ref());
        let upload_plan = {
            let guard = be.moe_pager().lock().unwrap();
            copy.plan_owned_upload(guard.as_ref().unwrap(), 0, &words)
                .unwrap()
        };
        let (push, leases) = upload_plan.execute(&be).unwrap();
        push.complete_without_recorder(&be).unwrap();
        upload(
            &be,
            logits.as_ref(),
            &(0..rows * EXPERTS)
                .map(|at| ((at % EXPERTS) as f32 * 0.25).to_bits())
                .collect::<Vec<_>>(),
        );
        let rec = be.recorder().unwrap();
        rec.publish_owned_cpu(&copy, journal.as_ref(), 0);
        rec.gpu_resident_check(
            ids.as_ref(),
            tape.as_ref(),
            control.as_ref(),
            journal.as_ref(),
            [0; 6],
        );
        let mut recovery = extra;
        recovery[3] = 1;
        assert_eq!(
            gpu_copy_prepare_push(params, true, recovery, copy.copy_groups)[12],
            1
        );
        let recovery_before = rec.dispatches();
        if rows >= 3 {
            assert!(
                !be.built_kernel_names()
                    .contains(&"gpu_owned_prepare_fused_overlap"),
                "recovery selection must not be inferred from an already-built kernel"
            );
        }
        // Even a retained route must not run again after the first admission.
        rec.gpu_resident_copy_prepare_overlap_topk(
            ids.as_ref(),
            tape.as_ref(),
            control.as_ref(),
            journal.as_ref(),
            &copy,
            params,
            recovery,
            vk::Event::null(),
            captured,
        );
        let fused_recovery = rows == 1
            || (rows == 2 && matches!(used, 8 | 10))
            || ((3..=8).contains(&rows) && matches!(used, 8 | 10))
            || (rows == 8 && used == 16);
        assert_eq!(
            rec.dispatches() - recovery_before,
            if fused_recovery { 1 } else { 2 },
            "real recovery must select fused or split without rerouting"
        );
        let names = be.built_kernel_names();
        assert_eq!(
            names.contains(&"gpu_owned_prepare_fused_overlap"),
            fused_recovery
        );
        if !fused_recovery {
            assert!(names.contains(&"gpu_owned_prepare_fast_split_overlap"));
            assert!(names.contains(&"gpu_owned_prepare_slow_subgroup_overlap"));
        }
        rec.gpu_owned_recovery_miss_masks(ids.as_ref(), count, used, &old_hits);
        rec.begin_immutable_conditional_at(control.as_ref(), 6);
        rec.gpu_resident_silu_bank(
            gate.as_ref(),
            up.as_ref(),
            ids.as_ref(),
            activation.as_ref(),
            [rows as u32, used as u32, used as u32 + 1, 1],
            false,
        );
        rec.end_conditional();
        rec.publish_host_reads();
        rec.finish().unwrap();
        drop(leases);
        assert_eq!(&read(&be, ids.as_ref())[..count], expected_ids.as_slice());
        assert_eq!(read(&be, weights.as_ref()), weights_before);
        assert_eq!(&read(&be, copy.candidates.as_ref())[8..12], &clock);
        assert_eq!(read(&be, journal.as_ref())[RESIDENT_JOURNAL_STATUS_WORD], 1);
        assert_eq!(
            read(&be, copy.miss_predicates.as_ref().unwrap().as_ref()),
            vec![0, 0, 1, 1]
        );
    }
    for bits in read(&be, activation.as_ref()) {
        assert!((f32::from_bits(bits) - 2.0 / (1.0 + (-1f32).exp())).abs() < 1e-5);
    }
    let counts = test_counts(&be);
    assert_eq!(counts.fused, usize::from(!fallback));
    assert_eq!(counts.routers, usize::from(fallback));
    assert_eq!(counts.admissions, 1 + usize::from(cpu));
    assert_eq!(counts.recovery, usize::from(cpu));
    copy.finish_owned_forward();
}

#[test]
#[ignore = "central serial GPU runner; real owner, mapped payload, no-notice and CPU recovery"]
fn gpu_owned_topk_copy_overlap_recording_entrypoint_and_recovery() {
    run(8, false, false, false);
    run(10, true, false, true);
}

#[test]
#[ignore = "central serial GPU runner; declined fusion executes original immutable route once"]
fn gpu_owned_topk_copy_overlap_recording_fallback_routes_once() {
    run(8, false, true, true);
}

#[test]
#[ignore = "central serial GPU runner; production two-row capture, general router bits and CPU recovery"]
fn gpu_owned_topk_two_row_recording_entrypoint_and_recovery() {
    run_route(8, false, false, false, 2, 0, true);
    run_route(10, true, false, true, 2, 1, false);
}

#[test]
#[ignore = "central serial GPU runner; deferred two-row fallback must run actual general TopK twice"]
fn gpu_owned_topk_two_row_recording_fallback_general_router() {
    run_route(8, false, true, true, 2, 1, false);
    run_route(10, false, true, false, 2, 0, true);
}

#[test]
#[ignore = "central serial GPU runner; explicit ManyRow opt-in, real entrypoint and general-router fallback"]
fn gpu_owned_topk_many_row_recording_entrypoint_and_fallback() {
    run_route(8, false, false, false, 3, 0, false);
    run_route(10, true, false, true, 4, 1, false);
    run_route(8, false, true, false, 3, 1, false);
}

#[test]
#[ignore = "central serial GPU runner; real status5 CPU upload/publication, recovery-only fused selection and ordinary split control"]
fn gpu_owned_recovery_fused_recording_entrypoint_qualified_shapes() {
    for (rows, used) in [(3, 8), (4, 10), (8, 16)] {
        run_route(used, true, false, true, rows, 0, false);
    }
    // The same real session must keep ordinary non-TopK admission split, while
    // its following status5 recovery selects the new one-dispatch owner path.
    run_route(10, true, true, true, 4, 0, false);
    run_route(16, true, false, true, 4, 0, false); // Unqualified used16 shape stays split.
}
