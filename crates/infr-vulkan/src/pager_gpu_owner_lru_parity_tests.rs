//! Characterization of the legacy schedule, not authorization to change LRU policy.
//! CPU tests use real Pager decisions; the ignored test also calls real session/upload methods.
use super::*;
use infr_core::pager::{Pager, Resolution};

fn restored(snapshot: &PagerSnapshot) -> Pager {
    let mut pager = Pager::new(snapshot.n_slots());
    pager.restore_handoff(snapshot).unwrap();
    pager
}

fn order(snapshot: &PagerSnapshot) -> Vec<u32> {
    snapshot.resident_lru().iter().map(|&(id, _)| id).collect()
}

#[derive(Debug)]
struct Touches {
    ids: Vec<u32>,
    victims: Vec<u32>,
    bytes: usize,
}

fn touch(pager: &mut Pager, block: u32, bytes: usize, result: &mut Touches) {
    result.ids.push(block);
    if let Resolution::Miss { evicted, .. } = pager.touch(block) {
        result.bytes += bytes;
        result.victims.extend(evicted);
    }
}

fn joint_hits(pagers: &[Pager], pools: [usize; 3], bases: [u32; 3], ids: &[u32]) -> Vec<bool> {
    ids.iter()
        .map(|&id| (0..3).all(|role| pagers[pools[role]].slot_of(bases[role] + id).is_some()))
        .collect()
}

// Mirrors touch_routed_hits followed by push_roles_cpu(false), or copy.replay's
// hits-first / admit_at-or-touch order. No dedup: partial-resident roles and aliases touch too.
fn legacy_pager_calls(
    pagers: &mut [Pager],
    pools: [usize; 3],
    bases: [u32; 3],
    ids: &[u32],
) -> Touches {
    let hits = joint_hits(pagers, pools, bases, ids);
    for pager in pagers.iter_mut() {
        pager.begin_batch();
    }
    let mut result = Touches {
        ids: vec![],
        victims: vec![],
        bytes: 0,
    };
    for phase_hit in [true, false] {
        for role in 0..3 {
            for (i, &id) in ids.iter().enumerate() {
                if hits[i] == phase_hit {
                    touch(
                        &mut pagers[pools[role]],
                        bases[role] + id,
                        SLOT_BYTES,
                        &mut result,
                    );
                }
            }
        }
    }
    result
}

// A real-Pager request-order counterpart, NOT a GPU execution or a manual LRU implementation.
// The fixture starts with enough older, unrequested victims for this admission.
fn request_order_pager_calls(
    pagers: &mut [Pager],
    pools: [usize; 3],
    bases: [u32; 3],
    ids: &[u32],
) -> Touches {
    for pager in pagers.iter_mut() {
        pager.begin_batch();
    }
    let mut result = Touches {
        ids: vec![],
        victims: vec![],
        bytes: 0,
    };
    for role in 0..3 {
        for &id in ids {
            touch(
                &mut pagers[pools[role]],
                bases[role] + id,
                SLOT_BYTES,
                &mut result,
            );
        }
    }
    result
}

fn legacy_replay_calls(
    pagers: &mut [Pager],
    pools: [usize; 3],
    bases: [u32; 3],
    ids: &[u32],
    placements: &std::collections::HashMap<u32, u32>,
) {
    let hits = joint_hits(pagers, pools, bases, ids);
    for pager in pagers.iter_mut() {
        pager.begin_batch();
    }
    for role in 0..3 {
        for (i, &id) in ids.iter().enumerate() {
            if hits[i] {
                assert!(matches!(
                    pagers[pools[role]].touch(bases[role] + id),
                    Resolution::Hit { .. }
                ));
            }
        }
    }
    for role in 0..3 {
        for (i, &id) in ids.iter().enumerate() {
            if hits[i] {
                continue;
            }
            let block = bases[role] + id;
            let pager = &mut pagers[pools[role]];
            if let Some(&slot) = placements.get(&block) {
                pager.admit_at(block, slot).unwrap();
            } else {
                assert!(matches!(pager.touch(block), Resolution::Hit { .. }));
            }
        }
    }
}

#[test]
fn legacy_joint_triplet_schedule_preserves_duplicate_ids_shared_pools_and_aliases() {
    for used in [8usize, 10] {
        for duplicate_rows in [false, true] {
            for (pools, bases) in [
                ([0, 0, 0], [0, 32, 64]),
                ([0, 0, 1], [0, 32, 64]),
                ([0, 1, 2], [0, 32, 64]),
                ([0, 0, 0], [0, 0, 64]),
            ] {
                let ids: Vec<u32> = (0..2 * used)
                    .map(|i| if duplicate_rows { i % used } else { i } as u32)
                    .collect();
                let mut wanted = vec![Vec::new(); pools.into_iter().max().unwrap() + 1];
                for role in 0..3 {
                    for &id in &ids {
                        let blocks = &mut wanted[pools[role]];
                        let block = bases[role] + id;
                        if !blocks.contains(&block) {
                            blocks.push(block);
                        }
                    }
                }
                let snapshots: Vec<_> = wanted
                    .iter()
                    .enumerate()
                    .map(|(pool, blocks)| {
                        let mut pager = Pager::new(blocks.len());
                        let missing = blocks.iter().filter(|&&block| block % 32 == 0).count();
                        for i in 0..missing {
                            pager.touch(1000 + pool as u32 * 100 + i as u32);
                        }
                        for &block in blocks.iter().filter(|&&block| block % 32 != 0) {
                            pager.touch(block);
                        }
                        pager.snapshot_for_handoff().unwrap()
                    })
                    .collect();
                let mut legacy: Vec<_> = snapshots.iter().map(restored).collect();
                let mut request: Vec<_> = snapshots.iter().map(restored).collect();
                let a = legacy_pager_calls(&mut legacy, pools, bases, &ids);
                let b = request_order_pager_calls(&mut request, pools, bases, &ids);
                assert_eq!(a.ids.len(), ids.len() * 3, "do not dedup actual calls");
                assert_eq!(a.bytes, b.bytes);
                assert_eq!(
                    a.victims, b.victims,
                    "first admission uses the same cold victims"
                );
                let mut replay: Vec<_> = snapshots.iter().map(restored).collect();
                let mut placements = std::collections::HashMap::new();
                for role in 0..3 {
                    for &id in &ids {
                        let block = bases[role] + id;
                        if replay[pools[role]].slot_of(block).is_none() {
                            placements.insert(block, legacy[pools[role]].slot_of(block).unwrap());
                        }
                    }
                }
                legacy_replay_calls(&mut replay, pools, bases, &ids, &placements);
                for (a, b) in legacy.iter().zip(&replay) {
                    assert_eq!(
                        a.snapshot_for_handoff().unwrap(),
                        b.snapshot_for_handoff().unwrap()
                    );
                }
                let old = legacy[0].snapshot_for_handoff().unwrap();
                let new = request[0].snapshot_for_handoff().unwrap();
                assert_ne!(order(&old), order(&new));
                assert_eq!(order(&old)[0], 1);
                assert_eq!(order(&new)[0], 0);
                legacy[0].begin_batch();
                request[0].begin_batch();
                assert!(matches!(
                    legacy[0].touch(9000),
                    Resolution::Miss {
                        evicted: Some(1),
                        ..
                    }
                ));
                assert!(matches!(
                    request[0].touch(9000),
                    Resolution::Miss {
                        evicted: Some(0),
                        ..
                    }
                ));
                let mut old_bytes = Touches {
                    ids: vec![],
                    victims: vec![],
                    bytes: 0,
                };
                let mut new_bytes = Touches {
                    ids: vec![],
                    victims: vec![],
                    bytes: 0,
                };
                legacy[0].begin_batch();
                request[0].begin_batch();
                touch(&mut legacy[0], 0, SLOT_BYTES, &mut old_bytes);
                touch(&mut request[0], 0, SLOT_BYTES, &mut new_bytes);
                assert_eq!((old_bytes.bytes, new_bytes.bytes), (0, SLOT_BYTES));
            }
        }
    }
}

#[test]
fn legacy_joint_miss_retouches_partial_resident_roles_in_miss_phase() {
    let bases = [0, 32, 64];
    let mut pager = Pager::new(6);
    for block in [1000, 1, 32, 33, 64, 65] {
        pager.touch(block);
    }
    let initial = pager.snapshot_for_handoff().unwrap();
    let mut pagers = [pager];
    // Expert 0 is a joint miss even though its Up and Down roles already exist.
    assert_eq!(
        joint_hits(&pagers, [0; 3], bases, &[0, 1, 0, 1]),
        [false, true, false, true]
    );
    let calls = legacy_pager_calls(&mut pagers, [0; 3], bases, &[0, 1, 0, 1]);
    assert_eq!(calls.ids, [1, 1, 33, 33, 65, 65, 0, 0, 32, 32, 64, 64]);
    assert_eq!(calls.victims, [1000]);
    assert_eq!(calls.bytes, SLOT_BYTES);
    assert_eq!(
        order(&pagers[0].snapshot_for_handoff().unwrap()),
        [1, 33, 65, 0, 32, 64]
    );
    let mut replay = [restored(&initial)];
    let placements = [(0, pagers[0].slot_of(0).unwrap())].into_iter().collect();
    legacy_replay_calls(&mut replay, [0; 3], bases, &[0, 1, 0, 1], &placements);
    assert_eq!(
        replay[0].snapshot_for_handoff().unwrap(),
        pagers[0].snapshot_for_handoff().unwrap()
    );
}

#[derive(Debug)]
struct Cohort {
    snapshot: PagerSnapshot,
    victims: Vec<u32>,
    bytes: usize,
}

fn removed(before: &PagerSnapshot, after: &PagerSnapshot) -> Vec<u32> {
    order(before)
        .into_iter()
        .filter(|id| !after.resident_lru().iter().any(|&(next, _)| next == *id))
        .collect()
}

fn legacy_session_cohort(f: &mut Fixture, be: &VulkanBackend, ids: &[u32]) -> Cohort {
    let before = f.snapshot();
    let mask = f.session.routed_roles_resident_mask(&ROLES, ids).unwrap();
    let hits: Vec<_> = ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| (mask & (1 << i) != 0).then_some(id))
        .collect();
    let misses: Vec<_> = ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| (mask & (1 << i) == 0).then_some(id))
        .collect();
    f.session.begin_role_batches(&ROLES).unwrap();
    f.session.touch_routed_hits(&ROLES, &hits).unwrap();
    let push = f
        .session
        .push_roles_cpu(be, &ROLES.map(|role| (role, misses.as_slice())), false)
        .unwrap();
    let bytes = push.bytes();
    push.complete_without_recorder(be).unwrap();
    let snapshot = f.snapshot();
    Cohort {
        victims: removed(&before, &snapshot),
        snapshot,
        bytes,
    }
}

fn owner_prepare(
    f: &Fixture,
    be: &VulkanBackend,
    ids: &dyn Buffer,
    routed: &[u32],
    used: usize,
) -> [u32; RESIDENT_JOURNAL_WORDS] {
    let rows = routed.len() / used;
    let mut words = routed.to_vec();
    words.resize(routed.len() + rows * 2, 0);
    be.upload(ids, bytemuck::cast_slice(&words)).unwrap();
    let mut control = vec![0u32; RESIDENT_CONTROL_BYTES / 4];
    control[0] = 1;
    control[1] = NONE;
    be.upload(f.control.as_ref(), bytemuck::cast_slice(&control))
        .unwrap();
    let notice = unsafe {
        be.shared
            .device
            .create_event(&vk::EventCreateInfo::default(), None)
    }
    .unwrap();
    let rec = be.recorder().unwrap();
    rec.gpu_resident_copy_check(
        ids,
        f.tape.as_ref(),
        f.control.as_ref(),
        f.journal.as_ref(),
        f.copy.as_ref(),
        [
            routed.len() as u32,
            EXPERTS as u32,
            f.windows[0],
            f.windows[1],
            f.windows[2],
            0,
            used as u32,
            0,
        ],
        f.extra,
        notice,
    );
    rec.publish_host_reads();
    rec.finish().unwrap();
    let signaled = unsafe { be.shared.device.get_event_status(notice) }.unwrap();
    unsafe { be.shared.device.destroy_event(notice, None) };
    assert!(signaled);
    read_mapped_words(f.journal.as_ref())
        .unwrap()
        .try_into()
        .unwrap()
}

#[test]
#[ignore = "central serial GPU runner only; actual legacy session vs GPU-owner LRU characterization, not parity acceptance"]
fn owned_session_legacy_lru_first_divergence_next_victims_and_bytes() {
    let be = backend();
    be.with_unified_exclusive(|| {
        // Alias Gate/Up, shared physical pool, and duplicate IDs within/across two rows.
        let routes: [&[u32]; 3] = [&[2, 1, 1, 2, 1, 1], &[0, 0], &[2, 2]];
        let (initial, legacy) = {
            let mut f = Fixture::new(&be);
            let initial = f.initial.clone();
            let legacy: Vec<_> = routes.iter().map(|ids| legacy_session_cohort(&mut f, &be, ids)).collect();
            (initial, legacy)
        };
        let mut f = Fixture::new(&be);
        assert_eq!(f.initial, initial, "same physical-slot snapshot before either schedule");
        let ids = be.alloc_mapped_arena_bda(10 * 4).unwrap().0;
        let mut gpu = Vec::new();
        for (index, routed) in routes.iter().enumerate() {
            let before = f.snapshot();
            f.copy.begin(&be, &mut f.session).unwrap();
            let words = owner_prepare(&f, &be, ids.as_ref(), routed, if index == 0 { 3 } else { 1 });
            assert_eq!(words[RESIDENT_JOURNAL_STATUS_WORD], 5);
            let jobs = read_mapped_range(f.copy.jobs.as_ref(), 0,
                words[RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize * 8).unwrap();
            let victims: Vec<_> = jobs.chunks_exact(8).map(|job| {
                before.resident_lru().iter()
                    .find(|&&(_, slot)| slot == job[6]).unwrap().0
            }).collect();
            let (push, leases) = f.copy.prepare_owned_upload(&f.session, &be, 0, &words).unwrap();
            let bytes = push.bytes();
            assert_eq!(bytes, words[RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize * SLOT_BYTES);
            push.complete_without_recorder(&be).unwrap();
            drop(leases);
            let rec = be.recorder().unwrap();
            rec.publish_owned_cpu(f.copy.as_ref(), f.journal.as_ref(), 0);
            rec.publish_host_reads();
            rec.finish().unwrap();
            f.copy.finish_owned_forward();
            // Real drained handoff observes the GPU FIFO; no hand-sorted stamp oracle.
            f.session.handoff_gpu_decode(&be).unwrap();
            let snapshot = f.snapshot();
            for &(block, slot) in snapshot.resident_lru() {
                f.assert_payload(&be, block, slot);
            }
            assert_eq!(victims, removed(&before, &snapshot), "job-order victims match drained ownership");
            gpu.push(Cohort { victims, snapshot, bytes });
        }
        assert_eq!(order(&legacy[0].snapshot), [1, 9, 2, 10]);
        assert_eq!(order(&gpu[0].snapshot), [2, 1, 10, 9]);
        assert_eq!(legacy[0].victims, gpu[0].victims);
        assert_eq!((legacy[0].bytes, gpu[0].bytes), (2 * SLOT_BYTES, 2 * SLOT_BYTES));
        assert_eq!(legacy[1].victims, [1, 9]);
        assert_eq!(gpu[1].victims, [2, 1]);
        assert_eq!((legacy[2].bytes, gpu[2].bytes), (0, SLOT_BYTES));
        let old_order = order(&legacy[0].snapshot);
        let gpu_order = order(&gpu[0].snapshot);
        let first_difference = old_order.iter().zip(&gpu_order).position(|(a, b)| a != b);
        assert_eq!(first_difference, Some(0));
        let mut model = [restored(&initial)];
        for (routed, actual) in routes.iter().zip(&legacy) {
            let calls = legacy_pager_calls(&mut model, [0; 3], [0, 0, 8], routed);
            assert_eq!(model[0].snapshot_for_handoff().unwrap(), actual.snapshot);
            assert_eq!(calls.bytes, actual.bytes);
        }
        eprintln!("LRU_LEGACY_PARITY_CHARACTERIZATION rows=2 alias_roles=true duplicate_ids=true initial_equal=true first_order_divergence={first_difference:?} legacy_order={old_order:?} gpu_order={gpu_order:?} legacy_next_victims={:?} gpu_next_victims={:?} third_legacy_bytes={} third_gpu_bytes={} actual_session_and_gpu_handoff=true drained_handoff_each_cohort=true no_policy_change=true", legacy[1].victims, gpu[1].victims, legacy[2].bytes, gpu[2].bytes);
    });
}

#[derive(Debug, PartialEq, Eq)]
struct AcceptanceJob {
    block: u32,
    slot: u32,
    victim: u32,
    bytes: usize,
}

#[derive(Debug)]
struct AcceptanceCohort {
    resident: Vec<(u32, u32)>,
    jobs: Vec<AcceptanceJob>,
    bytes: usize,
    status: Option<u32>,
}

fn acceptance_fixture(be: &VulkanBackend, partial: bool) -> Fixture {
    let mut f = Fixture::new(be);
    if partial {
        // One nonrouted cold victim, joint hit 1, and only Down:2 resident.
        // Legacy need not evict any protected role: placement policies agree here.
        f.session.handoff_gpu_decode(be).unwrap();
        for (role, expert) in [(GATE, 3u32), (DOWN, 1), (GATE, 1), (DOWN, 2)] {
            f.session.begin_role_batches(&[role]).unwrap();
            f.session
                .push_roles_cpu(be, &[(role, &[expert])], false)
                .unwrap()
                .complete_without_recorder(be)
                .unwrap();
        }
        f.session.gpu_decode_owner = Some(f.copy.clone());
    }
    f
}

fn legacy_acceptance_cohort(
    f: &mut Fixture,
    be: &VulkanBackend,
    routed: &[u32],
) -> AcceptanceCohort {
    // In-memory capture of REAL push decisions, not a second LRU simulator.
    // Take it before any stats/flush call; this test never writes a trace file.
    assert!(f.session.trace.is_none());
    f.session.trace = Some(PagerTrace {
        path: Default::default(),
        calls: 0,
        records: Vec::new(),
    });
    let result = legacy_session_cohort(f, be, routed);
    let trace = f.session.trace.take().unwrap();
    let jobs: Vec<_> = trace
        .records
        .iter()
        .filter(|record| !record.gpu_hit)
        .map(|record| AcceptanceJob {
            block: record.block_id,
            slot: result
                .snapshot
                .resident_lru()
                .iter()
                .find(|&&(block, _)| block == record.block_id)
                .unwrap()
                .1,
            victim: record.evicted,
            bytes: record.bytes as usize,
        })
        .collect();
    assert_eq!(
        jobs.iter().map(|job| job.bytes).sum::<usize>(),
        result.bytes
    );
    for &(block, slot) in result.snapshot.resident_lru() {
        f.assert_payload(be, block, slot);
    }
    AcceptanceCohort {
        resident: result.snapshot.resident_lru().to_vec(),
        jobs,
        bytes: result.bytes,
        status: None,
    }
}

// Test-only independent synchronous dispatch. Unlike eager_bind, descriptors retain
// arena suballocation offsets. Both built-in and external use the identical ABI/path.
fn acceptance_prepare(
    f: &Fixture,
    be: &VulkanBackend,
    ids: &dyn Buffer,
    routed: &[u32],
    used: usize,
    kernel: crate::ops::ComputeKernel,
) -> [u32; RESIDENT_JOURNAL_WORDS] {
    assert_eq!(routed.len(), 2 * used);
    let mut input = routed.to_vec();
    input.resize(routed.len() + 4, 0);
    be.upload(ids, bytemuck::cast_slice(&input)).unwrap();
    let mut control = vec![0u32; RESIDENT_CONTROL_BYTES / 4];
    control[0] = 1;
    control[1] = NONE;
    be.upload(f.control.as_ref(), bytemuck::cast_slice(&control))
        .unwrap();
    let mut push = [0u32; 31];
    push[..8].copy_from_slice(&[
        routed.len() as u32,
        EXPERTS as u32,
        f.windows[0],
        f.windows[1],
        f.windows[2],
        0,
        used as u32,
        0,
    ]);
    push[8] = 1;
    push[9..18].copy_from_slice(&f.extra);
    push[18] = f.copy.copy_groups;
    push[19..].copy_from_slice(
        &f.copy
            .owned_role_geometry([f.extra[0], f.extra[1], f.extra[2]], EXPERTS as u32)
            .expect("fixture must have certified role geometry"),
    );
    let buffers: [&dyn Buffer; 8] = [
        ids,
        f.tape.as_ref(),
        f.control.as_ref(),
        f.journal.as_ref(),
        f.copy.sources.as_ref(),
        f.copy.candidates.as_ref(),
        f.copy.overlay.as_ref(),
        f.copy.jobs.as_ref(),
    ];
    let infos: Vec<_> = buffers
        .iter()
        .map(|buffer| {
            let buffer = crate::as_vk_buf(*buffer).unwrap();
            vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .offset(buffer.sub_offset as u64)
                .range(buffer.size as u64)
        })
        .collect();
    let device = &be.shared.device;
    let set = if be.shared.push_descriptor.is_none() {
        Some(
            unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(kernel.desc_pool)
                        .set_layouts(&[kernel.ds_layout]),
                )
            }
            .unwrap()[0],
        )
    } else {
        None
    };
    let writes: Vec<_> = infos
        .iter()
        .enumerate()
        .map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set.unwrap_or(vk::DescriptorSet::null()))
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
    if set.is_some() {
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
    be.one_shot(|cmd| unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::TRANSFER
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::HOST_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
            &[],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
        if let Some(pd) = &be.shared.push_descriptor {
            pd.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &writes,
            );
        } else {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &[set.unwrap()],
                &[],
            );
        }
        device.cmd_push_constants(
            cmd,
            kernel.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push),
        );
        device.cmd_dispatch(cmd, 1, 1, 1);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::TRANSFER
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::HOST_READ
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE,
                )],
            &[],
            &[],
        );
    })
    .unwrap();
    // one_shot has drained the queue; no live descriptor or buffer reader remains.
    if set.is_some() {
        unsafe {
            device.reset_descriptor_pool(kernel.desc_pool, vk::DescriptorPoolResetFlags::empty())
        }
        .unwrap();
    }
    read_mapped_words(f.journal.as_ref())
        .unwrap()
        .try_into()
        .unwrap()
}

fn gpu_acceptance_cohort(
    f: &mut Fixture,
    be: &VulkanBackend,
    ids: &dyn Buffer,
    routed: &[u32],
    kernel: crate::ops::ComputeKernel,
) -> AcceptanceCohort {
    let before = f.snapshot();
    f.copy.begin(be, &mut f.session).unwrap();
    let words = acceptance_prepare(f, be, ids, routed, routed.len() / 2, kernel);
    let status = words[RESIDENT_JOURNAL_STATUS_WORD];
    let count = words[RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize;
    assert!(matches!(status, 1 | 5), "unexpected journal: {words:?}");
    let raw = read_mapped_range(f.copy.jobs.as_ref(), 0, count * 8).unwrap();
    let jobs: Vec<_> = raw
        .chunks_exact(8)
        .map(|job| {
            let block = job[5];
            let source = read_mapped_range(f.copy.sources.as_ref(), block as usize * 4, 4).unwrap();
            assert_eq!(
                &job[..2],
                &source[..2],
                "real source BDA, not a synthetic job"
            );
            assert_eq!(job[4], source[2]);
            let address = u64::from(job[2]) | (u64::from(job[3]) << 32);
            assert_eq!(address, f.session.pools[0].pager.slot_addr(job[6]).unwrap());
            AcceptanceJob {
                block,
                slot: job[6],
                bytes: job[4] as usize,
                victim: before
                    .resident_lru()
                    .iter()
                    .find(|&&(_, slot)| slot == job[6])
                    .unwrap()
                    .0,
            }
        })
        .collect();
    let bytes = if status == 5 {
        assert!(count > 0);
        let (push, leases) = f
            .copy
            .prepare_owned_upload(&f.session, be, 0, &words)
            .unwrap();
        let bytes = push.bytes();
        assert_eq!(bytes, jobs.iter().map(|job| job.bytes).sum::<usize>());
        push.complete_without_recorder(be).unwrap();
        drop(leases);
        let rec = be.recorder().unwrap();
        rec.publish_owned_cpu(f.copy.as_ref(), f.journal.as_ref(), 0);
        rec.publish_host_reads();
        rec.finish().unwrap();
        bytes
    } else {
        assert_eq!(count, 0, "all-hit must not service stale jobs");
        0
    };
    f.copy.finish_owned_forward();
    f.session.handoff_gpu_decode(be).unwrap();
    let snapshot = f.snapshot();
    for &(block, slot) in snapshot.resident_lru() {
        f.assert_payload(be, block, slot);
    }
    AcceptanceCohort {
        resident: snapshot.resident_lru().to_vec(),
        jobs,
        bytes,
        status: Some(status),
    }
}

#[test]
#[ignore = "parent serial GPU only; actual legacy/builtin/external ordered LRU and payload parity"]
fn owned_session_external_legacy_touch_order_acceptance() {
    let path = std::env::var_os("MOE4ALL_OWNER_LEGACY_TOUCH_ORDER_SPV")
        .expect("set to TMP legacy-touch-order-v73/legacy-touch-order-serial.spv");
    let external =
        ash::util::read_spv(&mut std::io::Cursor::new(std::fs::read(&path).unwrap())).unwrap();
    assert_eq!(external.first(), Some(&0x0723_0203));
    let be = backend();
    be.with_unified_exclusive(|| {
        let builtin = be.kernel_sg("owner_lru_acceptance_builtin", crate::gemm::gpu_owned_prepare_fused_spv(false), 8, 124, 32);
        let candidate = be.kernel_sg("owner_lru_acceptance_external_v73", &external, 8, 124, 32);
        // All cases share one physical pool and Gate/Up aliases. Partial cases leave
        // enough colder NONROUTED victims: this is not a universal protection-parity claim.
        for (case, partial, routes) in [
            ("historical_pressure", false, vec![vec![2, 1, 1, 2, 1, 1], vec![0, 0], vec![2, 2]]),
            ("duplicate_partial", true, vec![vec![2, 1, 1, 2, 1, 1], vec![1, 2, 1, 1, 2, 1], vec![2, 2]]),
            ("duplicate_hit_first", false, vec![vec![1, 2, 1, 1, 2, 1], vec![0, 0], vec![2, 2]]),
        ] {
            let (initial, legacy) = {
                let mut f = acceptance_fixture(&be, partial);
                let initial = f.snapshot().resident_lru().to_vec();
                let results: Vec<_> = routes.iter().map(|ids| legacy_acceptance_cohort(&mut f, &be, ids)).collect();
                (initial, results)
            };
            let mut results = Vec::new();
            for kernel in [builtin, candidate] {
                let mut f = acceptance_fixture(&be, partial);
                assert_eq!(f.snapshot().resident_lru(), initial.as_slice(), "{case}: same initial physical placement");
                let ids = be.alloc_mapped_arena_bda(10 * 4).unwrap().0;
                results.push(routes.iter().map(|routed| gpu_acceptance_cohort(&mut f, &be, ids.as_ref(), routed, kernel)).collect::<Vec<_>>());
            }
            let old = &results[0];
            let new = &results[1];
            for (index, (cpu, gpu)) in legacy.iter().zip(new).enumerate() {
                // Global GPU stamps advance by 3*count; legacy clocks are per pool.
                // Compare the actual logical order AND physical slots, never raw stamps.
                assert_eq!(gpu.resident, cpu.resident, "{case} cohort {index}: ordered (block,slot)");
                assert_eq!(gpu.jobs, cpu.jobs, "{case} cohort {index}: actual ordered copy/victim decisions");
                assert_eq!(gpu.bytes, cpu.bytes, "{case} cohort {index}: actual upload bytes");
                assert_eq!(gpu.status, Some(if cpu.bytes == 0 { 1 } else { 5 }));
            }
            for (index, (cpu, gpu)) in legacy.iter().zip(old).enumerate() {
                assert_eq!(gpu.resident, cpu.resident, "{case} builtin cohort {index}: ordered (block,slot)");
                assert_eq!(gpu.jobs, cpu.jobs, "{case} builtin cohort {index}: ordered copy/victim decisions");
                assert_eq!(gpu.bytes, cpu.bytes, "{case} builtin cohort {index}: upload bytes");
                assert_eq!(gpu.status, Some(if cpu.bytes == 0 { 1 } else { 5 }));
            }
            if case == "historical_pressure" {
                assert_eq!(old[0].jobs, new[0].jobs, "touch-only change must retain first placement");
                assert_eq!(old[0].resident.iter().map(|&(id, _)| id).collect::<Vec<_>>(), [1, 9, 2, 10]);
                assert_eq!((old[2].bytes, new[2].bytes), (0, 0));
                assert_eq!((old[2].status, new[2].status), (Some(1), Some(1)));
            }
            eprintln!("LRU_EXTERNAL_ACCEPTANCE case={case} external={path:?} cpu={legacy:?} builtin={old:?} candidate={new:?} actual_payload=true sorted=false drained_handoff_each_cohort=true touch_order_only=true");
        }
    });
}
