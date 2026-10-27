//! Real session/transport coverage for GPU-owned admission and drained host handoff.
use super::*;
use crate::recorder::{
    RESIDENT_CONTROL_BYTES, RESIDENT_JOURNAL_BYTES, RESIDENT_JOURNAL_JOB_COUNT_WORD,
    RESIDENT_JOURNAL_MASK_WORD, RESIDENT_JOURNAL_STATUS_WORD, RESIDENT_JOURNAL_WORDS,
};
use infr_core::{backend::Backend, blockio::BlockExtent, pager::PagerSnapshot};

#[path = "pager_gpu_owner_lru_parity_tests.rs"]
mod lru_parity_tests;

const SLOT_BYTES: usize = 64 * 1024;
const PHYSICAL_SLOTS: usize = 16;
const RUNTIME_BYTES: usize = (PHYSICAL_SLOTS - 4) * SLOT_BYTES;
const EXPERTS: usize = 4;
const GATE: usize = 101;
const DOWN: usize = 103;
const ROLES: [usize; 3] = [GATE, GATE, DOWN];

fn pattern(bank: u32, offset: usize) -> u8 {
    (bank as u8)
        .wrapping_mul(61)
        .wrapping_add((offset / SLOT_BYTES) as u8 * 7)
        .wrapping_add((offset % 251) as u8)
}

struct PatternIo;

impl BlockIo for PatternIo {
    fn read_block(&self, desc: &BlockDesc, dst: &mut [u8]) -> Result<()> {
        assert!(dst.len() >= desc.nbytes());
        for (offset, byte) in dst[..desc.nbytes()].iter_mut().enumerate() {
            *byte = pattern(desc.id, offset);
        }
        Ok(())
    }
}

fn backend() -> VulkanBackend {
    backend_with_stats(false)
}

fn backend_with_stats(stats: bool) -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(
        std::env::var("MOE4ALL_TEST_DEVICE")
            .expect("central GPU runner must explicitly select an idle device"),
    );
    cfg.paging.host_dma = false;
    cfg.paging.stats = stats;
    cfg.paging.expert_prefetch = false;
    cfg.kernels.vulkan.decode_gpu_resident = true;
    cfg.kernels.vulkan.decode_gpu_resident_copy = Some(true);
    cfg.kernels.vulkan.decode_gpu_resident_mapped_metadata = true;
    let be = VulkanBackend::new_with(Arc::new(cfg)).unwrap();
    be.prepare_moe_unified_vram(
        &[(SLOT_BYTES, PHYSICAL_SLOTS, 4)],
        0,
        0,
        0,
        RUNTIME_BYTES as u64,
    )
    .unwrap();
    be
}

#[derive(Debug)]
struct Job {
    block: u32,
    slot: u32,
}

struct Fixture {
    session: MoePagerSession,
    copy: Arc<GpuCopyBuffers>,
    initial: PagerSnapshot,
    tape: Arc<dyn Buffer>,
    windows: [u32; 3],
    extra: [u32; 9],
    ids: Box<dyn Buffer>,
    control: Box<dyn Buffer>,
    journal: Box<dyn Buffer>,
}

impl Fixture {
    fn new(be: &VulkanBackend) -> Self {
        let mut session = MoePagerSession::new(
            be,
            MoePagerLayout {
                load_reserve_bytes: 0,
                n_blocks: EXPERTS,
                pools: vec![MoePoolSpec {
                    slot_bytes: SLOT_BYTES,
                    n_slots: PHYSICAL_SLOTS,
                    min_enabled_slots: 3,
                }],
                host_tier: None,
                host_store_io: Some(Arc::new(PatternIo)),
                dynamic_state_reserve_bytes: 0,
                dynamic_state_max_allocation_bytes: 0,
                prefill_min_lane_bytes: 0,
                runtime_reserve_bytes: RUNTIME_BYTES as u64,
                host_chunks: vec![MoeHostChunkSpec {
                    base_offset: 0,
                    bytes: 2 * EXPERTS * SLOT_BYTES,
                }],
                prefill_target_lanes: 1,
                prefill_cache_bytes: (4 * SLOT_BYTES) as u64,
            },
        )
        .unwrap();
        // Leave four enabled cache slots and a free runtime corridor for the owned directory.
        // The session is deliberately direct, so metadata allocation cannot retire it implicitly.
        let claim = session
            .unified_pool
            .plan_owner_claim(
                &[RUNTIME_BYTES],
                crate::unified::UnifiedVramClass::LlmRuntime,
                &[],
            )
            .unwrap();
        drop(session.commit_unified_claim(be, claim).unwrap());
        assert_eq!(session.pools[0].pager.enabled_slots(), 4);
        for (index, (role, id)) in [(Role::Gate, GATE), (Role::Down, DOWN)]
            .into_iter()
            .enumerate()
        {
            session
                .register(
                    role,
                    id,
                    ExpertSource {
                        bank: Arc::new(vec![0u8; EXPERTS * SLOT_BYTES]),
                        stride_bytes: SLOT_BYTES,
                        layer_base: 0,
                        host_offset: index * EXPERTS * SLOT_BYTES,
                        file: Some(BlockDesc {
                            id: role.index() as u32,
                            extents: vec![BlockExtent {
                                offset: 0,
                                len: EXPERTS * SLOT_BYTES,
                            }],
                        }),
                    },
                    EXPERTS,
                )
                .unwrap();
        }
        let plan = be.build_session_transfer_plan(session.take_transfer_sources());
        session.install_transfer_plan(Arc::new(plan));

        // Down:1 is oldest and will be requested by the last role. Only Gate:0/Down:0 may go.
        for (role, expert) in [(DOWN, 1u32), (GATE, 1), (GATE, 0), (DOWN, 0)] {
            session.begin_role_batches(&[role]).unwrap();
            session
                .push_roles_cpu(be, &[(role, &[expert])], false)
                .unwrap()
                .complete_without_recorder(be)
                .unwrap();
        }
        let initial = session.pools[0].pager.pager.snapshot_for_handoff().unwrap();
        assert_eq!(initial.resident_lru().len(), 4);
        assert!(initial.free().is_empty());
        assert_eq!(initial.resident_lru()[0].0, (2 * EXPERTS + 1) as u32);

        let copy = GpuCopyBuffers::new_owned(
            be,
            session.gpu_copy_sources().unwrap().unwrap(),
            1,
            crate::pager::owned_seed(&session, 0).unwrap(),
        )
        .unwrap()
        .unwrap();
        assert!(copy.is_owned());
        // The real transfer-plan path, rather than patched metadata, must produce CPU jobs.
        let sources = read_mapped_words(copy.sources.as_ref()).unwrap();
        for role in [GATE, DOWN] {
            let source = &session.sources[&role].2;
            for expert in 0..EXPERTS {
                let offset = (source.block_base as usize + expert) * 4;
                assert_eq!(&sources[offset..offset + 3], &[0, 0, SLOT_BYTES as u32]);
            }
        }
        session.gpu_decode_owner = Some(copy.clone());
        let mut cursor = 0;
        let windows = ROLES.map(|role| {
            session
                .resident_lut_window(&mut cursor, role, EXPERTS)
                .unwrap()
        });
        let bases = ROLES.map(|role| session.sources[&role].2.block_base);
        let extra = [bases[0], bases[1], bases[2], 0, 0, 0, 0, 0, 0];
        let mapped = |bytes| be.alloc_mapped_arena_bda(bytes).unwrap().0;
        Self {
            tape: session.retained_tape(),
            session,
            copy,
            initial,
            windows,
            extra,
            ids: mapped(5 * 4),
            control: mapped(RESIDENT_CONTROL_BYTES),
            journal: mapped(RESIDENT_JOURNAL_BYTES),
        }
    }

    fn snapshot(&self) -> PagerSnapshot {
        self.session.pools[0]
            .pager
            .pager
            .snapshot_for_handoff()
            .unwrap()
    }

    fn initial_slot(&self, block: u32) -> u32 {
        self.initial
            .resident_lru()
            .iter()
            .find(|&&(owner, _)| owner == block)
            .unwrap()
            .1
    }

    fn prepare(&self, be: &VulkanBackend, routed: [u32; 3]) -> [u32; RESIDENT_JOURNAL_WORDS] {
        self.prepare_impl(be, routed, false)
    }

    fn prepare_recovery(
        &self,
        be: &VulkanBackend,
        routed: [u32; 3],
    ) -> [u32; RESIDENT_JOURNAL_WORDS] {
        self.prepare_impl(be, routed, true)
    }

    fn prepare_impl(
        &self,
        be: &VulkanBackend,
        routed: [u32; 3],
        recovery: bool,
    ) -> [u32; RESIDENT_JOURNAL_WORDS] {
        be.upload(
            self.ids.as_ref(),
            bytemuck::cast_slice(&[routed[0], routed[1], routed[2], 0, 0]),
        )
        .unwrap();
        let mut control = vec![0u32; RESIDENT_CONTROL_BYTES / 4];
        control[0] = 1;
        control[1] = NONE;
        be.upload(self.control.as_ref(), bytemuck::cast_slice(&control))
            .unwrap();
        // Even a retained session Tape containing stale nonzero addresses has no authority.
        be.upload(
            self.tape.as_ref(),
            bytemuck::cast_slice(&[0xdead_beef_dead_beefu64; 12]),
        )
        .unwrap();
        let notice = unsafe {
            be.shared
                .device
                .create_event(&vk::EventCreateInfo::default(), None)
        }
        .unwrap();
        let rec = be.recorder().unwrap();
        let mut params = self.extra;
        params[3] = u32::from(recovery);
        rec.gpu_resident_copy_check(
            self.ids.as_ref(),
            self.tape.as_ref(),
            self.control.as_ref(),
            self.journal.as_ref(),
            self.copy.as_ref(),
            [
                3,
                EXPERTS as u32,
                self.windows[0],
                self.windows[1],
                self.windows[2],
                0,
                3,
                0,
            ],
            params,
            notice,
        );
        // This fixture inspects private GPU metadata, unlike a production mailbox consumer.
        rec.publish_host_reads();
        rec.finish().unwrap();
        let signaled = unsafe { be.shared.device.get_event_status(notice) }.unwrap();
        unsafe { be.shared.device.destroy_event(notice, None) };
        assert!(signaled);
        read_mapped_words(self.journal.as_ref())
            .unwrap()
            .try_into()
            .unwrap()
    }

    fn cpu_jobs(&self, words: &[u32; RESIDENT_JOURNAL_WORDS], expert: u32) -> Vec<Job> {
        assert_eq!(words[RESIDENT_JOURNAL_STATUS_WORD], 5);
        assert_eq!(words[RESIDENT_JOURNAL_JOB_COUNT_WORD], 2);
        assert_eq!(words[RESIDENT_JOURNAL_MASK_WORD], 6);
        let control = read_mapped_words(self.control.as_ref()).unwrap();
        assert_eq!(&control[..7], &[0, 0, 1, 0, 1, 0, 1]);
        assert_eq!(
            &read_mapped_words(self.ids.as_ref()).unwrap()[3..5],
            &[6, 0]
        );
        let nodes = read_mapped_words(self.copy.candidates.as_ref()).unwrap();
        let address_base = nodes[7] as usize;
        let words = read_mapped_range(self.copy.jobs.as_ref(), 0, 16).unwrap();
        words
            .chunks_exact(8)
            .zip([expert, (2 * EXPERTS) as u32 + expert])
            .map(|(job, block)| {
                assert_eq!(&job[..2], &[0, 0]);
                assert_eq!(&job[4..6], &[SLOT_BYTES as u32, block]);
                let address = &nodes[(address_base + job[6] as usize) * 4..][..4];
                assert_eq!(&address[..2], &job[2..4]);
                assert_eq!(address[2], 0);
                let slot = address[3];
                assert_eq!(
                    u64::from(job[2]) | (u64::from(job[3]) << 32),
                    self.session.pools[0].pager.slot_addr(slot).unwrap()
                );
                let window = if block < EXPERTS as u32 {
                    self.windows[0]
                } else {
                    self.windows[2]
                };
                assert_eq!(job[7], window + expert);
                Job { block, slot }
            })
            .collect()
    }

    fn assert_payload(&self, be: &VulkanBackend, block: u32, slot: u32) {
        let target = self.session.pools[0].pager.slot_copy_target(slot).unwrap();
        let mut actual = vec![0u8; SLOT_BYTES];
        be.download_range(target.buffer(), target.buffer_offset(), &mut actual)
            .unwrap();
        let bank = block / EXPERTS as u32;
        let expert = block as usize % EXPERTS;
        let mismatch = actual
            .iter()
            .enumerate()
            .position(|(offset, &byte)| byte != pattern(bank, expert * SLOT_BYTES + offset));
        assert_eq!(
            mismatch, None,
            "block={block} slot={slot}: first payload mismatch"
        );
    }

    fn assert_protected_payloads(&self, be: &VulkanBackend) {
        for block in [1, (2 * EXPERTS + 1) as u32] {
            self.assert_payload(be, block, self.initial_slot(block));
        }
    }
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_service_mailbox_rejects_stale_and_malformed_jobs_before_upload() {
    let be = backend();
    be.with_unified_exclusive(|| {
        let mut fixture = Fixture::new(&be);
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        let words = fixture.prepare(&be, [2, 1, 1]);
        let jobs = fixture.cpu_jobs(&words, 2);
        let original = read_mapped_range(fixture.copy.jobs.as_ref(), 0, 16).unwrap();
        let before = jobs
            .iter()
            .map(|job| {
                let target = fixture.session.pools[0]
                    .pager
                    .slot_copy_target(job.slot)
                    .unwrap();
                let mut bytes = vec![0; SLOT_BYTES];
                be.download_range(target.buffer(), target.buffer_offset(), &mut bytes)
                    .unwrap();
                bytes
            })
            .collect::<Vec<_>>();
        for case in 0..8 {
            let mut notice = words;
            let mut records = original.clone();
            match case {
                0 => notice[RESIDENT_JOURNAL_JOB_COUNT_WORD + 1] ^= 1,
                1 => notice[RESIDENT_JOURNAL_STATUS_WORD] = 4,
                2 => notice[RESIDENT_JOURNAL_JOB_COUNT_WORD] = u32::MAX,
                3 => {
                    records[14] = records[6];
                    records[10] = records[2];
                    records[11] = records[3];
                }
                4 => records[13] = records[5],
                5 => records[2] ^= 16,
                6 => records[4] += 16,
                7 => records[0] = 16,
                _ => unreachable!(),
            }
            be.upload(fixture.copy.jobs.as_ref(), bytemuck::cast_slice(&records))
                .unwrap();
            assert!(
                fixture
                    .copy
                    .prepare_owned_upload(&fixture.session, &be, 0, &notice)
                    .is_err(),
                "case={case}"
            );
            assert_eq!(fixture.snapshot(), fixture.initial);
            for (job, expected) in jobs.iter().zip(&before) {
                let target = fixture.session.pools[0]
                    .pager
                    .slot_copy_target(job.slot)
                    .unwrap();
                let mut bytes = vec![0; SLOT_BYTES];
                be.download_range(target.buffer(), target.buffer_offset(), &mut bytes)
                    .unwrap();
                assert_eq!(&bytes, expected, "case={case}");
            }
        }
        be.upload(fixture.copy.jobs.as_ref(), bytemuck::cast_slice(&original))
            .unwrap();
        assert!(fixture
            .copy
            .prepare_owned_upload(&fixture.session, &be, usize::MAX, &words)
            .is_err());
        let (upload, leases) = fixture
            .copy
            .prepare_owned_upload(&fixture.session, &be, 0, &words)
            .unwrap();
        upload.complete_without_recorder(&be).unwrap();
        drop(leases);
        for job in jobs {
            fixture.assert_payload(&be, job.block, job.slot);
        }
        fixture.copy.finish_owned_forward();
        fixture.session.handoff_gpu_decode(&be).unwrap();
    });
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_statistics_preserve_gpu_authority_and_host_snapshot() {
    let be = backend_with_stats(true);
    let mut fixture = be.with_unified_exclusive(|| Fixture::new(&be));
    be.with_unified_exclusive(|| {
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        let words = fixture.prepare(&be, [1, 1, 1]);
        assert_eq!(words[RESIDENT_JOURNAL_STATUS_WORD], 1);
        fixture.copy.finish_owned_forward();
    });
    let before = fixture.snapshot();
    let owner = fixture.copy.clone();
    let statistics = be.with_unified_exclusive(|| {
        owner
            .owned_statistics(&be, &fixture.session)
            .unwrap()
            .unwrap()
    });
    assert_eq!(statistics.len(), 1);
    assert_eq!(statistics[0].hits, before.stats().hits + 9);
    assert_eq!(statistics[0].misses, before.stats().misses);
    assert_eq!(statistics[0].evictions, before.stats().evictions);
    let generation = owner
        .owned
        .as_ref()
        .unwrap()
        .generation
        .load(Ordering::Acquire);
    *be.moe_pager.lock().unwrap() = Some(fixture.session);
    be.print_moe_pager_stats();
    assert!(
        owner.owned_active(),
        "statistics must not hand ownership to CPU"
    );
    assert!(!owner.owned_in_forward());
    assert_eq!(
        owner
            .owned
            .as_ref()
            .unwrap()
            .generation
            .load(Ordering::Acquire),
        generation
    );
    let pager = be.moe_pager.lock().unwrap();
    let after = pager.as_ref().unwrap().pools[0]
        .pager
        .pager
        .snapshot_for_handoff()
        .unwrap();
    assert_eq!(
        after, before,
        "statistics must not replay GPU LRU into host state"
    );
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_cpu_upload_publish_and_handoff_match_payload_and_lru() {
    let be = backend();
    be.with_unified_exclusive(|| {
        let mut fixture = Fixture::new(&be);
        let protected_down = (2 * EXPERTS + 1) as u32;
        let mut generation = None;
        for expert in [2u32, 0] {
            fixture.copy.begin(&be, &mut fixture.session).unwrap();
            let current = fixture
                .copy
                .owned
                .as_ref()
                .unwrap()
                .generation
                .load(Ordering::Acquire);
            assert_eq!(
                *generation.get_or_insert(current),
                current,
                "no reseed between forwards"
            );
            let routed = [expert, 1, 1];
            let words = fixture.prepare(&be, routed);
            let jobs = fixture.cpu_jobs(&words, expert);
            assert_eq!(jobs[0].slot, fixture.initial_slot(0));
            assert_eq!(jobs[1].slot, fixture.initial_slot((2 * EXPERTS) as u32));
            fixture.assert_protected_payloads(&be);
            assert!(fixture.session.handoff_gpu_decode(&be).is_err());
            assert_eq!(fixture.snapshot(), fixture.initial);

            let (push, leases) = fixture
                .copy
                .prepare_owned_upload(&fixture.session, &be, 0, &words)
                .unwrap();
            assert_eq!(push.bytes(), 2 * SLOT_BYTES);
            assert_eq!(push.complete_without_recorder(&be).unwrap(), 2);
            // Bounded-tier sources, when present, must stay leased through transfer completion.
            drop(leases);
            assert_eq!(
                fixture.snapshot(),
                fixture.initial,
                "CPU upload must not replay host LRU"
            );
            let before_publish = read_mapped_words(fixture.copy.overlay.as_ref()).unwrap();
            for job in &jobs {
                assert_eq!(before_publish[job.block as usize * 4 + 2], COPYING);
                fixture.assert_payload(&be, job.block, job.slot);
            }
            let rec = be.recorder().unwrap();
            rec.publish_owned_cpu(fixture.copy.as_ref(), fixture.journal.as_ref(), 0);
            rec.publish_host_reads();
            rec.finish().unwrap();
            let published = read_mapped_words(fixture.copy.overlay.as_ref()).unwrap();
            for job in &jobs {
                assert_eq!(published[job.block as usize * 4 + 2], READY);
            }
            assert_eq!(
                read_mapped_words(fixture.journal.as_ref()).unwrap()[RESIDENT_JOURNAL_STATUS_WORD],
                5
            );
            let before_refresh = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
            let before_jobs = read_mapped_range(fixture.copy.jobs.as_ref(), 0, 16).unwrap();
            let replay = fixture.prepare_recovery(&be, routed);
            assert_eq!(replay[RESIDENT_JOURNAL_STATUS_WORD], 1);
            assert_eq!(replay[RESIDENT_JOURNAL_JOB_COUNT_WORD], 0);
            assert_eq!(replay[RESIDENT_JOURNAL_MASK_WORD], 7);
            assert_eq!(
                read_mapped_words(fixture.copy.candidates.as_ref()).unwrap(),
                before_refresh
            );
            assert_eq!(
                read_mapped_words(fixture.copy.overlay.as_ref()).unwrap(),
                published
            );
            assert_eq!(
                read_mapped_range(fixture.copy.jobs.as_ref(), 0, 16).unwrap(),
                before_jobs
            );
            assert_eq!(
                &read_mapped_words(fixture.control.as_ref()).unwrap()[..7],
                &[1, NONE, 1, 0, 1, 1, 1]
            );
            let tape = read_mapped_range(fixture.tape.as_ref(), 0, 24).unwrap();
            for (role, base) in [0, 0, 2 * EXPERTS].into_iter().enumerate() {
                for id in routed {
                    let word = (fixture.windows[role] as usize + id as usize) * 2;
                    let block = base + id as usize;
                    assert_eq!(&tape[word..word + 2], &published[block * 4..block * 4 + 2]);
                }
            }
            fixture.assert_protected_payloads(&be);
            assert!(fixture.session.handoff_gpu_decode(&be).is_err());
            fixture.copy.finish_owned_forward();
        }
        fixture.session.handoff_gpu_decode(&be).unwrap();
        let mut stats = fixture.initial.stats();
        stats.hits += 14;
        stats.misses += 4;
        stats.evictions += 4;
        let expected = PagerSnapshot::from_parts(
            fixture.initial.disabled().to_vec(),
            vec![],
            [0, 1, (2 * EXPERTS) as u32, protected_down]
                .map(|block| (block, fixture.initial_slot(block)))
                .to_vec(),
            stats,
        );
        assert_eq!(fixture.snapshot(), expected);
        for &(block, slot) in expected.resident_lru() {
            assert_eq!(
                fixture.session.pools[0].pager.lut_host[block as usize],
                slot
            );
            fixture.assert_payload(&be, block, slot);
        }
        for block in [2, (2 * EXPERTS + 2) as u32] {
            assert_eq!(
                fixture.session.pools[0].pager.lut_host[block as usize],
                NOT_RESIDENT
            );
        }
    });
}

fn bump_unrelated_runtime_generation(fixture: &Fixture) -> u64 {
    let before = fixture.snapshot();
    let pool = &fixture.session.unified_pool;
    let generation = pool.generation();
    // Bypass backend allocation hooks: exercise the global version changing while expert
    // ownership stays active, using only a genuinely free, non-Expert range.
    let lease = pool
        .allocate(256, crate::unified::UnifiedVramClass::LlmRuntime)
        .expect("fixture must retain a free runtime range");
    let allocated = pool.generation();
    assert!(allocated > generation);
    assert_eq!(fixture.snapshot(), before);
    drop(lease);
    let released = pool.generation();
    assert!(released > allocated);
    assert_eq!(fixture.snapshot(), before);
    released
}

fn check_benign_runtime_generation(during_forward: bool) {
    let be = backend();
    be.with_unified_exclusive(|| {
        let mut fixture = Fixture::new(&be);
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        let owner_generation = fixture
            .copy
            .owned
            .as_ref()
            .unwrap()
            .generation
            .load(Ordering::Acquire);
        let nodes = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
        let overlay = read_mapped_words(fixture.copy.overlay.as_ref()).unwrap();
        if !during_forward {
            fixture.copy.finish_owned_forward();
        }
        assert_eq!(fixture.copy.owned_in_forward(), during_forward);
        let global_generation = bump_unrelated_runtime_generation(&fixture);
        assert_eq!(fixture.session.restore_released_unified_slots(), 0);
        assert_eq!(fixture.snapshot(), fixture.initial);

        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        let state = fixture.copy.owned.as_ref().unwrap();
        assert!(state.active.load(Ordering::Acquire));
        assert!(state.in_forward.load(Ordering::Acquire));
        assert_eq!(
            state.generation.load(Ordering::Acquire),
            owner_generation,
            "benign frees must not reseed ownership"
        );
        assert_eq!(
            state.layout_generation.load(Ordering::Acquire),
            global_generation
        );
        assert_eq!(
            read_mapped_words(fixture.copy.candidates.as_ref()).unwrap(),
            nodes
        );
        assert_eq!(
            read_mapped_words(fixture.copy.overlay.as_ref()).unwrap(),
            overlay
        );
        fixture.copy.finish_owned_forward();
        fixture.session.handoff_gpu_decode(&be).unwrap();
        assert_eq!(fixture.snapshot(), fixture.initial);
    });
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_accepts_unrelated_runtime_generation_between_tokens() {
    check_benign_runtime_generation(false);
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_accepts_unrelated_runtime_generation_during_forward() {
    check_benign_runtime_generation(true);
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_rejects_changed_enabled_mask_after_global_generation_bump() {
    let be = backend();
    be.with_unified_exclusive(|| {
        let mut fixture = Fixture::new(&be);
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        fixture.copy.finish_owned_forward();
        let generation = fixture
            .copy
            .owned
            .as_ref()
            .unwrap()
            .generation
            .load(Ordering::Acquire);
        let nodes = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
        let (block, slot) = fixture.initial.resident_lru()[0];
        assert_eq!(
            fixture.session.pools[0].pager.pager.disable_slot(slot),
            Some(block)
        );
        let tampered = fixture.snapshot();
        // This direct Pager mutation intentionally bypasses production versioning. A real
        // allocator version bump is required to enter the enabled-geometry comparison.
        bump_unrelated_runtime_generation(&fixture);
        let error = fixture.copy.begin(&be, &mut fixture.session).unwrap_err();
        assert!(error
            .to_string()
            .contains("layout changed without a handoff"));
        assert!(!fixture.copy.owned_in_forward());
        assert!(fixture.copy.owned_active());
        assert_eq!(fixture.snapshot(), tampered);
        assert_eq!(
            read_mapped_words(fixture.copy.candidates.as_ref()).unwrap(),
            nodes
        );
        assert_eq!(
            fixture
                .copy
                .owned
                .as_ref()
                .unwrap()
                .generation
                .load(Ordering::Acquire),
            generation
        );

        // Undo the illicit geometry change before exporting authoritative GPU residency.
        fixture.session.pools[0].pager.pager.enable_slot(slot);
        fixture.session.handoff_gpu_decode(&be).unwrap();
        assert_eq!(
            fixture.snapshot().resident_lru(),
            fixture.initial.resident_lru()
        );
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        assert!(
            fixture
                .copy
                .owned
                .as_ref()
                .unwrap()
                .generation
                .load(Ordering::Acquire)
                > generation
        );
        fixture.copy.finish_owned_forward();
        fixture.session.handoff_gpu_decode(&be).unwrap();
    });
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_claim_rejects_active_forward_then_handoffs_and_reseeds() {
    let be = backend();
    be.with_unified_exclusive(|| {
        let mut fixture = Fixture::new(&be);
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        let generation = fixture
            .copy
            .owned
            .as_ref()
            .unwrap()
            .generation
            .load(Ordering::Acquire);
        let placement = fixture
            .session
            .unified_pool
            .expert_layout()
            .unwrap()
            .slots(0)
            .unwrap()
            .iter()
            .find(|slot| {
                fixture.session.pools[0]
                    .pager
                    .pager
                    .slot_enabled(slot.id.slot as u32)
            })
            .copied()
            .unwrap();
        let range = crate::unified::PlannedRange {
            shard: placement.shard,
            offset: placement.offset,
            len: placement.len,
            requested_len: placement.len,
        };
        // The remaining four enabled fixture slots are its dispatch floor. Prefill is the
        // production owner allowed to borrow one after a drained handoff.
        let class = crate::unified::UnifiedVramClass::Prefill;
        let plan = fixture
            .session
            .unified_pool
            .plan_exact_owner_claim(&[range], class, &[])
            .unwrap();
        assert_eq!(plan.victims(), &[placement.id]);
        let before = fixture.snapshot();
        let nodes = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
        let global_generation = fixture.session.unified_pool.generation();
        let error = fixture
            .session
            .commit_unified_claim(&be, plan)
            .err()
            .expect("active forward must reject a claim");
        assert!(error.to_string().contains("during a GPU-owned forward"));
        assert!(fixture.copy.owned_in_forward());
        assert_eq!(fixture.snapshot(), before);
        assert_eq!(fixture.session.unified_pool.generation(), global_generation);
        assert_eq!(
            read_mapped_words(fixture.copy.candidates.as_ref()).unwrap(),
            nodes
        );

        fixture.copy.finish_owned_forward();
        let plan = fixture
            .session
            .unified_pool
            .plan_exact_owner_claim(&[range], class, &[])
            .unwrap();
        let leases = fixture.session.commit_unified_claim(&be, plan).unwrap();
        assert!(
            !fixture.copy.owned_active(),
            "claim must hand off before retiring slots"
        );
        let after_claim = fixture.snapshot();
        assert!(after_claim.disabled()[placement.id.slot]);
        assert_eq!(fixture.session.pools[0].pager.enabled_slots(), 3);
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        assert!(
            fixture
                .copy
                .owned
                .as_ref()
                .unwrap()
                .generation
                .load(Ordering::Acquire)
                > generation
        );
        let nodes = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
        let slot_base = nodes[6] as usize;
        assert_eq!(nodes[(slot_base + placement.id.slot) * 4 + 3], DISABLED);
        fixture.copy.finish_owned_forward();
        fixture.session.handoff_gpu_decode(&be).unwrap();
        assert_eq!(fixture.snapshot(), after_claim);
        drop(leases);
    });
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_abort_frees_pending_slots_without_restoring_evicted_owners() {
    let be = backend();
    be.with_unified_exclusive(|| {
        let mut fixture = Fixture::new(&be);
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        let words = fixture.prepare(&be, [2, 1, 1]);
        let jobs = fixture.cpu_jobs(&words, 2);
        let before_abort = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
        let before_overlay = read_mapped_words(fixture.copy.overlay.as_ref()).unwrap();
        let before_jobs = read_mapped_range(fixture.copy.jobs.as_ref(), 0, 16).unwrap();
        let rejected = fixture.prepare_recovery(&be, [2, 1, 1]);
        assert_eq!(
            rejected[RESIDENT_JOURNAL_STATUS_WORD], 3,
            "COPYING is not recovery-ready"
        );
        assert_eq!(rejected[RESIDENT_JOURNAL_JOB_COUNT_WORD], 0);
        assert_eq!(
            read_mapped_words(fixture.copy.overlay.as_ref()).unwrap(),
            before_overlay
        );
        assert_eq!(
            read_mapped_range(fixture.copy.jobs.as_ref(), 0, 16).unwrap(),
            before_jobs
        );
        fixture.copy.abort(&mut fixture.session);
        assert_eq!(
            read_mapped_words(fixture.copy.candidates.as_ref()).unwrap(),
            before_abort
        );
        assert!(fixture.session.handoff_gpu_decode(&be).is_err());
        assert_eq!(fixture.snapshot(), fixture.initial);
        fixture.copy.finish_owned_forward();
        fixture.session.handoff_gpu_decode(&be).unwrap();

        let mut stats = fixture.initial.stats();
        stats.hits += 7;
        stats.misses += 2;
        stats.evictions += 2;
        let expected = PagerSnapshot::from_parts(
            fixture.initial.disabled().to_vec(),
            jobs.iter().map(|job| job.slot).collect(),
            [1, (2 * EXPERTS + 1) as u32]
                .map(|block| (block, fixture.initial_slot(block)))
                .to_vec(),
            stats,
        );
        assert_eq!(fixture.snapshot(), expected);
        for block in [0, 2, (2 * EXPERTS) as u32, (2 * EXPERTS + 2) as u32] {
            assert_eq!(fixture.session.pools[0].pager.pager.slot_of(block), None);
            assert_eq!(
                fixture.session.pools[0].pager.lut_host[block as usize],
                NOT_RESIDENT
            );
        }
        fixture.assert_protected_payloads(&be);

        // A fresh generation must seed the exported free slots, never the abandoned owners.
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        let nodes = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
        let slot_base = nodes[6] as usize;
        let pool_base = nodes[5] as usize;
        assert_eq!(&nodes[(pool_base + 1) * 4..(pool_base + 2) * 4], &[0; 4]);
        let overlay = read_mapped_words(fixture.copy.overlay.as_ref()).unwrap();
        for job in &jobs {
            assert_eq!(overlay[job.block as usize * 4 + 2], 0);
            let slot = &nodes[(slot_base + job.slot as usize) * 4..][..4];
            assert_eq!(&slot[2..4], &[NONE, FREE]);
        }
        fixture.copy.finish_owned_forward();
        fixture.session.handoff_gpu_decode(&be).unwrap();
        assert_eq!(
            fixture.snapshot(),
            expected,
            "handoff counters must not be counted twice"
        );
    });
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_fifo_handoff_accepts_stale_logs_and_discards_copying() {
    let be = backend();
    be.with_unified_exclusive(|| {
        let mut fixture = Fixture::new(&be);
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        fixture.copy.finish_owned_forward();
        let mut nodes = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
        let mut overlay = read_mapped_words(fixture.copy.overlay.as_ref()).unwrap();
        let pool = nodes[5] as usize * 4;
        let queue = nodes[pool] as usize * 4;
        let slot_base = nodes[6] as usize;
        let initial = fixture.initial.resident_lru();
        let (touched_block, touched_slot) = initial[0];
        let (copying_block, copying_slot) = initial[1];
        let clock = u64::from(nodes[10]) | (u64::from(nodes[11]) << 32);
        let next = clock + 1;
        let touched = (slot_base + touched_slot as usize) * 4;
        nodes[touched] = next as u32;
        nodes[touched + 1] = (next >> 32) as u32;
        let tail = nodes[queue + 1] as usize;
        let entry = queue + (1 + tail) * 4;
        nodes[entry..entry + 4].copy_from_slice(&[
            next as u32,
            (next >> 32) as u32,
            touched_slot,
            0,
        ]);
        nodes[queue + 1] += 1;
        nodes[10] = next as u32;
        nodes[11] = (next >> 32) as u32;
        nodes[(slot_base + copying_slot as usize) * 4 + 3] = COPYING;
        overlay[copying_block as usize * 4 + 2] = COPYING;
        be.upload(
            fixture.copy.candidates.as_ref(),
            bytemuck::cast_slice(&nodes),
        )
        .unwrap();
        be.upload(
            fixture.copy.overlay.as_ref(),
            bytemuck::cast_slice(&overlay),
        )
        .unwrap();
        fixture.session.handoff_gpu_decode(&be).unwrap();
        let snapshot = fixture.snapshot();
        let mut expected = initial[2..].to_vec();
        expected.push((touched_block, touched_slot));
        assert_eq!(snapshot.resident_lru(), expected);
        assert_eq!(snapshot.free(), &[copying_slot]);
        assert_eq!(snapshot.stats(), fixture.initial.stats());
        assert!(!fixture.copy.owned_active());
        // A new generation must reproduce the v4 buffer geometry after COPYING was discarded.
        let bytes = fixture.copy.candidates.len_bytes();
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        let reseeded = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
        assert_eq!(reseeded[0], 4);
        assert_eq!(reseeded.len() * 4, bytes);
        assert_eq!(reseeded[3], nodes[3] + 1);
        fixture.copy.finish_owned_forward();
        fixture.session.handoff_gpu_decode(&be).unwrap();
        assert_eq!(fixture.snapshot(), snapshot);
    });
}

#[test]
#[ignore = "central serial GPU runner only; requires MOE4ALL_TEST_DEVICE and mapped Vulkan BDA"]
fn owned_session_handoff_rejects_corrupt_queue_and_overlay_without_partial_import() {
    let be = backend();
    be.with_unified_exclusive(|| {
        let mut fixture = Fixture::new(&be);
        fixture.copy.begin(&be, &mut fixture.session).unwrap();
        fixture.copy.finish_owned_forward();
        let original = read_mapped_words(fixture.copy.candidates.as_ref()).unwrap();
        let overlay = read_mapped_words(fixture.copy.overlay.as_ref()).unwrap();
        assert_eq!(original[0], 4);
        let pool = original[5] as usize * 4;
        let queue = original[pool] as usize * 4;
        let first = queue + 4;
        for kind in 0..10 {
            let mut bad = original.clone();
            match kind {
                0 => {
                    let entry: [u32; 4] = bad[first..first + 4].try_into().unwrap();
                    bad[first + 4..first + 8].copy_from_slice(&entry);
                }
                1 => bad[queue] = u32::MAX,
                2 => {
                    let slot = bad[first + 2] as usize;
                    let node = (bad[6] as usize + slot) * 4;
                    bad[node + 3] = FREE;
                }
                3 => bad[pool + 1] = u32::MAX,
                4 => bad[queue + 1] = bad[queue + 2] + 1,
                5 => bad[queue + 2] -= 1,
                6 => bad[queue + 3] = 1,
                7 => bad[first] = bad[10] + 1,
                8 => {
                    let entry: [u32; 4] = bad[first..first + 4].try_into().unwrap();
                    let next: [u32; 4] = bad[first + 4..first + 8].try_into().unwrap();
                    bad[first..first + 4].copy_from_slice(&next);
                    bad[first + 4..first + 8].copy_from_slice(&entry);
                }
                _ => bad[queue + 1] -= 1,
            }
            be.upload(fixture.copy.candidates.as_ref(), bytemuck::cast_slice(&bad))
                .unwrap();
            assert!(
                fixture.session.handoff_gpu_decode(&be).is_err(),
                "corruption {kind}"
            );
            assert_eq!(fixture.snapshot(), fixture.initial);
            assert!(fixture.copy.owned_active());
        }
        be.upload(
            fixture.copy.candidates.as_ref(),
            bytemuck::cast_slice(&original),
        )
        .unwrap();
        let mut bad = overlay.clone();
        let block = fixture.initial.resident_lru()[0].0 as usize;
        bad[block * 4 + 3] = u32::MAX;
        be.upload(fixture.copy.overlay.as_ref(), bytemuck::cast_slice(&bad))
            .unwrap();
        assert!(fixture.session.handoff_gpu_decode(&be).is_err());
        assert_eq!(fixture.snapshot(), fixture.initial);
        be.upload(
            fixture.copy.overlay.as_ref(),
            bytemuck::cast_slice(&overlay),
        )
        .unwrap();
        // Corrupt both GPU copies consistently: handoff must still use physical host geometry.
        let slot = overlay[block * 4 + 3] as usize;
        let mut bad_nodes = original.clone();
        let address = (bad_nodes[7] as usize + slot) * 4;
        bad_nodes[address] ^= 16;
        let mut bad_overlay = overlay.clone();
        bad_overlay[block * 4] = bad_nodes[address];
        be.upload(
            fixture.copy.candidates.as_ref(),
            bytemuck::cast_slice(&bad_nodes),
        )
        .unwrap();
        be.upload(
            fixture.copy.overlay.as_ref(),
            bytemuck::cast_slice(&bad_overlay),
        )
        .unwrap();
        assert!(fixture.session.handoff_gpu_decode(&be).is_err());
        assert_eq!(fixture.snapshot(), fixture.initial);
        be.upload(
            fixture.copy.candidates.as_ref(),
            bytemuck::cast_slice(&original),
        )
        .unwrap();
        be.upload(
            fixture.copy.overlay.as_ref(),
            bytemuck::cast_slice(&overlay),
        )
        .unwrap();
        fixture.session.handoff_gpu_decode(&be).unwrap();
        assert_eq!(fixture.snapshot(), fixture.initial);
    });
}
