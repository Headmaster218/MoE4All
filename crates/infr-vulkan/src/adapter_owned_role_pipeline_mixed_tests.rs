//! Child of adapter_owned_role_pipeline_tests; one auto-C mixed-source success case.
use super::*;
use crate::host_ready_combined_test_support::{Fault, Registration};
use std::collections::HashSet;

fn read_words(backend: &VulkanBackend, buffer: &dyn Buffer, words: usize) -> Vec<u32> {
    let mut values = vec![0u32; words];
    backend
        .download_range(buffer, 0, bytemuck::cast_slice_mut(&mut values))
        .unwrap();
    values
}

pub(super) fn install_mixed_owner(
    fixture: &Fixture,
) -> (Arc<crate::pager::GpuCopyBuffers>, [u32; 3]) {
    let (sources, seed, bases) = {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        let previous = session.gpu_decode_owner.clone().expect("warm owner");
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        let params = previous.role_params(session, &fixture.roles).unwrap();
        let bases = [params[0], params[1], params[2]];
        // Capture imported BDAs AND their keepalive leases before removing the CPU
        // transfer import table. Otherwise expert 2 would silently become CPU-only too.
        let mut sources = session.gpu_copy_sources().unwrap().unwrap();
        for base in bases {
            sources.test_cpu_only_source(base as usize + fixture.routed[1] as usize);
        }
        session.discard_cpu_miss(fixture.roles, &[fixture.routed[1], fixture.routed[2]]);
        assert_eq!(
            session
                .routed_roles_resident_mask(&fixture.roles, &fixture.routed)
                .unwrap(),
            (if fixture.routed.len() == 32 {
                u32::MAX
            } else {
                (1u32 << fixture.routed.len()) - 1
            }) ^ (1 << 1)
                ^ (1 << 2),
        );
        session.gpu_decode_owner = None;
        session.test_unimport_host_uploads();
        (
            sources,
            crate::pager::owned_seed(session, 0).unwrap(),
            bases,
        )
    };
    // No pager mutex across allocation; the new owner retains GpuCopySources.keepalive.
    let owner = crate::pager::GpuCopyBuffers::new_owned(&fixture.backend, sources, 1, seed)
        .unwrap()
        .unwrap();
    assert!(owner.is_owned() && !owner.owned_self_service());
    let sources = read_words(
        &fixture.backend,
        owner.sources.as_ref(),
        owner.sources.len_bytes() / 4,
    );
    for base in bases {
        let cpu = (base as usize + fixture.routed[1] as usize) * 4;
        let gpu = (base as usize + fixture.routed[2] as usize) * 4;
        assert_eq!(&sources[cpu..cpu + 2], &[0, 0]);
        assert_ne!(
            &sources[gpu..gpu + 2],
            &[0, 0],
            "expert 2 must retain real imported BDA"
        );
        assert_ne!(sources[cpu + 2], 0);
        assert_eq!(sources[cpu + 2], sources[gpu + 2]);
    }
    // Padding poison distinguishes this admission's fresh jobs from stale allocation data.
    fixture
        .backend
        .upload(owner.jobs.as_ref(), &vec![0xa5; owner.jobs.len_bytes()])
        .unwrap();
    fixture
        .backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .gpu_decode_owner = Some(Arc::clone(&owner));
    fixture
        .backend
        .runtime_phase
        .lock()
        .unwrap()
        .resident_buffers
        .as_mut()
        .unwrap()
        .gpu_copy = Some(Arc::clone(&owner));
    (owner, bases)
}

struct RecoveryCountScope;

impl RecoveryCountScope {
    fn begin() -> Self {
        OWNED_RECOVERY_TEST_COUNTS.with(|state| assert!(state.replace(Some([0; 3])).is_none()));
        Self
    }
}

impl Drop for RecoveryCountScope {
    fn drop(&mut self) {
        OWNED_RECOVERY_TEST_COUNTS.with(|state| state.set(None));
    }
}

#[test]
#[ignore = "explicit MOE4ALL_TEST_DEVICE; one serialized auto Q5K mixed-source status5 host-ready success"]
fn gpu_owned_role_pipeline_adapter_auto_mixed_status5_host_ready_success() {
    let fixture =
        Fixture::with_profile_and_host_ready(2048, 512, 8, 1, [DType::Q5K; 3], true, true);
    let backend = &fixture.backend;
    assert!(
        backend
            .cfg()
            .kernels
            .vulkan
            .decode_gpu_resident_role_pipeline
    );
    assert!(backend.cfg().kernels.vulkan.decode_gpu_resident_host_ready);
    assert!(
        backend
            .cfg()
            .kernels
            .vulkan
            .decode_gpu_resident_upload_worker
    );
    assert!(!backend.cfg().spec.mtp);
    // Real backend Plan and the default auto gate, with no pipeline::test_scope override.
    let plan = backend.compile(&fixture.graph).unwrap();
    fixture.preload();
    backend.execute(plan.as_ref(), &fixture.bindings()).unwrap();
    let expected = fixture.read(fixture.buffer(fixture.output), fixture.ne);
    let expected_act = fixture.activation();
    assert!(expected.iter().all(|x| x.is_finite()));
    assert!(expected.iter().any(|x| x.abs() > 1e-7));
    assert!(expected_act.iter().all(|x| x.is_finite()));
    assert!(expected_act[fixture.nff..3 * fixture.nff]
        .iter()
        .any(|x| x.abs() > 1e-7));

    // Keep this Arc through completion/next request/handoff: it owns the retained
    // imported source leases even though CPU uploads now use an unimported plan.
    let (source_lease, bases) = install_mixed_owner(&fixture);
    fixture.reset_frontend();
    fixture.poison_role_scratch();
    let selection = pipeline::test_selection_scope(backend);
    let _recovery = RecoveryCountScope::begin();
    OWNED_HOST_READY_TEST_ARMS.with(|arms| arms.set(0));
    OWNED_HOST_READY_TEST_OUTCOME.with(|outcome| assert!(outcome.get().is_none()));
    let registration = Registration::new(
        &backend.shared,
        Arc::clone(&backend.unified_exec),
        Fault::None,
    );
    let result = backend.execute(plan.as_ref(), &fixture.bindings());
    let proof = registration.0.evidence();
    drop(registration);
    result.unwrap();

    assert!(proof.entered);
    assert_eq!(
        proof.jobs, 3,
        "one CPU-only expert, full Gate/Up/Down batch"
    );
    assert_eq!(proof.copied, 3);
    assert!(proof.submitted && proof.wait_value != 0);
    assert!(proof.unsignaled_at_submit && proof.fence_pending_at_submit);
    assert_eq!(proof.copied_at_submit, 0);
    assert!(proof.pending_owned && proof.shared_gate_blocked);
    assert!(!proof.controller_cancelled);
    OWNED_HOST_READY_TEST_ARMS.with(|arms| assert_eq!(arms.get(), 1));
    OWNED_RECOVERY_TEST_COUNTS.with(|state| {
        let counts = state.get().unwrap();
        assert_eq!(
            &counts[..2],
            &[1, 1],
            "one real status5 guarded recovery/submission"
        );
    });
    assert_eq!(
        selection.counts(),
        [1, 0],
        "auto C only on initial ordinary recording"
    );
    assert_eq!(
        selection.phase2_counts(),
        [0, 1],
        "initial C used the unchanged no-entry proof"
    );
    fixture.assert_result(&expected, &expected_act);
    assert_exact(
        &fixture.read(fixture.buffer(fixture.output), fixture.ne),
        &expected,
    );
    assert_exact(&fixture.activation(), &expected_act);
    assert!(!backend
        .moe_pager()
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .gpu_decode_in_forward());

    // Execute has drained GPU and CPU work. Recovery resets JobCount but does not
    // rewrite the job array. Inspect only now, never while a host wait is in flight.
    let jobs = read_words(backend, source_lease.jobs.as_ref(), 6 * 8);
    let mut seen = HashSet::new();
    let mut slots = HashSet::new();
    let mut gpu_jobs = 0;
    let mut cpu_jobs = 0;
    for job in jobs.chunks_exact(8) {
        assert!(
            seen.insert(job[5]) && slots.insert(job[6]),
            "unique actual block/slot"
        );
        assert_ne!(&job[2..4], &[0, 0]);
        assert!(job[4] > 0 && job[4] % 16 == 0);
        if bases.iter().any(|base| job[5] == base + fixture.routed[1]) {
            assert_eq!(&job[..2], &[0, 0], "CPU-only expert 1");
            cpu_jobs += 1;
        } else {
            assert!(bases.iter().any(|base| job[5] == base + fixture.routed[2]));
            assert_ne!(
                &job[..2],
                &[0, 0],
                "expert 2 actually used GPU-readable source"
            );
            gpu_jobs += 1;
        }
    }
    assert_eq!(
        (gpu_jobs, cpu_jobs),
        (3, 3),
        "mixed jobs, not an all-CPU surrogate"
    );
    let (journal, ids) = {
        let phase = backend.runtime_phase.lock().unwrap();
        let journal = Arc::clone(&phase.resident_buffers.as_ref().unwrap().journal);
        let ids = phase
            .pool
            .buffers
            .iter()
            .find(|((tag, _), _)| *tag == "moe_paged_ids")
            .unwrap()
            .1;
        (
            read_words(
                backend,
                journal.as_ref(),
                crate::recorder::RESIDENT_JOURNAL_WORDS,
            ),
            read_words(backend, ids.as_ref(), fixture.used + 2),
        )
    };
    assert_eq!(journal[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD], 1);
    assert_eq!(journal[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD], 0);
    assert_eq!(journal[crate::recorder::RESIDENT_JOURNAL_MASK_WORD], 0xff);
    assert_eq!(&ids[..fixture.used], fixture.routed.as_slice());
    assert_eq!(
        &ids[fixture.used..],
        &[0b110, 0],
        "recovery computes exactly the two original misses"
    );

    // Same plan/backend/producer, without the delay registration: the next request
    // and explicit ownership handoff must retain exact results and all eight experts.
    fixture.reset_frontend();
    fixture.poison_role_scratch();
    backend.execute(plan.as_ref(), &fixture.bindings()).unwrap();
    fixture.assert_result(&expected, &expected_act);
    assert_exact(
        &fixture.read(fixture.buffer(fixture.output), fixture.ne),
        &expected,
    );
    assert_exact(&fixture.activation(), &expected_act);
    assert_eq!(selection.counts(), [2, 0]);
    assert_eq!(selection.phase2_counts(), [0, 2]);
    OWNED_HOST_READY_TEST_ARMS.with(|arms| assert_eq!(arms.get(), 1));
    backend.with_unified_exclusive(|| {
        let mut guard = backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        assert!(!session.gpu_decode_in_forward());
        session.handoff_gpu_decode(backend).unwrap();
        assert_eq!(
            session
                .routed_roles_resident_mask(&fixture.roles, &fixture.routed)
                .unwrap(),
            0xff
        );
    });
    eprintln!(
        "AUTO_C_MIXED_STATUS5 gpu_jobs={gpu_jobs} cpu_jobs={cpu_jobs} host_ready={proof:?} selections={:?} phase2={:?} output_act_exact=true next_handoff_exact=true pre_ready=existing_raw_protocol_and_source_proof",
        selection.counts(),
        selection.phase2_counts()
    );
}
