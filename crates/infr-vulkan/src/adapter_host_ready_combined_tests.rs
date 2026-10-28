//! Child of adapter_owned_upload_tests: reuse its real pager/producer/recovery fixture.
use super::*;
use crate::host_ready_combined_test_support::{Fault, Registration};
use infr_core::backend::SegmentedKvSpec;

fn install_cpu_tail(fixture: &Fixture) {
    let (sources, seed) = {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        let previous = session.gpu_decode_owner.clone().expect("warm GPU owner");
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        let banks = fixture.roles.map(|roles| previous.role_params(session, &roles).unwrap());
        for roles in fixture.roles {
            session.discard_cpu_miss(roles, &[ROUTED[1]]);
        }
        session.gpu_decode_owner = None;
        session.test_unimport_host_uploads();
        let mut sources = session.gpu_copy_sources().unwrap().unwrap();
        for roles in banks {
            for base in &roles[..3] {
                sources.test_cpu_only_source(*base as usize + ROUTED[1] as usize);
            }
        }
        (sources, crate::pager::owned_seed(session, 0).unwrap())
    };
    // Allocation may consult the pager; never hold its mutex across new_owned.
    let owner = crate::pager::GpuCopyBuffers::new_owned(&fixture.backend, sources, LAYERS, seed)
        .unwrap().unwrap();
    assert!(!owner.owned_self_service());
    fixture.backend.moe_pager().lock().unwrap().as_mut().unwrap().gpu_decode_owner =
        Some(Arc::clone(&owner));
    fixture.backend.runtime_phase.lock().unwrap().resident_buffers.as_mut().unwrap().gpu_copy =
        Some(owner);
}

fn check_request(fixture: &Fixture, plan: &dyn infr_core::backend::Plan, expected: &[Vec<f32>; LAYERS]) {
    fixture.reset_frontend();
    for output in fixture.outputs {
        fixture.backend.upload(fixture.buffer(output), bytemuck::cast_slice(&[-777.0f32; NE])).unwrap();
    }
    fixture.backend.execute(plan, &fixture.bindings()).unwrap();
    fixture.assert_frontend(fixture.buffer(fixture.frontend), LAYERS as f32);
    assert_eq!(&fixture.outputs.map(|id| fixture.read(fixture.buffer(id))), expected);
    assert!(!fixture.backend.moe_pager().lock().unwrap().as_ref().unwrap().gpu_decode_in_forward());
}

fn assert_handoff(fixture: &Fixture) {
    fixture.backend.with_unified_exclusive(|| {
        let mut guard = fixture.backend.moe_pager().lock().unwrap();
        let session = guard.as_mut().unwrap();
        assert!(!session.gpu_decode_in_forward());
        session.handoff_gpu_decode(&fixture.backend).unwrap();
        for roles in fixture.roles {
            assert_eq!(session.routed_roles_resident_mask(&roles, &ROUTED).unwrap(), HIT_MASK);
        }
    });
}

fn auxiliary_projection(backend: &VulkanBackend, seed: f32) {
    let mut graph = Graph::new();
    let desc = TensorDesc::new(vec![NE], DType::F32);
    let input = graph.input(desc.clone());
    let scratch = graph.internal(desc.clone());
    let output = graph.output(desc);
    graph.push(Op::Copy { src: input, src_off: 0, dst: scratch, dst_off: 0, n: NE as u32 });
    graph.push(Op::Scale { x: scratch, dst: output, s: 2.0, n: NE as u32 });
    let src = backend.alloc(NE * 4, BufferUsage::Staging).unwrap();
    let dst = backend.alloc(NE * 4, BufferUsage::Readback).unwrap();
    backend.upload(src.as_ref(), bytemuck::cast_slice(&[seed; NE])).unwrap();
    backend.upload(dst.as_ref(), bytemuck::cast_slice(&[-777.0f32; NE])).unwrap();
    let plan = backend.compile(&graph).unwrap();
    let mut bindings = Bindings::new();
    bindings.bind(input, src.as_ref()).bind(output, dst.as_ref());
    backend.execute(plan.as_ref(), &bindings).unwrap();
    let mut actual = [0f32; NE];
    backend.download(dst.as_ref(), bytemuck::cast_slice_mut(&mut actual)).unwrap();
    assert_eq!(actual, [seed * 2.0; NE]);
    backend.release_auxiliary_runtime().unwrap();
}

fn run_combined(fault: Fault, transitions: bool) {
    let fixture = Fixture::new_with_combined_policy(1, true, Some(true), 16384, 4096, true, 3, true);
    let cfg = fixture.backend.cfg();
    assert!(cfg.device.decode_command_ahead);
    assert!(cfg.device.decode_command_early_submit);
    assert!(cfg.device.decode_command_resident);
    assert!(cfg.kernels.vulkan.decode_gpu_resident_copy_overlap);
    assert!(cfg.kernels.vulkan.decode_gpu_resident_host_ready);
    assert!(cfg.kernels.vulkan.decode_gpu_resident_upload_worker);
    assert_eq!(cfg.kernels.vulkan.decode_gpu_resident_copy, Some(true));
    assert!(!cfg.spec.mtp);
    let embedding = fixture.backend.fork_embedding_client().unwrap();
    let vision = fixture.backend.fork_vision_client().unwrap();
    assert!(Arc::ptr_eq(&embedding.unified_exec, &fixture.backend.unified_exec));
    assert!(Arc::ptr_eq(&vision.unified_exec, &fixture.backend.unified_exec));
    let plan = fixture.backend.compile(&fixture.graph).unwrap();
    fixture.preload();
    fixture.backend.execute(plan.as_ref(), &fixture.bindings()).unwrap();
    let expected = fixture.outputs.map(|id| fixture.read(fixture.buffer(id)));
    assert!(expected.iter().flatten().all(|value| value.is_finite() && *value > 0.0));
    install_cpu_tail(&fixture);
    fixture.reset_frontend();
    for output in fixture.outputs {
        fixture.backend.upload(fixture.buffer(output), bytemuck::cast_slice(&[-777.0f32; NE])).unwrap();
    }
    OWNED_HOST_READY_TEST_ARMS.with(|arms| arms.set(0));
    OWNED_HOST_READY_TEST_OUTCOME.with(|outcome| assert!(outcome.get().is_none()));
    let registration = Registration::new(
        &fixture.backend.shared, Arc::clone(&fixture.backend.unified_exec), fault,
    );
    let result = fixture.backend.execute(plan.as_ref(), &fixture.bindings());
    let proof = registration.0.evidence();
    drop(registration); // Do not affect retry or subsequent resource/phase transitions.
    eprintln!("host-ready combined {fault:?}: {proof:?}");
    assert!(proof.entered && proof.jobs > 1, "must intercept a real multi-job mapped upload");
    assert!(proof.submitted && proof.wait_value != 0);
    assert!(proof.unsignaled_at_submit && proof.fence_pending_at_submit);
    assert_eq!(proof.copied_at_submit, 0, "no memcpy before successful host-wait submit");
    assert!(proof.pending_owned && proof.shared_gate_blocked);
    assert_eq!(proof.controller_cancelled, fault == Fault::SubmittedCancel);
    match fault {
        Fault::None => {
            result.unwrap();
            assert_eq!(proof.copied, proof.jobs);
            OWNED_HOST_READY_TEST_ARMS.with(|arms| assert_eq!(arms.get(), LAYERS));
            assert_eq!(fixture.outputs.map(|id| fixture.read(fixture.buffer(id))), expected);
        }
        _ => {
            assert!(result.is_err(), "fault cannot produce a successful forward");
            assert_eq!(proof.copied, usize::from(fault != Fault::BeforeCopyError));
            OWNED_HOST_READY_TEST_ARMS.with(|arms| assert_eq!(arms.get(), 1));
            for output in fixture.outputs {
                assert_eq!(fixture.read(fixture.buffer(output)), vec![-777.0f32; NE],
                    "submitted recovery must not project partially uploaded weights");
            }
            let guard = fixture.backend.moe_pager().lock().unwrap();
            let session = guard.as_ref().unwrap();
            assert!(!session.gpu_decode_in_forward());
            assert_eq!(session.routed_roles_resident_mask(&fixture.roles[0], &ROUTED).unwrap() & 2, 0,
                "failed COPYING roles cannot be imported as completed joint residency");
        }
    }
    // Retry the same backend/pager/producer/worker; no queue-idle or reset of error policy.
    check_request(&fixture, plan.as_ref(), &expected);
    assert_handoff(&fixture);
    if !transitions { return; }

    let kv = fixture.backend.alloc_segmented_kv(SegmentedKvSpec {
        logical_bytes: 16384, segment_bytes: 4096, segment_elements: 1024, max_segments: 4,
    }).unwrap().unwrap();
    fixture.backend.ensure_segmented_kv(kv.as_ref(), 1).unwrap();
    let segmented = crate::as_segmented_kv(kv.as_ref()).unwrap();
    let identity = crate::pager::buffer_identity(kv.as_ref());
    let table_identity = segmented.table_buffer().device_addr();
    let payload = [0x5au8; 128];
    fixture.backend.upload_range(kv.as_ref(), 0, &payload).unwrap();
    check_request(&fixture, plan.as_ref(), &expected); // Reactivate GPU owner before growth.
    fixture.backend.ensure_segmented_kv(kv.as_ref(), 2).unwrap();
    assert_eq!(segmented.committed(), 2);
    assert_eq!(crate::pager::buffer_identity(kv.as_ref()), identity);
    assert_eq!(segmented.table_buffer().device_addr(), table_identity);
    let mut restored = [0u8; 128];
    fixture.backend.download_range(kv.as_ref(), 0, &mut restored).unwrap();
    assert_eq!(restored, payload);
    check_request(&fixture, plan.as_ref(), &expected);
    assert_eq!(fixture.backend.prepare_prefill(&[(1, 1024 * 1024)]).unwrap(), Some(1));
    assert!(fixture.backend.runtime_phase.lock().unwrap().resident_buffers.is_none());
    check_request(&fixture, plan.as_ref(), &expected);
    fixture.backend.release_transient_runtime(); // Real handoff + phase workspace release.
    assert!(fixture.backend.runtime_phase.lock().unwrap().resident_buffers.is_none());
    check_request(&fixture, plan.as_ref(), &expected);

    auxiliary_projection(&embedding, 1.25);
    assert_handoff(&fixture);
    check_request(&fixture, plan.as_ref(), &expected);
    auxiliary_projection(&vision, -0.75);
    assert_handoff(&fixture);
    check_request(&fixture, plan.as_ref(), &expected);
    fixture.backend.download_range(kv.as_ref(), 0, &mut restored).unwrap();
    assert_eq!(restored, payload, "phase handoff cannot replace live KV backing");
    fixture.backend.release_segmented_kv(kv.as_ref()).unwrap();
    assert_handoff(&fixture);
}

#[test]
#[ignore = "parent serial GPU only; all-five actual mapped fault/cancel after submitted host wait"]
fn gpu_host_ready_combined_submitted_copy_faults_retry() {
    for fault in [Fault::BeforeCopyError, Fault::AfterFirstError, Fault::AfterFirstPanic, Fault::SubmittedCancel] {
        run_combined(fault, false);
    }
}

#[test]
#[ignore = "parent serial GPU only; all-five host-ready followed by real KV/phase/auxiliary handoff"]
fn gpu_host_ready_combined_kv_phase_auxiliary_handoff() {
    run_combined(Fault::None, true);
}

#[path = "adapter_host_ready_prepared_tests.rs"]
mod prepared_tests;
