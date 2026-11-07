//! Serial owner payload reads may retire only at their proven compute publication boundary.
use super::*;

#[test]
fn serial_owned_payload_read_retirement_requires_proven_reset_and_clean_compute_state() {
    for bits in 0..16 {
        let reset = bits & 1 != 0;
        let normal = bits & 2 != 0;
        let writes_empty = bits & 4 != 0;
        let transfer = bits & 8 != 0;
        assert_eq!(
            serial_owned_payload_read_retirement_eligible(reset, normal, writes_empty, transfer),
            bits == 7,
            "bits={bits}"
        );
    }
}

fn backend() -> VulkanBackend {
    let mut cfg = infr_core::config::Config::default();
    cfg.device.dev = Some(
        std::env::var("MOE4ALL_TEST_DEVICE")
            .expect("parent must explicitly select an idle test GPU"),
    );
    VulkanBackend::new_with(Arc::new(cfg)).unwrap()
}

fn alias(owner: &dyn Buffer, offset: usize) -> VkBuffer {
    let original = Recorder::vk_of(owner);
    // Caller retains owner through every recording, submission and readback.
    let mut alias = unsafe { original.recording_alias() };
    alias.backing = Backing::RecordingAlias {
        mapped: original.mapped_ptr().map(|ptr| unsafe { ptr.add(offset) }),
        backing_len: Some(original.sub_offset + original.size),
    };
    alias.sub_offset = original.sub_offset + offset;
    alias.size = 256;
    alias.own_addr = original
        .device_addr()
        .map(|address| address + offset as u64);
    alias
}

fn record_payload(rec: &Recorder<'_>, metadata: &[VkBuffer; 4]) {
    let kernel = rec.be.kernel(
        "gpu_owned_copy_publish",
        crate::gemm::gpu_owned_copy_publish_spv(false),
        4,
        4,
    );
    let bindings: [vk::DescriptorBufferInfo; 4] =
        std::array::from_fn(|i| Recorder::vkb(&metadata[i]));
    rec.dispatch(kernel, &bindings, 0, bytemuck::bytes_of(&0u32), 1);
}

#[test]
#[ignore = "parent serial GPU only; real shared-VkBuffer metadata/output ranges and WAR count"]
fn gpu_serial_owned_payload_reads_retire_only_redundant_alias_war() {
    let be = backend();
    let (owner, _) = be.alloc_mapped_arena_bda(1536).unwrap();
    let metadata = std::array::from_fn(|i| alias(owner.as_ref(), i * 256));
    let output = alias(owner.as_ref(), 1024);
    assert_eq!(metadata[0].buffer, output.buffer);
    assert_ne!(metadata[0].sub_offset, output.sub_offset);

    for retire in [false, true] {
        let mut words = vec![0u32; 1536 / 4];
        words[1024 / 4] = 3.0f32.to_bits();
        be.upload(owner.as_ref(), bytemuck::cast_slice(&words))
            .unwrap();
        let rec = be.recorder().unwrap();
        rec.scale(&metadata[0], 1.0, 64);
        // Same handle, disjoint ranges: both payload metadata inputs have a tracked writer.
        let reset = rec.serial_owned_payload_reads_will_reset(&metadata[0], &metadata[1]);
        assert!(reset);
        let before_payload = *rec.barriers.borrow();
        record_payload(&rec, &metadata);
        assert_eq!(*rec.barriers.borrow(), before_payload + 1);
        assert!(rec.dirty_writes.borrow().is_empty());
        assert!(!rec.dirty_transfer.get());
        assert!(rec.dirty_reads.borrow().contains(&output.buffer));
        // This is the production post-payload barrier's compute access/stage scope.
        rec.owned_copy_barrier(false, false);
        if retire {
            assert!(rec.retire_serial_owned_payload_reads(reset));
            assert!(rec.dirty_reads.borrow().is_empty());
        }
        let before_output = *rec.barriers.borrow();
        rec.scale(&output, 2.0, 1);
        assert_eq!(*rec.barriers.borrow() - before_output, usize::from(!retire));
        // Real output writes MUST remain tracked after the optimized boundary.
        assert!(rec.dirty_writes.borrow().contains(&output.buffer));
        rec.scale(&output, 2.0, 1);
        assert_eq!(
            *rec.barriers.borrow() - before_output,
            usize::from(!retire) + 1
        );
        rec.finish().unwrap();
        let mut actual = vec![0u32; words.len()];
        be.download(owner.as_ref(), bytemuck::cast_slice_mut(&mut actual))
            .unwrap();
        words[1024 / 4] = 12.0f32.to_bits();
        assert_eq!(actual, words, "metadata/range canaries must remain exact");
    }
}

#[test]
#[ignore = "parent serial GPU only; rejected provenance, transfer/debug/indirect/independent tracking"]
fn gpu_serial_owned_payload_read_retirement_preserves_other_hazards() {
    let be = backend();
    let (owner, _) = be.alloc_mapped_arena_bda(1536).unwrap();
    be.upload(owner.as_ref(), &[0; 1536]).unwrap();
    let metadata = std::array::from_fn(|i| alias(owner.as_ref(), i * 256));
    let output = alias(owner.as_ref(), 1024);

    for mode in 0..8 {
        let mut rec = be.recorder().unwrap();
        record_payload(&rec, &metadata);
        assert!(!rec.serial_owned_payload_reads_will_reset(&metadata[0], &metadata[1]));
        rec.owned_copy_barrier(false, false);
        if mode == 2 {
            rec.no_barrier = true;
        } else if mode == 3 {
            rec.full_barrier = true;
        }
        let scope = if mode == 7 {
            Some(rec.independent_dispatch_scope())
        } else {
            None
        };
        match mode {
            0 => {} // No pre-payload RAW proof.
            1 => rec.copy(&metadata[0], 0, &output, 0, 4),
            2 | 3 => {}
            4 => rec.suppress_sync(true),
            5 => rec.indirect_pending.set(true),
            6 => rec.scale(&output, 1.0, 1), // A later real write cannot be retired.
            7 => record_payload(&rec, &metadata),
            _ => unreachable!(),
        }
        let reads = rec.dirty_reads.borrow().clone();
        let writes = rec.dirty_writes.borrow().clone();
        let transfer = rec.dirty_transfer.get();
        // All modes except 0 deliberately pretend that an earlier reset was proven;
        // the postconditions must still independently reject the retirement.
        assert!(
            !rec.retire_serial_owned_payload_reads(mode != 0),
            "mode={mode}"
        );
        assert_eq!(*rec.dirty_reads.borrow(), reads, "mode={mode}");
        assert_eq!(*rec.dirty_writes.borrow(), writes, "mode={mode}");
        assert_eq!(rec.dirty_transfer.get(), transfer, "mode={mode}");
        drop(scope);
        rec.discard().unwrap();
    }
}
