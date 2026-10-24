use super::*;
use ash::vk::Handle;

fn descriptor(buffer: u64, offset: u64, range: u64) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo {
        buffer: vk::Buffer::from_raw(buffer),
        offset,
        range,
    }
}

#[test]
fn independent_copy_scope_accepts_adjacent_ranges_and_shared_readers() {
    let mut batch = IndependentDispatchBatch::default();
    assert!(batch.accept(&[descriptor(1, 0, 256), descriptor(2, 0, 256)], 1));
    assert!(batch.accept(&[descriptor(1, 0, 256), descriptor(2, 256, 256)], 1));
    // Ordinary standalone descriptors cover their entire buffers conservatively.
    assert!(batch.accept(&[descriptor(3, 0, vk::WHOLE_SIZE)], 0));
    assert!(batch.accept(&[descriptor(3, 0, vk::WHOLE_SIZE)], 0));
    assert!(!batch.closed);
}

#[test]
fn independent_copy_scope_restores_sync_for_raw_war_and_waw_dependencies() {
    for (first_write, next_write) in [(true, false), (false, true), (true, true)] {
        let mut batch = IndependentDispatchBatch::default();
        assert!(batch.accept(&[descriptor(1, 256, 256)], usize::from(first_write)));
        assert!(!batch.accept(&[descriptor(1, 384, 256)], usize::from(next_write)));
        assert!(batch.closed);
        // A helper's intermediate output dependency closes the whole remaining window.
        assert!(!batch.accept(&[descriptor(2, 0, 256)], 1));
    }
    let mut batch = IndependentDispatchBatch::default();
    assert!(batch.accept(&[descriptor(1, 0, vk::WHOLE_SIZE)], 0));
    assert!(!batch.accept(&[descriptor(1, 512, 256)], 1));
}

#[test]
fn independent_copy_scope_rejects_invalid_or_unbounded_bookkeeping() {
    for invalid in [descriptor(1, 0, 0), descriptor(1, u64::MAX - 8, 16)] {
        assert!(!IndependentDispatchBatch::default().accept(&[invalid], 1));
    }
    let mut batch = IndependentDispatchBatch::default();
    for i in 0..64 {
        assert!(batch.accept(&[descriptor(i + 1, 0, 256)], 1));
    }
    assert!(!batch.accept(&[descriptor(65, 0, 256)], 1));
}

#[test]
fn gpu_copy_prepare_push_abi_appends_groups_without_shifting_existing_fields() {
    let params = [1, 2, 3, 4, 5, 6, 7, 8];
    let roles = [101, 102, 103, 104, 105, 106, 107, 108, 109];
    for immutable in [false, true] {
        for groups in [1, 2, 4, 8] {
            let push = gpu_copy_prepare_push(params, immutable, roles, groups);
            assert_eq!(bytemuck::cast_slice::<u32, u8>(&push).len(), 76);
            assert_eq!(GPU_COPY_PREPARE_PUSH_BYTES, 76);
            assert_eq!(&push[..8], &params);
            assert_eq!(push[8], u32::from(immutable));
            assert_eq!(&push[9..18], &roles);
            assert_eq!(push[18], groups);
        }
    }
    for invalid in [0, 9, u32::MAX] {
        assert!(
            std::panic::catch_unwind(|| gpu_copy_prepare_push(params, true, roles, invalid))
                .is_err()
        );
    }
}

#[test]
fn gpu_copy_overlap_publishes_only_current_indirect_args_range() {
    for ordinal in [0, 1, RESIDENT_MAX_LAYERS - 1] {
        let buffer = vk::Buffer::from_raw(7);
        let barrier = gpu_copy_payload_args_barrier(buffer, 256, ordinal);
        assert_eq!(barrier.buffer, buffer);
        assert_eq!(barrier.offset, (256 + (ordinal * 4 + 1) * 4) as u64);
        assert_eq!(barrier.size, 12);
        assert_eq!(barrier.src_access_mask, vk::AccessFlags::SHADER_WRITE);
        assert_eq!(
            barrier.dst_access_mask,
            vk::AccessFlags::INDIRECT_COMMAND_READ
        );
        assert_eq!(barrier.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(barrier.dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
    }
}

#[test]
fn gpu_copy_overlap_verify_is_single_lane_bounded_and_pair_exclusive() {
    for requested in [false, true] {
        for verify in [false, true] {
            for rows in 0..=infr_core::backend::MAX_DECODE_ROWS + 1 {
                for independent in [false, true] {
                    for pair in [false, true] {
                        for immutable in [false, true] {
                            for predicate in [false, true] {
                                for indirect in [false, true] {
                                    let shape = !verify || (!independent && matches!(rows, 2..=4));
                                    assert_eq!(
                                        gpu_copy_overlap_eligible(
                                            requested, verify, rows, independent, pair,
                                            immutable, predicate, indirect,
                                        ),
                                        requested && shape && !pair && immutable && predicate && !indirect,
                                        "verify={verify} rows={rows} independent={independent} pair={pair}",
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
