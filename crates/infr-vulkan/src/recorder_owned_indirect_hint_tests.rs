//! Explicit destination-residency hint probes; no runtime policy or external shader changes.
use super::*;
use crate::indirect_copy::{
    AddressCopyFlagsKHR, CopyMemoryIndirectInfoKHR, Device, ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR,
};

const FRAME_PAIRS: usize = 6;

fn hint_profiles(native: bool) -> [Profile; 2] {
    [
        Profile {
            name: "q5k_2048_512_used8",
            ne: if native { 2048 } else { 256 },
            nff: if native { 512 } else { 256 },
            used: 8,
            gate_up: DType::Q5K,
            down: DType::Q5K,
        },
        Profile {
            name: "iq2s_iq4nl_2560_640_used10",
            ne: if native { 2560 } else { 512 },
            nff: if native { 640 } else { 256 },
            used: 10,
            gate_up: DType::Iq2S,
            down: DType::Iq4Nl,
        },
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    CuPartial,
    Hint0,
    Hint1,
}

const ARMS: [Arm; 3] = [Arm::CuPartial, Arm::Hint0, Arm::Hint1];
const COMPARISONS: [(usize, usize); 3] = [(1, 2), (0, 1), (0, 2)];

fn pair_order(pair: usize, baseline: usize, candidate: usize) -> [usize; 2] {
    if pair.is_multiple_of(2) {
        [baseline, candidate]
    } else {
        [candidate, baseline]
    }
}

fn bounded_range(base: u64, len: u64, address: u64, size: u64) -> bool {
    size != 0
        && address >= base
        && address
            .checked_sub(base)
            .and_then(|offset| offset.checked_add(size))
            .is_some_and(|end| end <= len)
        && address.checked_add(size - 1).is_some()
}

fn assert_local_targets(f: &Fixture) {
    // Fixture::new uses ONLY alloc_mapped_arena_bda for these two buffers. That allocator
    // requires DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT and fails rather than spilling.
    // This assertion is specific to that provenance, not a generic mapped-buffer classifier.
    for buffer in [&f.weights, &f.dummy] {
        let buffer = as_vk_buf(buffer.as_ref()).unwrap();
        assert!(matches!(
            &buffer.backing,
            Backing::Vram { spilled: false, .. }
        ));
        assert!(buffer.mapped_ptr().is_some());
    }
    let base = f.weights.device_addr().unwrap();
    for j in 0..f.job_count {
        let at = (ORDINAL * CAPACITY + j) * 8;
        let job = &f.job_words[at..at + 8];
        let dst = job[2] as u64 | ((job[3] as u64) << 32);
        assert!(bounded_range(
            base,
            f.weights.len_bytes() as u64,
            dst,
            job[4] as u64
        ));
    }
    let dummy_base = f.dummy.device_addr().unwrap();
    let count = 3 * f.rows * f.p.used;
    assert!(bounded_range(
        dummy_base,
        f.dummy.len_bytes() as u64,
        dummy_base,
        4
    ));
    assert!(bounded_range(
        dummy_base,
        (f.dummy.len_bytes() - GUARD) as u64,
        dummy_base + (16 + (ORDINAL * CAPACITY + count - 1) * 16) as u64,
        4,
    ));
}

fn record_hint(f: &Fixture, rec: &Recorder<'_>, arm: Arm, hit_only: bool) {
    if arm == Arm::CuPartial {
        f.record(rec, Variant::CuPartial, hit_only);
        return;
    }
    let dst_flags = if arm == Arm::Hint1 {
        ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR
    } else {
        0
    };
    rec.begin_owned_indirect_copy_phase_flags(
        f.journal.as_ref(),
        f.jobs.as_ref(),
        f.commands.as_ref(),
        f.dummy.as_ref(),
        ORDINAL as u32,
        (3 * f.rows * f.p.used) as u32,
        [0; 4],
        0, // Real sources are imported ordinary RAM; dummy sources alone are device-local.
        dst_flags,
    )
    .unwrap();
    // Identical to the existing KhrFull schedule. Only dstCopyFlags differs between hints.
    f.projections(rec, 1, 0);
    f.tail(rec, 1, 0);
    rec.global_barrier();
    if !hit_only && f.job_count != 0 {
        f.projections(rec, 2, 0);
        f.tail(rec, 2, 0);
    }
}

fn run_hint(be: &VulkanBackend, f: &Fixture, arm: Arm, hit_only: bool) {
    assert_local_targets(f);
    let rec = be.recorder().unwrap();
    assert!(!rec.no_barrier && !rec.full_barrier && !rec.stages);
    assert!(!rec.prof_ops && rec.timeline_query.borrow().is_none());
    rec.seed_barrier();
    record_hint(f, &rec, arm, hit_only);
    rec.publish_host_reads();
    rec.finish().unwrap();
}

fn verify_commands_and_dummy(f: &Fixture) {
    let mut commands = vec![POISON; f.commands.len_bytes()];
    let mut dummy = vec![POISON; f.dummy.len_bytes()];
    dummy[..4].copy_from_slice(&[0x12, 0x34, 0x56, 0x78]);
    for j in 0..3 * f.rows * f.p.used {
        let at = (ORDINAL * CAPACITY + j) * 8;
        let words = if j < f.job_count {
            let job = &f.job_words[at..at + 8];
            [job[0], job[1], job[2], job[3], job[4], 0, 0, 0]
        } else {
            let offset = 16 + (ORDINAL * CAPACITY + j) * 16;
            dummy[offset..offset + 4].copy_from_slice(&[0x12, 0x34, 0x56, 0x78]);
            let src = f.dummy.device_addr().unwrap();
            let dst = src + offset as u64;
            [
                src as u32,
                (src >> 32) as u32,
                dst as u32,
                (dst >> 32) as u32,
                4,
                0,
                0,
                0,
            ]
        };
        commands[at * 4..at * 4 + 32].copy_from_slice(bytemuck::cast_slice(&words));
    }
    assert_eq!(
        read(f.commands.as_ref()),
        commands,
        "all command slots and padding"
    );
    assert_eq!(
        read(f.dummy.as_ref()),
        dummy,
        "nonzero unique dummy copies and guards"
    );
    assert_eq!(
        read(f.commands_late.as_ref()),
        vec![POISON; f.commands_late.len_bytes()],
        "unused phase buffer is untouched",
    );
}

#[test]
fn indirect_hint_record_with_flags_cpu_abi_and_default_zero() {
    use ash::vk::Handle;
    use std::cell::RefCell;
    use std::mem::{offset_of, size_of};

    thread_local! {
        static CAPTURE: RefCell<Option<(vk::CommandBuffer, CopyMemoryIndirectInfoKHR)>> =
            const { RefCell::new(None) };
    }
    unsafe extern "system" fn capture(
        cmd: vk::CommandBuffer,
        info: *const CopyMemoryIndirectInfoKHR,
    ) {
        CAPTURE.with(|slot| *slot.borrow_mut() = Some((cmd, *info)));
    }
    let device = Device {
        cmd_copy_memory_indirect: capture,
        supported_queues: vk::QueueFlags::COMPUTE,
        synchronization2: false,
    };
    assert_eq!(size_of::<AddressCopyFlagsKHR>(), 4);
    assert_eq!(ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR, 1);
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(size_of::<CopyMemoryIndirectInfoKHR>(), 56);
        assert_eq!(offset_of!(CopyMemoryIndirectInfoKHR, src_copy_flags), 16);
        assert_eq!(offset_of!(CopyMemoryIndirectInfoKHR, dst_copy_flags), 20);
    }
    for rows in 1..=8u32 {
        for (explicit, src, dst) in [(true, 0, 0), (true, 0, 1), (true, 1, 0), (false, 0, 0)] {
            let address = 0x1_0000_1000 + rows as u64 * 32;
            let command = vk::CommandBuffer::from_raw(0x2000 + rows as u64 * 16);
            let count = rows * 30;
            let size = count as u64 * 32;
            unsafe {
                if explicit {
                    device.record_with_flags(command, address, count, 32, size, src, dst);
                } else {
                    device.record(command, address, count, 32, size);
                }
            }
            let (cmd, info) = CAPTURE.with(|slot| slot.borrow_mut().take().unwrap());
            assert_eq!(cmd, command);
            assert_eq!(info.s_type.as_raw(), 1_000_549_002);
            assert!(info.p_next.is_null());
            assert_eq!(info.src_copy_flags, src);
            assert_eq!(info.dst_copy_flags, dst);
            assert_eq!(info.copy_count, count);
            assert_eq!(info.copy_address_range.address, address);
            assert_eq!(info.copy_address_range.size, size);
            assert_eq!(info.copy_address_range.stride, 32);
        }
    }
}

#[test]
fn indirect_hint_profiles_and_six_pair_schedule_cpu() {
    let [q5, iq] = hint_profiles(true);
    assert_eq!((q5.ne, q5.nff, q5.used), (2048, 512, 8));
    assert_eq!((q5.gate_up, q5.down), (DType::Q5K, DType::Q5K));
    assert_eq!((iq.ne, iq.nff, iq.used), (2560, 640, 10));
    assert_eq!((iq.gate_up, iq.down), (DType::Iq2S, DType::Iq4Nl));
    for (baseline, candidate) in COMPARISONS {
        let order: Vec<_> = (0..FRAME_PAIRS)
            .map(|p| pair_order(p, baseline, candidate))
            .collect();
        assert_eq!(
            order
                .iter()
                .filter(|&&v| v == [baseline, candidate])
                .count(),
            3
        );
        assert_eq!(
            order
                .iter()
                .filter(|&&v| v == [candidate, baseline])
                .count(),
            3
        );
    }
    for p in hint_profiles(false) {
        for rows in 1..=8 {
            assert!(3 * rows * p.used <= CAPACITY);
            for dtype in [p.gate_up, p.down] {
                assert_eq!(
                    infr_testkit::synth_weight(dtype, p.ne * p.nff, 7).len() % 16,
                    0
                );
            }
        }
    }
}

#[test]
fn indirect_hint_payload_range_proof_cpu() {
    assert!(bounded_range(0x1000, 64, 0x1000, 4));
    assert!(bounded_range(0x1000, 64, 0x103c, 4));
    assert!(!bounded_range(0x1000, 64, 0x0ffc, 4));
    assert!(!bounded_range(0x1000, 64, 0x1040, 4));
    assert!(!bounded_range(0x1000, 64, 0x1000, 0));
    assert!(!bounded_range(0, u64::MAX, u64::MAX - 1, 4));
    assert!(!bounded_range(u64::MAX - 3, 8, u64::MAX - 3, 8));
}

#[test]
#[ignore = "parent serial GPU; explicit MOE4ALL_TEST_DEVICE; hint0/1 real and dummy copies rows 1..8"]
fn gpu_owned_indirect_hint_correctness_rows_1_to_8() {
    let be = backend();
    for p in hint_profiles(false) {
        let source = Sources::new(&be, p, 0);
        for rows in 1..=8 {
            let mut f = Fixture::new(&be, &source, p, rows);
            let bank = rows % source.banks;
            let expected = f.reference(&be, &source, bank);
            for arm in ARMS {
                for hit_only in [true, false] {
                    f.prepare(&source, bank, false);
                    run_hint(&be, &f, arm, hit_only);
                    f.verify(&expected, hit_only);
                    if arm != Arm::CuPartial {
                        verify_commands_and_dummy(&f);
                    }
                }
                f.prepare_all_hit(&source, bank);
                run_hint(&be, &f, arm, false);
                f.verify(&expected, false);
                if arm != Arm::CuPartial {
                    verify_commands_and_dummy(&f);
                }
            }
        }
    }
}

#[test]
#[ignore = "parent serial GPU; native Q5K/IQ2S-IQ4NL geometry, continuous frames, rows 1/2, six AB/BA pairs"]
fn gpu_owned_indirect_hint_continuous_native_rows_1_2() {
    let be = backend();
    let query = Timestamps::new(&be);
    for p in hint_profiles(true) {
        let source = Sources::new(&be, p, SOURCE_RING_BYTES);
        for rows in [1, 2] {
            // Preflight EACH arm against poisoned real payload and fresh dummy/command guards.
            // No BAR readbacks or CPU fixture mutation occur between the timed frame arms.
            {
                let mut f = Fixture::new(&be, &source, p, rows);
                let expected = f.reference(&be, &source, 0);
                for arm in ARMS {
                    f.prepare(&source, 0, false);
                    run_hint(&be, &f, arm, false);
                    f.verify(&expected, false);
                    if arm != Arm::CuPartial {
                        verify_commands_and_dummy(&f);
                    }
                }
            }
            let mut fixtures = Vec::new();
            let mut expected = Vec::new();
            for bank in 0..source.banks {
                let mut f = Fixture::new(&be, &source, p, rows);
                expected.push(f.reference(&be, &source, bank));
                f.prepare(&source, bank, false);
                assert_local_targets(&f);
                fixtures.push(f);
            }
            let payload_bytes = source.banks * source.miss_bytes;
            assert!(
                payload_bytes > 128 * 1024 * 1024,
                "unique copied source exceeds cache"
            );
            let recordings: Vec<_> = ARMS
                .iter()
                .map(|&arm| {
                    let rec = be.recorder_persistent().unwrap();
                    assert!(!rec.no_barrier && !rec.full_barrier && !rec.stages);
                    assert!(!rec.prof_ops && rec.timeline_query.borrow().is_none());
                    rec.seed_barrier();
                    rec.global_barrier();
                    query.begin(&rec);
                    for f in &fixtures {
                        record_hint(f, &rec, arm, false);
                        rec.global_barrier();
                    }
                    query.end(&rec);
                    rec.publish_host_reads();
                    rec.finish_record().unwrap()
                })
                .collect();
            for recording in &recordings {
                recording.replay().unwrap();
                recording.replay().unwrap();
            }
            for (baseline, candidate) in COMPARISONS {
                let mut base_times = Vec::new();
                let mut candidate_times = Vec::new();
                let mut ratios = Vec::new();
                for pair in 0..FRAME_PAIRS {
                    let mut times = [0.0; 2];
                    for index in pair_order(pair, baseline, candidate) {
                        recordings[index].replay().unwrap();
                        let us = query.elapsed(&be) / fixtures.len() as f64;
                        times[usize::from(index == candidate)] = us;
                    }
                    base_times.push(times[0]);
                    candidate_times.push(times[1]);
                    ratios.push(times[0] / times[1]);
                }
                for (f, reference) in fixtures.iter().zip(&expected) {
                    assert_eq!(
                        read(f.output.as_ref()),
                        *reference,
                        "continuous output profile={} rows={rows} candidate={:?}",
                        p.name,
                        ARMS[candidate]
                    );
                }
                println!("OWNED_INDIRECT_HINT_CONTINUOUS profile={} synthetic_weights=true native_geometry=true ne={} nff={} used={} gate_up={:?} down={:?} rows={rows} layers={} source_bytes={} payload_bytes_per_frame={payload_bytes} baseline={:?} candidate={:?} src_flags=0 hint0_dst_flags=0 hint1_dst_flags=1 schedule=KhrFull_vs_CuPartial dummy_bytes=4 pairs={FRAME_PAIRS} order=AB_BA timestamps=outer_only source=ImportedHostBDA baseline_us_per_layer={base_times:?} candidate_us_per_layer={candidate_times:?} baseline_median_us={} candidate_median_us={} median_paired_speedup={} verified=true",
                    p.name, p.ne, p.nff, p.used, p.gate_up, p.down, fixtures.len(), source.span * source.banks,
                    ARMS[baseline], ARMS[candidate], median(&base_times), median(&candidate_times), median(&ratios));
            }
            // Verify retained payload/sidecars after continuous reuse, outside all timed pairs.
            for (f, reference) in fixtures.iter().zip(&expected) {
                f.verify(reference, false);
                verify_commands_and_dummy(f);
            }
            drop(recordings); // All replays drained before fixture/source leases are released.
        }
    }
}
