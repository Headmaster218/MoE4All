#![cfg(test)]

//! Touch-order oracle only: physical admission still protects the entire routed union.
use super::*;

// Build the legacy order as an explicit event list, independently of the shader's rank formula.
fn legacy_stamps(probe: &mut PoolProbe) {
    let ids = probe.initial[0][..probe.push[0] as usize].to_vec();
    legacy_stamps_for(probe, &ids);
}

pub(super) fn legacy_stamps_for(probe: &mut PoolProbe, ids: &[u32]) {
    let count = probe.push[0] as usize;
    assert_eq!(ids.len(), count, "use actual routed IDs, including repeats");
    let bases = [probe.push[9], probe.push[10], probe.push[11]];
    let hits: Vec<_> = ids
        .iter()
        .map(|&id| {
            bases
                .iter()
                .all(|&base| probe.initial[6][(base + id) as usize * 4 + 2] == 1)
        })
        .collect();
    let mut events = Vec::new();
    for phase in [true, false] {
        for &base in &bases {
            for (index, &id) in ids.iter().enumerate() {
                if hits[index] == phase {
                    events.push(base + id);
                }
            }
        }
    }
    assert_eq!(events.len(), count * 3);
    let clock = stamp(&probe.initial[5][10..12]);
    for (ordinal, block) in events.into_iter().enumerate() {
        let slot = probe.expected[6][block as usize * 4 + 3] as usize;
        let start = (SLOT_BASE + slot) * 4;
        probe.expected[5][start..start + 2].copy_from_slice(&pair(clock + ordinal as u64 + 1));
    }
}

#[test]
#[ignore = "central serial GPU only; external serial legacy touch-order module"]
fn gpu_owned_legacy_touch_external_oracle() {
    let path =
        std::env::var_os("MOE4ALL_OWNER_LEGACY_TOUCH_ORDER_SPV").expect("external SPV required");
    let spv =
        ash::util::read_spv(&mut std::io::Cursor::new(std::fs::read(&path).unwrap())).unwrap();
    let mut config = infr_core::config::Config::default();
    config.device.dev = Some(std::env::var("MOE4ALL_TEST_DEVICE").expect("explicit GPU required"));
    let be = VulkanBackend::new_with(Arc::new(config)).unwrap();
    let kernel = be.kernel_sg(
        "legacy_touch_external_oracle",
        &spv,
        8,
        GPU_OWNED_FAST_PUSH_BYTES,
        32,
    );
    for rows in [1, 2, 4, 8] {
        for workload in [
            Workload::AllHit,
            Workload::OneExpertMiss,
            Workload::MixedFresh,
            Workload::MixedStale,
            Workload::AliasedStale,
        ] {
            let slots = 512;
            let mut probe = PoolProbe::new(&be, slots, rows * probe_used(), workload);
            legacy_stamps(&mut probe);
            probe.buffers[5] = be
                .alloc_mapped_arena_bda((probe.heap_base + 2 + slots * 4) * 16)
                .unwrap()
                .0;
            probe.reset(&be, false, true);
            let geometry = [
                0,
                SLOT_BYTES[0],
                0,
                slots as u32,
                0,
                SLOT_BYTES[0],
                0,
                slots as u32,
                1,
                SLOT_BYTES[1],
                slots as u32,
                slots as u32,
            ];
            let push = gpu_owned_fast_push(&probe.push, &geometry);
            let bindings: Vec<_> = probe
                .buffers
                .iter()
                .map(|buffer| Recorder::vkb_exact(buffer.as_ref()))
                .collect();
            let rec = be.recorder().unwrap();
            rec.owned_copy_barrier(true, false);
            rec.dispatch(kernel, &bindings, 8, bytemuck::cast_slice(&push), 1);
            rec.owned_copy_barrier(false, true);
            rec.finish().unwrap();
            // All non-FIFO buffers and every directory header/slot/stamp/stat are exact;
            // FIFO representation is validated as a chronological complete live order.
            probe.assert_result(&be, false, true, false, false);
            eprintln!("LEGACY_TOUCH_ORACLE used={} rows={rows} workload={workload:?} blocks_slots_stamps_jobs_control=exact fifo_live_order=exact physical_fifo_padding_not_compared=true", probe_used());
        }
    }
}
