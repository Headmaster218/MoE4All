//! Real adapter qualification with routes above the standalone module's ten-expert set.
use super::*;

const NE4: usize = 2560;
const FF4: usize = 640;
const USED4: usize = 10;
const ROWS4: usize = 4;

fn routes(pattern: usize) -> Vec<Vec<u32>> {
    (0..ROWS4)
        .map(|row| {
            let start = match pattern {
                0 => 0,
                1 => row * 3,
                _ => row * USED4,
            };
            (start..start + USED4).map(|e| e as u32).collect()
        })
        .collect()
}

fn fixture(enabled: bool, selected: &[Vec<u32>]) -> Fixture {
    let mut f = Fixture::with_profile_modes(
        NE4,
        FF4,
        USED4,
        ROWS4,
        [DType::Iq2S, DType::Iq2S, DType::Iq4Nl],
        false,
        false,
        enabled,
        Some(40),
    );
    let Op::MoeFfn {
        x: input, router, router_x, ..
    } = f.graph.ops[1]
    else {
        unreachable!()
    };
    let mut x = vec![0.0f32; ROWS4 * NE4];
    let mut w = vec![0.0f32; 40 * NE4];
    for row in 0..ROWS4 {
        x[row * NE4 + row] = 1.0;
        for expert in 0..40 {
            w[expert * NE4 + row] = -3.0;
        }
        for (rank, &expert) in selected[row].iter().enumerate() {
            w[expert as usize * NE4 + row] = (USED4 - rank) as f32 / 8.0;
        }
    }
    f.backend
        .upload(f.buffer(router_x), bytemuck::cast_slice(&x))
        .unwrap();
    f.backend
        .upload(f.buffer(router), bytemuck::cast_slice(&w))
        .unwrap();
    // IQ synthetic scales are tiny: outputs must be well above the absolute tolerance.
    let values: Vec<f32> = (0..ROWS4 * NE4).map(|i| ((i % 17) as f32 - 8.0) * 4.0).collect();
    f.backend.upload(f.buffer(input), bytemuck::cast_slice(&values)).unwrap();
    f.routed = selected.iter().flatten().copied().collect();
    f.routed.sort_unstable();
    f.routed.dedup();
    f.preload();
    f
}

fn miss(f: &Fixture, selected: &[Vec<u32>]) {
    let mut pager = f.backend.moe_pager().lock().unwrap();
    let pager = pager.as_mut().unwrap();
    pager.handoff_gpu_decode(&f.backend).unwrap();
    let mut ids: Vec<_> = selected.iter().map(|r| r[0]).collect();
    ids.sort_unstable();
    ids.dedup();
    pager.discard_cpu_miss(f.roles, &ids);
}

fn verify(f: &Fixture, expected: &[f32], selected: &[Vec<u32>], mapped_miss: bool) -> f32 {
    let error = compare(&f.read(f.buffer(f.output), ROWS4 * NE4), expected);
    assert_eq!(
        f.read(f.buffer(f.frontend), ROWS4 * NE4),
        vec![1.0; ROWS4 * NE4]
    );
    for (_, buffer) in f
        .buffers
        .iter()
        .filter(|(_, b)| b.len_bytes() > GUARD_WORDS * 4)
    {
        let values = f.read(buffer.as_ref(), buffer.len_bytes() / 4);
        assert!(values[values.len() - GUARD_WORDS..]
            .iter()
            .all(|&v| v == GUARD));
    }
    if mapped_miss {
        let (journal, copy) = {
            let phase = f.backend.runtime_phase.lock().unwrap();
            let buffers = phase.resident_buffers.as_ref().unwrap();
            (buffers.journal.clone(), buffers.gpu_copy.clone().unwrap())
        };
        let mut words = vec![0u32; crate::recorder::RESIDENT_JOURNAL_WORDS];
        f.backend
            .download(journal.as_ref(), bytemuck::cast_slice_mut(&mut words))
            .unwrap();
        for row in 0..ROWS4 {
            assert_eq!(&words[row * USED4..(row + 1) * USED4], selected[row]);
        }
        let unique = selected
            .iter()
            .map(|row| row[0])
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(
            words[crate::recorder::RESIDENT_JOURNAL_JOB_COUNT_WORD] as usize,
            unique * 3
        );
        assert_eq!(words[crate::recorder::RESIDENT_JOURNAL_STATUS_WORD], 4);
        assert!(!copy.owned_in_forward());
    }
    error
}

fn verify_fresh_partials(f: &Fixture, selected: &[Vec<u32>]) {
    let phase = f.backend.runtime_phase.lock().unwrap();
    let (_, buffer) = phase.pool.buffers.iter()
        .find(|((tag, _), _)| *tag == crate::recorder::owned_splitk_hit::PARTIAL_TAG).unwrap();
    let values = f.read(buffer.as_ref(), 8_192_000 / 4);
    let missing: std::collections::HashSet<_> = selected.iter().map(|row| row[0]).collect();
    for row in 0..ROWS4 {
        for slot in 0..USED4 {
            for part in 0..20 {
                let value = values[((row * USED4 + slot) * 20 + part) * NE4];
                if missing.contains(&selected[row][slot]) {
                    assert!(value.is_nan(), "M partial must remain poisoned");
                } else {
                    assert!(value.is_finite(), "H partial must be newly written, including prepared replay");
                }
            }
        }
    }
}

fn poison_partial_and_y(f: &Fixture) -> (u64, usize) {
    let phase = f.backend.runtime_phase.lock().unwrap();
    let partials: Vec<_> = phase
        .pool
        .buffers
        .iter()
        .filter(|((tag, _), _)| *tag == crate::recorder::owned_splitk_hit::PARTIAL_TAG)
        .collect();
    assert_eq!(partials.len(), 1);
    let partial = partials[0].1.as_ref();
    assert!(partial.len_bytes() >= 8_192_000);
    let identity = (partial.device_addr().unwrap(), partial.len_bytes());
    for tag in [
        crate::recorder::owned_splitk_hit::PARTIAL_TAG,
        "moe_paged_y",
    ] {
        let (_, b) = phase
            .pool
            .buffers
            .iter()
            .find(|((t, _), _)| *t == tag)
            .unwrap();
        f.backend
            .upload(
                b.as_ref(),
                bytemuck::cast_slice(&vec![f32::NAN; b.len_bytes() / 4]),
            )
            .unwrap();
    }
    identity
}

#[test]
#[ignore = "parent serial GPU; actual four-row split-K, diverse routes, mapped miss and prepared replay"]
fn gpu_owned_splitk_adapter_diverse_routes_and_prepared() {
    for pattern in 0..3 {
        let selected = routes(pattern);
        let baseline = fixture(false, &selected);
        let old = VkDecodePlan::new_cpu(&baseline.graph, false, true);
        execute_static(&baseline.backend, &old, &baseline.bindings()).unwrap();
        let expected = baseline.read(baseline.buffer(baseline.output), ROWS4 * NE4);
        assert!(expected.iter().all(|v| v.is_finite()) && expected.iter().any(|v| v.abs() > 1e-3));
        drop(old);
        drop(baseline);
        let f = fixture(true, &selected);
        let current = VkDecodePlan::new_cpu(&f.graph, false, true);
        let mut next_graph = f.graph.clone();
        for span in &mut next_graph.sequence_spans {
            span.start_pos += 1;
        }
        let next = VkDecodePlan::new_cpu(&next_graph, false, true);
        execute_static(&f.backend, &current, &f.bindings()).unwrap();
        let mut error = verify(&f, &expected, &selected, false);
        assert!(
            f.backend
                .built_kernel_names()
                .iter()
                .any(|name| name.contains("hit_splitk")),
            "candidate must really be selected"
        );
        miss(&f, &selected);
        f.reset_frontend();
        let identity = poison_partial_and_y(&f);
        let epoch = f.epoch();
        execute_static_next(
            &f.backend,
            &current,
            &f.bindings(),
            Some((&next, &f.bindings(), &[1; ROWS4])),
        )
        .unwrap();
        error = error.max(verify(&f, &expected, &selected, true));
        verify_fresh_partials(&f, &selected);
        assert_eq!(
            f.epoch(),
            epoch,
            "no scratch growth after ownership capture"
        );
        assert!(
            next.prepared_prefix.lock().unwrap().is_some(),
            "actual prepared packet"
        );
        miss(&f, &selected);
        f.reset_frontend();
        assert_eq!(poison_partial_and_y(&f), identity);
        let trace = ActivationTrace::begin();
        execute_static(&f.backend, &next, &f.bindings()).unwrap();
        trace.assert_submitted(1);
        assert!(next.prepared_prefix.lock().unwrap().is_none());
        assert_eq!(f.epoch(), epoch);
        error = error.max(verify(&f, &expected, &selected, true));
        verify_fresh_partials(&f, &selected);
        println!("SPLITK_ADAPTER pattern={pattern} rows=4 experts=40 top10=true IDs_above9={} actual_router=true mapped_miss=true prepared=true fresh_H_partials=true untouched_M_partials=true guards=true epoch_stable=true weighted_max_abs={error} output_tolerance=2e-6_plus_1e-4_relative output_amplitude_above_1e-3=true", selected.iter().flatten().any(|&id| id > 9));
    }
}
