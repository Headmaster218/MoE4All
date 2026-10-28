//! Complete H + whole-payload copy comparisons. No role-split transfers in any arm.
use super::*;

const REPEATS: usize = 8;

#[derive(Clone, Copy)]
enum Candidate {
    Baseline,
    Tiled {
        kernel: ComputeKernel,
        local: usize,
        tiles: usize,
    },
    Split(SplitKernels),
}

impl std::fmt::Debug for Candidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Baseline => f.write_str("Baseline"),
            Self::Tiled { local, tiles, .. } => write!(f, "Tiled_local{local}_tiles{tiles}"),
            Self::Split(k) => write!(f, "Split_chunk{}", k.chunk),
        }
    }
}

fn profiles() -> [Profile; 3] {
    let [q5, iq2] = native_profiles();
    [
        q5,
        iq2,
        Profile {
            name: "iq3s_iq4nl_2560_640",
            gate_up: DType::Iq3S,
            ..iq2
        },
    ]
}

fn selected(name: &str, env: &str) -> bool {
    std::env::var(env).map_or(true, |value| value == "all" || name == value)
}

fn tiled_kernel(be: &VulkanBackend, p: Profile, local: usize, shared: bool) -> ComputeKernel {
    let root = PathBuf::from(std::env::var_os("MOE4ALL_HITFIRST_V3_DIR").unwrap());
    let suffix = if shared { "-shared" } else { "" };
    let file = format!("hit-tiled-{}-l{local}{suffix}.spv", format_name(p));
    let bytes = std::fs::read(root.join(&file)).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let key = Box::leak(format!("hitfirst_v3_{file}").into_boxed_str());
    be.kernel_sg(key, &words, 4, if shared { 64 } else { 40 }, 32)
}

fn record_one(f: &SplitFixture, rec: &Recorder<'_>, candidate: Candidate) {
    let inner = &f.ff.inner;
    // Pre-recorded production does not know whether the next route is all-hit.
    // The pure control is retained as a lower bound, not confused with this schedule.
    let copy_schedule = inner.job_count != 0
        || std::env::var("MOE4ALL_HITFIRST_RUNTIME_CONTROL").is_ok_and(|v| v == "1");
    let independent = copy_schedule.then(|| rec.independent_dispatch_scope());
    if copy_schedule {
        inner.cu_copy(rec);
    }
    match candidate {
        Candidate::Baseline => f.ff.projections(rec, 1, 0),
        Candidate::Tiled { kernel, tiles, .. } => {
            let mut push = f.push();
            push[9] = tiles as u32;
            let mut output = Recorder::vkb_exact(inner.output.as_ref());
            output.range -= GUARD as u64;
            rec.dispatch(
                kernel,
                &[
                    Recorder::vkb_exact(inner.input.as_ref()),
                    Recorder::vkb_exact(inner.ids.as_ref()),
                    Recorder::vkb_exact(inner.tape.as_ref()),
                    output,
                ],
                1,
                bytemuck::cast_slice(&push),
                (inner.rows * f.ff.physical_used() * tiles) as u32,
            );
        }
        Candidate::Split(k) => f.partial(rec, k),
    }
    drop(independent);
    match candidate {
        Candidate::Baseline => {
            f.ff.activation(rec);
            f.ff.down(rec, 1, 0);
        }
        Candidate::Split(k) => f.reduce(rec, k),
        Candidate::Tiled { .. } => {}
    }
    if inner.job_count != 0 {
        rec.global_barrier();
        f.ff.projections(rec, 2, 0);
        f.ff.activation(rec);
        f.ff.down(rec, 2, 0);
    }
}

fn frame_v3(
    be: &VulkanBackend,
    fixtures: &[SplitFixture],
    candidate: Candidate,
    query: &Timestamps<'_>,
) -> RecordedCmd {
    let rec = be.recorder_persistent().unwrap();
    rec.seed_barrier();
    query.begin(&rec);
    for _ in 0..REPEATS {
        for fixture in fixtures {
            record_one(fixture, &rec, candidate);
            rec.global_barrier();
        }
    }
    query.end(&rec);
    rec.publish_host_reads();
    rec.finish_record().unwrap()
}

fn strict_check(
    f: &SplitFixture,
    expected: &[u8],
    readonly: &[Vec<u8>],
    source: &Sources,
    bank: usize,
    candidate: Candidate,
) -> Errors {
    assert_eq!(
        f.readonly(),
        readonly,
        "readonly source, IDs, route and protocol"
    );
    assert_eq!(source.bytes(bank), f.ff.inner.expected_weights);
    if matches!(candidate, Candidate::Split(_)) {
        f.verify_partial();
    }
    let errors = f.ff.verify(expected);
    let actual = read(f.ff.inner.output.as_ref());
    let mut reference_max = 0.0f64;
    for (index, (a, b)) in actual[..actual.len() - GUARD]
        .chunks_exact(4)
        .zip(expected[..expected.len() - GUARD].chunks_exact(4))
        .enumerate()
    {
        let a = f32::from_le_bytes(a.try_into().unwrap()) as f64;
        let b = f32::from_le_bytes(b.try_into().unwrap()) as f64;
        reference_max = reference_max.max(b.abs());
        assert!(
            (a - b).abs() <= 2e-6 + 1e-4 * b.abs(),
            "strong H/M FFN {candidate:?} index={index} actual={a} expected={b}"
        );
    }
    assert!(
        reference_max > 1e-3,
        "avoid insignificant synthetic IQ values"
    );
    errors
}

fn candidates(be: &VulkanBackend, p: Profile, shared: bool) -> Vec<Candidate> {
    let mut out = Vec::new();
    if selected("tiled", "MOE4ALL_HITFIRST_V3_PATH") {
        let locals: &[usize] = match std::env::var("MOE4ALL_HITFIRST_V3_LOCAL").as_deref() {
            Ok("large") => &[512, 1024],
            Ok("512") => &[512],
            Ok("1024") => &[1024],
            _ => &[128, 256],
        };
        for &local in locals {
            let k = tiled_kernel(be, p, local, shared);
            for tiles in [1, 2, 4, 8, 16, 32] {
                if selected(&tiles.to_string(), "MOE4ALL_HITFIRST_V3_TILES") {
                    out.push(Candidate::Tiled {
                        kernel: k,
                        local,
                        tiles,
                    });
                }
            }
        }
    }
    if selected("split", "MOE4ALL_HITFIRST_V3_PATH") {
        // Only chunk32 can reproduce native Down's per32 K-block subgroup tree exactly.
        out.push(Candidate::Split(SplitKernels::new(be, p, 32, shared)));
    }
    out
}

#[test]
#[ignore = "parent serial GPU; native-shaped 35B Q5 and Flash IQ2/IQ3, rows1/2/4"]
fn gpu_owned_complete_hitfirst_v3_matrix() {
    let be = split_backend();
    let query = Timestamps::new(&be);
    for p in profiles() {
        if !selected(format_name(p), "MOE4ALL_HITFIRST_V3_FORMAT") {
            continue;
        }
        let shared = p.gate_up == DType::Q5K
            && !std::env::var("MOE4ALL_HITFIRST_NO_SHARED").is_ok_and(|v| v == "1");
        let source = Sources::new(&be, p, 0);
        for rows in [1, 2, 4] {
            if !selected(&rows.to_string(), "MOE4ALL_HITFIRST_V3_ROWS") {
                continue;
            }
            for mixed in [false, true] {
                let mode = if mixed { "one_miss" } else { "allhit" };
                if !selected(mode, "MOE4ALL_HITFIRST_V3_MODE") {
                    continue;
                }
                let mut fixtures = Vec::new();
                let mut expected = Vec::new();
                for bank in 0..source.banks {
                    let mut f = SplitFixture::new(&be, &source, p, rows, shared, 32);
                    let input: Vec<f32> = (0..rows * p.ne)
                        .map(|i| ((i % 17) as f32 - 8.0) * 4.0)
                        .collect();
                    write(f.ff.inner.input.as_ref(), bytemuck::cast_slice(&input));
                    expected.push(f.ff.reference(&be, &source, bank));
                    f.prepare(&source, bank, mixed, false);
                    fixtures.push(f);
                }
                let baseline = frame_v3(&be, &fixtures, Candidate::Baseline, &query);
                for candidate in candidates(&be, p, shared) {
                    for (bank, f) in fixtures.iter_mut().enumerate() {
                        f.prepare(&source, bank, mixed, false);
                    }
                    let readonly: Vec<_> = fixtures.iter().map(SplitFixture::readonly).collect();
                    let command = frame_v3(&be, &fixtures, candidate, &query);
                    command.replay().unwrap();
                    let mut numeric = true;
                    // A failed candidate remains evidence, not a reason to skip other layouts.
                    for (bank, f) in fixtures.iter().enumerate() {
                        numeric &= std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            strict_check(
                                f,
                                &expected[bank],
                                &readonly[bank],
                                &source,
                                bank,
                                candidate,
                            )
                        }))
                        .is_ok();
                    }
                    if !numeric {
                        println!(
                            "HITFIRST_V3_REJECT format={} rows={rows} mode={mode} candidate={candidate:?} reason=numeric_or_guard",
                            format_name(p)
                        );
                        continue;
                    }
                    for cmd in [&baseline, &command] {
                        let start = std::time::Instant::now();
                        while start.elapsed().as_millis() < 120 {
                            cmd.replay().unwrap();
                        }
                    }
                    let (base, new) =
                        measure_pair(&be, &baseline, &command, &query, fixtures.len() * REPEATS);
                    for (bank, f) in fixtures.iter_mut().enumerate() {
                        f.prepare(&source, bank, mixed, false);
                    }
                    command.replay().unwrap();
                    let mut errors = Errors::default();
                    for (bank, f) in fixtures.iter().enumerate() {
                        errors.merge(strict_check(
                            f,
                            &expected[bank],
                            &readonly[bank],
                            &source,
                            bank,
                            candidate,
                        ));
                    }
                    let (path, local, tiles) = match candidate {
                        Candidate::Tiled { local, tiles, .. } => ("tiled", local, tiles),
                        Candidate::Split(_) => ("split", 128, 0),
                        Candidate::Baseline => unreachable!(),
                    };
                    println!(
                        "HITFIRST_V3_MODULE format={} profile={} rows={rows} mode={mode} path={path} local={local} tiles={tiles} shared={shared} whole_payload=true input_scale=512 copy_local=1024 copy_groups=8 base_us={base:?} candidate_us={new:?} before_us={} after_us={} max_abs={} max_rel={} numeric=true banks={} repeats={REPEATS} scope=complete_H_M_no_trace_subtraction",
                        format_name(p),
                        p.name,
                        median(&base),
                        median(&new),
                        errors.max_abs,
                        errors.max_rel,
                        source.banks
                    );
                }
            }
        }
    }
}

fn event_frame(
    be: &VulkanBackend,
    fixtures: &[SplitFixture],
    resources: &[super::super::hit_event::HitEventResources],
    kind: &str,
    split: SplitKernels,
    query: &Timestamps<'_>,
) -> RecordedCmd {
    use super::super::hit_event;
    let rec = be.recorder_persistent().unwrap();
    rec.seed_barrier();
    query.begin(&rec);
    let batch = resources.len() != 1;
    assert_eq!(
        resources.len(),
        if batch { REPEATS * fixtures.len() } else { 1 }
    );
    if batch {
        hit_event::record_batch_preamble(&rec, resources);
    }
    let mut ordinal = 0;
    for _ in 0..REPEATS {
        for f in fixtures {
            let resources = &resources[if batch { ordinal } else { 0 }];
            ordinal += 1;
            match kind {
                "compact" => hit_event::record_complete_hit(&rec, &f.ff, resources),
                "native" => hit_event::record_producer_copy_consumer(
                    &rec,
                    &f.ff,
                    resources,
                    f.ff.inner.activation.as_ref(),
                    |r| {
                        f.ff.projections(r, 1, 0);
                        f.ff.activation(r);
                    },
                    |r| f.ff.down(r, 1, 0),
                ),
                "split" => hit_event::record_producer_copy_consumer(
                    &rec,
                    &f.ff,
                    resources,
                    f.partial.as_ref(),
                    |r| f.partial(r, split),
                    |r| f.reduce(r, split),
                ),
                _ => unreachable!(),
            }
            if f.ff.inner.job_count != 0 {
                f.ff.projections(&rec, 2, 0);
                f.ff.activation(&rec);
                f.ff.down(&rec, 2, 0);
            }
            rec.global_barrier();
        }
    }
    query.end(&rec);
    rec.publish_host_reads();
    rec.finish_record().unwrap()
}

#[test]
#[ignore = "parent serial GPU; complete H native Down or split reduction overlaps whole copy"]
fn gpu_owned_complete_hitfirst_v3_events() {
    use super::super::hit_event;
    let be = split_backend();
    let families = unsafe {
        be.shared
            .instance
            .get_physical_device_queue_family_properties(be.shared.physical_device)
    };
    eprintln!(
        "HITFIRST_QUEUES main={} families={:?}",
        be.shared.queue_family_index,
        families
            .iter()
            .map(|f| (f.queue_count, f.queue_flags))
            .collect::<Vec<_>>()
    );
    let query = Timestamps::new(&be);
    for p in profiles() {
        if !selected(format_name(p), "MOE4ALL_HITFIRST_V3_FORMAT") {
            continue;
        }
        let shared = p.gate_up == DType::Q5K
            && !std::env::var("MOE4ALL_HITFIRST_NO_SHARED").is_ok_and(|v| v == "1");
        let source = Sources::new(&be, p, 0);
        let split = SplitKernels::new(&be, p, 32, shared);
        for rows in [1, 2, 4] {
            if !selected(&rows.to_string(), "MOE4ALL_HITFIRST_V3_ROWS") {
                continue;
            }
            for mixed in [false, true] {
                let mode = if mixed { "one_miss" } else { "allhit" };
                if !selected(mode, "MOE4ALL_HITFIRST_V3_MODE") {
                    continue;
                }
                let mut fixtures = Vec::new();
                let mut expected = Vec::new();
                for bank in 0..source.banks {
                    let mut f = SplitFixture::new(&be, &source, p, rows, shared, 32);
                    let input: Vec<f32> = (0..rows * p.ne)
                        .map(|i| ((i % 17) as f32 - 8.0) * 4.0)
                        .collect();
                    write(f.ff.inner.input.as_ref(), bytemuck::cast_slice(&input));
                    expected.push(f.ff.reference(&be, &source, bank));
                    f.prepare(&source, bank, mixed, false);
                    fixtures.push(f);
                }
                let split_baseline =
                    std::env::var("MOE4ALL_HITFIRST_EVENT_BASELINE").is_ok_and(|v| v == "split");
                let baseline = frame_v3(
                    &be,
                    &fixtures,
                    if split_baseline {
                        Candidate::Split(split)
                    } else {
                        Candidate::Baseline
                    },
                    &query,
                );
                for kind in ["native", "compact", "split"] {
                    if !selected(kind, "MOE4ALL_HITFIRST_EVENT_KIND") {
                        continue;
                    }
                    let variants: &[(usize, usize)] = if kind == "compact" {
                        &[(64, 2), (64, 8), (128, 4), (256, 2)]
                    } else {
                        &[(0, 0)]
                    };
                    for &(local, nr) in variants {
                        if kind == "compact"
                            && !selected(&format!("{local}x{nr}"), "MOE4ALL_HITFIRST_EVENT_SHAPE")
                        {
                            continue;
                        }
                        let batch_events =
                            std::env::var("MOE4ALL_HITFIRST_BATCH_EVENTS").is_ok_and(|v| v == "1");
                        let count = if batch_events {
                            REPEATS * fixtures.len()
                        } else {
                            1
                        };
                        let resources: Vec<_> = (0..count)
                            .map(|_| {
                                if kind == "compact" {
                                    hit_event::HitEventResources::new(&be, p, shared, local, nr)
                                } else {
                                    hit_event::HitEventResources::event_only(&be)
                                }
                            })
                            .collect();
                        for (bank, f) in fixtures.iter_mut().enumerate() {
                            f.prepare(&source, bank, mixed, false);
                        }
                        let readonly: Vec<_> =
                            fixtures.iter().map(SplitFixture::readonly).collect();
                        let candidate = if kind == "split" {
                            Candidate::Split(split)
                        } else {
                            Candidate::Baseline
                        };
                        let command = event_frame(&be, &fixtures, &resources, kind, split, &query);
                        command.replay().unwrap();
                        let mut numeric = true;
                        for (bank, f) in fixtures.iter().enumerate() {
                            numeric &=
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    strict_check(
                                        f,
                                        &expected[bank],
                                        &readonly[bank],
                                        &source,
                                        bank,
                                        candidate,
                                    )
                                }))
                                .is_ok();
                        }
                        if !numeric {
                            println!(
                                "HITFIRST_V3_REJECT format={} rows={rows} mode={mode} candidate=event_{kind}_l{local}_nr{nr} reason=numeric_or_guard",
                                format_name(p)
                            );
                            continue;
                        }
                        for cmd in [&baseline, &command] {
                            let start = std::time::Instant::now();
                            while start.elapsed().as_millis() < 120 {
                                cmd.replay().unwrap();
                            }
                        }
                        let (base, new) = measure_pair(
                            &be,
                            &baseline,
                            &command,
                            &query,
                            fixtures.len() * REPEATS,
                        );
                        for (bank, f) in fixtures.iter_mut().enumerate() {
                            f.prepare(&source, bank, mixed, false);
                        }
                        command.replay().unwrap();
                        let mut errors = Errors::default();
                        for (bank, f) in fixtures.iter().enumerate() {
                            errors.merge(strict_check(
                                f,
                                &expected[bank],
                                &readonly[bank],
                                &source,
                                bank,
                                candidate,
                            ));
                        }
                        println!(
                            "HITFIRST_V3_MODULE format={} profile={} rows={rows} mode={mode} path=event_{kind} local={local} tiles={nr} shared={shared} batch_events={batch_events} split_baseline={split_baseline} whole_payload=true input_scale=512 copy_local=1024 copy_groups=8 base_us={base:?} candidate_us={new:?} before_us={} after_us={} max_abs={} max_rel={} numeric=true banks={} repeats={REPEATS} scope=complete_H_M_reset_wait_and_join_timed",
                            format_name(p),
                            p.name,
                            median(&base),
                            median(&new),
                            errors.max_abs,
                            errors.max_rel,
                            source.banks
                        );
                    }
                }
            }
        }
    }
}
